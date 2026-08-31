/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package causalsloexternality

import (
	"sort"
	"sync"
	"time"
)

// THE SHADOW TABLE -- signals 12-18, degradations D2, D4, D6.
//
// The externality is a sum over INDIVIDUAL residents' deadline slack
// (sim/edpp_var.go:615-641 reads per-resident StepsDone, ArrivalUs, FirstTokenUs,
// TTFTSet). vLLM exports resident state only in AGGREGATE, so this index tracks the
// requests this EPP placed.
//
// Three design points are load-bearing rather than stylistic:
//
//   - A MUTEX-GUARDED LONG-LIVED INDEX, not framework per-request state. Entries are
//     written from the async response-body goroutine (non-final chunks drain from a
//     queue, director.go:601-632) and read from the scheduling path, and the index
//     outlives any one request. llm-d's PluginState is explicitly not an option: its
//     own doc says "PluginState is not a cross-plugin handoff channel"
//     (plugin_state.go:75-77) and it is keyed per request.
//
//   - A TTL SWEEP IS REQUIRED, NOT HYGIENE. Requests terminate without a final chunk
//     (client disconnect, upstream error) and the ResponseBodyProcessor signature
//     carries no error or termination state, so a disconnect is indistinguishable
//     from a completion. Without the sweep those entries are charged as residents
//     forever and permanently inflate every externality on their endpoint.
//
//   - A REPEATED REQUEST ID MUST REPLACE, so a retried request is not counted twice.
//
// StepsDone additionally requires the engine flag --enable-force-include-usage
// (config.md section 11). Without it, usage arrives only in the FINAL chunk, so StepsDone
// stays 0 for every request's whole lifetime and every remaining-steps estimate is wrong
// WHILE THE TABLE LOOKS PRESENT AND CORRECT.
//
// THE PLUGIN CANNOT DETECT THAT, and nothing here claims to: a population whose StepsDone
// is entirely zero is also exactly what a freshly started fleet looks like, so there is no
// threshold that separates the two from inside. It is an operator requirement, listed in
// the package README's runtime-requirements table, not a runtime check. The nearest
// available evidence is indirect -- residents accumulate in the shadow-table gauge while
// contributing no decode-side externality.

// residentEntry is one tracked request.
type residentEntry struct {
	requestID string
	// decodeID is the endpoint the decode leg runs on -- the endpoint this entry is a
	// resident OF.
	decodeID string
	// prefillID is the endpoint the prefill leg ran on, empty for a local placement.
	// A resident whose prefill ran REMOTELY occupies no prefill capacity on its
	// decode endpoint and must be skipped there rather than double-counted, so this
	// field is what keeps the prefill index honest.
	prefillID string

	sloClass     string
	promptTokens int64

	arrivalUs    int64
	firstTokenUs int64
	ttftSet      bool
	stepsDone    int64

	// kvBlocks is the request's KV footprint in ENGINE blocks, used by the
	// rollforward estimator's departure accounting.
	kvBlocks int64

	lastSeen time.Time
}

// shadowTable is the mutex-guarded index. Keyed by request ID; the per-endpoint views
// are computed on read, because a request's endpoint never changes after placement and
// the populations are small (bounded by max_num_seqs per endpoint in the healthy case).
type shadowTable struct {
	mu      sync.Mutex
	entries map[string]*residentEntry

	ttl              time.Duration
	prefillTokensCap int64
	// blockSize is the ENGINE block size, needed to keep each resident's KV footprint
	// current as it decodes -- see observeChunk.
	blockSize int64

	// now is injectable so tests can drive the TTL sweep deterministically.
	now func() time.Time

	metrics *pluginMetrics

	stopOnce sync.Once
	stopCh   chan struct{}
}

func newShadowTable(cfg ShadowTable, blockSize int64, metrics *pluginMetrics) *shadowTable {
	return &shadowTable{
		entries:          map[string]*residentEntry{},
		ttl:              time.Duration(cfg.EntryTTLSeconds) * time.Second,
		prefillTokensCap: cfg.ResidentPrefillTokensCap,
		blockSize:        blockSize,
		now:              time.Now,
		metrics:          metrics,
		stopCh:           make(chan struct{}),
	}
}

// startSweeper runs the TTL reaper until stop is called.
func (t *shadowTable) startSweeper(interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-t.stopCh:
				return
			case <-ticker.C:
				t.sweep()
			}
		}
	}()
}

func (t *shadowTable) stop() {
	t.stopOnce.Do(func() { close(t.stopCh) })
}

// place records a new resident at decision time. A repeated request ID REPLACES, so a
// retried request is not counted twice.
func (t *shadowTable) place(e residentEntry) {
	t.mu.Lock()
	defer t.mu.Unlock()
	e.lastSeen = t.now()
	entry := e
	t.entries[e.requestID] = &entry
	t.metrics.setShadowTableSize(len(t.entries))
}

// observeChunk advances a resident from a response chunk.
//
// completionTokens is Usage.CompletionTokens, which is MONOTONIC per request and
// requires --enable-force-include-usage to arrive before the final chunk. The first
// non-zero value marks the realized first token -- degradation D2c: that instant is a
// DEQUEUE instant, so realized TTFT is late, which shrinks a common positive
// multiplier on every charge and biases toward local.
func (t *shadowTable) observeChunk(requestID string, completionTokens int, atUs int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	entry := t.entries[requestID]
	if entry == nil {
		return
	}
	entry.lastSeen = t.now()
	if completionTokens <= 0 {
		return
	}
	if !entry.ttftSet {
		entry.ttftSet = true
		entry.firstTokenUs = atUs
	}
	if int64(completionTokens) > entry.stepsDone {
		entry.stepsDone = int64(completionTokens)
	}
	// ADVANCE THE KV FOOTPRINT as the resident decodes. Its context grows one token per
	// output token, so a resident 300 tokens into decode holds ceil((prompt+300)/blockSize)
	// blocks, not the prompt-only count it was placed with.
	//
	// This is not cosmetic bookkeeping: the value feeds rollforwardEstimateTAdm's
	// `freeKV += d.kv`, so a stale prompt-only count UNDER-states the KV a departure
	// reclaims => the KV condition is satisfied later or falls through to the wave form =>
	// tAdm is over-stated => admissionSteps grows => more residents finish inside the
	// admission window => the externality is under-counted => the local candidate is
	// UNDER-priced, biasing toward LOCAL.
	//
	// Unlike the declared degradations this one is avoidable: prompt length and StepsDone
	// are both already tracked, so the simulation's value is computable here.
	if entry.kvBlocks < ceilBlocks(entry.promptTokens+entry.stepsDone, t.blockSize) {
		entry.kvBlocks = ceilBlocks(entry.promptTokens+entry.stepsDone, t.blockSize)
	}
}

// complete removes a resident on its terminal response-body observation and reports the
// output length seen, so the caller can fold it into the per-class mean.
//
// It CANNOT confirm the request completed -- degradation D9, documented at the call site in
// Handler.ResponseBody. An aborted request reaches here too, carrying a truncated count.
func (t *shadowTable) complete(requestID string) (class string, outputTokens int, ok bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	entry := t.entries[requestID]
	if entry == nil {
		return "", 0, false
	}
	delete(t.entries, requestID)
	t.metrics.setShadowTableSize(len(t.entries))
	return entry.sloClass, int(entry.stepsDone), true
}

// sweep reaps entries with no activity inside the TTL.
//
// Every reaped entry is COUNTED. A reap is not necessarily an error -- a client
// disconnect is legitimate -- but a rising reap rate means residents are being charged
// for requests that are gone, so the count is what distinguishes a healthy table from
// one silently inflating every externality on an endpoint.
func (t *shadowTable) sweep() {
	t.mu.Lock()
	defer t.mu.Unlock()
	cutoff := t.now().Add(-t.ttl)
	reaped := 0
	for id, entry := range t.entries {
		if entry.lastSeen.Before(cutoff) {
			delete(t.entries, id)
			reaped++
		}
	}
	if reaped > 0 {
		t.metrics.addShadowTableReaped(reaped)
	}
	t.metrics.setShadowTableSize(len(t.entries))
}

// residentsFor returns the two resident populations for one decode endpoint.
//
// The split is by first-token state, not by phase label, and it is forced by the value
// kernels: gDecodeComposite returns 0 for a resident with no first token, so such a
// request would be MISSED ENTIRELY if it were not moved to the prefill population,
// where its first token is still live.
//
// A resident whose prefill ran REMOTELY is skipped in the prefill population: it
// occupies no prefill capacity on its decode endpoint, so counting it there would
// double-charge the placement that already paid for it on the pool.
func (t *shadowTable) residentsFor(decodeID string) (decode, prefill []RunningReqState) {
	t.mu.Lock()
	defer t.mu.Unlock()
	decode, prefill = t.residentsForLocked(decodeID)
	return decode, prefill
}

// residentsForLocked is residentsFor's body. Caller must hold t.mu.
func (t *shadowTable) residentsForLocked(decodeID string) (decode, prefill []RunningReqState) {
	for _, e := range t.entries {
		if e.decodeID != decodeID {
			continue
		}
		if e.ttftSet {
			decode = append(decode, RunningReqState{
				RequestID:     e.requestID,
				StepsDone:     e.stepsDone,
				KVBlocks:      e.kvBlocks,
				TrueRemaining: -1, // deployable: the oracle is never read
				SLOClass:      e.sloClass,
				ArrivalUs:     e.arrivalUs,
				FirstTokenUs:  e.firstTokenUs,
				TTFTSet:       true,
			})
			continue
		}
		// Pre-first-token occupant -- degradation D6. Skipped here when its prefill
		// is remote.
		if e.prefillID != "" {
			continue
		}
		prefill = append(prefill, RunningReqState{
			RequestID: e.requestID,
			StepsDone: 0,
			KVBlocks:  e.kvBlocks,
			// On the PREFILL slice TrueRemaining carries remaining PROMPT tokens,
			// which is known input and needs no censoring. The occupant has produced
			// no output, so its whole prompt is still outstanding.
			TrueRemaining: e.promptTokens,
			SLOClass:      e.sloClass,
			ArrivalUs:     e.arrivalUs,
			TTFTSet:       false,
		})
	}
	// SORT BY REQUEST ID BEFORE RETURNING. The map above iterates in randomized order, and
	// "stable" in sort.SliceStable only means "preserves input order" -- so a randomized
	// input makes the estimator's documented determinism contract vacuous. It bites hardest
	// on the PREFILL slice, where TrueRemaining carries per-occupant prompt tokens and
	// therefore genuinely differs: two occupants with equal prompt length but different KV
	// holdings would otherwise resolve in a random order and change which departure first
	// satisfies the KV condition. It also removes float-summation order variance from the
	// externality sums.
	sortResidents(decode)
	sortResidents(prefill)
	return decode, prefill
}

// sortResidents orders a population deterministically by request ID.
func sortResidents(rs []RunningReqState) {
	sort.Slice(rs, func(i, j int) bool { return rs[i].RequestID < rs[j].RequestID })
}

// prefillOccupantsFor returns the pre-first-token population on a PREFILL POOL
// endpoint -- the requests a prior decision deflected there that have not yet produced
// a first token.
func (t *shadowTable) prefillOccupantsFor(prefillID string) []RunningReqState {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.prefillOccupantsForLocked(prefillID)
}

// prefillOccupantsForLocked is prefillOccupantsFor's body. Caller must hold t.mu.
func (t *shadowTable) prefillOccupantsForLocked(prefillID string) []RunningReqState {
	var out []RunningReqState
	for _, e := range t.entries {
		if e.prefillID != prefillID || e.ttftSet {
			continue
		}
		out = append(out, RunningReqState{
			RequestID:     e.requestID,
			StepsDone:     0,
			KVBlocks:      e.kvBlocks,
			TrueRemaining: e.promptTokens,
			SLOClass:      e.sloClass,
			ArrivalUs:     e.arrivalUs,
			TTFTSet:       false,
		})
	}
	sortResidents(out)
	return out
}

// residentPrefillTokens returns S_pf for an endpoint -- signal 14, degradation D4.
//
// S_pf is what is being prefilled THIS STEP, not the outstanding backlog, and the EPP
// cannot know what the engine scheduled. So the sum is capped PER OCCUPANT and capped
// IN TOTAL at the configured cap (= max_num_batched_tokens). It remains an
// OVER-estimate, which over-states local prefill inflation and biases toward remote.
func (t *shadowTable) residentPrefillTokens(endpointID string) int64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.residentPrefillTokensLocked(endpointID)
}

// residentPrefillTokensLocked is residentPrefillTokens' body. Caller must hold t.mu.
func (t *shadowTable) residentPrefillTokensLocked(endpointID string) int64 {
	cap0 := t.prefillTokensCap
	var total int64
	for _, e := range t.entries {
		if e.ttftSet {
			continue
		}
		// Count the occupant against the endpoint actually doing its prefill.
		prefillHost := e.prefillID
		if prefillHost == "" {
			prefillHost = e.decodeID
		}
		if prefillHost != endpointID {
			continue
		}
		remaining := e.promptTokens
		if remaining > cap0 {
			remaining = cap0
		}
		total += remaining
		if total >= cap0 {
			return cap0
		}
	}
	return total
}

// viewFor returns everything one endpoint's Snapshot needs under a SINGLE lock hold.
//
// Taking the lock three times would let a response chunk land between the calls, so S_pf
// could describe a resident population the same snapshot does not contain -- an
// inconsistency the objective would then price as if it were real.
func (t *shadowTable) viewFor(endpointID string) (decode, prefill []RunningReqState, prefillTokens int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	decode, prefill = t.residentsForLocked(endpointID)
	// A prefill-pool endpoint's occupants are the requests deflected TO it, which are
	// indexed by prefillID rather than decodeID.
	if pool := t.prefillOccupantsForLocked(endpointID); len(pool) > 0 {
		prefill = append(prefill, pool...)
		sortResidents(prefill)
	}
	return decode, prefill, t.residentPrefillTokensLocked(endpointID)
}

// size reports the entry count, for the gauge and for tests.
func (t *shadowTable) size() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.entries)
}

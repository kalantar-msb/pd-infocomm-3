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
	"sync"
	"testing"
	"time"
)

func testTable(t *testing.T) *shadowTable {
	t.Helper()
	return newShadowTable(ShadowTable{
		EntryTTLSeconds:          900,
		SweepIntervalSeconds:     30,
		ResidentPrefillTokensCap: 2048,
	}, newPluginMetrics("test", HandlerPluginType))
}

func TestShadowTablePlaceAndSize(t *testing.T) {
	tbl := testTable(t)
	tbl.place(residentEntry{requestID: "r1", decodeID: "d1", sloClass: sloClassStandard, promptTokens: 1000})
	tbl.place(residentEntry{requestID: "r2", decodeID: "d1", sloClass: sloClassStandard, promptTokens: 2000})
	if got := tbl.size(); got != 2 {
		t.Errorf("size = %d, want 2", got)
	}
}

// TestShadowTableRepeatedRequestIDReplaces is the retry property: a repeated request ID
// must REPLACE so a retried request is not counted twice, which would double-charge every
// externality on its endpoint.
func TestShadowTableRepeatedRequestIDReplaces(t *testing.T) {
	tbl := testTable(t)
	tbl.place(residentEntry{requestID: "r1", decodeID: "d1", promptTokens: 1000})
	tbl.place(residentEntry{requestID: "r1", decodeID: "d2", promptTokens: 4000})

	if got := tbl.size(); got != 1 {
		t.Fatalf("a repeated request ID must replace, not add: size = %d", got)
	}
	// And the replacement's fields win.
	if decode, _ := tbl.residentsFor("d2"); len(decode) != 0 {
		t.Error("the replaced entry has no first token yet, so it is not a decode resident")
	}
	_, prefill := tbl.residentsFor("d2")
	if len(prefill) != 1 || prefill[0].TrueRemaining != 4000 {
		t.Errorf("replacement fields must win, got %+v", prefill)
	}
}

// TestShadowTableSplitsByFirstTokenState pins the split the value kernels force: a
// resident with no first token returns 0 from gDecodeComposite, so it must be carried in
// the PREFILL population where its first token is still live, or it would be missed
// entirely.
func TestShadowTableSplitsByFirstTokenState(t *testing.T) {
	tbl := testTable(t)
	tbl.place(residentEntry{requestID: "decoding", decodeID: "d1", promptTokens: 1000, sloClass: sloClassStandard})
	tbl.place(residentEntry{requestID: "prefilling", decodeID: "d1", promptTokens: 2000, sloClass: sloClassStandard})

	// Advance one of them past its first token.
	tbl.observeChunk("decoding", 5, 1_000_000)

	decode, prefill := tbl.residentsFor("d1")
	if len(decode) != 1 || !decode[0].TTFTSet || decode[0].StepsDone != 5 {
		t.Errorf("decode population wrong: %+v", decode)
	}
	if len(prefill) != 1 || prefill[0].TTFTSet {
		t.Errorf("prefill population wrong: %+v", prefill)
	}
	// The prefill occupant carries remaining PROMPT tokens in TrueRemaining, which is
	// known input and needs no censoring.
	if prefill[0].TrueRemaining != 2000 {
		t.Errorf("prefill occupant TrueRemaining = %d, want the 2000 prompt tokens", prefill[0].TrueRemaining)
	}
	// The decode resident is always censored: the oracle is never read.
	if decode[0].TrueRemaining != -1 {
		t.Errorf("decode resident TrueRemaining = %d, want -1 (censored)", decode[0].TrueRemaining)
	}
}

// TestRemotelyPrefillingResidentNotChargedOnDecodeEndpoint pins the double-count guard: a
// resident whose prefill runs REMOTELY occupies no prefill capacity on its decode
// endpoint, so it must be skipped there.
func TestRemotelyPrefillingResidentNotChargedOnDecodeEndpoint(t *testing.T) {
	tbl := testTable(t)
	// A disaggregated request: decode on d1, prefill on p1, no first token yet.
	tbl.place(residentEntry{requestID: "remote", decodeID: "d1", prefillID: "p1", promptTokens: 4000})

	_, prefillOnDecode := tbl.residentsFor("d1")
	if len(prefillOnDecode) != 0 {
		t.Errorf("a remotely-prefilling resident must not occupy prefill capacity on its decode endpoint, got %+v",
			prefillOnDecode)
	}
	// It IS an occupant of the prefill pool endpoint.
	occupants := tbl.prefillOccupantsFor("p1")
	if len(occupants) != 1 || occupants[0].TrueRemaining != 4000 {
		t.Errorf("prefill pool occupants wrong: %+v", occupants)
	}

	// A locally-placed request, by contrast, DOES occupy prefill capacity on its decode
	// endpoint.
	tbl.place(residentEntry{requestID: "local", decodeID: "d1", promptTokens: 1500})
	_, localPrefill := tbl.residentsFor("d1")
	if len(localPrefill) != 1 || localPrefill[0].TrueRemaining != 1500 {
		t.Errorf("a locally-placed resident must occupy prefill capacity on its decode endpoint, got %+v",
			localPrefill)
	}
}

// TestObserveChunkRecordsFirstTokenOnce pins D2c's mechanics: the FIRST non-zero
// completion count marks the realized first token, and later chunks must not move it.
func TestObserveChunkRecordsFirstTokenOnce(t *testing.T) {
	tbl := testTable(t)
	tbl.place(residentEntry{requestID: "r1", decodeID: "d1", promptTokens: 100})

	// A zero-usage chunk does not set the first token. This is the shape of every chunk
	// when --enable-force-include-usage is missing.
	tbl.observeChunk("r1", 0, 1_000)
	if decode, _ := tbl.residentsFor("d1"); len(decode) != 0 {
		t.Fatal("a zero-usage chunk must not mark a first token")
	}

	tbl.observeChunk("r1", 1, 5_000)
	tbl.observeChunk("r1", 7, 9_000)

	decode, _ := tbl.residentsFor("d1")
	if len(decode) != 1 {
		t.Fatalf("expected one decode resident, got %d", len(decode))
	}
	if decode[0].FirstTokenUs != 5_000 {
		t.Errorf("first token instant = %d, want the first non-zero chunk at 5000", decode[0].FirstTokenUs)
	}
	if decode[0].StepsDone != 7 {
		t.Errorf("StepsDone = %d, want the latest count 7", decode[0].StepsDone)
	}
}

// TestObserveChunkStepsDoneIsMonotonic guards against a stale repeated Usage value pulling
// the count backwards: reqCtx.Usage retains its previous value when a chunk carried no
// usage block, so the same number can arrive twice.
func TestObserveChunkStepsDoneIsMonotonic(t *testing.T) {
	tbl := testTable(t)
	tbl.place(residentEntry{requestID: "r1", decodeID: "d1", promptTokens: 100})
	tbl.observeChunk("r1", 10, 1_000)
	tbl.observeChunk("r1", 3, 2_000) // a stale or out-of-order value
	decode, _ := tbl.residentsFor("d1")
	if decode[0].StepsDone != 10 {
		t.Errorf("StepsDone must not regress: got %d, want 10", decode[0].StepsDone)
	}
}

func TestObserveChunkUnknownRequestIsIgnored(t *testing.T) {
	tbl := testTable(t)
	tbl.observeChunk("never-placed", 5, 1_000) // must not panic or create an entry
	if got := tbl.size(); got != 0 {
		t.Errorf("size = %d, want 0", got)
	}
}

// TestCompleteRemovesAndReportsOutputLength is how the per-class N_out mean is fed --
// signal 17, and only from requests that actually completed.
func TestCompleteRemovesAndReportsOutputLength(t *testing.T) {
	tbl := testTable(t)
	tbl.place(residentEntry{requestID: "r1", decodeID: "d1", sloClass: sloClassStandard, promptTokens: 100})
	tbl.observeChunk("r1", 42, 1_000)

	class, outputTokens, ok := tbl.complete("r1")
	if !ok {
		t.Fatal("complete must report a tracked request")
	}
	if class != sloClassStandard || outputTokens != 42 {
		t.Errorf("complete = (%q, %d), want (%q, 42)", class, outputTokens, sloClassStandard)
	}
	if got := tbl.size(); got != 0 {
		t.Errorf("the entry must be removed, size = %d", got)
	}
	if _, _, ok := tbl.complete("r1"); ok {
		t.Error("completing twice must report not-found")
	}
}

// THE TTL SWEEP, and it is required rather than hygiene.
//
// Requests terminate without a final chunk (client disconnect, upstream error) and the
// ResponseBodyProcessor signature carries no termination state, so without the sweep those
// entries are charged as residents FOREVER and permanently inflate every externality on
// their endpoint.
func TestSweepReapsStrandedEntriesAndCountsThem(t *testing.T) {
	tbl := testTable(t)
	base := time.Now()
	tbl.now = func() time.Time { return base }

	tbl.place(residentEntry{requestID: "stranded", decodeID: "d1", promptTokens: 100})
	tbl.place(residentEntry{requestID: "healthy", decodeID: "d1", promptTokens: 100})

	// Advance past the TTL, then refresh only the healthy one.
	tbl.now = func() time.Time { return base.Add(1000 * time.Second) }
	tbl.observeChunk("healthy", 3, 1_000)

	tbl.sweep()

	if got := tbl.size(); got != 1 {
		t.Fatalf("expected exactly the stranded entry to be reaped, size = %d", got)
	}
	if _, _, ok := tbl.complete("healthy"); !ok {
		t.Error("the healthy entry must survive the sweep")
	}
}

// TestSweepIsANoOpWhenNothingIsStale guards against reaping live residents.
func TestSweepIsANoOpWhenNothingIsStale(t *testing.T) {
	tbl := testTable(t)
	base := time.Now()
	tbl.now = func() time.Time { return base }
	tbl.place(residentEntry{requestID: "r1", decodeID: "d1", promptTokens: 100})
	tbl.sweep()
	if got := tbl.size(); got != 1 {
		t.Errorf("a fresh entry must not be reaped, size = %d", got)
	}
}

// TestResidentPrefillTokensIsCappedPerOccupantAndInTotal pins degradation D4's two caps.
// S_pf is what is being prefilled THIS STEP, not the outstanding backlog, and the EPP
// cannot know what the engine scheduled -- so the sum is capped and remains an
// over-estimate, biasing toward remote.
func TestResidentPrefillTokensIsCappedPerOccupantAndInTotal(t *testing.T) {
	tbl := testTable(t) // cap = 2048

	// One occupant with a deep-research-scale prompt: capped per occupant.
	tbl.place(residentEntry{requestID: "huge", decodeID: "d1", promptTokens: 45000})
	if got := tbl.residentPrefillTokens("d1"); got != 2048 {
		t.Errorf("per-occupant cap: got %d, want 2048", got)
	}

	// Several modest occupants: capped in total.
	tbl2 := testTable(t)
	for _, id := range []string{"a", "b", "c", "d", "e"} {
		tbl2.place(residentEntry{requestID: id, decodeID: "d1", promptTokens: 900})
	}
	if got := tbl2.residentPrefillTokens("d1"); got != 2048 {
		t.Errorf("total cap: got %d, want 2048", got)
	}

	// Under the cap the true sum is reported.
	tbl3 := testTable(t)
	tbl3.place(residentEntry{requestID: "a", decodeID: "d1", promptTokens: 300})
	tbl3.place(residentEntry{requestID: "b", decodeID: "d1", promptTokens: 500})
	if got := tbl3.residentPrefillTokens("d1"); got != 800 {
		t.Errorf("under the cap: got %d, want 800", got)
	}
}

// TestResidentPrefillTokensCountsAgainstThePrefillingEndpoint confirms a deflected
// request's prefill tokens land on the POOL endpoint, not on its decode endpoint.
func TestResidentPrefillTokensCountsAgainstThePrefillingEndpoint(t *testing.T) {
	tbl := testTable(t)
	tbl.place(residentEntry{requestID: "remote", decodeID: "d1", prefillID: "p1", promptTokens: 500})
	if got := tbl.residentPrefillTokens("d1"); got != 0 {
		t.Errorf("a deflected request must not charge prefill tokens to its decode endpoint, got %d", got)
	}
	if got := tbl.residentPrefillTokens("p1"); got != 500 {
		t.Errorf("it must charge them to the prefill pool endpoint, got %d", got)
	}
}

// TestResidentPrefillTokensExcludesDecodingResidents pins that S_pf counts only
// pre-first-token occupants: a request already decoding is not being prefilled.
func TestResidentPrefillTokensExcludesDecodingResidents(t *testing.T) {
	tbl := testTable(t)
	tbl.place(residentEntry{requestID: "r1", decodeID: "d1", promptTokens: 500})
	tbl.observeChunk("r1", 1, 1_000) // now decoding
	if got := tbl.residentPrefillTokens("d1"); got != 0 {
		t.Errorf("a decoding resident contributes no prefill tokens, got %d", got)
	}
}

// TestShadowTableIsRaceFree exercises the concurrency the target's own dispatch forces:
// non-final response chunks drain on a background goroutine while the scheduling path
// reads the table. Run under -race, which the build gate does.
func TestShadowTableIsRaceFree(t *testing.T) {
	tbl := testTable(t)
	const n = 200
	var wg sync.WaitGroup

	// Writers: place and advance, as PreRequest and ResponseBody would.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < n; i++ {
			id := string(rune('a' + i%26))
			tbl.place(residentEntry{requestID: id, decodeID: "d1", promptTokens: 100, sloClass: sloClassStandard})
			tbl.observeChunk(id, i, int64(i)*1000)
		}
	}()

	// Readers: the scheduling path.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < n; i++ {
			tbl.residentsFor("d1")
			tbl.residentPrefillTokens("d1")
			tbl.prefillOccupantsFor("p1")
			tbl.size()
		}
	}()

	// A concurrent sweeper.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 20; i++ {
			tbl.sweep()
		}
	}()

	wg.Wait()
}

func TestShadowTableStopIsIdempotent(t *testing.T) {
	tbl := testTable(t)
	tbl.startSweeper(time.Millisecond)
	tbl.stop()
	tbl.stop() // must not panic on a second close
}

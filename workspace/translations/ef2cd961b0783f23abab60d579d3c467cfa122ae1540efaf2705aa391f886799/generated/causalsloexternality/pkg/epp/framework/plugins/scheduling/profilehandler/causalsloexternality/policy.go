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
	"math"
	"sort"
	"sync"
)

// sloClassStandard is the one SLO class every cohort of all three workloads
// declares (workloads/interactive-chat-single-turn.yaml:21 and the other two). It is
// the fallback when a request carries no objective, so an unlabelled request resolves
// the same triple as a labelled one rather than an empty-class zero triple.
const sloClassStandard = "standard"

// Policy holds the arm's long-lived routing state.
//
// It is shared by BOTH plugin registrations -- the handler owns it and the picker
// holds a pointer, resolved at factory time by name. Every field below is guarded by
// mu, because the response-body hooks that write nHatOut run on a different goroutine
// from the scheduling path that reads it (non-final chunks drain from an async queue,
// director.go:541-552, while the final chunk runs on the request goroutine at :539).
type Policy struct {
	cfg Config

	// table is the EPP-side resident index -- signals 12-18, degradation D2.
	table *shadowTable

	// metrics carries the counters every declared degradation needs to be visible.
	metrics *pluginMetrics

	mu sync.Mutex

	// nHatOut is the per-class running mean of realized output lengths.
	// sim/edpp.go:412-425.
	//
	// SEEDED AT 1, NOT 0 (mean() returns 1 when n == 0): a zero seed makes the
	// decode demand vanish and collapses every remaining-steps floor.
	//
	// Updated ONLY on requests that actually COMPLETED. A request reaching a
	// terminal state without completing carries no realized output length, and
	// folding its truncated count in would drag the estimate down.
	nHatOut map[string]*runningMean

	// sloCapacity holds the per-endpoint virtual workload queues. DEAD STATE in this
	// arm, which disables the capacity term.
	sloCapacity         map[string]*capacityState
	sloCapacityClock    int64
	sloCapacityClockSet bool

	// gpuTypeByID maps endpoint ID to its GPU-type label value for the current
	// decision. Rebuilt per decision from the snapshots, so the candidate score and
	// the capacity commit cannot select different physics for the same endpoint.
	gpuTypeByID map[string]string
}

// newPolicy builds a Policy over a validated Config.
func newPolicy(cfg Config, table *shadowTable, metrics *pluginMetrics) *Policy {
	return &Policy{
		cfg:         cfg,
		table:       table,
		metrics:     metrics,
		nHatOut:     map[string]*runningMean{},
		sloCapacity: map[string]*capacityState{},
		gpuTypeByID: map[string]string{},
	}
}

// runningMean is a per-class running mean. sim/edpp.go:412-425.
type runningMean struct {
	n   int64
	sum float64
}

func (r *runningMean) update(v float64) { r.n++; r.sum += v }

func (r *runningMean) mean() float64 {
	if r.n == 0 {
		return 1 // no completions yet: conservative 1-token seed, NOT 0
	}
	return r.sum / float64(r.n)
}

// observeCompletedOutput folds a COMPLETED request's realized output length into the
// per-class mean. Callers must not call it for a request that terminated early.
func (p *Policy) observeCompletedOutput(class string, outputTokens int) {
	if outputTokens <= 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	m := p.nHatOut[class]
	if m == nil {
		m = &runningMean{}
		p.nHatOut[class] = m
	}
	m.update(float64(outputTokens))
}

// nHatFor reads the per-class mean. Caller must hold p.mu, or be inside a decision
// that already snapshotted it -- see nHatSnapshot.
func (p *Policy) nHatFor(class string) float64 {
	m := p.nHatOut[class]
	if m == nil {
		return 1
	}
	return m.mean()
}

// sloFor resolves a class's SLO thresholds. sim/edpp.go:682-708 via
// sim/edpp_var.go:597-608.
//
// Because every cohort declares one constant class, this resolves the SELECTED
// workload triple for every request. It is a function of config alone, so it needs no
// lock.
func (p *Policy) sloFor(_ string) slo {
	t := p.cfg.activeTargets()
	return slo{
		tauTTFTUs: float64(t.TauTTFTUs),
		tauITLUs:  float64(t.TauITLUs),
		tauE2EUs:  float64(t.TauE2EUs),
	}
}

// coeffsFor is the SINGLE selection point for per-endpoint heterogeneous
// coefficients. sim/edpp.go:709-719.
//
// The key is the pod-label value. An unmapped label must have ALREADY caused the
// endpoint to be rejected by the filter, so reaching the zero-value return here is a
// bug rather than a fallback -- it is counted, not silently tolerated.
func (p *Policy) coeffsFor(gpuType string) Coeffs {
	if c, ok := p.cfg.CoeffsByGPUType[gpuType]; ok {
		return c
	}
	p.metrics.recordUnmappedGPUTypeAtScore(gpuType)
	return Coeffs{}
}

// varDecodeInputs converts an endpoint's decode resident slice into value inputs.
// sim/edpp_var.go:615-641.
//
// THIS ARM ALWAYS TAKES THE DEPLOYABLE (CENSORED) BRANCH -- varDeployable is forced
// true at sim/edpp.go:1710 and :1746 and is not a config knob here. The oracle branch
// reads TrueRemaining, which is hidden output length, and is never a deployable
// policy.
//
// The censoring: a resident that has produced StepsDone tokens has total output
// >= StepsDone, so the class mean is FLOORED BY StepsDone before subtracting, and the
// remainder floored at 1 because it is still decoding.
func (p *Policy) varDecodeInputs(running []RunningReqState) []decodeResident {
	if len(running) == 0 {
		return nil
	}
	out := make([]decodeResident, 0, len(running))
	for _, r := range running {
		nHat := p.nHatFor(r.SLOClass)
		rem := int64(math.Max(math.Max(nHat, float64(r.StepsDone))-float64(r.StepsDone), 1))
		out = append(out, decodeResident{
			rem:          rem,
			arrivalUs:    r.ArrivalUs,
			firstTokenUs: r.FirstTokenUs,
			ttftSet:      r.TTFTSet,
			slo:          p.sloFor(r.SLOClass),
		})
	}
	return out
}

// varPrefillInputs converts an endpoint's prefill occupant slice into value inputs.
// sim/edpp_var.go:643-673.
//
// remPrefillTokens is remaining PROMPT tokens -- known input, never oracle-gated.
//
// remDecodeSteps is the occupant's decode horizon once it reaches its first token. It
// has produced no output yet, so the deployable estimate is the FULL censored class
// mean with no StepsDone to subtract, floored at 1.
func (p *Policy) varPrefillInputs(running []RunningReqState) []prefillResident {
	if len(running) == 0 {
		return nil
	}
	out := make([]prefillResident, 0, len(running))
	for _, r := range running {
		remDec := int64(math.Max(p.nHatFor(r.SLOClass), 1))
		out = append(out, prefillResident{
			remPrefillTokens: r.TrueRemaining,
			remDecodeSteps:   remDec,
			arrivalUs:        r.ArrivalUs,
			slo:              p.sloFor(r.SLOClass),
		})
	}
	return out
}

// apForEndpoint is a_p: the UNCACHED SUFFIX of the prompt on this endpoint.
// sim/edpp.go:1055-1064.
//
//	a_p = a_r - cachedBlocks * prefixBlockSizeTokens
//
// The value is computed in the filter, where the request and the per-endpoint prefix
// attribute are both in scope, and carried here through evalCtx. Picker.Pick receives
// no request, so there is no other route.
//
// a_p may be NEGATIVE if the cache reports more blocks than the prompt covers;
// callers clamp with maxInt(ap, 0) at every use, and chunkTerms returns (0,0) for
// ap <= 0.
//
// TWO BLOCK SIZES ARE REQUIRED, AND THIS SITE NEEDS THE PREFIX PRODUCER'S. The engine
// reports 16 while the prefix producer clamps its own block size up to a 64-token
// floor, so multiplying the producer's block count by the ENGINE's block size would
// understate the cached span by 4x -- inflating a_p and OVER-pricing prefill work,
// biasing toward remote prefill. The filter therefore reads the block size off the
// same PrefixCacheMatchInfo it reads the count from, so the two cannot drift.
// Config.Engine.BlockSize stays the engine's, because reqKVNeed and the rollout need
// that one.
func (p *Policy) apForEndpoint(ec *evalCtx, id string) int {
	if ap, ok := ec.apByEndpoint[id]; ok {
		return ap
	}
	// No prefix observation for this endpoint: treat the prompt as fully UNCACHED.
	// A miss means "no information", which is not "nothing cached"; charging the full
	// prompt over-prices the candidate rather than asserting a cold cache as fact,
	// and leaves it in the argmin.
	return ec.inputLen
}

// chunkTerms returns (nChunks, deltaPfChunk) for ap uncached prefill tokens under the
// batched-token budget. sim/edpp.go:1220-1228.
//
// deltaPfChunk = theta.CPf*chunk is charged on the pool the prefill runs on: decode
// theta when local, prefill theta when disaggregated. ap <= 0 (fully cached or empty)
// yields (0, 0) -- no prefill work and no per-chunk ITL inflation.
func (p *Policy) chunkTerms(theta Coeffs, ap int) (nChunks, deltaPfChunk float64) {
	if ap <= 0 {
		return 0, 0
	}
	chunk := ap
	if p.cfg.Engine.ChunkTokens > 0 && p.cfg.Engine.ChunkTokens < chunk {
		chunk = p.cfg.Engine.ChunkTokens
	}
	return math.Ceil(float64(ap) / float64(chunk)), theta.CPf * float64(chunk)
}

// projectedLocalTTFT ends at the LOCAL request's client-visible first token.
// sim/edpp.go:1245-1250.
//
// Local execution samples the first token when prefill completes, so there is NO
// separate decode iteration before post-processing -- unlike the disaggregated form
// below. That asymmetry is real, not an oversight.
func (p *Policy) projectedLocalTTFT(tAdmD, nChunks, tIterD, wpLoc float64) float64 {
	return tAdmD + nChunks*tIterD + wpLoc + p.cfg.OutputTokenProcessingUs
}

// projectedDisaggTTFT ends at the decode endpoint's first client-visible token.
// sim/edpp.go:1252-1255.
//
// decodeJoinUs covers prefill admission and work, transfer, and decode admission; the
// first B+1 decode iteration and post-processing follow.
func (p *Policy) projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode float64) float64 {
	return decodeJoinUs + tIterFirstDecode + p.cfg.OutputTokenProcessingUs
}

// reqKVNeed is ceil(a_r / BlockSize) in ENGINE blocks. sim/edpp.go:1257-1262.
func (p *Policy) reqKVNeed(inputLen int) int64 {
	if p.cfg.Engine.BlockSize <= 0 {
		return 0
	}
	return int64((inputLen + p.cfg.Engine.BlockSize - 1) / p.cfg.Engine.BlockSize)
}

// cXferUsFor is the KV-transfer cost for routing this request remotely.
// sim/edpp.go:1126-1140.
//
// DEGRADATION D5: XferBaseUs and XferBandwidthGBps are UNMEASURED on this target.
// This is the ONLY size-dependent price of going remote, entering the objective at
// exactly one place (remoteLeadUs), so a wrong value mis-prices SYSTEMATICALLY rather
// than noisily.
func (p *Policy) cXferUsFor(reqKVNeed int64) float64 {
	if !p.cfg.Transfer.SizeAware {
		return float64(p.cfg.Transfer.FlatCXferUs) // the FLAT field, not XferBaseUs
	}
	bwBytesPerUs := p.cfg.Transfer.XferBandwidthGBps * 1000.0 // GB/s -> bytes/us
	if bwBytesPerUs <= 0 || p.cfg.Engine.BlockSize <= 0 {
		return p.cfg.Transfer.XferBaseUs
	}
	transferBytes := float64(reqKVNeed) * float64(p.cfg.Engine.BlockSize) * p.cfg.Transfer.KVBytesPerTokenPerGPU
	return p.cfg.Transfer.XferBaseUs + transferBytes/bwBytesPerUs
}

// decodeRemStepsEst is the mean censored remaining-steps estimate over an endpoint's
// decode residents. sim/edpp.go:1070-1092.
//
// Deliberately NOT a mean that can go negative: the class estimate is floored by the
// LONGEST in-flight elapsed count before per-request subtraction, and each per-request
// remainder is floored at 1. An endpoint with no residents returns 1.
func (p *Policy) decodeRemStepsEst(snap Snapshot, class string) float64 {
	n := len(snap.RunningDecode)
	if n == 0 {
		return 1.0
	}
	nHatOut := p.nHatFor(class)
	var maxSteps int64
	for _, r := range snap.RunningDecode {
		if r.StepsDone > maxSteps {
			maxSteps = r.StepsDone
		}
	}
	nHatEff := math.Max(nHatOut, float64(maxSteps))
	var sum float64
	for _, r := range snap.RunningDecode {
		sum += math.Max(nHatEff-float64(r.StepsDone), 1)
	}
	return sum / float64(n)
}

// prefillRemStepsEst is the symmetric estimate for a prefill endpoint, over its
// occupants' remaining prefill-chunk counts floored at 1. sim/edpp.go:1094-1114.
func (p *Policy) prefillRemStepsEst(snap Snapshot) float64 {
	n := len(snap.RunningPrefill)
	if n == 0 {
		return 1.0
	}
	var sum float64
	for _, r := range snap.RunningPrefill {
		rem := r.TrueRemaining
		if rem < 1 {
			rem = 1
		}
		sum += float64(rem)
	}
	return sum / float64(n)
}

// decodeAdmissionCtx assembles the admission context for a decode candidate.
// sim/edpp.go:1601-1613.
//
// QWork is left ZERO and that is DECISION-NEUTRAL for this arm: it is read only by
// waitingEstimateTAdm, and this arm selects rollforward. Stated rather than removed so
// the omission reads as a decision.
func (p *Policy) decodeAdmissionCtx(ec *evalCtx, ds Snapshot) AdmissionContext {
	thetaD := p.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	return AdmissionContext{
		QWork:             0, // see note above
		Mu:                thetaD.muDecode(bDec, kv, sPfD),
		BatchSize:         ds.BatchSize,
		MaxBatchSize:      int(ds.MaxBatchSize),
		FreeKVBlocks:      ds.FreeKVBlocks,
		ReqKVNeed:         ec.reqKVNeed,
		TIter:             thetaD.tIterDecode(bDec, kv, sPfD),
		QueueDepth:        ds.QueueDepth,
		AdmissionRate:     ds.AdmissionRate, // read only by `little`; unset here
		RemainingStepsEst: p.decodeRemStepsEst(ds, ec.class),
		// The oracle remaining must be CENSORED to -1 before it reaches an
		// estimator. It is already -1 here, so this documents the invariant.
		Running: censorOracleRemaining(ds.RunningDecode),
	}
}

// prefillAdmissionCtx is the prefill-pool counterpart. sim/edpp.go:1616-1628.
//
// Note it uses muPrefill and tIterPrefill -- a dedicated prefill endpoint runs no
// decode work -- and passes RunningPrefill, whose TrueRemaining is remaining PROMPT
// tokens and therefore needs no censoring.
func (p *Policy) prefillAdmissionCtx(ec *evalCtx, ps Snapshot) AdmissionContext {
	thetaP := p.coeffsFor(ps.GPUType)
	sPfP := ps.ResidentPrefillTokens
	return AdmissionContext{
		QWork:             0, // see decodeAdmissionCtx
		Mu:                thetaP.muPrefill(sPfP),
		BatchSize:         ps.BatchSize,
		MaxBatchSize:      int(ps.MaxBatchSize),
		FreeKVBlocks:      ps.FreeKVBlocks,
		ReqKVNeed:         ec.reqKVNeed,
		TIter:             thetaP.tIterPrefill(sPfP),
		QueueDepth:        ps.QueueDepth,
		AdmissionRate:     ps.AdmissionRate,
		RemainingStepsEst: p.prefillRemStepsEst(ps),
		Running:           ps.RunningPrefill,
	}
}

// estimateTAdm dispatches to the configured estimator, which config validation has
// already pinned to rollforward.
//
// EVERY CALL INCREMENTS THE D1 SUBSTITUTION COUNTER, because every call is a call
// that the published policy would have served with schedulerRollout. The fallback is
// acceptable; the silence would be the defect -- without the counter, running a
// different TTFT estimator than the one that produced every number in sim_results/ is
// invisible in the goodput figure.
func (p *Policy) estimateTAdm(ctx AdmissionContext) float64 {
	p.metrics.recordEstimatorSubstitution()
	return rollforwardEstimateTAdm(ctx)
}

// selfGood is the arriving request's own projected good under a candidate.
// sim/edpp.go:1666-1669.
//
// The prefill chunk does not affect tIterAfter -- only the overlap term -- so chunk is
// passed as 0. Decode always happens on ds, so ds's theta sets tIterAfter for BOTH
// local and disaggregated candidates.
func (p *Policy) selfGood(ec *evalCtx, thetaD Coeffs, ds Snapshot, tHat float64) float64 {
	rt := reTimingFor(ec.inputLen, thetaD, ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens, 0)
	return goodSelf(p.sloFor(ec.class), tHat, rt.tIterAfter, ec.nHatOut)
}

// externalityLocal is the LOCAL branch of sim/edpp_var.go:851-880, with this arm's
// forced settings (exact overlap, deployable, colloc-prefill on) already applied.
func (p *Policy) externalityLocal(ec *evalCtx, ds Snapshot, thetaD Coeffs, apLoc int, decodeAdmissionUs float64) pathBreakdown {
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	decode := p.varDecodeInputs(ds.RunningDecode)

	chunkLoc := apLoc
	if p.cfg.Engine.ChunkTokens > 0 && p.cfg.Engine.ChunkTokens < chunkLoc {
		chunkLoc = p.cfg.Engine.ChunkTokens
	}
	rt := reTimingFor(ec.inputLen, thetaD, bDec, kv, sPfD, chunkLoc)
	// Forced true by this arm (sim/edpp.go:1709), overriding config.
	rt.exactPrefillOverlap = true
	rt.cPf = thetaD.CPf
	rt.ap = float64(maxInt(apLoc, 0))
	rt.ar = float64(ec.inputLen)

	nChunksLoc, _ := p.chunkTerms(thetaD, apLoc)
	// THE ADMISSION GATE, in baseline iterations. max(tIter0, 1) guards a degenerate
	// zero iteration time.
	admissionSteps := math.Ceil(decodeAdmissionUs / math.Max(rt.tIter0, 1))

	v := pathBreakdown{
		decode: varDecodeLocalAfter(ec.nowUs, decode, rt, nChunksLoc, admissionSteps),
	}
	if len(ds.RunningPrefill) > 0 {
		colloc := p.varPrefillInputs(ds.RunningPrefill)
		v.collocPrefill = varCollocPrefillLocalAfter(ec.nowUs, colloc, rt, float64(chunkLoc), nChunksLoc, admissionSteps)
	}
	// No prefillPool term: a local placement puts no work on any prefill endpoint.
	return v
}

// externalityDisagg is the DISAGGREGATED branch of sim/edpp_var.go:881-941.
//
// decodeJoinOverride is the join clock computed by the caller as
// max(remoteLeadUs, tAdmD) -- the OVERLAP form. The published function also contains a
// serialized fallback,
//
//	nChunksP*tIterP + Wp + prefillAdmissionUs + cXfer + decodeAdmissionUs
//
// used when no override is supplied (sim/edpp_var.go:899-905). This arm ALWAYS
// supplies the override (sim/edpp.go:1750-1752), so the serialized form is unreachable
// here; it is recorded so a reader does not mistake the override for the only model.
func (p *Policy) externalityDisagg(ec *evalCtx, ds Snapshot, ps *Snapshot, thetaD, thetaP Coeffs, apP int, decodeJoinOverride, prefillAdmissionUs float64) pathBreakdown {
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	decode := p.varDecodeInputs(ds.RunningDecode)

	chunkP := apP
	if p.cfg.Engine.ChunkTokens > 0 && p.cfg.Engine.ChunkTokens < chunkP {
		chunkP = p.cfg.Engine.ChunkTokens
	}
	// DECODE re-timing uses ds's theta even on a disaggregated candidate: decode
	// happens on ds in both placements. Only the prefill-pool term uses thetaP.
	rt := reTimingFor(ec.inputLen, thetaD, bDec, kv, sPfD, chunkP)
	rt.exactPrefillOverlap = true
	rt.cPf = thetaD.CPf
	rt.ap = float64(maxInt(apP, 0))
	rt.ar = float64(ec.inputLen)

	// NOTE: upstream also computes nChunksP here (sim/edpp_var.go:889), but it is an
	// operand of the LEGACY prefill-pool term only -- varPrefillDisaggAfter needs
	// rPrefillUs = nChunksP*tIterP + Wp. This arm forces the exact form, which charges
	// each occupant only R's chunks that execute before that occupant's first token, so
	// nChunksP is dead here and Go will not allow an unused local. It is recomputed at
	// the call site in scoreCandidate, where it IS live (the disaggregated TTFT
	// projection needs it), so nothing is lost.
	sPfP := ps.ResidentPrefillTokens
	tIterP := thetaP.tIterPrefill(sPfP)

	// Decode interference starts when R is ADMITTED, not when its first token
	// completes. This join-time clock is deliberately separate from the
	// client-visible TTFT projection.
	arrivalSteps := math.Ceil(decodeJoinOverride / math.Max(rt.tIter0, 1))

	v := pathBreakdown{
		decode: varDecodeDisagg(ec.nowUs, decode, rt, arrivalSteps),
	}
	// Collocated prefill occupants on ds are undisturbed until R arrives from the
	// pool; only those still prefilling past arrivalSteps have their first token
	// delayed.
	if len(ds.RunningPrefill) > 0 {
		colloc := p.varPrefillInputs(ds.RunningPrefill)
		v.collocPrefill = varCollocPrefillDisagg(ec.nowUs, colloc, rt, float64(chunkP), arrivalSteps)
	}

	prefill := p.varPrefillInputs(ps.RunningPrefill)
	prefillAdmissionSteps := math.Ceil(prefillAdmissionUs / math.Max(tIterP, 1))
	// Forced exact form (sim/edpp.go:1745). The legacy alternative
	// varPrefillDisaggAfter would charge each occupant R's ENTIRE prefill duration,
	// rPrefillUs = nChunksP*tIterP + Wp.
	v.prefillPool = varPrefillDisaggExactAfter(
		ec.nowUs, prefill, tIterP, float64(chunkP), prefillAdmissionSteps,
		maxInt(apP, 0), ec.inputLen, thetaP,
	)
	return v
}

// scoreCandidate implements the policy contract exactly:
//
//	total = V*(externality - ownGood) + capacity
//
// sim/edpp.go:1688-1779. ps == nil means LOCAL (prefill co-resident on the decode
// endpoint); otherwise decode on ds and prefill on *ps.
//
// It carries NO historical TTFT/ITL deficit term, no standalone transfer residue, and
// no per-decision normalization. Those belong to other rules in the published decider
// and are deliberately absent here (sim/edpp.go:1683-1687 states that contract).
func (p *Policy) scoreCandidate(ec *evalCtx, ds Snapshot, ps *Snapshot) candidateScore {
	thetaD := p.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	tIterD := thetaD.tIterDecode(bDec, kv, sPfD)
	// The B+1 re-timed FIRST decode iteration: batch grows by one and KV grows by the
	// arriving request's full input length.
	tIterFirstDecode := thetaD.tIterDecode(bDec+1, kv+int64(ec.inputLen), sPfD)
	wd := thetaD.Wd(ec.inputLen, ec.nHatOut)
	tAdmD := p.estimateTAdm(p.decodeAdmissionCtx(ec, ds))

	score := candidateScore{}

	if ps == nil {
		// ---- LOCAL candidate ----
		apLoc := p.apForEndpoint(ec, ds.ID)
		nChunksLoc, _ := p.chunkTerms(thetaD, apLoc)
		wpLoc := thetaD.Wp(maxInt(apLoc, 0), ec.inputLen)
		tHatLocal := p.projectedLocalTTFT(tAdmD, nChunksLoc, tIterD, wpLoc)
		// D1: unreachable at this pin, so the closed-form tAdmD and tHatLocal stand.
		if rolloutAdm, rolloutTTFT, ok := p.rolloutLocalTTFT(ec, ds, thetaD); ok {
			tAdmD, tHatLocal = rolloutAdm, rolloutTTFT
		}

		score.externalityBreakdown = p.externalityLocal(ec, ds, thetaD, apLoc, tAdmD)

		if !p.cfg.Ablation.NoOwnGood {
			score.ownGood = p.selfGood(ec, thetaD, ds, tHatLocal)
		}
		if !p.cfg.Ablation.NoCapacity {
			demand := wpLoc + wd
			if p.cfg.Ablation.OccupancyCapacity {
				demand = p.localOccupancy(thetaD, apLoc, ec.inputLen, ec.nHatOut)
			}
			score.capacityQueueDecode = p.capacityQueue(ds.ID)
			score.capacityDemandDecode = demand
			score.capacityDecode = p.capacityTerm(ds.ID, demand)
		}
	} else {
		// ---- DISAGGREGATED candidate ----
		if rolloutAdm, ok := p.rolloutDecodeAdmission(ec, ds, thetaD); ok {
			tAdmD = rolloutAdm // D1: unreachable at this pin
		}
		thetaP := p.coeffsFor(ps.GPUType)
		apP := p.apForEndpoint(ec, ps.ID)
		nChunksP, _ := p.chunkTerms(thetaP, apP)
		wpP := thetaP.Wp(maxInt(apP, 0), ec.inputLen)
		tIterP := thetaP.tIterPrefill(ps.ResidentPrefillTokens)
		tAdmP := p.estimateTAdm(p.prefillAdmissionCtx(ec, *ps))
		prefillCompletionUs := tAdmP + nChunksP*tIterP + wpP
		if rolloutAdm, rolloutCompletion, ok := p.rolloutPrefillCompletion(ec, *ps, thetaP); ok {
			tAdmP, prefillCompletionUs = rolloutAdm, rolloutCompletion // D1
		}

		// THE ONLY PLACE c_xfer ENTERS THE OBJECTIVE -- degradation D5.
		remoteLeadUs := prefillCompletionUs + p.cXferUsFor(ec.reqKVNeed)

		// OVERLAP, NOT SERIALIZATION. The decode admission estimate is an absolute
		// wait measured at the routing instant. Remote prefill and transfer consume
		// part or all of that interval while the decode queue continues to drain, so
		// they must NOT be serialized with the full estimate a second time.
		//
		// UNCONDITIONAL. There is a --edpp-ttft-overlap-aware flag upstream, but it
		// gates only the REDUCED path (sim/edpp.go:909) and is never consulted on the
		// joint path. Do NOT make this max() conditional on it: that reintroduces the
		// serialized form and over-prices every disaggregated candidate by up to a
		// full decode admission delay.
		decodeJoinUs := math.Max(remoteLeadUs, tAdmD)
		tHatDisagg := p.projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode)

		score.externalityBreakdown = p.externalityDisagg(ec, ds, ps, thetaD, thetaP, apP, decodeJoinUs, tAdmP)

		if !p.cfg.Ablation.NoOwnGood {
			score.ownGood = p.selfGood(ec, thetaD, ds, tHatDisagg)
		}
		if !p.cfg.Ablation.NoCapacity {
			decodeDemand, prefillDemand := wd, wpP
			if p.cfg.Ablation.OccupancyCapacity {
				decodeDemand = p.decodeOccupancy(thetaD, ec.inputLen, ec.nHatOut)
				prefillDemand = p.prefillOccupancy(thetaP, apP, ec.inputLen)
			}
			score.capacityQueueDecode = p.capacityQueue(ds.ID)
			score.capacityQueuePrefill = p.capacityQueue(ps.ID)
			score.capacityDemandDecode = decodeDemand
			score.capacityDemandPrefill = prefillDemand
			score.capacityDecode = p.capacityTerm(ds.ID, decodeDemand)
			score.capacityPrefill = p.capacityTerm(ps.ID, prefillDemand)
		}
	}

	if !p.cfg.Ablation.NoExternality {
		score.externality = score.externalityBreakdown.total()
	}
	score.netGoodCost = score.externality - score.ownGood
	score.capacityTotal = score.capacityDecode + score.capacityPrefill
	score.total = p.cfg.V*score.netGoodCost + score.capacityTotal
	return score
}

// Decide enumerates every candidate and returns the argmin.
// sim/edpp.go:1334-1400, SLO-externality branch at :1380-1398.
//
// THE SHAPE: D local candidates PLUS D*P disaggregated candidates, scored on ONE scale
// and compared in ONE argmin.
//
// DETERMINISM: candidates are enumerated over endpoints sorted by ID, and the argmin
// uses a strict improvement threshold, so ties resolve to the first-enumerated
// candidate rather than to map iteration order.
//
// THE DECODE CHOICE IS PART OF THE OUTPUT ON BOTH OUTCOMES. Upstream encodes a local
// win as {Disaggregate:false, DecodePodOverride:dID} and a disaggregated win as
// {Disaggregate:true, DecodePodOverride:dID, PrefillPodHint:pID}
// (sim/edpp.go:1470-1475): the decode pod is overridden EVEN WHEN THE RULE DECLINES TO
// DISAGGREGATE. A port that applies the decode selection only on the disaggregated
// branch, and otherwise lets the inherited scorer place decode, has silently discarded
// half the joint argmin -- what remains is closer to the `decomposed` control than to
// this arm.
func (p *Policy) decide(ec *evalCtx, decodeSnaps, prefillSnaps []Snapshot, scorerDecodeID, scorerPrefillID string) (candidate, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()

	decodeSnaps = sortedByID(decodeSnaps)
	prefillSnaps = sortedByID(prefillSnaps)

	// Rebuild the ID -> GPU type map for this decision so the candidate score and the
	// capacity commit cannot select different physics for the same endpoint.
	p.gpuTypeByID = make(map[string]string, len(decodeSnaps)+len(prefillSnaps))
	for _, s := range decodeSnaps {
		p.gpuTypeByID[s.ID] = s.GPUType
	}
	for _, s := range prefillSnaps {
		p.gpuTypeByID[s.ID] = s.GPUType
	}

	// nHatOut is candidate-invariant, so it is resolved once per decision -- every
	// candidate evaluation then uses identical operands.
	ec.nHatOut = p.nHatFor(ec.class)

	if !p.cfg.Ablation.NoCapacity {
		p.refreshCapacity(int64(ec.nowUs), decodeSnaps, prefillSnaps)
	}

	// The inherited scorer's decode pick is enumerated FIRST, so that restricting the
	// enumeration to it reproduces the decomposed rule exactly.
	orderedD := scorerFirstSnapshots(decodeSnaps, scorerDecodeID)

	// AND THE SAME APPLIES ON THE PREFILL SIDE. Upstream reorders the prefill
	// snapshots too whenever a prefill scorer is injected (sim/edpp.go:1387-1390).
	// Iterating plain prefillSnaps instead changes which candidate wins an exact tie.
	// With one prefill endpoint the two orders coincide, which is exactly why this is
	// easy to port wrong and hard to notice: config.md section 2 deploys 1P, so a
	// second prefill endpoint would silently change tie resolution.
	orderedP := scorerFirstSnapshots(prefillSnaps, scorerPrefillID)

	if p.cfg.Ablation.Decomposed && len(orderedD) > 1 {
		// The matched decomposition CONTROL, not the focal arm: decode is fixed to the
		// inherited scorer's choice. config.md section 10 prices this at +0.0485
		// equal-cell mean goodput against the full joint shape -- and flipping it is
		// the cheapest on-cluster check that the joint shape survived the port.
		orderedD = orderedD[:1]
	}

	var best *candidate
	consider := func(c candidate) {
		if best == nil || c.J < best.J-1e-12 {
			cc := c
			best = &cc
		}
	}

	for _, ds := range orderedD {
		s := p.scoreCandidate(ec, ds, nil)
		consider(candidate{dID: ds.ID, local: true, J: s.total})
		for i := range orderedP {
			ps := orderedP[i]
			s = p.scoreCandidate(ec, ds, &ps)
			consider(candidate{dID: ds.ID, pID: ps.ID, local: false, J: s.total})
		}
	}
	if best == nil {
		return candidate{}, false
	}

	// The COMMIT half of the capacity account, at the winning endpoints, fed by the
	// same demand expression the candidate score used.
	p.bookCapacityWork(ec, !best.local, best.dID, best.pID)

	return *best, true
}

// sortedByID returns the snapshots sorted ascending by endpoint ID.
// sim/edpp.go:2160-2168.
//
// Required for decision determinism: without it the enumeration order follows the
// caller's slice, and an exact tie resolves differently between two otherwise
// identical decisions. It COPIES before sorting -- the input slice must not be
// reordered in place, since the caller may hold it for other purposes.
func sortedByID(snaps []Snapshot) []Snapshot {
	if len(snaps) == 0 {
		return nil
	}
	out := make([]Snapshot, len(snaps))
	copy(out, snaps)
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// scorerFirstSnapshots preserves ascending-ID order except that the inherited
// scorer's preferred endpoint is moved to the FRONT. sim/edpp.go:2180-2196.
//
// It is a TIE-BREAK ORDER, not a filter: every endpoint is still enumerated, and
// because the argmin uses a strict improvement threshold, moving one to the front only
// changes which candidate wins an EXACT tie. That is also what makes restricting the
// enumeration to the first element reproduce the decomposed rule.
//
// Three early-return cases return the input UNCHANGED rather than a copy: no
// preference, fewer than two endpoints, or the preference already first.
func scorerFirstSnapshots(snaps []Snapshot, preferred string) []Snapshot {
	if preferred == "" || len(snaps) < 2 || snaps[0].ID == preferred {
		return snaps
	}
	out := make([]Snapshot, 0, len(snaps))
	for _, s := range snaps {
		if s.ID == preferred {
			out = append(out, s)
		}
	}
	for _, s := range snaps {
		if s.ID != preferred {
			out = append(out, s)
		}
	}
	return out
}

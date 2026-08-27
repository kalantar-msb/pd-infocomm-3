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

package leastttftjoint

import (
	"math"
	"sort"
)

// sloClassStandard is the one SLO class every cohort of all three workloads
// declares (workloads/interactive-chat-single-turn.yaml:21 and the other two). It is
// the fallback when a request carries no objective, so an unlabelled request resolves
// the same triple as a labelled one rather than an empty-class zero triple.
const sloClassStandard = "standard"

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

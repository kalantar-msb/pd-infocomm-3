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
)

// Admission-delay estimators, ported from sim/admission_estimator.go.
//
// ALL FOUR ARE STATED because degradation D1 makes the choice among them a port
// decision rather than an inherited one: this arm selects `rollforward`
// (config.md section 9.1), and a reader must be able to see what the published
// policy would have done instead. Only `rollforward` is reachable through
// Config.AdmissionEstimator, which rejects every other value at startup.
//
// An estimator is a pure function of AdmissionContext.

// AdmissionContext mirrors sim/admission_estimator.go:57-70. Times and work in us.
type AdmissionContext struct {
	// QWork is the waiting-backlog work in us, read ONLY by waitingEstimateTAdm.
	// It is left zero on this target and that is decision-neutral for this arm,
	// which selects rollforward. Stated rather than removed so the omission reads
	// as a decision: reconstructing the per-instance work account would be dead
	// state that implies a consumer.
	QWork float64
	// Mu is the occupancy-aware drain rate. Reaches here but rollforward never
	// reads it; live if the capacity account or the `waiting` estimator is enabled.
	Mu                float64
	BatchSize         int
	MaxBatchSize      int
	FreeKVBlocks      int64
	ReqKVNeed         int64
	TIter             float64
	QueueDepth        int
	AdmissionRate     float64 // req/us -- read ONLY by littleEstimateTAdm
	RemainingStepsEst float64
	Running           []RunningReqState
}

// flooredTAdm lower-bounds an estimate by one iteration: even with a free slot, a
// request waits for the current decode step to finish before the next batch
// formation admits it. No-op when TIter is unavailable or already exceeded.
// sim/admission_estimator.go:79-86.
func flooredTAdm(est float64, ctx AdmissionContext) float64 {
	if ctx.TIter > est {
		return ctx.TIter
	}
	return est
}

// waitingEstimateTAdm is the `waiting` estimator. sim/admission_estimator.go:89-95.
//
// NOT SELECTED by any registered arm. It is the only consumer of ctx.QWork, which
// is why QWork is a declared-but-unread input on this target.
func waitingEstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.Mu <= 0 {
		return 0
	}
	return ctx.QWork / ctx.Mu
}

// littleEstimateTAdm is the `little` estimator. sim/admission_estimator.go:99-105.
//
// NOT SELECTED. It is the only consumer of ctx.AdmissionRate, for the same reason
// as QWork above.
func littleEstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.AdmissionRate <= 0 {
		return flooredTAdm(0, ctx)
	}
	return flooredTAdm(float64(ctx.QueueDepth)/ctx.AdmissionRate, ctx)
}

// fluidEstimateTAdm is the `fluid` estimator. sim/admission_estimator.go:109-124.
//
// NOT SELECTED, but its WAVE FORM IS REACHED: rollforward falls back to it for the
// deep tail, so the arithmetic below is live on this target.
func fluidEstimateTAdm(ctx AdmissionContext) float64 {
	// Admit next iteration if a slot AND enough KV already fit.
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return flooredTAdm(0, ctx)
	}
	if ctx.BatchSize <= 0 || ctx.RemainingStepsEst <= 0 || ctx.TIter <= 0 {
		return flooredTAdm(0, ctx)
	}
	// Synchronized batch: occupants finish ~R steps together, so slots free in
	// WAVES of BatchSize every ~R iterations. A request at queue position
	// QueueDepth waits ceil((QueueDepth+1)/BatchSize) waves. This is deliberately
	// NOT the naive fluid drain /BatchSize.
	waves := math.Ceil(float64(ctx.QueueDepth+1) / float64(ctx.BatchSize))
	return flooredTAdm(waves*ctx.RemainingStepsEst*ctx.TIter, ctx)
}

// rollforwardEstimateTAdm is the `rollforward` estimator, WHICH THIS ARM SELECTS
// (config.md section 9.1). sim/admission_estimator.go:127-176.
//
// Deterministic look-ahead: each running request departs after its remaining steps
// (oracle TrueRemaining if >= 0, else the N_out estimate), freeing its KV.
// Accumulate departureStep * T_iter until a slot AND enough free KV exist.
//
// DEGRADATION D1 LIVES HERE. This is the SUBSTITUTE for the scheduler rollout, not
// the published estimator -- every number in sim_results/ was produced by
// schedulerRollout. Its bias: past one batch drain it UNDERSTATES admission delay
// => smaller admissionSteps => residents are charged for more of the arrival's
// interference => the local candidate is OVER-priced => biased toward REMOTE
// prefill.
//
// Every call that reaches this function instead of schedulerRollout increments a
// substitution counter at the call site (Policy.estimateTAdm). The fallback is
// acceptable; the silence would be the defect.
func rollforwardEstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return flooredTAdm(0, ctx)
	}
	type dep struct{ step, kv int64 }
	deps := make([]dep, 0, len(ctx.Running))
	for _, r := range ctx.Running {
		rem := r.TrueRemaining
		if rem < 0 {
			// The deployable branch, and the ONLY one reachable on this target:
			// TrueRemaining is -1 because the oracle reads hidden output length.
			rem = int64(ctx.RemainingStepsEst)
			if rem < 1 {
				rem = 1
			}
		}
		deps = append(deps, dep{step: rem, kv: r.KVBlocks})
	}
	// Sort by departure step ascending, STABLY. Stability is not cosmetic: two
	// running requests with equal remaining steps but different KV holdings are
	// accumulated in slice order, so an unstable sort can change which departure
	// first satisfies the KV condition, and therefore change the returned delay.
	sort.SliceStable(deps, func(i, j int) bool { return deps[i].step < deps[j].step })

	// The request sits at queue position QueueDepth: the QueueDepth requests ahead
	// fill the first QueueDepth freed slots and ours takes the next, so
	// QueueDepth+1 slots must free (plus our KV) before we are admitted.
	needSlots := int64(ctx.QueueDepth + 1)
	freeSlots := int64(ctx.MaxBatchSize - ctx.BatchSize)
	freeKV := ctx.FreeKVBlocks
	for _, d := range deps {
		freeSlots++
		freeKV += d.kv
		if freeSlots >= needSlots && freeKV >= ctx.ReqKVNeed {
			return flooredTAdm(float64(d.step)*ctx.TIter, ctx)
		}
	}
	// The running set's departures were exhausted before freeing QueueDepth+1
	// slots (queue deeper than one batch drain). Fall back to the fluid wave form.
	// THIS is the branch that understates delay -- see D1 above.
	if ctx.BatchSize > 0 {
		waves := math.Ceil(float64(ctx.QueueDepth+1) / float64(ctx.BatchSize))
		return flooredTAdm(waves*ctx.RemainingStepsEst*ctx.TIter, ctx)
	}
	// BatchSize <= 0: no wave cadence available -- cap at the last known departure.
	if len(deps) > 0 {
		return flooredTAdm(float64(deps[len(deps)-1].step)*ctx.TIter, ctx)
	}
	return flooredTAdm(0, ctx)
}

// censorOracleRemaining forces TrueRemaining to -1 so no estimator can read a
// hidden output length. sim/edpp.go:2234-2248.
//
// On this target TrueRemaining is already -1, so this is a no-op that documents
// the invariant rather than a transformation that does work.
func censorOracleRemaining(running []RunningReqState) []RunningReqState {
	if len(running) == 0 {
		return nil
	}
	out := make([]RunningReqState, len(running))
	copy(out, running)
	for i := range out {
		out[i].TrueRemaining = -1
	}
	return out
}

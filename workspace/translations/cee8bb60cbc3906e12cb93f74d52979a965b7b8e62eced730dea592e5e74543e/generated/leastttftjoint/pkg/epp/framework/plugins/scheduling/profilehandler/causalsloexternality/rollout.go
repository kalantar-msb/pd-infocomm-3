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

import "math"

// THE SCHEDULER ROLLOUT -- THE PUBLISHED ESTIMATOR.
//
// Ported in full from sim/edpp_scheduler_rollout.go. Every number in
// sim_results/ was produced by this, not by rollforwardEstimateTAdm.
//
// ITS INPUTS ARE UNOBTAINABLE AT THIS PIN (degradation D1): Snapshot's
// SchedulerStateObserved is permanently false, so this code is unreachable here
// and rollforwardEstimateTAdm substitutes. It is ported REGARDLESS, because an
// unobtainable INPUT does not license omitting the algebra that consumes it --
// the two are independent facts and both are on the record. A future engine patch
// exporting the wait queue makes this branch reachable without re-deriving
// anything, which is exactly the property config.md section 5 asks the port to
// preserve.
//
// It is exercised by unit tests through rolloutTimes with a synthesised observed
// snapshot, so "unreachable in production" does not mean "untested".

// rolloutReq is a mutable copy used by the replay.
//
// outputRemaining counts only FUTURE decode grants. Prefill progress is
// represented exactly by prompt-computed, because token-budget contention can
// make an actual prefill grant smaller than the nominal chunk cap; combining the
// two into one grant count would let extra prefill steps incorrectly consume the
// predicted output lifetime. sim/edpp_scheduler_rollout.go:11-20.
type rolloutReq struct {
	id              string
	prompt          int64
	computed        int64
	kvBlocks        int64
	outputRemaining int64
	target          bool
}

type rolloutGrant struct {
	req     *rolloutReq
	before  int64
	grant   int64
	prefill bool
}

type rolloutResult struct {
	admissionUs  float64
	firstTokenUs float64
	admitted     bool
	firstToken   bool
}

type rolloutContext struct {
	running, waiting   []*rolloutReq
	target             *rolloutReq
	currentScheduled   []SchedulerReqState
	currentStepStartUs int64
	nowUs              float64
	freeKVBlocks       int64
	tokenBudget        int64
	prefillChunkCap    int64
	blockSize          int64
	maxBatch           int
	maxSteps           int
	theta              Coeffs
	alpha              float64
}

// ceilBlocks is ceil(tokens/blockSize), 0 for non-positive tokens.
// sim/edpp_scheduler_rollout.go:35-43.
func ceilBlocks(tokens, blockSize int64) int64 {
	if tokens <= 0 {
		return 0
	}
	if blockSize <= 0 {
		blockSize = 1
	}
	return (tokens + blockSize - 1) / blockSize
}

// rolloutGrantTime is one replayed step's duration: the per-iteration intercept
// plus each grant's marginal charge. A prefill grant is charged causally against
// the prefix already computed; a decode grant is charged C0 + C1*context.
// sim/edpp_scheduler_rollout.go:45-57.
func rolloutGrantTime(grants []rolloutGrant, theta Coeffs, alpha float64) float64 {
	t := alpha
	for _, item := range grants {
		if item.prefill {
			grant := float64(item.grant)
			before := float64(item.before)
			t += theta.CPf*grant + theta.CAttn*grant*(before+grant/2)
		} else {
			t += theta.C0 + theta.C1*float64(item.before)
		}
	}
	return math.Max(t, 0)
}

// currentScheduledTime is the duration of the step ALREADY IN FLIGHT at the
// decision instant, reconstructed from its grants.
// sim/edpp_scheduler_rollout.go:59-77.
func currentScheduledTime(states []SchedulerReqState, theta Coeffs, alpha float64) float64 {
	grants := make([]rolloutGrant, 0, len(states))
	for i := range states {
		state := states[i]
		if state.ScheduledTokens <= 0 {
			continue
		}
		grants = append(grants, rolloutGrant{
			req:     &rolloutReq{},
			before:  state.ComputedTokens,
			grant:   state.ScheduledTokens,
			prefill: state.ComputedTokens < state.PromptTokens,
		})
	}
	if len(grants) == 0 {
		return 0
	}
	return rolloutGrantTime(grants, theta, alpha)
}

// schedulerRollout replays future scheduler steps until the target request is
// admitted and reaches its first token. sim/edpp_scheduler_rollout.go:79-243.
//
// The replay models, in order per step: continuing the running set under the token
// budget, KV-pressure preemption from the tail, then admitting from the wait queue
// while a slot and budget remain. The target request is appended to the WAIT QUEUE,
// so its admission is predicted rather than assumed.
func schedulerRollout(ctx rolloutContext) rolloutResult {
	result := rolloutResult{}
	// The in-flight step's REMAINING time: its full duration less however much of
	// it has already elapsed.
	elapsed := math.Max(
		currentScheduledTime(ctx.currentScheduled, ctx.theta, ctx.alpha)-
			math.Max(ctx.nowUs-float64(ctx.currentStepStartUs), 0),
		0,
	)
	running := append([]*rolloutReq(nil), ctx.running...)
	waiting := append([]*rolloutReq(nil), ctx.waiting...)
	waiting = append(waiting, ctx.target)
	freeKV := ctx.freeKVBlocks
	if freeKV < 0 {
		freeKV = 0
	}

	for step := 0; step < ctx.maxSteps; step++ {
		budget := ctx.tokenBudget
		grants := make([]rolloutGrant, 0, len(running)+len(waiting))
		preempted := make([]*rolloutReq, 0)

		// --- Continue the running set under the token budget.
		for index := 0; index < len(running) && budget > 0; index++ {
			req := running[index]
			before := req.computed
			isPrefill := before < req.prompt
			demand := int64(1) // a decode step demands exactly one token
			if isPrefill {
				demand = req.prompt - before
				if ctx.prefillChunkCap > 0 {
					demand = minI64(demand, ctx.prefillChunkCap)
				}
			}
			grant := minI64(demand, budget)
			if grant <= 0 {
				continue
			}
			newBlocks := ceilBlocks(before+grant, ctx.blockSize)
			deltaBlocks := newBlocks - req.kvBlocks
			if deltaBlocks < 0 {
				deltaBlocks = 0
			}
			canSchedule := true
			// KV pressure: preempt from the TAIL until the delta fits.
			for deltaBlocks > freeKV {
				if len(running) == 0 {
					canSchedule = false
					break
				}
				victim := running[len(running)-1]
				running = running[:len(running)-1]
				freeKV += victim.kvBlocks
				victim.kvBlocks = 0
				// A preempted request must RECOMPUTE. One already in decode
				// recomputes its prompt plus the output it had produced, so its
				// effective prompt grows by one; one still prefilling recomputes
				// its prompt.
				resumeTokens := victim.prompt
				if victim.computed >= victim.prompt {
					resumeTokens = victim.computed + 1
				}
				victim.prompt = resumeTokens
				victim.computed = 0
				preempted = append([]*rolloutReq{victim}, preempted...)
				if victim == req {
					canSchedule = false
					break
				}
			}
			if !canSchedule {
				break
			}
			freeKV -= deltaBlocks
			req.kvBlocks = newBlocks
			grants = append(grants, rolloutGrant{req: req, before: before, grant: grant, prefill: isPrefill})
			budget -= grant
		}

		// Preempted requests go to the FRONT of the wait queue, and no new
		// admission happens on a step that preempted.
		if len(preempted) > 0 {
			waiting = append(preempted, waiting...)
		}
		for len(preempted) == 0 && len(waiting) > 0 && budget > 0 && len(running) < ctx.maxBatch {
			req := waiting[0]
			before := req.computed
			isPrefill := before < req.prompt
			demand := int64(1)
			if isPrefill {
				demand = req.prompt - before
				if ctx.prefillChunkCap > 0 {
					demand = minI64(demand, ctx.prefillChunkCap)
				}
			}
			grant := minI64(demand, budget)
			if grant <= 0 {
				break
			}
			newBlocks := ceilBlocks(before+grant, ctx.blockSize)
			deltaBlocks := newBlocks - req.kvBlocks
			if deltaBlocks < 0 {
				deltaBlocks = 0
			}
			if deltaBlocks > freeKV {
				break
			}
			// THE ADMISSION INSTANT: recorded when the target first receives a
			// grant from the wait queue.
			if req.target {
				result.admissionUs = elapsed
				result.admitted = true
			}
			waiting = waiting[1:]
			freeKV -= deltaBlocks
			req.kvBlocks = newBlocks
			running = append(running, req)
			grants = append(grants, rolloutGrant{req: req, before: before, grant: grant, prefill: isPrefill})
			budget -= grant
		}

		// A step that only preempted makes no progress but is not a dead end.
		if len(grants) == 0 && len(preempted) > 0 {
			continue
		}
		if len(grants) == 0 {
			return result
		}

		// THE FIRST-TOKEN CONDITION: the target either took a decode grant, or
		// took the prefill grant that COMPLETED its prompt.
		targetFirstToken := false
		for _, item := range grants {
			if item.req.target && ((!item.prefill) || item.before+item.grant >= item.req.prompt) {
				targetFirstToken = true
			}
		}
		elapsed += rolloutGrantTime(grants, ctx.theta, ctx.alpha)

		// Advance every granted request, retire the finished, reclaim their KV.
		kept := running[:0]
		for _, req := range running {
			var grant *rolloutGrant
			for i := range grants {
				if grants[i].req == req {
					grant = &grants[i]
					break
				}
			}
			if grant == nil {
				kept = append(kept, req)
				continue
			}
			req.computed = grant.before + grant.grant
			if !grant.prefill {
				req.outputRemaining--
			}
			if grant.prefill || req.outputRemaining > 0 {
				kept = append(kept, req)
			} else {
				freeKV += req.kvBlocks
			}
		}
		running = kept

		if targetFirstToken {
			result.firstTokenUs = elapsed
			result.firstToken = true
			return result
		}
	}
	return result
}

// rolloutReqFor converts one observed scheduler request into a replay copy,
// supplying the censored per-class output estimate.
// sim/edpp_scheduler_rollout.go:244-260.
//
// The output estimate is CENSORED: a request that has already decoded decodeDone
// tokens has total output >= decodeDone, so the class mean is floored by that
// before subtracting, and the remainder floored at 1.
func (p *Policy) rolloutReqFor(state SchedulerReqState) *rolloutReq {
	computed := state.ComputedTokens
	if computed < 0 {
		computed = 0
	}
	prompt := state.PromptTokens
	if prompt < 0 {
		prompt = 0
	}
	outputEstimate := p.nHatFor(state.SLOClass)
	decodeDone := computed - prompt
	if decodeDone < 0 {
		decodeDone = 0
	}
	totalOutput := math.Max(outputEstimate, float64(decodeDone))
	remainingOutput := int64(math.Ceil(totalOutput)) - decodeDone
	if remainingOutput < 1 {
		remainingOutput = 1
	}
	return &rolloutReq{
		id:              state.ID,
		prompt:          prompt,
		computed:        computed,
		kvBlocks:        maxI64(state.KVBlocks, 0),
		outputRemaining: remainingOutput,
	}
}

// rolloutTimes applies the replay to one candidate endpoint.
// sim/edpp_scheduler_rollout.go:262-307.
//
// cachedTokens is the target's known prefix on that endpoint. decodeOnly models a
// TRANSFERRED request: its prompt is already computed elsewhere, but its KV blocks
// still have to fit at the candidate decoder. prefillPool selects the prefill
// intercept alpha_p instead of the decode intercept alpha.
//
// THE GUARD ON LINE ONE IS DEGRADATION D1: SchedulerStateObserved is permanently
// false at this pin, so this returns (zero, false) on every production call and
// every caller falls through to rollforwardEstimateTAdm.
func (p *Policy) rolloutTimes(ec *evalCtx, snap Snapshot, theta Coeffs, cachedTokens int, decodeOnly, prefillPool bool) (rolloutResult, bool) {
	if !snap.SchedulerStateObserved || snap.MaxScheduledTokens <= 0 || snap.MaxBatchSize <= 0 {
		return rolloutResult{}, false
	}
	blockSize := snap.BlockSizeTokens
	if blockSize <= 0 {
		blockSize = int64(maxInt(p.cfg.Engine.BlockSize, 1))
	}
	chunkCap := snap.MaxScheduledTokens
	if snap.LongPrefillTokenThreshold > 0 {
		chunkCap = minI64(chunkCap, snap.LongPrefillTokenThreshold)
	}
	running := make([]*rolloutReq, 0, len(snap.SchedulerRunning))
	for _, state := range snap.SchedulerRunning {
		running = append(running, p.rolloutReqFor(state))
	}
	waiting := make([]*rolloutReq, 0, len(snap.SchedulerWaiting))
	for _, state := range snap.SchedulerWaiting {
		waiting = append(waiting, p.rolloutReqFor(state))
	}
	prompt := int64(ec.inputLen)
	computed := int64(maxInt(cachedTokens, 0))
	targetKV := ceilBlocks(computed, blockSize)
	if decodeOnly {
		computed = prompt
		targetKV = 0
	}
	target := &rolloutReq{
		id:              ec.requestID,
		prompt:          prompt,
		computed:        computed,
		kvBlocks:        targetKV,
		outputRemaining: maxI64(int64(math.Ceil(math.Max(ec.nHatOut, 1))), 1),
		target:          true,
	}
	alpha := theta.AlphaD
	if prefillPool {
		alpha = theta.AlphaP
	}
	result := schedulerRollout(rolloutContext{
		running: running, waiting: waiting, target: target,
		currentScheduled: snap.CurrentScheduled, currentStepStartUs: snap.CurrentStepStartUs,
		nowUs: ec.nowUs, freeKVBlocks: snap.FreeKVBlocks,
		tokenBudget: snap.MaxScheduledTokens, prefillChunkCap: chunkCap,
		blockSize: blockSize, maxBatch: int(snap.MaxBatchSize), maxSteps: 100000,
		theta: theta, alpha: alpha,
	})
	return result, result.admitted
}

// rolloutLocalTTFT predicts (admission, TTFT) for a LOCAL placement.
// sim/edpp_scheduler_rollout.go:309-316.
//
// The client-visible TTFT adds the output-token post-processing latency, which is
// outside theta.
func (p *Policy) rolloutLocalTTFT(ec *evalCtx, ds Snapshot, theta Coeffs) (tAdm, ttft float64, ok bool) {
	cached := ec.inputLen - maxInt(p.apForEndpoint(ec, ds.ID), 0)
	result, ok := p.rolloutTimes(ec, ds, theta, cached, false, false)
	if !ok || !result.firstToken {
		return 0, 0, false
	}
	return result.admissionUs, result.firstTokenUs + p.cfg.OutputTokenProcessingUs, true
}

// rolloutDecodeAdmission predicts decode admission for a TRANSFERRED request: its
// prompt counts as already computed. sim/edpp_scheduler_rollout.go:318-324.
func (p *Policy) rolloutDecodeAdmission(ec *evalCtx, ds Snapshot, theta Coeffs) (float64, bool) {
	result, ok := p.rolloutTimes(ec, ds, theta, ec.inputLen, true, false)
	if !ok {
		return 0, false
	}
	return result.admissionUs, true
}

// rolloutPrefillCompletion predicts (admission, completion) on a prefill pool
// endpoint. sim/edpp_scheduler_rollout.go:326-333.
//
// No post-processing term: this clock ends at prefill completion, not at a
// client-visible token.
func (p *Policy) rolloutPrefillCompletion(ec *evalCtx, ps Snapshot, theta Coeffs) (tAdm, completion float64, ok bool) {
	cached := ec.inputLen - maxInt(p.apForEndpoint(ec, ps.ID), 0)
	result, ok := p.rolloutTimes(ec, ps, theta, cached, false, true)
	if !ok || !result.firstToken {
		return 0, 0, false
	}
	return result.admissionUs, result.firstTokenUs, true
}

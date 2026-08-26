// Package kairospaper specifies the Kairos load-aware prefill-deflection
// comparator, as PRINTED, for the INFOCOM 2027 transfer.
//
// THIS FILE IS A SPECIFICATION LAYER, NOT COMPILING CODE.
//
// # THIS ARM MUST NOT BE REGISTERED AT THIS PIN
//
// It is specified so the algebra is on record and so it can be added with a single
// append once the missing signal exists. It is NOT in transfer.yaml. See K1.
//
// # PINS
//
//	simulation  vishakha-ramani/inference-sim  871b169bb13934ca8dd1e002638e1f6bf490b3b5  (infocom-implementation)
//	target      llm-d/llm-d-router             5f4e762f341a5196393ce79f8a57c3e1900c4a6b  (v0.9.0)
//	engine      vllm-project/vllm              v0.26.0
//
// Upstream: "Towards Load-Aware Prefill Deflection for Disaggregated LLM Serving",
// arXiv:2607.02043, implemented at sim/edpp_kairos.go.
//
// CAMPAIGN POLICY: `kairos_paper_alpha_1p3` (run_public_load_static_benchmark.py:42),
// the arm in the bundle's deployable minimax ranking. Its operating point is
// HARDCODED at run_decisive_campaign.py:398-403:
//
//	--edpp-rule kairos-paper  --kairos-alpha 1.3  --kairos-beta 1.0
//
// DO NOT CONFUSE IT WITH `kairos_beta_0p5`. That policy resolves to
// `--edpp-rule kairos --kairos-beta 0.5` (run_decisive_campaign.py:396-397) -- the
// ADAPTED alias, a different algorithm, at a beta this arm never runs. The two are
// mutually exclusive: the `kairos-paper` rule only ever ships with beta 1.0, and beta
// 0.5 belongs exclusively to the adapted branch. Deploying beta 0.5 into a paper-mode
// plugin halves the TBT budget and silently suppresses deflection.
//
// The simulation deliberately exposes TWO identities (sim/edpp_kairos.go:9-19):
// `kairos-paper` follows the printed rule, and `kairos-adapted` keeps this study's
// earlier relaxations. THIS FILE SPECIFIES `kairos-paper` ONLY -- the campaign
// selects it, and the adapted mode is a different algorithm with a
// continuous-chunk relaxation, an admission-aware decode estimate, and a
// transfer-aware prefill estimate.
//
// # THE POLARITY, WHICH IS OPPOSITE TO THE OTHER TWO ARMS
//
// Kairos's DEFAULT is disaggregation. It DEFLECTS a prefill onto a decode node
// when doing so is predicted to be both faster and SLO-safe. So in the
// simulation's decision vocabulary a deflection is `Disaggregate: false` with a
// decode-pod override (sim/edpp_kairos.go:391-396), and the fallback is
// `Disaggregate: true` with a prefill-pool hint. A port that reads
// "Disaggregate: false" as "the policy declined to act" has the rule backwards.
//
// # THE DECISION
//
// Deflect onto the best eligible decode node iff BOTH inequalities hold
// (marginPassed at sim/edpp_kairos.go:387, gatePassed at :388):
//
//	marginPassed:  bestTTFT <= alpha * ttftPrefill      (alpha = 1.3)
//	gatePassed:    bestTTFT <= tau_ttft                 (arriving request's class)
//
// Otherwise disaggregate to the best prefill path, or keep local if no prefill
// path exists.
//
// # BLOCKING DEGRADATION K1 -- PrefillTokensAhead
//
// The arm reads an instance's REMAINING PROMPT TOKENS in TWO places, and neither is
// available from the target's scraped metrics surface.
//
// WHAT THE QUANTITY ACTUALLY IS. Upstream sums remaining prompt tokens over the
// RUNNING BATCH **and** the wait queue -- both, not the queue alone
// (sim/simulator.go:660-673) -- as sum of max(0, InputLen - ProgressIndex). Decode-only
// requests contribute exactly zero, since their ProgressIndex is already at or beyond
// their input length. So it needs per-request prompt length AND per-request prefill
// PROGRESS, for running and queued requests alike.
//
// IT IS PER-INSTANCE AND POPULATED FOR EVERY ROLE (sim/cluster/snapshot.go:158), so
// only ONE of this arm's two reads is on the prefill pool. The other -- the
// eligibility gate below -- reads it on each DECODE candidate. A port that supplies it
// for prefill endpoints alone leaves the gate reading zero, which is the leak
// described next.
//
// READING ZERO DOES NOT ADD NOISE -- IT INVERTS THE COMPARATOR:
//
//  1. THE ELIGIBILITY GATE LEAKS. It is a DISJUNCTION
//     (sim/edpp_kairos.go:371):
//
//     if ds.ResidentPrefillTokens > 0 || ds.PrefillTokensAhead > 0 { continue }
//
//     so zeroing the second operand does not hold the gate open -- the first
//     still excludes. What it does is DEGRADE the gate from "exclude any node with
//     a deflected prefill anywhere between enqueue and prefill completion" to
//     "exclude only a node being chunk-prefilled THIS STEP". It therefore leaks on
//     every step where a deflected prefill is admitted but not scheduled, and
//     entirely for one still waiting in the queue. The paper's invariant -- at most
//     ONE deflected prefill in flight per decode node -- is broken.
//
//  2. THE POOL QUEUE-WAIT TERM GOES TO ZERO (kairosPaperPrefillTTFT below,
//     sim/edpp_kairos.go:248-252). With sumL == 0 the queueWait is 0, so
//     ttftPrefill collapses to execution time alone, making REMOTE PREFILL LOOK
//     FREE. Since the margin test is bestTTFT <= alpha*ttftPrefill, a smaller
//     ttftPrefill makes the test HARDER to pass -- fewer deflections, more
//     disaggregation.
//
// BOTH EFFECTS PUSH THE SAME WAY. A zero-filled run measures a MORE AGGRESSIVE
// RELATIVE of Kairos, in the direction that makes it look worse. That is a
// MISREPORT, NOT A PARTIAL RESULT -- which is why the arm is omitted rather than
// registered and caveated.
//
// WHAT IS AND IS NOT AVAILABLE, stated precisely. The narrow claim is the true one:
// there is no route through the SCRAPED METRICS surface, whose whole content is the
// seven fields at pkg/epp/framework/interface/datalayer/metrics.go:26-42. It would be
// too strong to say the field cannot be populated at all -- the target already
// maintains a per-endpoint in-flight prompt-token counter of nearly this shape,
// incremented by each request's input token count at dispatch on BOTH the prefill and
// the decode pod
// (pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/requestcontrol_hooks.go:100-106).
//
// THE ARM STILL MUST NOT BE REGISTERED, for reasons that survive that counter:
//   - it counts tokens DISPATCHED, not tokens REMAINING. Without per-request prefill
//     progress it cannot subtract completed prefill work, so it over-states the
//     backlog and drains only on completion.
//   - it sees only requests THIS EPP placed, and only this replica's share -- the same
//     D2-class limitation as the focal arm's shadow table, but here it lands on a GATE
//     rather than on a smooth price, so the error is a WRONG BRANCH rather than a
//     small mis-price.
//   - upstream's sum includes the engine's own running batch and wait queue, which no
//     EPP-side counter observes.
//
// A faithful route is a new engine gauge summing max(0, InputLen - ProgressIndex) over
// the running batch and wait queue, which then rides the existing metrics path into
// endpoint attributes through custom-metrics config with no llm-d-router change. Add
// the arm at that point, not before.
//
// # NON-BLOCKING NOTES, recorded so nobody "improves" them
//
// K2  IT IGNORES PREFIX-CACHE RESIDENCY BY CONSTRUCTION. Algorithm 1 takes the
//     request's FULL PROMPT LENGTH (sim/edpp_kairos.go:228-230, and the deflection
//     search below), not the uncached suffix a_p. Substituting a_p would make it a
//     DIFFERENT ALGORITHM and would flatter it relative to the focal arm on
//     exactly the cache-heavy cells the workloads emphasize (all three declare
//     prefix groups). Do not "fix" this.
//
// K3  ITS RESIDENT-TBT SCAN IS IMMUNE TO D2 UNDER THIS BUNDLE'S CONFIGURATION.
//     kairosResidentTBTTarget minimizes tau_itl over decode residents, falling
//     back to the arriving class when the list is empty
//     (sim/edpp_kairos.go:170-183). Every cohort in all three workloads declares
//     slo_class: standard (config.md §6), so both branches resolve to the
//     IDENTICAL configured tau_itl and a degraded resident list cannot move the
//     TBT budget. D2 becomes live here only if a future campaign introduces a
//     second SLO class.
//
//     ONE SHARP EDGE REGARDLESS: upstream's RunningDecode slice is EMPTY unless
//     admission detail is enabled, and an empty slice silently takes the
//     arriving-class branch instead of the minimize-over-residents branch
//     (sim/edpp_kairos.go:171-174). The two coincide only because of the
//     single-class configuration above. A port must not read an empty resident
//     population as "no constraint".
//
// K4  WHICH DEGRADATIONS IT DOES AND DOES NOT INHERIT.
//     INHERITS the KV-occupancy derivation (config.md §5 signal 6), since
//     KvTokensInUse feeds the deflection search's baseline iteration time.
//     NOT D4: S_pf is passed as a HARD ZERO in that iteration-time call
//     (sim/edpp_kairos.go:118), and ResidentPrefillTokens enters the arm at exactly
//     ONE place -- the eligibility gate -- where only its SIGN is read. So the D4
//     magnitude error is irrelevant here; only "is it nonzero" matters.
//     NOT D7: paper mode never reads FreeKVBlocks at all. The only FreeKVBlocks read
//     in this file is on the ADAPTED path (sim/edpp_kairos.go:328), unused by this arm.
//
// K5  PRINTED EQUATION 1 HAS NO TRANSFER TERM, so D5 (c_xfer unmeasured) does NOT
//     apply to paper mode: `ttft := queueWait + exec` at sim/edpp_kairos.go:258, with
//     no third term. Contrast the ADAPTED mode, which adds c_xfer at
//     sim/edpp_kairos.go:213. A port that adds it has silently switched identities.
//
// D1 (scheduler rollout unobtainable) does NOT apply to paper mode's deflection
// search either: the printed rule has no decode-admission term. The ADAPTED mode
// adds one (sim/edpp_kairos.go:325-333). Paper mode's bestTTFT is pure deflected
// prefill execution.
package kairospaper

import "math"

// Config holds the knobs this arm reads.
type Config struct {
	// KairosAlpha is the TTFT margin. This arm runs 1.3.
	// SANITIZED WITH <= 0, NOT == 0 (sim/edpp.go:600-602): any non-positive value
	// resolves to 1.3. A port testing == 0 lets a NEGATIVE alpha survive, making
	// alpha*ttftPrefill negative, so marginPassed is false for every request and the
	// arm silently never deflects.
	KairosAlpha float64

	// KairosBeta is the TBT safety margin: a deflected prefill chunk must keep the
	// decode step within beta * the strictest resident TBT target. LOWER beta is
	// MORE conservative.
	//
	// THIS ARM RUNS 1.0, hardcoded (run_decisive_campaign.py:402). Not 0.5 -- see the
	// package comment.
	//
	// SANITIZED WITH <= 0 (sim/edpp.go:604-606). Unlike alpha this has no
	// config-validation counterpart upstream, so that guard is the only defence: a
	// non-positive beta gives a non-positive tbtBudget, no chunk candidate satisfies
	// the safety test, every node is infeasible, and the arm degenerates to
	// always-disaggregate while still reporting results.
	KairosBeta float64

	// ChunkTokens caps the profiled chunk candidates. MUST equal the engine's
	// max_num_batched_tokens = 2048 (config.md §1).
	ChunkTokens int

	// TauTTFTUs / TauITLUs and their per-class overrides. Paper mode reads
	// tau_ttft for the arriving request's gate and tau_itl for the resident TBT
	// target. It reads NO tau_e2e.
	TauTTFTUs, TauITLUs              int64
	TauTTFTByClassUs, TauITLByClassUs map[string]int64

	Coeffs      Coeffs
	CoeffsByGPU map[string]Coeffs
}

// maxKairosSteps bounds the greedy schedule search. sim/edpp_kairos.go:361.
const maxKairosSteps = 4096

// paperChunkCandidates is the discrete set the paper profiles, consumed in
// DESCENDING order by Algorithm 1. sim/edpp_kairos.go:24.
var paperChunkCandidates = []float64{2048, 1024, 512, 256, 128}

// discreteCandidates filters the profiled set to those at or below the engine's
// token cap, preserving descending order. sim/edpp_kairos.go:90-103.
//
// The fallback matters: an engine configured BELOW the paper's smallest profiled
// point (128) would otherwise have no executable candidate, so the cap itself is
// appended. Explicit and deterministic rather than an empty-list failure.
func discreteCandidates(chunkCap float64) []float64 {
	out := make([]float64, 0, len(paperChunkCandidates))
	for _, c := range paperChunkCandidates {
		if chunkCap <= 0 || c <= chunkCap {
			out = append(out, c)
		}
	}
	if len(out) == 0 && chunkCap > 0 {
		out = append(out, chunkCap)
	}
	return out
}

// stepPrefill is the prefill-node step time for a chunk of chi tokens attending
// over a resident context of k tokens. sim/edpp_kairos.go:28-33.
//
// An empty chunk still costs the per-iteration intercept.
func stepPrefill(c Coeffs, chi, k float64) float64 {
	if chi <= 0 {
		return c.AlphaP
	}
	return c.AlphaP + c.CPf*chi + c.CAttn*chi*(k+chi/2)
}

// discreteDeflectTTFT implements Algorithm 1's greedy LargestSafe search: at each
// step take the LARGEST profiled chunk whose resulting decode-step time still fits
// the TBT budget. sim/edpp_kairos.go:107-140.
//
// Returns (elapsed, schedule, true) only if the WHOLE prompt can be deflected
// within budget. A step where no candidate fits, or exhausting maxSteps with work
// remaining, makes the node ineligible -- ok == false.
//
// TWO DETAILS THAT ARE LOAD-BEARING:
//   - S_pf IS PASSED AS ZERO in the iteration-time call. That is K4: the baseline
//     is the decode batch's own iteration time with NO resident prefill, and the
//     deflected chunk's cost is added explicitly. Do not substitute the endpoint's
//     ResidentPrefillTokens here -- the eligibility gate has already established
//     it is zero.
//   - THE KV TERM GROWS WITH `done`. The baseline is recomputed each step at
//     kv+done, so the deflected prompt's own accumulating KV inflates subsequent
//     steps. This is why a long prompt can start with large chunks and be forced
//     down to small ones.
//   - THE FINAL CHUNK IS SHORTENED to the remaining tokens: the executor cannot
//     process padding beyond the prompt.
func discreteDeflectTTFT(c Coeffs, bDec int, kv int64, tokens, tbt float64, candidates []float64, maxSteps int) (float64, []float64, bool) {
	if tokens <= 0 {
		return 0, nil, true
	}
	if len(candidates) == 0 {
		return 0, nil, false
	}
	var elapsed, done float64
	schedule := make([]float64, 0, int(math.Ceil(tokens/candidates[len(candidates)-1])))
	for step := 0; step < maxSteps && done < tokens; step++ {
		base := c.tIterDecode(bDec, kv+int64(done), 0) // S_pf = 0 -- see K4
		remaining := tokens - done
		chosen := 0.0
		for _, candidate := range candidates {
			chi := math.Min(candidate, remaining)
			stepTime := base + c.CPf*chi + c.CAttn*chi*(done+chi/2)
			if stepTime <= tbt {
				chosen = chi
				break // LargestSafe: candidates are descending, so first fit wins
			}
		}
		if chosen <= 0 {
			return 0, nil, false // no safe chunk exists at this step
		}
		elapsed += base + c.CPf*chosen + c.CAttn*chosen*(done+chosen/2)
		done += chosen
		schedule = append(schedule, chosen)
	}
	if done < tokens {
		return 0, nil, false
	}
	return elapsed, schedule, true
}

// residentTBTTarget returns the STRICTEST TBT target among the decode residents
// whose latency the safety constraint protects. sim/edpp_kairos.go:170-183.
//
// The ARRIVING class is used only when the node has no decode residents. See K3:
// under this bundle's single-SLO-class workloads both branches resolve to the same
// value, so a degraded resident list cannot move the budget.
func (p *Policy) residentTBTTarget(ds Snapshot, arrivingClass string) float64 {
	if len(ds.RunningDecode) == 0 {
		_, tau := p.targetsFor(arrivingClass)
		return float64(tau)
	}
	strictest := math.Inf(1)
	for _, resident := range ds.RunningDecode {
		_, tau := p.targetsFor(resident.SLOClass)
		if float64(tau) < strictest {
			strictest = float64(tau)
		}
	}
	return strictest
}

// paperPrefillTTFT is PRINTED EQUATION 1: the predicted first-token time on the
// prefill pool. sim/edpp_kairos.go:223-266.
//
//	queueWait = (sumL / chi) * stepPrefill(theta, chi, sumL/2)
//	exec      = sum over chunks of stepPrefill(theta, chunk_i, done_i)
//	ttft      = queueWait + exec
//
// ROUNDING, AND THE TWO STEP COUNTS ARE NOT THE SAME KIND:
//   - the QUEUE step count sumL/chi is DELIBERATELY FRACTIONAL -- no floor, no
//     ceil, no round. Applying ceil() here inflates the queue wait, which makes
//     ttftPrefill larger, which makes the margin test EASIER to pass and produces
//     more deflection than the printed rule.
//   - the EXEC step count is INTEGRAL with a shortened tail: the loop advances by
//     chi and the final chunk is min(chi, tokens-done).
//
// FOUR THINGS TO NOT GET WRONG:
//   - tokens IS THE FULL PROMPT LENGTH, not a_p. That is K2 -- by construction.
//   - sumL IS PrefillTokensAhead, over the running batch AND the wait queue. THIS IS
//     THE UNOBTAINABLE SIGNAL (K1). Equation 2's context is left LITERAL at
//     sumL/2 rather than capped at the prompt length -- the adapted mode caps it
//     (sim/edpp_kairos.go:205), paper mode does not.
//   - NO c_xfer TERM. That is K5: printed Equation 1 is queue wait plus prefill
//     execution and nothing else.
//   - The published algorithm has ONE prefill path. In a multi-prefill topology
//     the same printed estimator is applied to every path and the MINIMUM taken,
//     with the winning path returned as an execution hint so the request is
//     actually served where it was scored.
func (p *Policy) paperPrefillTTFT(req Request, prefillSnaps []Snapshot, chunkCap float64) (float64, string) {
	snaps := sortedByID(prefillSnaps)
	tokens := float64(p.inputLen(req)) // FULL prompt -- K2
	if tokens <= 0 {
		return math.Inf(1), ""
	}
	bestTTFT := math.Inf(1)
	bestPrefillID := ""
	for _, ps := range snaps {
		theta := p.coeffsFor(ps.GPUType)
		chi := chunkCap
		if chi <= 0 || chi > tokens {
			chi = tokens
		}
		// K1 -- NO NATIVE ROUTE ON THE TARGET. Upstream computes this EXACTLY, as
	//   sum over (running batch + wait queue) of max(0, InputLen - ProgressIndex)
	// (sim/simulator.go:660-673). Decode-only requests contribute exactly ZERO,
	// because their ProgressIndex is already at or beyond their input length. So it
	// needs per-request prompt length AND per-request prefill progress for every
	// queued and running request; engine metrics expose only a queue-depth COUNT.
	//
	// Do NOT substitute the adapted mode's approximation
	// (ResidentPrefillTokens + QueueDepth*InputLen, sim/edpp_kairos.go:202): that is
	// a different algorithm's input, and it assumes every queued request has the
	// arriving request's prompt length.
	sumL := float64(ps.PrefillTokensAhead)
		queueWait := 0.0
		if sumL > 0 {
			queueWait = (sumL / chi) * stepPrefill(theta, chi, sumL/2)
		}
		exec := 0.0
		for done := 0.0; done < tokens; done += chi {
			stepChi := math.Min(chi, tokens-done)
			exec += stepPrefill(theta, stepChi, done)
		}
		ttft := queueWait + exec // no transfer term -- K5
		// snaps is ID-sorted and the comparison is STRICT, which preserves
		// deterministic ID tie-breaking when two paths predict equal TTFT.
		if ttft < bestTTFT {
			bestTTFT, bestPrefillID = ttft, ps.ID
		}
	}
	return bestTTFT, bestPrefillID
}

// Decision is the arm's output. Note the polarity described in the package
// comment: Deflect true means prefill runs ON DecodeID, alongside decode.
type Decision struct {
	Deflect       bool
	DecodeID      string
	PrefillID     string
	ChunkSchedule []int
}

// Decide implements decideKairosPaper. sim/edpp_kairos.go:352-402.
//
// THE STALE-SCHEDULE RESET IS PART OF THE RULE, not hygiene. The dispatcher clears
// the request's chunk-schedule metadata BEFORE evaluating
// (sim/edpp_kairos.go:287-291):
//
//	req.PrefillChunkSchedule = nil
//	req.resetPrefillChunkSchedule()   // chunk cursor and remaining -> 0
//
// A Request object can already carry an executable schedule from a previous
// routing decision. If the fresh decision does NOT end in deflection, the request
// must carry NO schedule, so the engine-wide token budget applies. A port that
// leaves a stale schedule attached silently EXECUTES THE PREVIOUS DECISION'S
// CHUNKING -- the request is routed by this decision and chunked by the last one.
func (p *Policy) Decide(req Request, decodeSnaps, prefillSnaps []Snapshot) Decision {
	// Clear stale policy metadata before evaluating. sim/edpp_kairos.go:289-290.
	p.clearPrefillChunkSchedule(req)

	tauTTFTUs, _ := p.targetsFor(p.requestSLOClass(req))

	if p.inputLen(req) == 0 {
		// Empty prompt: nothing to deflect, keep local.
		return Decision{Deflect: false}
	}

	chunkCap := float64(p.cfg.ChunkTokens)
	candidates := discreteCandidates(chunkCap)
	ttftPrefill, prefillID := p.paperPrefillTTFT(req, prefillSnaps, chunkCap)

	// <= 0, matching sim/edpp.go:600-606. NOT == 0 -- see Config.KairosAlpha.
	alpha := p.cfg.KairosAlpha
	if alpha <= 0 {
		alpha = 1.3 // the paper's setting
	}
	beta := p.cfg.KairosBeta
	if beta <= 0 {
		beta = 1.0 // this arm's operating point
	}

	bestTTFT := math.Inf(1)
	bestDecode := ""
	var bestSchedule []float64

	for _, ds := range sortedByID(decodeSnaps) {
		// THE ELIGIBILITY GATE. At most ONE deflected prefill may be in flight on a
		// decode node. THIS DISJUNCTION IS K1: zeroing PrefillTokensAhead degrades
		// it to a step-local check rather than opening it, so the leak is silent.
		// BOTH OPERANDS FAIL PERMISSIVELY. ResidentPrefillTokens is itself
		// simulator-internal -- the sum of in-flight prefill token counts on the
		// instance (sim/cluster/snapshot.go:164) -- and reads 0 when unavailable
		// rather than erroring. So an unpopulated snapshot does not close the gate,
		// it OPENS it. Combined with a zero PrefillTokensAhead the gate degrades to
		// "always eligible", which compounds K1 rather than mitigating it.
		if ds.ResidentPrefillTokens > 0 || ds.PrefillTokensAhead > 0 {
			continue
		}
		residentTau := p.residentTBTTarget(ds, p.requestSLOClass(req))
		tbtBudget := beta * residentTau
		theta := p.coeffsFor(ds.GPUType)
		t, schedule, ok := discreteDeflectTTFT(
			theta, ds.BatchSize, ds.KvTokensInUse,
			float64(p.inputLen(req)), tbtBudget, candidates, maxKairosSteps,
		)
		if !ok {
			continue // no safe schedule on this node
		}
		if t < bestTTFT {
			bestTTFT, bestDecode, bestSchedule = t, ds.ID, schedule
		}
	}

	// THE TWO PRINTED INEQUALITIES. sim/edpp_kairos.go:388-390.
	marginPassed := bestDecode != "" && bestTTFT <= alpha*ttftPrefill
	gatePassed := bestTTFT <= float64(tauTTFTUs)
	if marginPassed && gatePassed {
		return Decision{
			Deflect:       true,
			DecodeID:      bestDecode,
			ChunkSchedule: executableSchedule(bestSchedule),
		}
	}
	if prefillID == "" || math.IsInf(ttftPrefill, 1) {
		return Decision{Deflect: false} // no prefill path -- keep local
	}
	return Decision{Deflect: false, PrefillID: prefillID} // disaggregate
}

// executableSchedule rounds the chunk schedule to integer token counts for the
// executor. sim/edpp_kairos.go:155-164.
func executableSchedule(schedule []float64) []int {
	if len(schedule) == 0 {
		return nil
	}
	out := make([]int, len(schedule))
	for i, chi := range schedule {
		out[i] = int(math.Round(chi))
	}
	return out
}

// targetsFor resolves (tau_ttft, tau_itl) for a class: a per-class override if
// present, else the default. sim/edpp.go:682-695.
func (p *Policy) targetsFor(class string) (tauTTFTUs, tauITLUs int64) {
	tauTTFTUs, tauITLUs = p.cfg.TauTTFTUs, p.cfg.TauITLUs
	if v, ok := p.cfg.TauTTFTByClassUs[class]; ok {
		tauTTFTUs = v
	}
	if v, ok := p.cfg.TauITLByClassUs[class]; ok {
		tauITLUs = v
	}
	return
}

// coeffsFor selects per-endpoint theta by GPU type. sim/edpp.go:709-719. On the
// target the key is the POD LABEL value (config.md §2, §5 signal 8).
func (p *Policy) coeffsFor(gpuType string) Coeffs {
	if gpuType != "" {
		if c, ok := p.cfg.CoeffsByGPU[gpuType]; ok {
			return c
		}
	}
	return p.cfg.Coeffs
}

// Policy holds this arm's state. It needs no per-class output-length mean and no
// shadow-table value inputs: paper mode reads no output-length estimate and no
// resident deadline slack. It DOES need resident SLO classes, for residentTBTTarget.
type Policy struct {
	cfg Config
}

// ---------------------------------------------------------------------------
// Shared symbols -- copy verbatim from the focal arm
// (algorithms/causal_slo_externality.go), package clause aside.
// ---------------------------------------------------------------------------

// Coeffs and tIterDecode are the focal arm's, unchanged. Only AlphaP, CPf, CAttn,
// and tIterDecode are reached by this arm.
type Coeffs struct{ AlphaD, AlphaP, C0, C1, CPf, CAttn float64 }

func (c Coeffs) tIterDecode(bDec int, kv, sPf int64) float64

// Snapshot is the focal arm's, PLUS PrefillTokensAhead.
//
// PrefillTokensAhead HAS NO ROUTE AT THIS PIN (K1). It is the prefill pool's total
// remaining prompt tokens -- an O(queue) sum. Supplying it needs a new vLLM gauge,
// after which it rides the existing metrics path into endpoint attributes through
// custom-metrics config, with no llm-d-router change. Until then this field cannot
// be populated and the arm must not be registered.
type Snapshot struct {
	ID                    string
	GPUType               string
	BatchSize             int
	QueueDepth            int
	KvTokensInUse         int64
	ResidentPrefillTokens int64 // K4: only its SIGN is read, in the eligibility gate
	RunningDecode         []RunningReqState

	PrefillTokensAhead int64 // K1 -- UNOBTAINABLE
}

// RunningReqState is the focal arm's. This arm reads only SLOClass, for
// residentTBTTarget.
type RunningReqState struct {
	StepsDone     int64
	KVBlocks      int64
	TrueRemaining int64
	SLOClass      string
	ArrivalUs     int64
	FirstTokenUs  int64
	TTFTSet       bool
}

type Request any

// clearPrefillChunkSchedule clears any chunk schedule left on the request by a
// previous routing decision. Target-API adapter: the storage is whatever carries
// per-request policy metadata on the target. See the note on Decide -- this is
// part of the rule, and skipping it executes the previous decision's chunking.
func (p *Policy) clearPrefillChunkSchedule(req Request)

// DELIBERATE OMISSION, decision-neutral for this arm: the published dispatcher
// also calls creditAwaiting before dispatching (sim/edpp.go:744), which accrues
// observed first-token lateness into the per-class SLO virtual queues z. Paper mode
// reads no z anywhere, so omitting it changes no decision. Named so its absence
// reads as a decision. A port adding a rule that reads z must add it back.

func sortedByID(snaps []Snapshot) []Snapshot
func (p *Policy) inputLen(req Request) int
func (p *Policy) requestSLOClass(req Request) string

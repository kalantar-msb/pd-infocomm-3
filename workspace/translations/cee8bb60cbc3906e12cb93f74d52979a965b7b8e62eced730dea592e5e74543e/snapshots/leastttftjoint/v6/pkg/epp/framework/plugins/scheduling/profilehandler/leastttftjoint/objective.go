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

// Package leastttftjoint is the least-TTFT joint comparator arm of the INFOCOM 2027
// transfer: `--edpp-rule least-ttft` with `--edpp-ttft-overlap-aware`
// (config.md section 9.3, run_decisive_campaign.py:325-331).
//
// # WHY THIS ARM EXISTS
//
// It shares the candidate set, the estimators, the physics, and the prefix reads with
// the focal arm, and differs IN THE OBJECTIVE ONLY. That is what attributes the measured
// effect to the MECHANISM rather than to the MACHINERY: a weaker comparator would leave
// open that the focal arm won because it computes better physics, not because it prices
// SLO externality.
//
// Cited evidence for the comparison (sim_results/main/): worst static-plan regret 0.0100
// for the focal arm against 0.0542 here, and 0.0031 against 0.1110 in the stress cohort.
// The README's pre-registered expectation is an ORDERING claim, not a margin.
//
// # THE OBJECTIVE
//
// The argmin of the ARRIVING REQUEST'S OWN projected time-to-first-token, over the same
// candidate set. It carries no backlog or congestion drift, no SLO virtual queues, no
// externality over residents, NO tau AT ALL, and no transfer penalty beyond the transfer
// LATENCY already inside the disaggregated TTFT.
//
// Each side uses its own candidate's theta, so the arm is hardware-aware by
// construction. That is deliberate: a hardware-BLIND least-TTFT baseline would lose to
// the focal arm partly because it cannot see the fleet, which would confound the
// mechanism with hardware awareness.
//
// # HOW THE VERBATIM-COPY CONTRACT IS SATISFIED
//
// The specification layer requires roughly forty symbols to be VERBATIM COPIES of the
// focal arm's, package clause aside (algorithms/least_ttft_joint.go:47-96): Coeffs and
// every method, the admission contexts and every estimator, the whole scheduler rollout,
// the projection helpers, the shadow-table reads, the types, and the target adapters. Its
// stated reason is that "a re-derived-but-slightly-different estimator would silently
// destroy the attribution argument while every test still passed".
//
// THIS PORT SATISFIES THAT BY SHARING, NOT BY COPYING, which is strictly stronger: a copy
// can drift under an edit to one side, and drift is precisely the failure named. Every
// listed symbol has exactly ONE implementation, in package causalsloexternality, and both
// arms reach it through causalsloexternality.Eval. One Coeffs, one rollforward estimator,
// one shadow table, one candidate enumeration, one argmin.
//
// The consequence for a reader is worth stating: THIS PACKAGE CONTAINS NO PHYSICS. It
// contains one cost function whose every operand comes from Eval, plus the two plugin
// registrations that switch it on. If you find yourself adding a coefficient, a queue
// estimate, or a unit conversion here, the shared side is missing an accessor -- add it
// there.
//
// Both arms also read the SAME prefix-cache and token producers, bound BY PRODUCER NAME
// in plugin config (config.md section 5 signals 10 and 11). A data key's identity
// includes its producer name, so an unbound read can silently become a second,
// differently-configured signal -- and then the two arms are pricing different prompts.
//
// THAT SHARING IS NOT INHERITED, AND IT IS IMPORTANT NOT TO BELIEVE IT IS.
// router.epp.pluginsCustomConfig["custom-plugins.yaml"] is a SCALAR STRING, so the
// treatment overlay's value REPLACES the baseline's outright rather than deep-merging into
// it. Nothing in the baseline's inner plugin config survives into a treatment, and both
// treatment overlays therefore re-declare token-producer and approx-prefix-cache-producer
// themselves. The binding is shared only because the baseline and the two treatment
// overlays restate identical producer names and identical producer parameters -- an
// agreement maintained BY HAND across three files, and asserted by test
// (TestOverlayIsCommonModeWithTheFocalArmOutsideTheObjective), not guaranteed by the merge.
//
// The practical consequence, which is the reason this is spelled out rather than
// summarised: editing the producers in the BASELINE overlay does NOT change either arm.
// Both treatment overlays must be edited too, and the two must still agree with each other
// afterwards. A reader who believes otherwise makes one edit, assumes both arms followed,
// and reintroduces exactly the differently-configured-signal hazard that signal 11 exists
// to prevent.
//
// # WHAT THIS ARM DOES NOT READ
//
// It drops fields rather than populating unread ones, because a populated-but-never-read
// field implies a consumer exists somewhere. tau_ttft, tau_itl, and tau_e2e are dropped
// ENTIRELY -- no SLO threshold reaches this arm's score anywhere, so the tau selector of
// config.md section 6 is IRRELEVANT here and cannot be flattened by a zero triple the way
// the focal arm can. So are the resident value inputs: ArrivalUs, FirstTokenUs, and
// TTFTSet are never read, so degradation D2c (the late-first-token bias) does not affect
// this arm. There is no capacity account and no value kernel.
//
// That absence is ENFORCED rather than documented: Objective.ReadsSLOValueConfig returns
// false, and the shared Config validation then REFUSES to start if v, ablation,
// activeWorkload, workloadTargets, or capacity appear in this arm's parameters.
//
// What it does still read is the resident POPULATIONS, in three places, so their
// degradations are inherited: D4 (ResidentPrefillTokens enters both iteration-time laws
// as S_pf, on every candidate), D2 (RunningDecode feeds decodeRemStepsEst and the
// rollforward KV walk, so StepsDone and KVBlocks are read), and D6 (RunningPrefill feeds
// prefillRemStepsEst). D7 is inherited through the admission context.
//
// # DECLARED DEGRADATIONS THAT DIFFER FROM THE FOCAL ARM'S
//
// D1 (the scheduler rollout is unobtainable) is inherited in full, with the same
// substitution counter -- BUT ITS DIRECTION IS NOT A SINGLE SIGN HERE, and the focal
// arm's one-line summary must not be copied over. The two placements join their clocks
// differently: local ADDS tAdmD with slope 1, while disaggregated takes
// max(remoteLead, tAdmD). So when tAdmD dominates the max, understating it is charged
// identically to both placements and CANCELS; when remoteLead dominates, the
// disaggregated candidate is insensitive to tAdmD altogether, so understating it lowers
// ONLY the local candidate -- biasing toward LOCAL, the opposite of the focal arm's
// direction. The toward-remote direction survives only through the tAdmP channel, and
// only while remoteLead dominates.
//
// THIS MATTERS FOR THE ATTRIBUTION ARGUMENT. It would be convenient to say both arms
// inherit D1 identically so the comparison stays fair, but a load-dependent direction
// cannot support that claim: under decode-side congestion -- the regime the high-load and
// burst cohorts target -- D1 shifts the two arms' local/remote splits by different
// amounts. Treat the comparison as fair only where the substitution counter shows the
// same estimator regime on both arms.
//
// AND THE ROLLOUT WAS THE LIVE PATH UPSTREAM FOR THIS ARM, not a fallback: selecting this
// rule makes the simulation record the resident and scheduler detail that activates the
// rollout, and the closed-form projection is reached only when that guard fails. On the
// target the guard is permanently false, so BOTH arms run an estimator that produced none
// of the numbers in sim_results/.
//
// D5 (c_xfer unmeasured) is inherited and matters MORE here than in the focal arm: there
// is no externality term to partially offset a mis-priced transfer, so an unmeasured
// c_xfer moves this arm's decisions systematically. It enters at exactly one place,
// remoteLeadUs below. It is NOT, however, the only size-dependent price of going remote:
// the disaggregated projection also adds the re-timed B+1 first decode iteration, whose
// KV component scales with the request's own input length, so prompt size prices the
// remote path through two channels rather than one.
//
// D8 (tokenization can be unavailable) is inherited, and its own counter is per arm
// because the two arms are separate plugin instances with separate producer bindings. If
// the FALLBACK RATE differs between them it confounds the very comparison the arms exist
// to make, and it is likeliest on long prompts -- exactly where local-versus-remote is
// most contested. Compare the two arms' rates before reading any result.
//
// D9 (the per-class N_out mean folds requests that never completed) is inherited
// unchanged from the shared handler, and it applies here for a narrower reason than in
// the focal arm: this arm reads no resident value, but nHatOut still feeds
// decodeRemStepsEst and the rollout's censored output estimate, so a truncated count
// still moves the admission estimate. See the shared package's README.
package leastttftjoint

import (
	"math"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality"
)

// Objective is THE ONLY THING THIS ARM CONTRIBUTES: the deciding request's own forward
// time-to-first-token for one candidate, minimised over the shared candidate set.
//
// It is the trailing-else branch of upstream's joint decider, where costFn is set to
// jointCandidateTTFT (sim/edpp.go:1448-1464, assignment at :1453).
type Objective struct{}

var _ causalsloexternality.Objective = Objective{}

// Cost is the arriving request's projected TTFT under one candidate, in MICROSECONDS.
// sim/edpp.go:1891-1940.
//
// ps == nil means LOCAL (prefill and decode co-resident on ds); otherwise decode on ds
// and prefill on *ps.
//
// The arithmetic reproduces EXACTLY the focal arm's tHatLocal and tHatDisagg, and that
// exactness IS the attribution argument: the two arms compute the same TTFT and differ
// only in what they do with it. Here it is the objective; there it is one input to a
// value kernel. Every operand below comes from the shared Eval, so "exactly" is a
// structural property rather than a claim maintained by hand.
//
// THE SCALE IS A LATENCY: non-negative and unbounded above. It must not pass through the
// profile's score-range enforcement, which clamps to [0,1]
// (pkg/epp/scheduling/scheduler_profile.go:212, :364) and would collapse essentially
// every candidate to a single score. That is one of the two reasons this arm attaches as
// a Picker rather than a Scorer -- the other being that a Scorer returns a per-endpoint
// map and cannot name a PAIR. The focal arm faces the same clamp hazard for a different
// reason: its J is signed and unbounded.
//
// THE REDUCED-RULE EQUIVALENCE HOLDS ONLY UNDER HOMOGENEOUS theta. Upstream asserts that
// restricting this to a single decode endpoint reproduces the reduced least-ttft decision
// (sim/edpp.go:1889), and that is true when one coefficient set is in play. It is NOT
// true with per-GPU coefficients loaded: the reduced path prices remote prefill with the
// GLOBAL coefficients (sim/edpp.go:906, and again in its prefill admission context at
// :882), while this objective uses the prefill endpoint's own thetaP (:1915). config.md
// section 4 puts AlphaP at 16617.85 us on H100 against 25568.35 us on A100, so the two
// disagree by roughly 8950 us per prefill iteration -- larger than the entire transfer
// term. So that equivalence is the cheapest single-endpoint check available for
// validating this port, and it WILL diverge on the h100_a100_realistic cohorts, which are
// the fleets this experiment is about. Run it on a homogeneous fleet, or a divergence
// leaves no way to tell whether the port or the equivalence claim is at fault.
func (Objective) Cost(e causalsloexternality.Eval, ds causalsloexternality.Snapshot,
	ps *causalsloexternality.Snapshot) float64 {
	thetaD := e.CoeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	tIterD := e.TIterDecode(thetaD, bDec, kv, sPfD)
	// The B+1 re-timed FIRST decode iteration: the batch grows by one and KV grows by
	// the arriving request's full input length. Read by the DISAGGREGATED branch only.
	tIterFirstDecode := e.TIterDecode(thetaD, bDec+1, kv+int64(e.InputLen()), sPfD)
	tAdmD := e.DecodeAdmissionUs(ds)

	if ps == nil {
		// ---- LOCAL: prefill and decode co-resident on ds, so prefill uses the DECODE
		// theta.
		apLoc := e.APForEndpoint(ds.ID)
		nChunksLoc, _ := e.ChunkTerms(thetaD, apLoc)
		// Wp, NOT nChunks*deltaPfChunk, so the estimate keeps the QUADRATIC attention
		// cost and matches the executor. max(apLoc, 0) is the shared maxInt guard from
		// the copy contract -- identical arithmetic, expressed as the language builtin.
		wpLoc := thetaD.Wp(max(apLoc, 0), e.InputLen())
		// LOCAL ADDS tAdmD WITH SLOPE 1, and no decode iteration precedes the first
		// token: local execution samples it when prefill completes (sim/edpp.go:1245
		// against :1252). That asymmetry against the disaggregated branch is real.
		tHat := e.ProjectedLocalTTFT(tAdmD, nChunksLoc, tIterD, wpLoc)
		// Degradation D1: unreachable at this pin, so the closed-form tHat stands. The
		// branch is kept so a future engine patch supplying the wait queue activates
		// BOTH arms at once.
		if _, rolloutTTFT, ok := e.RolloutLocalTTFT(ds, thetaD); ok {
			tHat = rolloutTTFT
		}
		return tHat
	}

	// ---- DISAGGREGATED: decode on ds, prefill on *ps, so prefill uses the PREFILL
	// ENDPOINT'S OWN theta. That is what makes the arm hardware-aware, and it is where
	// it parts company with upstream's reduced least-ttft path.
	thetaP := e.CoeffsFor(ps.GPUType)
	apP := e.APForEndpoint(ps.ID)
	nChunksP, _ := e.ChunkTerms(thetaP, apP)
	wpP := thetaP.Wp(max(apP, 0), e.InputLen())
	tIterP := e.TIterPrefill(thetaP, ps.ResidentPrefillTokens)
	if rolloutAdm, ok := e.RolloutDecodeAdmission(ds, thetaD); ok {
		tAdmD = rolloutAdm // D1
	}
	tAdmP := e.PrefillAdmissionUs(*ps)
	prefillCompletionUs := tAdmP + nChunksP*tIterP + wpP
	if _, rolloutCompletion, ok := e.RolloutPrefillCompletion(*ps, thetaP); ok {
		prefillCompletionUs = rolloutCompletion // D1
	}
	// THE ONLY PLACE c_xfer ENTERS THIS OBJECTIVE -- degradation D5, unmeasured.
	remoteLeadUs := prefillCompletionUs + e.CXferUs()

	// MAX, NOT SUM: remote preparation and decode-queue drainage are CONCURRENT clocks
	// from the routing instant. Adding them serially over-prices disaggregation by up to
	// a full decode admission delay.
	//
	// THIS max() IS UNCONDITIONAL, AND THE FLAG THAT LOOKS LIKE ITS SWITCH IS NOT. The
	// campaign passes --edpp-ttft-overlap-aware (run_decisive_campaign.py:325-331), but
	// cfg.TTFTOverlapAware gates only the REDUCED path (sim/edpp.go:909) and is never
	// consulted on the joint path -- its only two occurrences upstream are the config
	// field (:110) and that one gate. So the flag is a NO-OP for this arm, and there is
	// deliberately no configuration knob for it here: exposing one would invite making
	// this max() conditional, which reintroduces the serialized form whenever the flag is
	// off. That is a different algorithm, and one that systematically under-selects
	// remote prefill.
	decodeJoinUs := math.Max(remoteLeadUs, tAdmD)
	return e.ProjectedDisaggTTFT(decodeJoinUs, tIterFirstDecode)
}

// ScorerFirstEnumeration is FALSE: this arm enumerates decode endpoints in PLAIN
// ASCENDING-ID ORDER.
//
// The focal arm moves the inherited scorer's pick to the front so that restricting
// enumeration to it reproduces the decomposed rule; THIS arm does not, and upstream does
// not either -- the least-ttft branch iterates decodeSnaps directly (sim/edpp.go:1456),
// and scorerFirstSnapshots is applied only in the SLO-externality and causal-VaR
// branches, its own doc calling it "the causal-VaR equality tie-break order"
// (sim/edpp.go:2176-2178).
//
// Reordering here would change which candidate wins an EXACT tie, silently and only
// sometimes -- which is exactly why it is asserted by a test rather than left to be
// noticed. It also makes the decomposition ablation unreachable for this arm by
// construction, since that control depends on the first enumerated endpoint being the
// scorer's pick; that is correct, because config.md section 10's decomposition knob is
// `DecomposedSLOExternality`, a focal-arm switch.
func (Objective) ScorerFirstEnumeration() bool { return false }

// ReadsSLOValueConfig is FALSE: no tau, no V, no ablation, no capacity.
//
// It is the enforcement point for "what this arm does not read". The shared validation
// REJECTS those keys rather than ignoring them, so an operator who copied the focal arm's
// overlay and tuned tau in it gets a startup error instead of a silently inert setting --
// see Config.rejectSLOValueConfig. The specification layer is explicit that adding either
// a tau field or a nominal prefill token count "would imply a consumer that does not
// exist" (algorithms/least_ttft_joint.go:237-242).
func (Objective) ReadsSLOValueConfig() bool { return false }

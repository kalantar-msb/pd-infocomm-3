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
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality"
)

// The tests in this file assert the OBJECTIVE'S ARITHMETIC against values composed from
// the SHARED primitives -- Eval.TIterDecode, Eval.DecodeAdmissionUs, Coeffs.Wp, and the
// rest. They deliberately do not restate a physics law: every operand is read back from
// the same implementation the plugin uses, and what is asserted is the COMPOSITION.
//
// That is the right division. A test oracle that re-derived the laws would be the very
// drift the verbatim-copy contract forbids, one layer down. A test that only checked which
// endpoint won would catch a dropped post-processing term or a serialized clock join only
// when the error happened to flip a ranking on the fixture at hand -- and for an objective
// whose entire content is an arithmetic expression in microseconds, that is not adequate.
//
// A NOTE ON THE FIXTURES. An endpoint with an empty batch and abundant free KV takes the
// rollforward estimator's immediate-admission branch, which is floored at one iteration
// rather than zero: even with a free slot a request waits for the current step to finish
// before the next batch formation admits it. Several tests assert that identity explicitly,
// so the reason the numbers below are what they are stays visible.

const testInputLen = 1024

// idleDecode is a decode endpoint with nothing running: immediate admission, floored at
// one decode iteration.
func idleDecode(id, gpuType string) causalsloexternality.Snapshot {
	return causalsloexternality.Snapshot{
		ID:              id,
		GPUType:         gpuType,
		BatchSize:       0,
		QueueDepth:      0,
		KvTokensInUse:   0,
		FreeKVBlocks:    10000,
		MaxBatchSize:    256,
		BlockSizeTokens: 16,
	}
}

// idlePrefill is a dedicated prefill endpoint with nothing running. A prefill server runs
// no decode work, so its iteration law reads only S_pf.
func idlePrefill(id, gpuType string) causalsloexternality.Snapshot {
	snap := idleDecode(id, gpuType)
	snap.ID = id
	return snap
}

// congestedDecode is a decode endpoint at full batch with a queue deeper than one drain,
// which is the regime in which decode admission dominates the disaggregated clock join.
func congestedDecode(id, gpuType string) causalsloexternality.Snapshot {
	snap := idleDecode(id, gpuType)
	snap.BatchSize = 256
	snap.QueueDepth = 1000
	snap.FreeKVBlocks = 0
	return snap
}

// evalFor builds the shared physics bound to one decision.
func evalFor(t *testing.T, cfg causalsloexternality.Config, inputLen int, apByEndpoint map[string]int,
	snaps ...causalsloexternality.Snapshot) causalsloexternality.Eval {
	t.Helper()
	eval, stop, err := (causalsloexternality.EvalFixture{
		Config:       cfg,
		Objective:    Objective{},
		Snapshots:    snaps,
		SLOClass:     "standard",
		InputLen:     inputLen,
		APByEndpoint: apByEndpoint,
		PluginType:   HandlerPluginType,
	}).Build()
	if err != nil {
		t.Fatalf("build eval fixture: %v", err)
	}
	t.Cleanup(stop)
	return eval
}

// closeEnough compares two microsecond quantities. The tolerance is relative because the
// magnitudes here are tens of thousands of microseconds and the arithmetic is a sum of
// products, so bit-exact equality would be asserting float association order rather than
// the formula.
func closeEnough(t *testing.T, what string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-6*math.Max(1, math.Abs(want)) {
		t.Errorf("%s = %.6f us, want %.6f us (delta %.6f)", what, got, want, got-want)
	}
}

// ---------------------------------------------------------------------------
// The LOCAL branch
// ---------------------------------------------------------------------------

// TestLocalCostIsAdmissionPlusChunkedIterationsPlusPrefillWork pins the local projection
// in full, including the quadratic attention cost.
//
// Wp is used rather than nChunks*deltaPfChunk so the estimate keeps the QUADRATIC
// attention term and matches the executor; a port that charged the linear per-chunk term
// instead would be cheaper by 0.5*CAttn*ar^2, which at 1024 tokens is only ~53 us but
// grows as the square of the prompt and would silently favour local placement on exactly
// the long prompts where the local/remote boundary is contested.
func TestLocalCostIsAdmissionPlusChunkedIterationsPlusPrefillWork(t *testing.T) {
	cfg := comparatorConfig()
	ds := idleDecode("default/d1", gpuH100)
	eval := evalFor(t, cfg, testInputLen, nil, ds)

	theta := h100Coeffs()
	tIterD := eval.TIterDecode(theta, ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens)
	tAdmD := eval.DecodeAdmissionUs(ds)

	// The immediate-admission branch is floored at ONE iteration, not zero.
	closeEnough(t, "idle decode admission", tAdmD, tIterD)

	// a_p is the full prompt: no prefix observation was attached, and a miss means "no
	// information", not "nothing cached".
	nChunks, _ := eval.ChunkTerms(theta, testInputLen)
	if nChunks != 1 {
		t.Fatalf("a %d-token prompt under a %d-token chunk budget is one chunk, got %g",
			testInputLen, cfg.Engine.ChunkTokens, nChunks)
	}
	want := tAdmD + nChunks*tIterD + theta.Wp(testInputLen, testInputLen) + cfg.OutputTokenProcessingUs

	closeEnough(t, "local cost", Objective{}.Cost(eval, ds, nil), want)
}

// TestCachedPrefixReducesTheLocalCostToAdmissionAlone covers the a_p channel: a fully
// cached prompt has no prefill work at all, so the local projection collapses to the
// admission delay plus post-processing.
func TestCachedPrefixReducesTheLocalCostToAdmissionAlone(t *testing.T) {
	cfg := comparatorConfig()
	ds := idleDecode("default/d1", gpuH100)

	cached := evalFor(t, cfg, testInputLen, map[string]int{ds.ID: 0}, ds)
	uncached := evalFor(t, cfg, testInputLen, nil, ds)

	cachedCost := Objective{}.Cost(cached, ds, nil)
	uncachedCost := Objective{}.Cost(uncached, ds, nil)

	closeEnough(t, "fully cached local cost", cachedCost,
		cached.DecodeAdmissionUs(ds)+cfg.OutputTokenProcessingUs)
	if !(cachedCost < uncachedCost) {
		t.Errorf("a cached prompt must cost less than an uncached one: cached %.3f, uncached %.3f",
			cachedCost, uncachedCost)
	}
}

// ---------------------------------------------------------------------------
// The DISAGGREGATED branch
// ---------------------------------------------------------------------------

// TestDisaggregatedCostIsRemoteLeadPlusTheRetimedFirstDecodeIteration pins the whole
// disaggregated projection when the remote lead dominates the clock join.
//
// It also pins the asymmetry against the local branch: the disaggregated form adds the
// B+1 re-timed first decode iteration and the local form does not, because local samples
// its first token when prefill completes and no decode iteration precedes it. That term's
// KV component scales with the request's own input length, so prompt size prices the
// remote path through TWO channels, not just c_xfer.
func TestDisaggregatedCostIsRemoteLeadPlusTheRetimedFirstDecodeIteration(t *testing.T) {
	cfg := comparatorConfig()
	ds := idleDecode("default/d1", gpuH100)
	ps := idlePrefill("default/p1", gpuH100)
	eval := evalFor(t, cfg, testInputLen, nil, ds, ps)

	thetaD, thetaP := h100Coeffs(), h100Coeffs()
	tAdmD := eval.DecodeAdmissionUs(ds)
	tAdmP := eval.PrefillAdmissionUs(ps)
	tIterP := eval.TIterPrefill(thetaP, ps.ResidentPrefillTokens)
	nChunksP, _ := eval.ChunkTerms(thetaP, testInputLen)
	remoteLead := tAdmP + nChunksP*tIterP + thetaP.Wp(testInputLen, testInputLen) + eval.CXferUs()

	if !(remoteLead > tAdmD) {
		t.Fatalf("this fixture is meant to have the remote lead dominate: remoteLead %.3f, tAdmD %.3f",
			remoteLead, tAdmD)
	}

	// bDec+1 and KV grown by the arriving request's FULL input length.
	tIterFirstDecode := eval.TIterDecode(thetaD, ds.BatchSize+1,
		ds.KvTokensInUse+int64(testInputLen), ds.ResidentPrefillTokens)
	want := remoteLead + tIterFirstDecode + cfg.OutputTokenProcessingUs

	closeEnough(t, "disaggregated cost", Objective{}.Cost(eval, ds, &ps), want)

	// The local branch on the same endpoint must NOT carry that term.
	localTIterD := eval.TIterDecode(thetaD, ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens)
	nChunksLoc, _ := eval.ChunkTerms(thetaD, testInputLen)
	closeEnough(t, "local cost carries no first decode iteration", Objective{}.Cost(eval, ds, nil),
		tAdmD+nChunksLoc*localTIterD+thetaD.Wp(testInputLen, testInputLen)+cfg.OutputTokenProcessingUs)
}

// TestClocksJoinWithMaxNotSum is the decisive shape test for the disaggregated branch.
//
// Remote preparation and decode-queue drainage are CONCURRENT clocks from the routing
// instant, so they join with max(). Adding them serially over-prices disaggregation by up
// to a full decode admission delay and systematically under-selects remote prefill -- a
// different algorithm, and one that still runs and still reports goodput.
//
// The max is UNCONDITIONAL. Upstream has a --edpp-ttft-overlap-aware flag, but it gates
// only the REDUCED path (sim/edpp.go:909) and is never consulted on the joint path, so it
// is a no-op for this arm; there is deliberately no knob for it here, because a knob
// invites making the max conditional.
//
// The fixture puts decode admission ON TOP, which is the half a serialized port would get
// wrong by the largest margin, and it is the regime the high-load and burst cohorts
// target.
func TestClocksJoinWithMaxNotSum(t *testing.T) {
	cfg := comparatorConfig()
	ds := congestedDecode("default/d1", gpuH100)
	ps := idlePrefill("default/p1", gpuH100)
	eval := evalFor(t, cfg, testInputLen, nil, ds, ps)

	thetaD, thetaP := h100Coeffs(), h100Coeffs()
	tAdmD := eval.DecodeAdmissionUs(ds)
	tAdmP := eval.PrefillAdmissionUs(ps)
	tIterP := eval.TIterPrefill(thetaP, ps.ResidentPrefillTokens)
	nChunksP, _ := eval.ChunkTerms(thetaP, testInputLen)
	remoteLead := tAdmP + nChunksP*tIterP + thetaP.Wp(testInputLen, testInputLen) + eval.CXferUs()
	tIterFirstDecode := eval.TIterDecode(thetaD, ds.BatchSize+1,
		ds.KvTokensInUse+int64(testInputLen), ds.ResidentPrefillTokens)

	if !(tAdmD > remoteLead) {
		t.Fatalf("this fixture is meant to have decode admission dominate: tAdmD %.3f, remoteLead %.3f",
			tAdmD, remoteLead)
	}

	got := Objective{}.Cost(eval, ds, &ps)
	closeEnough(t, "disaggregated cost under decode-side congestion", got,
		tAdmD+tIterFirstDecode+cfg.OutputTokenProcessingUs)

	serial := tAdmD + remoteLead + tIterFirstDecode + cfg.OutputTokenProcessingUs
	if got >= serial {
		t.Errorf("the two clocks were serialized: cost %.3f is not below the serial form %.3f, so "+
			"every disaggregated candidate is over-priced by up to a full decode admission delay",
			got, serial)
	}
}

// TestRemotePrefillIsPricedWithThePrefillEndpointsOwnTheta is the HARDWARE-AWARENESS test,
// and it is the one that separates this port from upstream's reduced least-ttft path.
//
// Upstream asserts that restricting the joint rule to a single decode endpoint reproduces
// the reduced rule (sim/edpp.go:1889), and that holds under homogeneous theta. It does NOT
// hold with per-GPU coefficients loaded: the reduced path prices remote prefill with the
// GLOBAL coefficients (sim/edpp.go:906, :882) while this objective uses the prefill
// endpoint's own thetaP (:1915). config.md section 4 puts AlphaP at 16617.85 us on H100
// against 25568.35 us on A100, so the two disagree by roughly 8950 us per prefill
// iteration -- larger than the entire transfer term.
//
// A port that used the decode theta, or one global set, for remote prefill would still
// run, still report goodput, and would be hardware-BLIND on exactly the heterogeneous
// fleet where the mechanism has room. This test fails in that case, and it is the only
// test here that would.
func TestRemotePrefillIsPricedWithThePrefillEndpointsOwnTheta(t *testing.T) {
	cfg := comparatorConfig()
	ds := idleDecode("default/d1", gpuH100)
	fast := idlePrefill("default/p-h100", gpuH100)
	slow := idlePrefill("default/p-a100", gpuA100)
	eval := evalFor(t, cfg, testInputLen, nil, ds, fast, slow)

	fastCost := Objective{}.Cost(eval, ds, &fast)
	slowCost := Objective{}.Cost(eval, ds, &slow)

	if !(slowCost > fastCost) {
		t.Fatalf("an A100 prefill endpoint must be priced above an H100 one: A100 %.3f, H100 %.3f. "+
			"Equal costs mean remote prefill is priced with a single theta and the arm is "+
			"hardware-blind on the heterogeneous fleet", slowCost, fastCost)
	}

	// The gap must be exactly the difference in the remote lead, since decode is on the
	// SAME endpoint in both candidates and the first-decode term therefore cancels.
	remoteLead := func(theta causalsloexternality.Coeffs, ps causalsloexternality.Snapshot) float64 {
		nChunks, _ := eval.ChunkTerms(theta, testInputLen)
		return eval.PrefillAdmissionUs(ps) + nChunks*eval.TIterPrefill(theta, ps.ResidentPrefillTokens) +
			theta.Wp(testInputLen, testInputLen) + eval.CXferUs()
	}
	wantGap := remoteLead(a100Coeffs(), slow) - remoteLead(h100Coeffs(), fast)
	closeEnough(t, "A100-minus-H100 prefill cost gap", slowCost-fastCost, wantGap)
}

// TestTransferPriceEntersOnlyTheDisaggregatedCandidate pins degradation D5's single entry
// point. c_xfer is the only size-dependent remote price that this objective states
// explicitly, and it is UNMEASURED, so a wrong value mis-prices systematically rather than
// noisily. It matters more here than in the focal arm: there is no externality term to
// partially offset it.
func TestTransferPriceEntersOnlyTheDisaggregatedCandidate(t *testing.T) {
	base := comparatorConfig()
	dearer := comparatorConfig()
	const bump = 1000.0
	dearer.Transfer.XferBaseUs += bump

	ds := idleDecode("default/d1", gpuH100)
	ps := idlePrefill("default/p1", gpuH100)

	baseEval := evalFor(t, base, testInputLen, nil, ds, ps)
	dearerEval := evalFor(t, dearer, testInputLen, nil, ds, ps)

	closeEnough(t, "local cost is transfer-independent",
		Objective{}.Cost(dearerEval, ds, nil), Objective{}.Cost(baseEval, ds, nil))
	closeEnough(t, "disaggregated cost rises by exactly the transfer bump",
		Objective{}.Cost(dearerEval, ds, &ps)-Objective{}.Cost(baseEval, ds, &ps), bump)
}

// TestPostProcessingIsAddedOnceToEachPlacement pins config.md signal 25.
//
// OutputTokenProcessingUs sits OUTSIDE the calibrated theta and must be added to every
// TTFT projection. The shipped overlay carries 0.0, which is exactly why this test sets a
// non-zero value: at 0.0 a port that dropped the term entirely would pass every other test
// in this file.
func TestPostProcessingIsAddedOnceToEachPlacement(t *testing.T) {
	const postProcessing = 250.0
	base := comparatorConfig()
	withPost := comparatorConfig()
	withPost.OutputTokenProcessingUs = postProcessing

	ds := idleDecode("default/d1", gpuH100)
	ps := idlePrefill("default/p1", gpuH100)
	baseEval := evalFor(t, base, testInputLen, nil, ds, ps)
	postEval := evalFor(t, withPost, testInputLen, nil, ds, ps)

	closeEnough(t, "local placement adds it once",
		Objective{}.Cost(postEval, ds, nil)-Objective{}.Cost(baseEval, ds, nil), postProcessing)
	closeEnough(t, "disaggregated placement adds it once",
		Objective{}.Cost(postEval, ds, &ps)-Objective{}.Cost(baseEval, ds, &ps), postProcessing)
}

// TestCostIsAnUnboundedMicrosecondLatency pins the reason this arm cannot be a Scorer.
//
// Every scorer contribution passes through the profile's score-range enforcement, which
// clamps to [0,1] (pkg/epp/scheduling/scheduler_profile.go:212, :364). This objective is a
// latency in microseconds -- non-negative and unbounded above -- so clamping would collapse
// essentially every candidate to a single score and destroy the ranking rather than degrade
// it. The focal arm faces the same hazard because its J is signed and unbounded. A Picker's
// own objective does not pass through the clamp, which is why both arms attach there.
func TestCostIsAnUnboundedMicrosecondLatency(t *testing.T) {
	cfg := comparatorConfig()
	ds := idleDecode("default/d1", gpuH100)
	ps := idlePrefill("default/p1", gpuH100)
	eval := evalFor(t, cfg, testInputLen, nil, ds, ps)

	for name, cost := range map[string]float64{
		"local":         Objective{}.Cost(eval, ds, nil),
		"disaggregated": Objective{}.Cost(eval, ds, &ps),
	} {
		if cost <= 1.0 {
			t.Errorf("%s cost %.6f is inside the unit interval; the fixture should produce a "+
				"realistic microsecond latency so the clamp hazard is concrete", name, cost)
		}
		if cost < 0 {
			t.Errorf("%s cost %.6f is negative; this objective is a latency", name, cost)
		}
	}
}

// TestLongerPromptsCostMoreOnBothPlacements is a monotonicity sanity check over the one
// operand every term in the objective reads.
func TestLongerPromptsCostMoreOnBothPlacements(t *testing.T) {
	cfg := comparatorConfig()
	ds := idleDecode("default/d1", gpuH100)
	ps := idlePrefill("default/p1", gpuH100)

	short := evalFor(t, cfg, 512, nil, ds, ps)
	long := evalFor(t, cfg, 8192, nil, ds, ps)

	if !(Objective{}.Cost(long, ds, nil) > Objective{}.Cost(short, ds, nil)) {
		t.Error("a longer prompt must not cost less locally")
	}
	if !(Objective{}.Cost(long, ds, &ps) > Objective{}.Cost(short, ds, &ps)) {
		t.Error("a longer prompt must not cost less remotely")
	}
}

// TestObjectiveDeclaresWhatItDoesNotRead pins the two declared properties an arm carries
// besides its cost function. Both are silent-failure surfaces: the enumeration order shows
// up only on an exact tie, and the config surface only in what an operator can set.
func TestObjectiveDeclaresWhatItDoesNotRead(t *testing.T) {
	if (Objective{}).ScorerFirstEnumeration() {
		t.Error("this arm enumerates decode endpoints in plain ascending-ID order: upstream's " +
			"least-ttft branch iterates decodeSnaps directly and applies the scorer-first " +
			"reordering only in the SLO-externality and causal-VaR branches")
	}
	if (Objective{}).ReadsSLOValueConfig() {
		t.Error("this arm reads no tau, no V, no ablation, and no capacity")
	}
}

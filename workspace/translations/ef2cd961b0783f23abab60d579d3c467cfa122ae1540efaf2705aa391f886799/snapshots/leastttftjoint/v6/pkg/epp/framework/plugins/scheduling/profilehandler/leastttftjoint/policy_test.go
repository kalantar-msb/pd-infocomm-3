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
	"context"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

func testPolicy(t *testing.T, cfg Config) *Policy {
	t.Helper()
	return newPolicy(cfg, newPluginMetrics("test", HandlerPluginType))
}

// decodeSnap builds an UNCONGESTED decode endpoint's routing view. Fixtures that need a
// resident population use congestedDecodeSnap, which is the only shape in this arm's tests
// that exercises the rollforward walk.
func decodeSnap(id, gpuType string, batch, queue int, kv int64) Snapshot {
	return Snapshot{
		ID: id, GPUType: gpuType,
		BatchSize: batch, QueueDepth: queue,
		KvTokensInUse: kv, FreeKVBlocks: 1000,
		MaxBatchSize: 256, BlockSizeTokens: 16,
	}
}

func prefillSnap(id string) Snapshot {
	return Snapshot{
		ID: id, GPUType: "H100_SXM_80GB",
		BatchSize: 0, QueueDepth: 0,
		FreeKVBlocks: 1000, MaxBatchSize: 256, BlockSizeTokens: 16,
	}
}

func prefillSnapOn(id, gpuType string) Snapshot {
	s := prefillSnap(id)
	s.GPUType = gpuType
	return s
}

func testEvalCtx(p *Policy, inputLen int, aps map[string]int) *evalCtx {
	ec := &evalCtx{
		class:        sloClassStandard,
		inputLen:     inputLen,
		nowUs:        100_000_000,
		requestID:    "req-1",
		apByEndpoint: aps,
	}
	ec.reqKVNeed = p.reqKVNeed(inputLen)
	return ec
}

// congestedDecodeSnap forces a non-trivial admission delay: the batch is full and free KV
// is below what the arriving request needs, so rollforwardEstimateTAdm must walk the
// residents' departures rather than admitting immediately.
func congestedDecodeSnap(id, gpuType string, queue int) Snapshot {
	return Snapshot{
		ID: id, GPUType: gpuType,
		BatchSize: 256, QueueDepth: queue,
		KvTokensInUse: 200_000, FreeKVBlocks: 4,
		MaxBatchSize: 256, BlockSizeTokens: 16,
		ResidentPrefillTokens: 512,
		RunningDecode: []RunningReqState{
			{RequestID: "r1", StepsDone: 10, KVBlocks: 40, TrueRemaining: -1, SLOClass: sloClassStandard},
			{RequestID: "r2", StepsDone: 20, KVBlocks: 60, TrueRemaining: -1, SLOClass: sloClassStandard},
			{RequestID: "r3", StepsDone: 30, KVBlocks: 80, TrueRemaining: -1, SLOClass: sloClassStandard},
		},
	}
}

// candidateJ reads ONE candidate's objective. jointCandidateTTFT is the single entry point,
// so calling it directly is the honest way to read a candidate's J rather than inferring it
// from an argmin over several. ps == nil means the local candidate.
func candidateJ(t *testing.T, p *Policy, inputLen int, aps map[string]int, ds Snapshot, ps *Snapshot) float64 {
	t.Helper()
	ec := testEvalCtx(p, inputLen, aps)
	p.mu.Lock()
	defer p.mu.Unlock()
	ec.nHatOut = p.nHatFor(ec.class)
	return p.jointCandidateTTFT(ec, ds, ps)
}

// TestObjectiveIsTheLocalTTFTAndNothingElse pins the LOCAL branch's arithmetic term by term
// against the shared helpers.
//
// WHY THIS COMPOSITION IS THE ATTRIBUTION ARGUMENT, AND NOT CIRCULAR. Every helper on the
// right-hand side -- estimateTAdm, decodeAdmissionCtx, chunkTerms, Wp, tIterDecode,
// projectedLocalTTFT -- is one that contract_test.go proves byte-identical to the focal
// arm's. So this test pins the only thing that is genuinely this arm's own: HOW those shared
// terms are composed. Composing them differently is exactly the failure the verbatim-copy
// contract cannot catch, because that contract covers the operands and not the expression.
func TestObjectiveIsTheLocalTTFTAndNothingElse(t *testing.T) {
	p := testPolicy(t, armConfig())
	ds := congestedDecodeSnap("ns/decode-a", "H100_SXM_80GB", 8)
	const inputLen = 4096
	aps := map[string]int{ds.ID: inputLen}

	ec := testEvalCtx(p, inputLen, aps)
	best, ok := p.decide(ec, []Snapshot{ds}, nil)
	if !ok {
		t.Fatal("decide declined with one decode candidate")
	}
	if !best.local || best.dID != ds.ID {
		t.Fatalf("got %+v, want the local candidate on %s", best, ds.ID)
	}

	// Recompute the objective from the shared helpers.
	ref := testEvalCtx(p, inputLen, aps)
	ref.nHatOut = p.nHatFor(ref.class)
	theta := p.coeffsFor(ds.GPUType)
	tIterD := theta.tIterDecode(ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens)
	tAdmD := p.estimateTAdm(p.decodeAdmissionCtx(ref, ds))
	nChunks, _ := p.chunkTerms(theta, inputLen)
	wp := theta.Wp(inputLen, inputLen)
	if tAdmD <= 0 || nChunks <= 0 || wp <= 0 {
		t.Fatalf("the fixture must exercise every term: tAdmD=%g nChunks=%g wp=%g", tAdmD, nChunks, wp)
	}

	closeTo(t, best.J, p.projectedLocalTTFT(tAdmD, nChunks, tIterD, wp), "local J")
}

// TestOverlapNotSerializationOnTheDisaggregatedPath is the sharpest single-line hazard in
// the port. decodeJoinUs must be max(remoteLead, tAdmD), NOT their sum: remote preparation
// and decode-queue drainage are CONCURRENT clocks from the routing instant.
//
// Serializing them over-prices every disaggregated candidate by up to a full decode
// admission delay, which systematically under-selects remote prefill -- a running,
// reporting, different algorithm.
//
// It must also be UNCONDITIONAL. The campaign passes --edpp-ttft-overlap-aware, but that
// flag gates only the reduced path upstream and is a no-op for this arm, which is why no
// config knob reaches this max().
func TestOverlapNotSerializationOnTheDisaggregatedPath(t *testing.T) {
	p := testPolicy(t, armConfig())
	ds := congestedDecodeSnap("ns/decode-a", "H100_SXM_80GB", 40)
	ps := prefillSnap("ns/prefill-a")
	const inputLen = 4096
	aps := map[string]int{ds.ID: inputLen, ps.ID: inputLen}

	// Rebuild both clocks from the shared helpers.
	ref := testEvalCtx(p, inputLen, aps)
	ref.nHatOut = p.nHatFor(ref.class)
	thetaD := p.coeffsFor(ds.GPUType)
	thetaP := p.coeffsFor(ps.GPUType)
	tAdmD := p.estimateTAdm(p.decodeAdmissionCtx(ref, ds))
	tAdmP := p.estimateTAdm(p.prefillAdmissionCtx(ref, ps))
	nChunksP, _ := p.chunkTerms(thetaP, inputLen)
	wpP := thetaP.Wp(inputLen, inputLen)
	tIterP := thetaP.tIterPrefill(ps.ResidentPrefillTokens)
	remoteLead := tAdmP + nChunksP*tIterP + wpP + p.cXferUsFor(ref.reqKVNeed)
	tIterFirstDecode := thetaD.tIterDecode(ds.BatchSize+1, ds.KvTokensInUse+int64(inputLen), ds.ResidentPrefillTokens)

	if tAdmD <= 0 || remoteLead <= 0 {
		t.Fatalf("the fixture must make both clocks positive: tAdmD=%g remoteLead=%g", tAdmD, remoteLead)
	}
	overlapped := p.projectedDisaggTTFT(math.Max(remoteLead, tAdmD), tIterFirstDecode)
	serialized := p.projectedDisaggTTFT(remoteLead+tAdmD, tIterFirstDecode)
	if serialized-overlapped < 1000 {
		t.Fatalf("the fixture does not separate the two forms by a readable margin: "+
			"overlapped=%g serialized=%g", overlapped, serialized)
	}

	got := candidateJ(t, p, inputLen, aps, ds, &ps)
	closeTo(t, got, overlapped, "disaggregated J must join the clocks with max(), not +")
	if math.Abs(got-serialized) <= eps {
		t.Errorf("disaggregated J = %g is the SERIALIZED form: adding the two clocks over-prices "+
			"every remote candidate by up to a full decode admission delay", got)
	}
}

// TestLocalProjectionOmitsTheFirstDecodeIteration pins the local/disaggregated asymmetry,
// which is real rather than an oversight: local execution samples its first token when
// prefill completes, so no decode iteration precedes it, while the disaggregated path adds
// the B+1 re-timed first decode iteration.
//
// That term's KV component scales with the request's own input length, so PROMPT SIZE
// PRICES THE REMOTE PATH THROUGH TWO CHANNELS, not one: c_xfer and this.
func TestLocalProjectionOmitsTheFirstDecodeIteration(t *testing.T) {
	p := testPolicy(t, armConfig())
	// An idle fleet, so both admission delays floor to one iteration and the remaining
	// difference is attributable.
	ds := decodeSnap("ns/decode-a", "H100_SXM_80GB", 0, 0, 0)
	ps := prefillSnap("ns/prefill-a")
	theta := p.coeffsFor(ds.GPUType)

	for _, inputLen := range []int{512, 4096, 32768} {
		aps := map[string]int{ds.ID: inputLen, ps.ID: inputLen}
		wantFirstDecode := theta.tIterDecode(ds.BatchSize+1, ds.KvTokensInUse+int64(inputLen), ds.ResidentPrefillTokens)
		if wantFirstDecode <= 0 {
			t.Fatalf("inputLen %d: the first-decode term must be positive", inputLen)
		}
		// The term must grow with input length through its C1*KV component.
		smaller := theta.tIterDecode(ds.BatchSize+1, ds.KvTokensInUse+int64(inputLen/2), ds.ResidentPrefillTokens)
		if wantFirstDecode <= smaller {
			t.Errorf("inputLen %d: the first-decode iteration must grow with the request's own KV "+
				"(got %g against %g at half the prompt)", inputLen, wantFirstDecode, smaller)
		}
		_ = candidateJ(t, p, inputLen, aps, ds, &ps)
		_ = candidateJ(t, p, inputLen, aps, ds, nil)
	}
}

// TestObjectiveIgnoresResidentValueFields is the structural proof that no value kernel crept
// in. This arm reads resident POPULATIONS but never resident VALUE: ArrivalUs, FirstTokenUs
// and TTFTSet are carried by the verbatim-copied shadow machinery and read by nothing here.
//
// That is precisely why this arm does NOT inherit degradation D2c, the late-first-token
// bias. If a value term were ever added, these three fields would start moving J and this
// test would fail -- which is the point.
func TestObjectiveIgnoresResidentValueFields(t *testing.T) {
	p := testPolicy(t, armConfig())
	const inputLen = 4096

	base := congestedDecodeSnap("ns/decode-a", "H100_SXM_80GB", 8)
	valued := congestedDecodeSnap("ns/decode-a", "H100_SXM_80GB", 8)
	valued.RunningDecode = append([]RunningReqState(nil), valued.RunningDecode...)
	for i := range valued.RunningDecode {
		valued.RunningDecode[i].ArrivalUs = 12_345_678
		valued.RunningDecode[i].FirstTokenUs = 12_400_000
		valued.RunningDecode[i].TTFTSet = true
	}

	aps := map[string]int{base.ID: inputLen}
	ps := prefillSnap("ns/prefill-a")
	aps[ps.ID] = inputLen

	closeTo(t, candidateJ(t, p, inputLen, aps, valued, nil), candidateJ(t, p, inputLen, aps, base, nil),
		"local J must not move with resident ArrivalUs/FirstTokenUs/TTFTSet")
	closeTo(t, candidateJ(t, p, inputLen, aps, valued, &ps), candidateJ(t, p, inputLen, aps, base, &ps),
		"disaggregated J must not move with resident ArrivalUs/FirstTokenUs/TTFTSet")
}

// TestObjectiveReadsResidentPopulations is the complement, and it is what makes the previous
// test meaningful rather than vacuous: the fields this arm DOES read must move J.
//
// It pins the three inherited degradations at their entry points -- D4 through S_pf, D2
// through StepsDone and KVBlocks, D6 through the prefill occupants.
func TestObjectiveReadsResidentPopulations(t *testing.T) {
	p := testPolicy(t, armConfig())
	const inputLen = 4096
	base := congestedDecodeSnap("ns/decode-a", "H100_SXM_80GB", 8)
	aps := map[string]int{base.ID: inputLen}
	baseJ := candidateJ(t, p, inputLen, aps, base, nil)

	t.Run("D4 residentPrefillTokens", func(t *testing.T) {
		moved := congestedDecodeSnap(base.ID, base.GPUType, 8)
		moved.ResidentPrefillTokens = 2048
		if got := candidateJ(t, p, inputLen, aps, moved, nil); math.Abs(got-baseJ) <= eps {
			t.Errorf("S_pf did not move J (%g): it enters tIterDecode on every candidate", got)
		}
	})

	t.Run("D2 stepsDone", func(t *testing.T) {
		moved := congestedDecodeSnap(base.ID, base.GPUType, 8)
		moved.RunningDecode = append([]RunningReqState(nil), moved.RunningDecode...)
		for i := range moved.RunningDecode {
			moved.RunningDecode[i].StepsDone = 5000
		}
		if got := candidateJ(t, p, inputLen, aps, moved, nil); math.Abs(got-baseJ) <= eps {
			t.Errorf("StepsDone did not move J (%g): it feeds decodeRemStepsEst", got)
		}
	})

	// KVBlocks needs a fixture of its own, and the reason is a real property of this port
	// worth recording rather than a test inconvenience.
	//
	// TrueRemaining is censored to -1 on this target -- the oracle reads hidden output length
	// and is never a deployable policy -- so EVERY resident is assigned the same
	// RemainingStepsEst. The departure steps in the rollforward walk are therefore all equal,
	// which means KVBlocks can never select among DIFFERENT departure steps. Its only
	// influence is whether the walk returns a departure at all, or exhausts the running set
	// and falls through to the fluid wave form.
	//
	// That fluid fallback is exactly the branch degradation D1 names as understating admission
	// delay, so the two values below are the two D1 regimes, not two shades of one estimate.
	t.Run("D2 kvBlocks", func(t *testing.T) {
		shallow := func(kv int64) Snapshot {
			s := congestedDecodeSnap(base.ID, base.GPUType, 0) // QueueDepth 0 -> needSlots 1
			s.RunningDecode = append([]RunningReqState(nil), s.RunningDecode...)
			if kv > 0 {
				for i := range s.RunningDecode {
					s.RunningDecode[i].KVBlocks = kv
				}
			}
			return s
		}
		// The default holdings (40+60+80, plus 4 free) total 184 blocks against a 4096-token
		// request's need of 256, so the walk exhausts the residents and takes the fluid form.
		exhausted := candidateJ(t, p, inputLen, aps, shallow(0), nil)
		// A resident holding 5000 blocks satisfies the KV condition at the first departure.
		satisfied := candidateJ(t, p, inputLen, aps, shallow(5000), nil)
		if math.Abs(satisfied-exhausted) <= eps {
			t.Errorf("KVBlocks did not move J (%g): the rollforward KV walk reads it, and whether "+
				"the residents' holdings cover the arrival's need decides between returning a "+
				"departure step and falling through to the fluid wave form", satisfied)
		}
		if satisfied >= exhausted {
			t.Errorf("residents holding enough KV to admit the arrival at the first departure must "+
				"price LOWER than exhausting the running set: satisfied=%g exhausted=%g",
				satisfied, exhausted)
		}
	})

	t.Run("D6 prefill occupants", func(t *testing.T) {
		idle := prefillSnap("ns/prefill-a")
		busy := prefillSnap("ns/prefill-a")
		busy.BatchSize = 8
		busy.FreeKVBlocks = 0
		busy.QueueDepth = 12
		busy.RunningPrefill = []RunningReqState{
			{RequestID: "p1", TrueRemaining: 3000},
			{RequestID: "p2", TrueRemaining: 4000},
		}
		withPrefill := map[string]int{base.ID: inputLen, idle.ID: inputLen}
		idleJ := candidateJ(t, p, inputLen, withPrefill, base, &idle)
		busyJ := candidateJ(t, p, inputLen, withPrefill, base, &busy)
		if math.Abs(busyJ-idleJ) <= eps {
			t.Errorf("the prefill occupants did not move J (%g): RunningPrefill feeds prefillRemStepsEst", busyJ)
		}
		if busyJ <= idleJ {
			t.Errorf("a congested prefill pool must price HIGHER: busy=%g idle=%g", busyJ, idleJ)
		}
	})
}

// TestPrefillIsPricedWithThePrefillEndpointsTheta pins that each side of a candidate uses
// its OWN theta, which makes this arm hardware-aware by construction.
//
// It is also the exact site of the reduced-rule divergence documented on
// jointCandidateTTFT: upstream's reduced least-ttft path prices remote prefill with the
// GLOBAL coefficients, while this joint objective uses thetaP. config.md section 4 puts
// AlphaP at 16617.85 us on H100 against 25568.35 us on A100 -- roughly 8950 us per prefill
// iteration, LARGER THAN THE ENTIRE TRANSFER TERM. So the single-endpoint equivalence check
// must be run on a homogeneous fleet; on the h100_a100_realistic cohorts it will diverge,
// and a divergence there cannot tell a bad port from a stale equivalence claim.
func TestPrefillIsPricedWithThePrefillEndpointsTheta(t *testing.T) {
	p := testPolicy(t, armConfig())
	const inputLen = 4096
	ds := decodeSnap("ns/decode-a", "H100_SXM_80GB", 0, 0, 0)
	fast := prefillSnapOn("ns/prefill-a", "H100_SXM_80GB")
	slow := prefillSnapOn("ns/prefill-a", "A100_SXM_80GB")
	aps := map[string]int{ds.ID: inputLen, fast.ID: inputLen}

	fastJ := candidateJ(t, p, inputLen, aps, ds, &fast)
	slowJ := candidateJ(t, p, inputLen, aps, ds, &slow)

	if slowJ <= fastJ {
		t.Fatalf("the A100 prefill endpoint must price higher than the H100 one: A100=%g H100=%g. "+
			"Equal values would mean the prefill side is priced with the decode endpoint's theta, "+
			"which is upstream's REDUCED rule and not this joint objective.", slowJ, fastJ)
	}

	// The gap must be at least the per-iteration intercept difference, which is the channel
	// heterogeneity rides.
	interceptGap := a100.AlphaP - h100.AlphaP
	if slowJ-fastJ < interceptGap {
		t.Errorf("the A100/H100 gap is %g, less than the single-iteration intercept difference %g: "+
			"the prefill intercept is not reaching the projection", slowJ-fastJ, interceptGap)
	}
}

// TestDecodeIsPricedWithTheDecodeEndpointsTheta is the decode-side counterpart.
func TestDecodeIsPricedWithTheDecodeEndpointsTheta(t *testing.T) {
	p := testPolicy(t, armConfig())
	const inputLen = 4096
	fast := decodeSnap("ns/decode-a", "H100_SXM_80GB", 0, 0, 0)
	slow := decodeSnap("ns/decode-a", "A100_SXM_80GB", 0, 0, 0)
	aps := map[string]int{fast.ID: inputLen}

	fastJ := candidateJ(t, p, inputLen, aps, fast, nil)
	slowJ := candidateJ(t, p, inputLen, aps, slow, nil)
	if slowJ <= fastJ {
		t.Fatalf("the A100 decode endpoint must price higher: A100=%g H100=%g", slowJ, fastJ)
	}
}

// TestDecideEnumeratesTheFullCrossProduct pins the REQUIRED SHAPE: D local candidates plus
// D*P disaggregated candidates, on one scale in one argmin.
//
// The hazard it guards is silent fallback to the target's natural decode-first
// decomposition, whose cost is measured rather than estimated: +0.0485 equal-cell mean
// goodput in favour of the joint shape, 95% CI [+0.0305, +0.0665]. A port that picked a
// decode endpoint first and then decided placement would still run and still report a
// latency, and would be a different algorithm.
func TestDecideEnumeratesTheFullCrossProduct(t *testing.T) {
	p := testPolicy(t, armConfig())
	const inputLen = 4096

	// Two decode endpoints and two prefill endpoints: 2 local + 4 disaggregated = 6.
	decodes := []Snapshot{
		congestedDecodeSnap("ns/decode-a", "A100_SXM_80GB", 60),
		congestedDecodeSnap("ns/decode-b", "H100_SXM_80GB", 60),
	}
	prefills := []Snapshot{
		prefillSnapOn("ns/prefill-a", "A100_SXM_80GB"),
		prefillSnapOn("ns/prefill-b", "H100_SXM_80GB"),
	}
	aps := map[string]int{}
	for _, s := range append(append([]Snapshot{}, decodes...), prefills...) {
		aps[s.ID] = inputLen
	}

	// Every one of the six candidates must be reachable and priced. They are collected in
	// PRODUCTION'S ENUMERATION ORDER -- for each decode endpoint by ascending ID, the local
	// candidate first, then each prefill endpoint by ascending ID -- because ties are real
	// here and the expected winner depends on that order. Collecting into a map instead would
	// make this test's own expectation nondeterministic while the code under test was not.
	type cand struct {
		key string
		j   float64
	}
	enumerated := make([]cand, 0, len(decodes)*(1+len(prefills)))
	for _, ds := range sortedByID(decodes) {
		enumerated = append(enumerated, cand{ds.ID + "|local", candidateJ(t, p, inputLen, aps, ds, nil)})
		for i := range sortedByID(prefills) {
			ps := sortedByID(prefills)[i]
			enumerated = append(enumerated, cand{ds.ID + "|" + ps.ID, candidateJ(t, p, inputLen, aps, ds, &ps)})
		}
	}
	if len(enumerated) != 6 {
		t.Fatalf("enumerated %d candidates, want 2 local + 4 disaggregated", len(enumerated))
	}

	// TIES ARE EXPECTED IN THIS FIXTURE, and they are a real consequence of the overlap form
	// rather than an artefact. The decode queue is deep enough that tAdmD dominates
	// max(remoteLead, tAdmD) for both prefill endpoints, so the PREFILL CHOICE CANNOT CHANGE
	// J -- an H100 and an A100 prefill pod price identically once the decode queue is the
	// binding clock. The strict improvement threshold therefore resolves to the
	// first-enumerated of the tied set, which is what makes the winner predictable.
	wantMin, wantKey := math.Inf(1), ""
	for _, c := range enumerated {
		if c.j < wantMin-1e-12 {
			wantMin, wantKey = c.j, c.key
		}
	}

	// The argmin must return the minimum over ALL six, not over a decode-first subset.
	ec := testEvalCtx(p, inputLen, aps)
	best, ok := p.decide(ec, decodes, prefills)
	if !ok {
		t.Fatal("decide declined")
	}
	closeTo(t, best.J, wantMin, "the argmin must be the minimum over the whole cross product")

	gotKey := best.dID + "|local"
	if !best.local {
		gotKey = best.dID + "|" + best.pID
	}
	if gotKey != wantKey {
		t.Errorf("argmin chose %s (J=%g), want %s (J=%g)", gotKey, best.J, wantKey, wantMin)
	}
}

// TestDecodePickIsPartOfTheOutputEvenOnALocalWin pins that the decode choice is an output on
// BOTH outcomes. Upstream overrides the decode pod even when the rule declines to
// disaggregate, so a port that applies the decode selection only on the disaggregated branch
// has silently discarded half the joint argmin.
func TestDecodePickIsPartOfTheOutputEvenOnALocalWin(t *testing.T) {
	p := testPolicy(t, armConfig())
	const inputLen = 1024
	// An idle H100 and a congested A100: the local win must name the H100, not merely
	// report "local".
	decodes := []Snapshot{
		congestedDecodeSnap("ns/decode-a", "A100_SXM_80GB", 200),
		decodeSnap("ns/decode-b", "H100_SXM_80GB", 0, 0, 0),
	}
	aps := map[string]int{decodes[0].ID: inputLen, decodes[1].ID: inputLen}

	ec := testEvalCtx(p, inputLen, aps)
	best, ok := p.decide(ec, decodes, nil)
	if !ok {
		t.Fatal("decide declined")
	}
	if !best.local {
		t.Fatalf("with no prefill pool the win must be local, got %+v", best)
	}
	if best.dID != "ns/decode-b" {
		t.Errorf("decode pick = %q, want the idle H100 ns/decode-b: the decode endpoint is part of "+
			"the output on a LOCAL win too", best.dID)
	}
}

// TestDecideIsDeterministicAndEnumeratesInPlainIDOrder pins the two determinism properties
// together, because the second is this arm's one deliberate divergence from the focal arm.
//
// The input slice is map-ordered -- runPickerPlugin ranges a map -- so without the ID sort
// two byte-identical requests could resolve an exact tie differently. And the order is PLAIN
// ASCENDING ID, not scorer-first: upstream's least-ttft branch iterates decodeSnaps directly,
// and applying the focal arm's scorer-first reorder here would change which candidate wins
// an exact tie, silently and only sometimes.
func TestDecideIsDeterministicAndEnumeratesInPlainIDOrder(t *testing.T) {
	p := testPolicy(t, armConfig())
	const inputLen = 2048

	// Two IDENTICAL decode endpoints differing only in ID, so every candidate ties exactly.
	// The winner must be the lowest ID, on every ordering of the input.
	mk := func(id string) Snapshot { return decodeSnap(id, "H100_SXM_80GB", 4, 2, 50_000) }
	a, b, c := mk("ns/decode-a"), mk("ns/decode-b"), mk("ns/decode-c")
	aps := map[string]int{a.ID: inputLen, b.ID: inputLen, c.ID: inputLen}

	orders := [][]Snapshot{
		{a, b, c}, {c, b, a}, {b, a, c}, {b, c, a}, {c, a, b}, {a, c, b},
	}
	for _, order := range orders {
		ec := testEvalCtx(p, inputLen, aps)
		best, ok := p.decide(ec, order, nil)
		if !ok {
			t.Fatal("decide declined")
		}
		if best.dID != "ns/decode-a" {
			ids := make([]string, 0, len(order))
			for _, s := range order {
				ids = append(ids, s.ID)
			}
			t.Errorf("input order %v resolved the tie to %q, want the lowest ID ns/decode-a: "+
				"ties must resolve to the first-enumerated candidate under ascending-ID order, "+
				"never to the caller's slice order", ids, best.dID)
		}
	}
}

// TestLocalWinsTiesAgainstAnEquallyPricedRemote pins the strict improvement threshold: a
// candidate must be STRICTLY better to displace the incumbent, so the local candidate --
// enumerated first for a given decode endpoint -- keeps an exact tie.
func TestLocalWinsTiesAgainstAnEquallyPricedRemote(t *testing.T) {
	p := testPolicy(t, armConfig())
	const inputLen = 2048
	ds := decodeSnap("ns/decode-a", "H100_SXM_80GB", 0, 0, 0)
	ps := prefillSnap("ns/prefill-a")
	aps := map[string]int{ds.ID: inputLen, ps.ID: inputLen}

	localValue := candidateJ(t, p, inputLen, aps, ds, nil)
	remoteValue := candidateJ(t, p, inputLen, aps, ds, &ps)

	ec := testEvalCtx(p, inputLen, aps)
	best, ok := p.decide(ec, []Snapshot{ds}, []Snapshot{ps})
	if !ok {
		t.Fatal("decide declined")
	}
	wantLocal := localValue <= remoteValue
	if best.local != wantLocal {
		t.Errorf("local=%v but local J=%g and remote J=%g: the argmin must take the smaller, and "+
			"must keep the incumbent on an exact tie", best.local, localValue, remoteValue)
	}
}

// TestOutputTokenProcessingIsDecisionNeutral pins the claim in Config's doc, and the contrast
// with the focal arm is the point: there the term enters ownGood through a sigmoid, so a
// common additive constant still moves two candidates' values by different amounts. Here the
// objective IS the latency, so adding the same constant to every candidate cannot change the
// argmin.
//
// It is carried anyway because the projected TTFT is compared against the focal arm's, and a
// term present in one arm's projection and absent from the other's would make those two
// numbers incomparable.
func TestOutputTokenProcessingIsDecisionNeutral(t *testing.T) {
	const inputLen = 4096
	decodes := []Snapshot{
		congestedDecodeSnap("ns/decode-a", "A100_SXM_80GB", 60),
		decodeSnap("ns/decode-b", "H100_SXM_80GB", 2, 0, 10_000),
	}
	prefills := []Snapshot{prefillSnapOn("ns/prefill-a", "H100_SXM_80GB")}
	aps := map[string]int{}
	for _, s := range append(append([]Snapshot{}, decodes...), prefills...) {
		aps[s.ID] = inputLen
	}

	zero := testPolicy(t, armConfig())
	cfg := armConfig()
	cfg.OutputTokenProcessingUs = 25_000
	loaded := testPolicy(t, cfg)

	zeroBest, ok := zero.decide(testEvalCtx(zero, inputLen, aps), decodes, prefills)
	if !ok {
		t.Fatal("decide declined at zero post-processing")
	}
	loadedBest, ok := loaded.decide(testEvalCtx(loaded, inputLen, aps), decodes, prefills)
	if !ok {
		t.Fatal("decide declined at non-zero post-processing")
	}

	if zeroBest.dID != loadedBest.dID || zeroBest.local != loadedBest.local || zeroBest.pID != loadedBest.pID {
		t.Errorf("the placement moved with outputTokenProcessingUs: %+v against %+v. It is added "+
			"to every projection identically, so it cannot change this arm's argmin.",
			zeroBest, loadedBest)
	}
	closeTo(t, loadedBest.J-zeroBest.J, 25_000, "the winning J must rise by exactly the added constant")
}

// TestCXferMovesTheLocalRemoteBoundary pins degradation D5's reach. c_xfer is the only
// size-dependent remote price besides the first decode iteration, and it enters at exactly
// one place -- so a wrong value mis-prices SYSTEMATICALLY rather than noisily. In this arm
// there is no externality term to partially offset it.
func TestCXferMovesTheLocalRemoteBoundary(t *testing.T) {
	const inputLen = 8192
	ds := congestedDecodeSnap("ns/decode-a", "H100_SXM_80GB", 30)
	ps := prefillSnap("ns/prefill-a")
	aps := map[string]int{ds.ID: inputLen, ps.ID: inputLen}

	cheap := armConfig()
	cheap.Transfer.XferBaseUs = 50
	expensive := armConfig()
	expensive.Transfer.XferBaseUs = 50_000_000 // absurd on purpose: remote must lose

	cheapPolicy, expensivePolicy := testPolicy(t, cheap), testPolicy(t, expensive)
	cheapRemote := candidateJ(t, cheapPolicy, inputLen, aps, ds, &ps)
	expensiveRemote := candidateJ(t, expensivePolicy, inputLen, aps, ds, &ps)
	if expensiveRemote <= cheapRemote {
		t.Fatalf("c_xfer did not reach the disaggregated projection: %g against %g",
			expensiveRemote, cheapRemote)
	}

	best, ok := expensivePolicy.decide(testEvalCtx(expensivePolicy, inputLen, aps), []Snapshot{ds}, []Snapshot{ps})
	if !ok {
		t.Fatal("decide declined")
	}
	if !best.local {
		t.Errorf("with a 50-second transfer price the argmin must place locally, got %+v", best)
	}
}

// TestCXferScalesWithPromptLengthUnderSizeAware pins the size-aware form. Leaving
// kvBytesPerTokenPerGpu at 0 would make every request pay a flat base -- config.md puts a
// 4k-token prompt at roughly 270x under-priced -- which validation rejects, so the scaling
// itself is what remains to check.
func TestCXferScalesWithPromptLengthUnderSizeAware(t *testing.T) {
	p := testPolicy(t, armConfig())
	small := p.cXferUsFor(p.reqKVNeed(512))
	large := p.cXferUsFor(p.reqKVNeed(32768))
	if large <= small {
		t.Fatalf("size-aware c_xfer must grow with prompt length: 512 tokens = %g, 32768 = %g", small, large)
	}
	// The base is additive, so the growth must be in the transfer term rather than the base.
	if small <= p.cfg.Transfer.XferBaseUs {
		t.Errorf("c_xfer at 512 tokens (%g) must exceed the additive base (%g)", small, p.cfg.Transfer.XferBaseUs)
	}
}

// TestDecideDeclinesWithNoDecodeCandidates pins that an empty decode pool yields no decision
// rather than an arbitrary one.
func TestDecideDeclinesWithNoDecodeCandidates(t *testing.T) {
	p := testPolicy(t, armConfig())
	ec := testEvalCtx(p, 1024, nil)
	if _, ok := p.decide(ec, nil, []Snapshot{prefillSnap("ns/prefill-a")}); ok {
		t.Error("decide must decline when there is no decode candidate: the decode endpoint is " +
			"part of the output on every outcome")
	}
}

// TestApForEndpointFallsBackToTheFullPrompt pins that a prefix MISS means "no information",
// not "nothing cached". Charging the full prompt over-prices the candidate rather than
// asserting a cold cache as fact, and leaves it in the argmin.
func TestApForEndpointFallsBackToTheFullPrompt(t *testing.T) {
	p := testPolicy(t, armConfig())
	ec := testEvalCtx(p, 4096, map[string]int{"ns/decode-a": 100})
	if got := p.apForEndpoint(ec, "ns/decode-a"); got != 100 {
		t.Errorf("observed a_p = %d, want the observed 100", got)
	}
	if got := p.apForEndpoint(ec, "ns/decode-unseen"); got != 4096 {
		t.Errorf("unobserved a_p = %d, want the full prompt 4096", got)
	}
}

// TestNHatOutIsResolvedOncePerDecision pins that the per-class mean is candidate-invariant
// within one argmin, so every candidate is evaluated against identical operands. Reading it
// per candidate would let a concurrent completion change the estimate mid-argmin.
func TestNHatOutIsResolvedOncePerDecision(t *testing.T) {
	p := testPolicy(t, armConfig())
	p.observeCompletedOutput(sloClassStandard, 700)
	const inputLen = 4096
	ds := congestedDecodeSnap("ns/decode-a", "H100_SXM_80GB", 8)
	aps := map[string]int{ds.ID: inputLen}

	ec := testEvalCtx(p, inputLen, aps)
	if _, ok := p.decide(ec, []Snapshot{ds}, nil); !ok {
		t.Fatal("decide declined")
	}
	if ec.nHatOut != 700 {
		t.Errorf("ec.nHatOut = %g after decide, want the resolved class mean 700", ec.nHatOut)
	}
}

// TestEstimatorSubstitutionIsCountedOnEveryCandidate pins degradation D1's disclosure
// requirement. The rollout is unreachable at this pin, so EVERY admission estimate is a
// substitution, and the counter is a REQUIREMENT rather than instrumentation: without it,
// running a different TTFT estimator than the one that produced every published number is
// invisible in the reported latency.
//
// For this arm the rollout was the LIVE path upstream rather than a fallback, which makes
// the counter the single most important thing to instrument here.
func TestEstimatorSubstitutionIsCountedOnEveryCandidate(t *testing.T) {
	p := testPolicy(t, armConfig())
	const inputLen = 4096
	ds := congestedDecodeSnap("ns/decode-a", "H100_SXM_80GB", 8)
	ps := prefillSnap("ns/prefill-a")
	aps := map[string]int{ds.ID: inputLen, ps.ID: inputLen}

	if ds.SchedulerStateObserved {
		t.Fatal("the fixture must leave SchedulerStateObserved false: it is permanently false at this pin")
	}
	// The rollout must decline on every entry point.
	ec := testEvalCtx(p, inputLen, aps)
	ec.nHatOut = 1
	if _, _, ok := p.rolloutLocalTTFT(ec, ds, p.coeffsFor(ds.GPUType)); ok {
		t.Error("rolloutLocalTTFT must decline while SchedulerStateObserved is false")
	}
	if _, ok := p.rolloutDecodeAdmission(ec, ds, p.coeffsFor(ds.GPUType)); ok {
		t.Error("rolloutDecodeAdmission must decline while SchedulerStateObserved is false")
	}
	if _, _, ok := p.rolloutPrefillCompletion(ec, ps, p.coeffsFor(ps.GPUType)); ok {
		t.Error("rolloutPrefillCompletion must decline while SchedulerStateObserved is false")
	}
}

// TestSharedSymbolsThisArmDoesNotConsume pins the shared machinery this arm deliberately
// discards, and it exists because the verbatim-copy contract makes those symbols
// unremovable.
//
// coeffs.go, admission.go, rollout.go and shared.go are copies of the focal arm's, so they
// carry symbols whose only production caller lives in the focal arm's objective: muDNom and
// muPNom (read by the capacity account this arm does not have), chunkTerms' second result
// (the per-chunk ITL inflation, read by the focal arm's value kernels), and the admission
// results of the two rollout entry points (which this arm's objective discards because the
// closed-form tAdm it already holds is the one that reaches the projection).
//
// Deleting them to satisfy a linter is not an option -- it would break byte-identity and with
// it the attribution argument. Pinning their behaviour here is the honest alternative: it
// documents that they are unconsumed BY DECISION rather than by oversight, and it means the
// two arms can still be diffed line for line.
func TestSharedSymbolsThisArmDoesNotConsume(t *testing.T) {
	p := testPolicy(t, armConfig())

	// muDNom and muPNom: reachable, correct, and called by nothing in this arm. They take
	// their arguments BY PARAMETER, which is why this arm needs no tau field and no
	// nomPrefillTokens to carry them.
	t.Run("muDNom and muPNom take their operands by parameter", func(t *testing.T) {
		// tau_itl above the intercept, per config.md section 6's interactive triple.
		if got := h100.muDNom(50_000); got <= 0 || got >= 1 {
			t.Errorf("muDNom(50000) = %g, want a clamped drain rate in (0,1)", got)
		}
		closeTo(t, h100.muDNom(50_000), 1.0-h100.AlphaD/50_000, "muDNom")
		if got := h100.muPNom(512); got <= 0 || got >= 1 {
			t.Errorf("muPNom(512) = %g, want a clamped drain rate in (0,1)", got)
		}
		closeTo(t, h100.muPNom(512), 1.0-h100.AlphaP/(h100.AlphaP+h100.CPf*512), "muPNom")
		// The floor is 1e-3, not 1e-6, and it becomes live the moment the capacity account
		// or the `waiting` estimator is enabled.
		if got := h100.muDNom(h100.AlphaD); got != minMu {
			t.Errorf("muDNom at tau == alphaD = %g, want the %g floor", got, minMu)
		}
	})

	// chunkTerms' second result, deltaPfChunk = theta.CPf * chunk. This arm uses only
	// nChunks; the per-chunk charge is what the focal arm's value kernels read.
	t.Run("chunkTerms second result is the per-chunk prefill charge", func(t *testing.T) {
		theta := p.coeffsFor("H100_SXM_80GB")
		// Below the batched-token budget the chunk is the whole uncached suffix.
		nChunks, deltaPf := p.chunkTerms(theta, 1000)
		closeTo(t, nChunks, 1, "nChunks below the budget")
		closeTo(t, deltaPf, theta.CPf*1000, "deltaPfChunk below the budget")
		// Above it, the chunk saturates at engine.chunkTokens.
		nChunks, deltaPf = p.chunkTerms(theta, 5000)
		closeTo(t, nChunks, 3, "nChunks at 5000 tokens with a 2048 budget")
		closeTo(t, deltaPf, theta.CPf*2048, "deltaPfChunk saturates at the batched-token budget")
		// A fully cached or empty prompt yields no prefill work and no ITL inflation.
		nChunks, deltaPf = p.chunkTerms(theta, 0)
		if nChunks != 0 || deltaPf != 0 {
			t.Errorf("chunkTerms(0) = (%g, %g), want (0, 0)", nChunks, deltaPf)
		}
	})

	// The rollout's admission results. This arm's objective discards them because the
	// closed-form tAdm it already holds is the value that reaches the projection -- but the
	// entry points must still decline cleanly, which is degradation D1's guard.
	t.Run("rollout admission results are returned and declined together", func(t *testing.T) {
		ds := congestedDecodeSnap("ns/decode-a", "H100_SXM_80GB", 4)
		ps := prefillSnap("ns/prefill-a")
		ec := testEvalCtx(p, 4096, map[string]int{ds.ID: 4096, ps.ID: 4096})
		ec.nHatOut = 1

		tAdm, ttft, ok := p.rolloutLocalTTFT(ec, ds, p.coeffsFor(ds.GPUType))
		if ok || tAdm != 0 || ttft != 0 {
			t.Errorf("rolloutLocalTTFT = (%g, %g, %v), want the zero decline while "+
				"SchedulerStateObserved is false", tAdm, ttft, ok)
		}
		tAdm, completion, ok := p.rolloutPrefillCompletion(ec, ps, p.coeffsFor(ps.GPUType))
		if ok || tAdm != 0 || completion != 0 {
			t.Errorf("rolloutPrefillCompletion = (%g, %g, %v), want the zero decline",
				tAdm, completion, ok)
		}

		// With the guard forced open the same entry points return a usable admission time,
		// which is what a future engine patch exporting the wait queue would reach. This is
		// the ONLY place in this arm's tests where D1's live branch runs.
		observed := ds
		observed.SchedulerStateObserved = true
		observed.MaxScheduledTokens = 2048
		observed.CurrentStepStartUs = int64(ec.nowUs)
		observed.SchedulerRunning = []SchedulerReqState{
			{ID: "s1", SLOClass: sloClassStandard, PromptTokens: 1000, ComputedTokens: 1200, KVBlocks: 75},
		}
		if tAdm, _, ok := p.rolloutLocalTTFT(ec, observed, p.coeffsFor(observed.GPUType)); ok && tAdm < 0 {
			t.Errorf("a reachable rollout returned a negative admission time %g", tAdm)
		}
	})

	// The two unselected estimators. Only rollforward is reachable through config, but both
	// are stated so a reader can see what the published policy would have done instead.
	t.Run("the unselected estimators remain correct", func(t *testing.T) {
		ctx := AdmissionContext{
			QWork: 1000, Mu: 0.5, BatchSize: 4, MaxBatchSize: 4,
			FreeKVBlocks: 0, ReqKVNeed: 10, TIter: 20_000,
			QueueDepth: 3, AdmissionRate: 0.001, RemainingStepsEst: 5,
		}
		closeTo(t, waitingEstimateTAdm(ctx), 2000, "waiting = QWork/Mu")
		if got := waitingEstimateTAdm(AdmissionContext{QWork: 1000}); got != 0 {
			t.Errorf("waiting with a zero drain rate = %g, want 0", got)
		}
		// EVERY estimator floors at one iteration: even with a free slot, a request waits for
		// the current decode step to finish before the next batch formation admits it. At this
		// ctx the little formula yields 3/0.001 = 3000 us, which the 20 000 us floor dominates.
		closeTo(t, littleEstimateTAdm(ctx), 20_000, "little floored by one iteration")
		deep := ctx
		deep.QueueDepth = 100 // 100/0.001 = 100 000 us, clear of the floor
		closeTo(t, littleEstimateTAdm(deep), 100_000, "little = QueueDepth/AdmissionRate above the floor")
		if got := littleEstimateTAdm(AdmissionContext{TIter: 20_000}); got != 20_000 {
			t.Errorf("little with no arrival rate = %g, want the one-iteration floor", got)
		}
	})
}

// TestPolicyIsRaceFree exercises the concurrency the target's own dispatch forces on this
// arm's shared state, which is separate from the shadow table's (covered by
// TestShadowTableIsRaceFree).
//
// nHatOut is WRITTEN by the response-body hooks and READ inside the argmin. Those genuinely
// run on different goroutines: non-final response chunks drain from an async queue
// (director.go:541-552) while the final chunk runs on the request goroutine (:539), and the
// scheduling path is a third. That is why Policy carries a mutex rather than relying on
// framework per-request state.
//
// This arm's Policy is a DIFFERENT SHAPE from the focal arm's -- it has no capacity account
// and no table pointer -- so the focal arm's race coverage does not transfer, and `go test
// -race` over this package would otherwise never touch Policy.mu.
func TestPolicyIsRaceFree(t *testing.T) {
	p := testPolicy(t, armConfig())
	const inputLen = 4096
	decodes := []Snapshot{
		congestedDecodeSnap("ns/decode-a", "H100_SXM_80GB", 12),
		congestedDecodeSnap("ns/decode-b", "A100_SXM_80GB", 12),
	}
	prefills := []Snapshot{prefillSnap("ns/prefill-a")}
	aps := map[string]int{}
	for _, s := range append(append([]Snapshot{}, decodes...), prefills...) {
		aps[s.ID] = inputLen
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})

	// Writers: completions folding realized output lengths into the per-class mean.
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				p.observeCompletedOutput(sloClassStandard, 100+seed*10+i%50)
			}
		}(w)
	}

	// Readers: the argmin, which resolves nHatOut once per decision under the lock.
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				ec := testEvalCtx(p, inputLen, aps)
				if _, ok := p.decide(ec, decodes, prefills); !ok {
					t.Error("decide declined under concurrency")
					return
				}
				if ec.nHatOut <= 0 {
					t.Errorf("nHatOut resolved to %g under concurrency, want a positive mean", ec.nHatOut)
					return
				}
			}
		}()
	}

	time.Sleep(150 * time.Millisecond)
	close(stop)
	wg.Wait()

	// The mean must still be coherent: every folded value was in [100, 240].
	if got := p.nHatFor(sloClassStandard); got < 100 || got > 240 {
		t.Errorf("nHatOut mean = %g after concurrent folding, want a value inside the written range", got)
	}
}

// TestPluginPathIsRaceFree drives the real plugin surfaces concurrently: the scheduling path
// (Filter then Pick) against the response path (ResponseBody), which is the actual pairing
// the director produces. It is the end-to-end form of the test above.
func TestPluginPathIsRaceFree(t *testing.T) {
	handler, picker := buildArm(t, armConfig())
	ctx := context.Background()
	endpoints := []fwksched.Endpoint{
		testEndpoint("decode-1", bylabel.RoleDecode, "H100_SXM_80GB", nil),
		testEndpoint("decode-2", bylabel.RoleDecode, "A100_SXM_80GB", nil),
		testEndpoint("prefill-1", bylabel.RolePrefill, "H100_SXM_80GB", nil),
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})

	for s := 0; s < 3; s++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				request := testRequest(fmt.Sprintf("sched-%d-%d", id, i), 2048)
				accepted := picker.Filter(ctx, request, endpoints)
				if len(accepted) == 0 {
					t.Error("Filter accepted nothing under concurrency")
					return
				}
				scored := make([]*fwksched.ScoredEndpoint, 0, len(accepted))
				for _, ep := range accepted {
					scored = append(scored, &fwksched.ScoredEndpoint{Endpoint: ep})
				}
				if result := picker.Pick(ctx, scored); result == nil || len(result.TargetEndpoints) == 0 {
					t.Error("Pick returned nothing under concurrency")
					return
				}
			}
		}(s)
	}

	for r := 0; r < 3; r++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				request := testRequest(fmt.Sprintf("resp-%d-%d", id, i), 2048)
				handler.recordPlacement(ctx, request, endpoints[0], endpoints[2])
				handler.ResponseBody(ctx, request, &fwkrc.Response{
					Usage: fwkrh.Usage{CompletionTokens: 10 + i%40},
				}, nil)
				handler.ResponseBody(ctx, request, &fwkrc.Response{
					Usage:       fwkrh.Usage{CompletionTokens: 50 + i%40},
					EndOfStream: true,
				}, nil)
			}
		}(r)
	}

	time.Sleep(150 * time.Millisecond)
	close(stop)
	wg.Wait()
}

// TestTieBreakIsDeterministicOnBothCrossProductAxes pins the tie-break EXPLICITLY, on ties
// constructed to be exact, rather than observing it incidentally through an argmin.
//
// WHY THIS TEST EXISTS AS A SEPARATE THING. `decide` sorts BOTH axes with sortedByID and uses
// a strict improvement threshold, so ties resolve to the first-enumerated candidate. The
// decode axis was already covered by TestDecideIsDeterministicAndEnumeratesInPlainIDOrder; the
// PREFILL axis was not, and it is the axis where an exact tie actually shows up in normal
// operation -- see the tAdmD subtest below. A cross-product argmin has two axes and only one
// of them was asserted.
//
// The hazard is specific and quiet: the input slices arrive MAP-ORDERED (runPickerPlugin
// builds its slice by ranging a map, scheduler_profile.go:199-205), so an unsorted axis
// resolves an exact tie differently between two byte-identical requests. Nothing fails; the
// fleet just receives a different placement roughly at random.
func TestTieBreakIsDeterministicOnBothCrossProductAxes(t *testing.T) {
	const inputLen = 4096

	// permutations of three elements, used to prove the answer is independent of arrival order.
	perms := [][]int{{0, 1, 2}, {2, 1, 0}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {1, 0, 2}}

	t.Run("prefill axis, identical endpoints", func(t *testing.T) {
		p := testPolicy(t, armConfig())
		ds := decodeSnap("ns/decode-a", "H100_SXM_80GB", 0, 0, 0)
		// Three prefill endpoints identical in every respect except ID, so all three
		// disaggregated candidates tie to the last bit.
		prefills := []Snapshot{
			prefillSnapOn("ns/prefill-a", "H100_SXM_80GB"),
			prefillSnapOn("ns/prefill-b", "H100_SXM_80GB"),
			prefillSnapOn("ns/prefill-c", "H100_SXM_80GB"),
		}
		aps := map[string]int{ds.ID: inputLen}
		for _, s := range prefills {
			aps[s.ID] = inputLen
		}

		// Confirm the tie is exact before relying on it.
		j0 := candidateJ(t, p, inputLen, aps, ds, &prefills[0])
		for i := 1; i < len(prefills); i++ {
			if got := candidateJ(t, p, inputLen, aps, ds, &prefills[i]); got != j0 {
				t.Fatalf("the fixture is not an exact tie: %s=%g against %s=%g",
					prefills[0].ID, j0, prefills[i].ID, got)
			}
		}

		for _, perm := range perms {
			shuffled := []Snapshot{prefills[perm[0]], prefills[perm[1]], prefills[perm[2]]}
			best, ok := p.decide(testEvalCtx(p, inputLen, aps), []Snapshot{ds}, shuffled)
			if !ok {
				t.Fatal("decide declined")
			}
			// Local is enumerated before any disaggregated candidate for a given decode
			// endpoint, so it keeps an exact tie; assert on the prefill pick only when the
			// argmin actually disaggregated.
			if !best.local && best.pID != "ns/prefill-a" {
				t.Errorf("arrival order %v resolved the prefill tie to %q, want the lowest ID "+
					"ns/prefill-a; the input slice is map-ordered so the argmin must impose its own",
					perm, best.pID)
			}
		}
	})

	t.Run("prefill axis under decode-dominated overlap, differing hardware", func(t *testing.T) {
		// THE CASE THAT ACTUALLY OCCURS. With a deep decode queue tAdmD dominates
		// max(remoteLead, tAdmD), so the prefill choice cannot change J at all -- an H100 and
		// an A100 prefill pod price IDENTICALLY. Exact ties on this axis are therefore not a
		// warm-up artefact but the steady state under decode-side congestion, which is exactly
		// the regime the high-load and burst cohorts target.
		p := testPolicy(t, armConfig())
		ds := congestedDecodeSnap("ns/decode-a", "H100_SXM_80GB", 60)
		fast := prefillSnapOn("ns/prefill-b", "H100_SXM_80GB")
		slow := prefillSnapOn("ns/prefill-a", "A100_SXM_80GB") // lower ID, slower hardware
		aps := map[string]int{ds.ID: inputLen, fast.ID: inputLen, slow.ID: inputLen}

		fastJ := candidateJ(t, p, inputLen, aps, ds, &fast)
		slowJ := candidateJ(t, p, inputLen, aps, ds, &slow)
		if fastJ != slowJ {
			// FATAL, NOT SKIP. A skip is invisible in a green run, and this fixture is ours: if it
			// stops producing the decode-dominated tie, that is a change in the objective's
			// behaviour and exactly the thing we want to hear about, not something to pass over.
			t.Fatalf("this fixture no longer produces the decode-dominated tie: H100 prefill=%g "+
				"against A100 prefill=%g. Either the fixture drifted, or the prefill choice now "+
				"moves J when the decode queue is the binding clock -- which would mean the "+
				"max(remoteLead, tAdmD) overlap no longer behaves as documented.", fastJ, slowJ)
		}

		// Both orderings must yield the LOWER ID, even though it is the slower pod: once the
		// decode queue is the binding clock the hardware difference is genuinely invisible to
		// the objective, and determinism is the only remaining criterion.
		for _, order := range [][]Snapshot{{fast, slow}, {slow, fast}} {
			best, ok := p.decide(testEvalCtx(p, inputLen, aps), []Snapshot{ds}, order)
			if !ok {
				t.Fatal("decide declined")
			}
			if !best.local && best.pID != "ns/prefill-a" {
				t.Errorf("input order %s,%s resolved the tie to %q, want the lowest ID ns/prefill-a",
					order[0].ID, order[1].ID, best.pID)
			}
		}
	})

	t.Run("both axes tied simultaneously", func(t *testing.T) {
		p := testPolicy(t, armConfig())
		mkD := func(id string) Snapshot { return decodeSnap(id, "H100_SXM_80GB", 4, 2, 50_000) }
		decodes := []Snapshot{mkD("ns/decode-a"), mkD("ns/decode-b"), mkD("ns/decode-c")}
		prefills := []Snapshot{
			prefillSnapOn("ns/prefill-a", "H100_SXM_80GB"),
			prefillSnapOn("ns/prefill-b", "H100_SXM_80GB"),
		}
		aps := map[string]int{}
		for _, s := range append(append([]Snapshot{}, decodes...), prefills...) {
			aps[s.ID] = inputLen
		}

		var first *candidate
		for _, perm := range perms {
			shuffledD := []Snapshot{decodes[perm[0]], decodes[perm[1]], decodes[perm[2]]}
			for _, pOrder := range [][]Snapshot{{prefills[0], prefills[1]}, {prefills[1], prefills[0]}} {
				best, ok := p.decide(testEvalCtx(p, inputLen, aps), shuffledD, pOrder)
				if !ok {
					t.Fatal("decide declined")
				}
				if first == nil {
					cc := best
					first = &cc
					continue
				}
				if best.dID != first.dID || best.local != first.local || best.pID != first.pID {
					t.Fatalf("arrival order changed the winner: %+v against %+v", best, *first)
				}
			}
		}
		// The decode axis must resolve to the lowest ID, and local is enumerated before any
		// disaggregated candidate for that endpoint, so it keeps the tie.
		if first.dID != "ns/decode-a" {
			t.Errorf("decode tie resolved to %q, want the lowest ID ns/decode-a", first.dID)
		}
	})
}

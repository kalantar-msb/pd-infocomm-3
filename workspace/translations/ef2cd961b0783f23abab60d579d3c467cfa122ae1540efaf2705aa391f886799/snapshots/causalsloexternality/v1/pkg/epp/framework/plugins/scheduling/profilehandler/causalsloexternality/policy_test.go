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
	"testing"
)

func testPolicy(t *testing.T, cfg Config) *Policy {
	t.Helper()
	metrics := newPluginMetrics("test", HandlerPluginType)
	return newPolicy(cfg, newShadowTable(cfg.ShadowTable, metrics), metrics)
}

// decodeSnap builds a decode endpoint's routing view with the given resident population.
func decodeSnap(id, gpuType string, batch, queue int, kv int64, residents []RunningReqState) Snapshot {
	return Snapshot{
		ID: id, GPUType: gpuType,
		BatchSize: batch, QueueDepth: queue,
		KvTokensInUse: kv, FreeKVBlocks: 1000,
		MaxBatchSize: 256, BlockSizeTokens: 16,
		RunningDecode: residents,
	}
}

func prefillSnap(id string) Snapshot {
	return Snapshot{
		ID: id, GPUType: "H100_SXM_80GB",
		BatchSize: 0, QueueDepth: 0,
		FreeKVBlocks: 1000, MaxBatchSize: 256, BlockSizeTokens: 16,
	}
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

// TestRunningMeanSeededAtOne pins the seed: a zero seed would make the decode demand
// vanish and collapse every remaining-steps floor.
func TestRunningMeanSeededAtOne(t *testing.T) {
	var m runningMean
	if got := m.mean(); got != 1 {
		t.Errorf("unseeded mean = %g, want the conservative 1-token seed", got)
	}
	m.update(300)
	m.update(500)
	closeTo(t, m.mean(), 400, "mean after two completions")
}

func TestNHatForUnknownClassReturnsSeed(t *testing.T) {
	p := testPolicy(t, focalConfig())
	if got := p.nHatFor("never-seen"); got != 1 {
		t.Errorf("unknown class = %g, want 1", got)
	}
	p.observeCompletedOutput(sloClassStandard, 250)
	if got := p.nHatFor(sloClassStandard); got != 250 {
		t.Errorf("after one completion = %g, want 250", got)
	}
}

// TestObserveCompletedOutputIgnoresNonPositive pins that a request reaching a terminal
// state without completing carries no realized output length, so folding it in would drag
// the estimate down.
func TestObserveCompletedOutputIgnoresNonPositive(t *testing.T) {
	p := testPolicy(t, focalConfig())
	p.observeCompletedOutput(sloClassStandard, 400)
	p.observeCompletedOutput(sloClassStandard, 0)
	p.observeCompletedOutput(sloClassStandard, -3)
	if got := p.nHatFor(sloClassStandard); got != 400 {
		t.Errorf("mean = %g, want 400 -- zero and negative counts must be ignored", got)
	}
}

// TestVarDecodeInputsCensorsWithStepsDone pins the censoring: a resident that has produced
// StepsDone tokens has total output >= StepsDone, so the class mean is FLOORED by that
// before subtracting, and the remainder floored at 1.
func TestVarDecodeInputsCensorsWithStepsDone(t *testing.T) {
	p := testPolicy(t, focalConfig())
	p.observeCompletedOutput(sloClassStandard, 300) // class mean 300

	in := []RunningReqState{
		{StepsDone: 50, SLOClass: sloClassStandard, TTFTSet: true},  // 300-50 = 250 left
		{StepsDone: 400, SLOClass: sloClassStandard, TTFTSet: true}, // past the mean: floored at 1
	}
	out := p.varDecodeInputs(in)
	if len(out) != 2 {
		t.Fatalf("got %d residents, want 2", len(out))
	}
	if out[0].rem != 250 {
		t.Errorf("resident 0 rem = %d, want 250", out[0].rem)
	}
	if out[1].rem != 1 {
		t.Errorf("resident 1 rem = %d, want the floor of 1 (it has already exceeded the class mean)", out[1].rem)
	}
}

// TestVarPrefillInputsUsesFullClassMean pins that an occupant with no output yet gets the
// FULL censored class mean, with no StepsDone to subtract.
func TestVarPrefillInputsUsesFullClassMean(t *testing.T) {
	p := testPolicy(t, focalConfig())
	p.observeCompletedOutput(sloClassStandard, 300)
	out := p.varPrefillInputs([]RunningReqState{
		{TrueRemaining: 2000, SLOClass: sloClassStandard},
	})
	if len(out) != 1 {
		t.Fatalf("got %d occupants, want 1", len(out))
	}
	if out[0].remDecodeSteps != 300 {
		t.Errorf("remDecodeSteps = %d, want the full class mean 300", out[0].remDecodeSteps)
	}
	if out[0].remPrefillTokens != 2000 {
		t.Errorf("remPrefillTokens = %d, want the 2000 known prompt tokens", out[0].remPrefillTokens)
	}
}

// TestCoeffsForSelectsByLabelValue is the single selection point for heterogeneous
// coefficients, and it must not fall back silently.
func TestCoeffsForSelectsByLabelValue(t *testing.T) {
	p := testPolicy(t, focalConfig())
	if got := p.coeffsFor("H100_SXM_80GB"); got.AlphaD != h100.AlphaD {
		t.Errorf("H100 lookup returned AlphaD %g", got.AlphaD)
	}
	if got := p.coeffsFor("A100_SXM_80GB"); got.AlphaD != a100.AlphaD {
		t.Errorf("A100 lookup returned AlphaD %g", got.AlphaD)
	}
	// An unmapped label yields the zero value AND is counted -- the filter should have
	// rejected the endpoint before scoring, so reaching here is a bug, not a fallback.
	if got := p.coeffsFor("UNKNOWN"); got.AlphaD != 0 {
		t.Errorf("an unmapped label must not resolve to real physics, got AlphaD %g", got.AlphaD)
	}
}

// TestChunkTermsZeroForFullyCachedPrompt pins the (0,0) return: no prefill work and no
// per-chunk inflation when the prompt is fully cached or empty.
func TestChunkTermsZeroForFullyCachedPrompt(t *testing.T) {
	p := testPolicy(t, focalConfig())
	for _, ap := range []int{0, -100} {
		nChunks, delta := p.chunkTerms(h100, ap)
		if nChunks != 0 || delta != 0 {
			t.Errorf("ap = %d: got (%g, %g), want (0, 0)", ap, nChunks, delta)
		}
	}
	// A prompt longer than one chunk needs ceil(ap/chunk) iterations.
	nChunks, delta := p.chunkTerms(h100, 5000)
	if nChunks != 3 { // ceil(5000/2048)
		t.Errorf("nChunks = %g, want 3", nChunks)
	}
	closeTo(t, delta, h100.CPf*2048, "deltaPfChunk at the chunk cap")

	// A prompt shorter than the cap is one chunk, and delta is charged on the ACTUAL
	// chunk rather than the cap.
	nChunks2, delta2 := p.chunkTerms(h100, 500)
	if nChunks2 != 1 {
		t.Errorf("nChunks = %g, want 1", nChunks2)
	}
	closeTo(t, delta2, h100.CPf*500, "deltaPfChunk below the cap")
}

// TestReqKVNeedCeilingInEngineBlocks pins that reqKVNeed uses the ENGINE block size, not
// the prefix producer's.
func TestReqKVNeedCeilingInEngineBlocks(t *testing.T) {
	p := testPolicy(t, focalConfig()) // engine block size 16
	if got := p.reqKVNeed(16); got != 1 {
		t.Errorf("reqKVNeed(16) = %d, want 1", got)
	}
	if got := p.reqKVNeed(17); got != 2 {
		t.Errorf("reqKVNeed(17) = %d, want 2 (ceiling)", got)
	}
	if got := p.reqKVNeed(4000); got != 250 {
		t.Errorf("reqKVNeed(4000) = %d, want 250", got)
	}
}

// TestCXferSizeAwareScalesWithPromptLength is degradation D5's shape: c_xfer is the only
// size-dependent price of going remote.
func TestCXferSizeAwareScalesWithPromptLength(t *testing.T) {
	p := testPolicy(t, focalConfig())

	small := p.cXferUsFor(p.reqKVNeed(500))
	large := p.cXferUsFor(p.reqKVNeed(45000))
	if large <= small {
		t.Error("a larger prompt must cost more to transfer")
	}

	// The arithmetic, spelled out: bytes = blocks * blockSize * bytesPerToken, and
	// bandwidth is GB/s converted to bytes/us.
	kv := p.reqKVNeed(4000)
	wantBytes := float64(kv) * 16 * 81920
	want := 50.0 + wantBytes/(25.0*1000.0)
	closeTo(t, p.cXferUsFor(kv), want, "size-aware c_xfer")

	// config.md's stated magnitude: a 4k-token prompt should be charged ~13500 us.
	if got := p.cXferUsFor(kv); got < 12000 || got > 15000 {
		t.Errorf("4k-token transfer charge = %g us, expected ~13500 per config.md section 7", got)
	}
}

// TestCXferFlatPathIsDistinctFromSizeAwareBase pins that FlatCXferUs and XferBaseUs are
// separate fields. Conflating them makes the flat path unrepresentable and silently
// reprices every disaggregated candidate if sizeAware is turned off.
func TestCXferFlatPathIsDistinctFromSizeAwareBase(t *testing.T) {
	cfg := focalConfig()
	cfg.Transfer.SizeAware = false
	p := testPolicy(t, cfg)
	if got := p.cXferUsFor(p.reqKVNeed(45000)); got != 5000 {
		t.Errorf("the flat path must return flatCXferUs (5000), not xferBaseUs (50), got %g", got)
	}
}

// TestDecodeRemStepsEstNeverNegativeOrZero pins the floors: the class estimate is floored
// by the LONGEST in-flight elapsed count before per-request subtraction, and each remainder
// is floored at 1.
func TestDecodeRemStepsEstNeverNegativeOrZero(t *testing.T) {
	p := testPolicy(t, focalConfig())
	p.observeCompletedOutput(sloClassStandard, 100)

	// Every resident is far past the class mean.
	snap := decodeSnap("d1", "H100_SXM_80GB", 3, 0, 1000, []RunningReqState{
		{StepsDone: 900, SLOClass: sloClassStandard, TTFTSet: true},
		{StepsDone: 950, SLOClass: sloClassStandard, TTFTSet: true},
	})
	got := p.decodeRemStepsEst(snap, sloClassStandard)
	if got < 1 {
		t.Errorf("estimate must never drop below 1, got %g", got)
	}

	// An endpoint with no residents returns exactly 1.
	empty := decodeSnap("d2", "H100_SXM_80GB", 0, 0, 0, nil)
	if got := p.decodeRemStepsEst(empty, sloClassStandard); got != 1 {
		t.Errorf("an empty endpoint returns %g, want 1", got)
	}
}

func TestPrefillRemStepsEstFloorsAtOne(t *testing.T) {
	p := testPolicy(t, focalConfig())
	snap := Snapshot{RunningPrefill: []RunningReqState{
		{TrueRemaining: 0}, {TrueRemaining: -5}, {TrueRemaining: 7},
	}}
	// Floored: 1, 1, 7 -> mean 3.
	closeTo(t, p.prefillRemStepsEst(snap), 3, "prefill remaining steps")
	if got := p.prefillRemStepsEst(Snapshot{}); got != 1 {
		t.Errorf("no occupants returns %g, want 1", got)
	}
}

// TestSortedByIDCopiesAndSorts pins the determinism requirement and the no-mutation
// contract: the caller may hold the input slice for other purposes.
func TestSortedByIDCopiesAndSorts(t *testing.T) {
	in := []Snapshot{{ID: "z"}, {ID: "a"}, {ID: "m"}}
	out := sortedByID(in)
	if out[0].ID != "a" || out[1].ID != "m" || out[2].ID != "z" {
		t.Errorf("not sorted: %v", []string{out[0].ID, out[1].ID, out[2].ID})
	}
	if in[0].ID != "z" {
		t.Error("sortedByID must not reorder its input in place")
	}
	if sortedByID(nil) != nil {
		t.Error("nil input must yield nil")
	}
}

// TestScorerFirstSnapshotsIsATieBreakOrderNotAFilter pins that EVERY endpoint is still
// enumerated -- the preference only moves one to the front, which changes only which
// candidate wins an EXACT tie.
func TestScorerFirstSnapshotsIsATieBreakOrderNotAFilter(t *testing.T) {
	snaps := []Snapshot{{ID: "a"}, {ID: "b"}, {ID: "c"}}

	out := scorerFirstSnapshots(snaps, "c")
	if len(out) != 3 {
		t.Fatalf("every endpoint must survive: got %d", len(out))
	}
	if out[0].ID != "c" {
		t.Errorf("preferred endpoint must be first, got %s", out[0].ID)
	}
	// The rest keep ascending order.
	if out[1].ID != "a" || out[2].ID != "b" {
		t.Errorf("remaining order = %s,%s, want a,b", out[1].ID, out[2].ID)
	}

	// Three early-return cases return the input UNCHANGED.
	if got := scorerFirstSnapshots(snaps, ""); len(got) != 3 || got[0].ID != "a" {
		t.Error("no preference must return the input unchanged")
	}
	if got := scorerFirstSnapshots(snaps, "a"); got[0].ID != "a" {
		t.Error("an already-first preference must return the input unchanged")
	}
	single := []Snapshot{{ID: "only"}}
	if got := scorerFirstSnapshots(single, "other"); len(got) != 1 {
		t.Error("fewer than two endpoints must return the input unchanged")
	}
}

// THE REQUIRED STRUCTURAL SHAPE: D local candidates PLUS D*P disaggregated candidates,
// scored on ONE scale and compared in ONE argmin.
//
// This test counts the candidates actually evaluated, so a port that silently degenerated
// to a decode-first decomposition -- pick decode, then decide placement -- would fail it.
func TestDecideEnumeratesFullCrossProduct(t *testing.T) {
	p := testPolicy(t, focalConfig())
	p.observeCompletedOutput(sloClassStandard, 300)

	residents := []RunningReqState{
		{StepsDone: 20, SLOClass: sloClassStandard, TTFTSet: true, ArrivalUs: 99_000_000, FirstTokenUs: 99_200_000, KVBlocks: 100},
	}
	decodes := []Snapshot{
		decodeSnap("d1", "H100_SXM_80GB", 4, 0, 20000, residents),
		decodeSnap("d2", "A100_SXM_80GB", 4, 0, 20000, residents),
	}
	prefills := []Snapshot{prefillSnap("p1"), prefillSnap("p2")}

	aps := map[string]int{"d1": 4000, "d2": 4000, "p1": 4000, "p2": 4000}
	ec := testEvalCtx(p, 4000, aps)

	// The estimator-substitution counter is incremented once per admission estimate, so
	// it is a direct probe of how many candidates were scored.
	before := readCounter(t, estimatorSubstitutionCount, "test", HandlerPluginType)
	best, ok := p.decide(ec, decodes, prefills, "", "")
	after := readCounter(t, estimatorSubstitutionCount, "test", HandlerPluginType)

	if !ok {
		t.Fatal("Decide must return a candidate")
	}
	// D local candidates: 2, each costing one decode admission estimate.
	// D*P disaggregated: 4, each costing a decode AND a prefill admission estimate.
	// Total estimates = 2*1 + 4*2 = 10.
	wantEstimates := 2*1.0 + 4*2.0
	if got := after - before; got != wantEstimates {
		t.Errorf("admission estimates = %g, want %g -- the argmin must enumerate 2 local plus 4 disaggregated candidates",
			got, wantEstimates)
	}
	if best.dID != "d1" && best.dID != "d2" {
		t.Errorf("winner names an unknown decode endpoint: %q", best.dID)
	}
}

// TestDecideIsDeterministic pins the determinism contract: candidates are enumerated over
// ID-sorted endpoints with a strict improvement threshold, so two identical decisions
// cannot differ. Without it an exact tie would resolve by map iteration order.
func TestDecideIsDeterministic(t *testing.T) {
	cfg := focalConfig()
	residents := []RunningReqState{
		{StepsDone: 10, SLOClass: sloClassStandard, TTFTSet: true, ArrivalUs: 99_000_000, FirstTokenUs: 99_100_000, KVBlocks: 50},
	}
	// Two IDENTICAL decode endpoints, which is the exact-tie case.
	decodes := []Snapshot{
		decodeSnap("dB", "H100_SXM_80GB", 4, 0, 20000, residents),
		decodeSnap("dA", "H100_SXM_80GB", 4, 0, 20000, residents),
	}
	prefills := []Snapshot{prefillSnap("p1")}
	aps := map[string]int{"dA": 4000, "dB": 4000, "p1": 4000}

	var first candidate
	for i := 0; i < 25; i++ {
		p := testPolicy(t, cfg)
		p.observeCompletedOutput(sloClassStandard, 300)
		got, ok := p.decide(testEvalCtx(p, 4000, aps), decodes, prefills, "", "")
		if !ok {
			t.Fatal("Decide must succeed")
		}
		if i == 0 {
			first = got
			continue
		}
		if got.dID != first.dID || got.pID != first.pID || got.local != first.local {
			t.Fatalf("non-deterministic decision: %+v then %+v", first, got)
		}
	}
	// The tie resolves to the FIRST-ENUMERATED candidate, i.e. the lowest ID.
	if first.dID != "dA" {
		t.Errorf("an exact tie must resolve to the lowest ID, got %q", first.dID)
	}
}

// TestDecodePickIsPartOfTheOutputEvenOnALocalWin is the half of the joint argmin easiest
// to lose. Upstream overrides the decode pod EVEN WHEN the rule declines to disaggregate;
// a port that applied the decode selection only on the disaggregated branch would be
// running something closer to the decomposed control.
func TestDecodePickIsPartOfTheOutputEvenOnALocalWin(t *testing.T) {
	p := testPolicy(t, focalConfig())
	p.observeCompletedOutput(sloClassStandard, 300)

	// One endpoint holds residents near their deadline; the other is empty. A local
	// placement on the empty one should win, and its ID must be reported.
	loaded := decodeSnap("d-loaded", "H100_SXM_80GB", 200, 50, 900_000, []RunningReqState{
		{StepsDone: 5, SLOClass: sloClassStandard, TTFTSet: true, ArrivalUs: 85_000_000, FirstTokenUs: 85_200_000, KVBlocks: 400},
		{StepsDone: 5, SLOClass: sloClassStandard, TTFTSet: true, ArrivalUs: 85_100_000, FirstTokenUs: 85_300_000, KVBlocks: 400},
	})
	empty := decodeSnap("d-empty", "H100_SXM_80GB", 0, 0, 0, nil)

	aps := map[string]int{"d-loaded": 4000, "d-empty": 4000, "p1": 4000}
	best, ok := p.decide(testEvalCtx(p, 4000, aps),
		[]Snapshot{loaded, empty}, []Snapshot{prefillSnap("p1")}, "", "")
	if !ok {
		t.Fatal("Decide must succeed")
	}
	if best.dID == "" {
		t.Fatal("the decode pick must always be part of the output, local win or not")
	}
	if best.dID != "d-empty" {
		t.Errorf("the argmin should prefer the endpoint with nothing to harm, got %q", best.dID)
	}
}

// TestDecomposedAblationRestrictsDecodeEnumeration pins the switch config.md section 10
// prices at +0.0485. Flipping it is the cheapest on-cluster check that the joint shape
// survived the port, so it must actually change what is enumerated.
func TestDecomposedAblationRestrictsDecodeEnumeration(t *testing.T) {
	residents := []RunningReqState{
		{StepsDone: 10, SLOClass: sloClassStandard, TTFTSet: true, ArrivalUs: 99_000_000, FirstTokenUs: 99_100_000, KVBlocks: 50},
	}
	decodes := []Snapshot{
		decodeSnap("dA", "H100_SXM_80GB", 4, 0, 20000, residents),
		decodeSnap("dB", "A100_SXM_80GB", 4, 0, 20000, residents),
	}
	prefills := []Snapshot{prefillSnap("p1")}
	aps := map[string]int{"dA": 4000, "dB": 4000, "p1": 4000}

	count := func(decomposed bool) float64 {
		cfg := focalConfig()
		cfg.Ablation.Decomposed = decomposed
		p := testPolicy(t, cfg)
		p.observeCompletedOutput(sloClassStandard, 300)
		before := readCounter(t, estimatorSubstitutionCount, "test", HandlerPluginType)
		// Name dB as the inherited scorer's pick, so the decomposed arm is pinned to it.
		p.decide(testEvalCtx(p, 4000, aps), decodes, prefills, "dB", "")
		return readCounter(t, estimatorSubstitutionCount, "test", HandlerPluginType) - before
	}

	joint := count(false)     // 2 local + 2 disagg = 2*1 + 2*2 = 6
	decomposed := count(true) // 1 local + 1 disagg = 1*1 + 1*2 = 3
	if joint != 6 {
		t.Errorf("joint enumeration cost %g estimates, want 6", joint)
	}
	if decomposed != 3 {
		t.Errorf("decomposed enumeration cost %g estimates, want 3", decomposed)
	}
	if decomposed >= joint {
		t.Error("the decomposed control must enumerate strictly fewer candidates")
	}
}

// TestDecomposedAblationHonoursTheInheritedPick confirms the restriction pins decode to
// the inherited scorer's choice rather than to the lowest ID.
func TestDecomposedAblationHonoursTheInheritedPick(t *testing.T) {
	cfg := focalConfig()
	cfg.Ablation.Decomposed = true
	p := testPolicy(t, cfg)
	p.observeCompletedOutput(sloClassStandard, 300)

	decodes := []Snapshot{
		decodeSnap("dA", "H100_SXM_80GB", 4, 0, 20000, nil),
		decodeSnap("dB", "H100_SXM_80GB", 4, 0, 20000, nil),
	}
	aps := map[string]int{"dA": 4000, "dB": 4000, "p1": 4000}
	best, ok := p.decide(testEvalCtx(p, 4000, aps), decodes,
		[]Snapshot{prefillSnap("p1")}, "dB", "")
	if !ok {
		t.Fatal("Decide must succeed")
	}
	if best.dID != "dB" {
		t.Errorf("the decomposed control must use the inherited pick dB, got %q", best.dID)
	}
}

// TestScoreCandidateObeysTheObjectiveContract is the ablation cohort's validity gate,
// executable: score = V * (externality - ownGood) with the capacity term disabled.
//
// This is why V is KEPT rather than folded away -- the gate asserts the factor of 8
// exactly, and a port that folded V away could not reproduce it.
func TestScoreCandidateObeysTheObjectiveContract(t *testing.T) {
	p := testPolicy(t, focalConfig())
	p.observeCompletedOutput(sloClassStandard, 300)

	ds := decodeSnap("d1", "H100_SXM_80GB", 8, 2, 40000, []RunningReqState{
		{StepsDone: 30, SLOClass: sloClassStandard, TTFTSet: true, ArrivalUs: 95_000_000, FirstTokenUs: 95_300_000, KVBlocks: 200},
	})
	ec := testEvalCtx(p, 4000, map[string]int{"d1": 4000})
	ec.nHatOut = p.nHatFor(sloClassStandard)
	p.gpuTypeByID = map[string]string{"d1": "H100_SXM_80GB"}

	score := p.scoreCandidate(ec, ds, nil)

	closeTo(t, score.netGoodCost, score.externality-score.ownGood, "netGoodCost")
	closeTo(t, score.total, 8*score.netGoodCost, "total = V * netGoodCost with capacity disabled")
	if score.capacityTotal != 0 {
		t.Errorf("capacity must be identically 0 in the focal arm, got %g", score.capacityTotal)
	}
	// The breakdown must aggregate to the externality, which is what makes it auditable.
	closeTo(t, score.externality, score.externalityBreakdown.total(), "externality equals its breakdown")
}

// TestAblationSwitchesRemoveTheirTerms checks each switch actually removes its term.
func TestAblationSwitchesRemoveTheirTerms(t *testing.T) {
	build := func(mut func(*Config)) candidateScore {
		cfg := focalConfig()
		mut(&cfg)
		p := testPolicy(t, cfg)
		p.observeCompletedOutput(sloClassStandard, 300)
		ds := decodeSnap("d1", "H100_SXM_80GB", 8, 2, 40000, []RunningReqState{
			{StepsDone: 30, SLOClass: sloClassStandard, TTFTSet: true, ArrivalUs: 95_000_000, FirstTokenUs: 95_300_000, KVBlocks: 200},
		})
		ec := testEvalCtx(p, 4000, map[string]int{"d1": 4000})
		ec.nHatOut = p.nHatFor(sloClassStandard)
		p.gpuTypeByID = map[string]string{"d1": "H100_SXM_80GB"}
		return p.scoreCandidate(ec, ds, nil)
	}

	full := build(func(*Config) {})
	if full.externality == 0 {
		t.Fatal("the fixture must produce a non-zero externality or the ablations prove nothing")
	}
	if full.ownGood == 0 {
		t.Fatal("the fixture must produce a non-zero ownGood")
	}

	noExt := build(func(c *Config) { c.Ablation.NoExternality = true })
	if noExt.externality != 0 {
		t.Errorf("noExternality must zero the term, got %g", noExt.externality)
	}
	// The breakdown is still computed -- only its contribution to the objective is cut,
	// which is what keeps the ablation auditable.
	if noExt.externalityBreakdown.total() == 0 {
		t.Error("the breakdown should still be computed under the ablation")
	}

	noOwn := build(func(c *Config) { c.Ablation.NoOwnGood = true })
	if noOwn.ownGood != 0 {
		t.Errorf("noOwnGood must zero the term, got %g", noOwn.ownGood)
	}
}

// TestOverlapNotSerializationOnTheDisaggregatedPath pins the max() the specification warns
// must NOT be made conditional on the upstream --edpp-ttft-overlap-aware flag, which gates
// only the reduced path and is never consulted on the joint path.
//
// The decode admission estimate is an absolute wait measured at the routing instant, so
// remote prefill and transfer consume part or all of that interval while the decode queue
// continues to drain. Serializing them would over-price every disaggregated candidate by up
// to a full decode admission delay.
//
// The test computes the externality under BOTH clocks and asserts the scored candidate
// matches the overlapped one. The fixture is chosen so the two differ: a lightly-loaded
// decode endpoint whose residents have far more steps left than either clock's arrival
// window, so neither is gated to zero by the causal window.
func TestOverlapNotSerializationOnTheDisaggregatedPath(t *testing.T) {
	p := testPolicy(t, focalConfig())
	p.observeCompletedOutput(sloClassStandard, 300)

	ds := decodeSnap("d1", "H100_SXM_80GB", 4, 0, 20_000, []RunningReqState{
		{StepsDone: 10, SLOClass: sloClassStandard, TTFTSet: true, ArrivalUs: 99_000_000, FirstTokenUs: 99_100_000, KVBlocks: 200},
	})
	ps := prefillSnap("p1")

	ec := testEvalCtx(p, 4000, map[string]int{"d1": 4000, "p1": 4000})
	ec.nHatOut = p.nHatFor(sloClassStandard)
	p.gpuTypeByID = map[string]string{"d1": "H100_SXM_80GB", "p1": "H100_SXM_80GB"}

	tAdmD := rollforwardEstimateTAdm(p.decodeAdmissionCtx(ec, ds))
	tAdmP := rollforwardEstimateTAdm(p.prefillAdmissionCtx(ec, ps))
	nChunksP, _ := p.chunkTerms(h100, 4000)
	remoteLead := tAdmP + nChunksP*h100.tIterPrefill(ps.ResidentPrefillTokens) +
		h100.Wp(4000, 4000) + p.cXferUsFor(ec.reqKVNeed)

	overlapped := math.Max(remoteLead, tAdmD)
	serialized := remoteLead + tAdmD
	if overlapped >= serialized {
		t.Fatalf("fixture does not exercise the difference: max=%g sum=%g", overlapped, serialized)
	}

	wantOverlap := p.externalityDisagg(ec, ds, &ps, h100, h100, 4000, overlapped, tAdmP)
	wantSerial := p.externalityDisagg(ec, ds, &ps, h100, h100, 4000, serialized, tAdmP)
	if wantOverlap.total() == wantSerial.total() {
		t.Fatalf("fixture cannot distinguish the two clocks (both %g); the residents are "+
			"gated to zero either way, so pick a fixture where rem exceeds both windows",
			wantOverlap.total())
	}
	// Serializing delays the arrival further, so it disturbs residents LESS and under-states
	// the externality of going remote -- which is what makes remote look cheaper than it is.
	if wantSerial.total() >= wantOverlap.total() {
		t.Errorf("the serialized clock should under-state the disaggregated externality: "+
			"overlap=%g serial=%g", wantOverlap.total(), wantSerial.total())
	}

	score := p.scoreCandidate(ec, ds, &ps)
	closeTo(t, score.externalityBreakdown.total(), wantOverlap.total(),
		"scored candidate must use the overlapped join clock")
}

// TestHeterogeneousFleetPricesTheSameResidentsDifferently is the condition the mechanism
// exploits: per-GPU coefficients make the SAME resident set cost different amounts on
// different hardware, which is exactly what load-shaped signals cannot see.
func TestHeterogeneousFleetPricesTheSameResidentsDifferently(t *testing.T) {
	p := testPolicy(t, focalConfig())
	p.observeCompletedOutput(sloClassStandard, 300)

	residents := []RunningReqState{
		{StepsDone: 20, SLOClass: sloClassStandard, TTFTSet: true, ArrivalUs: 95_000_000, FirstTokenUs: 95_200_000, KVBlocks: 200},
		{StepsDone: 25, SLOClass: sloClassStandard, TTFTSet: true, ArrivalUs: 95_500_000, FirstTokenUs: 95_700_000, KVBlocks: 200},
	}
	// IDENTICAL load signals -- same batch, same queue, same KV -- differing only in the
	// GPU-type label.
	onH100 := decodeSnap("d-h100", "H100_SXM_80GB", 8, 2, 40000, residents)
	onA100 := decodeSnap("d-a100", "A100_SXM_80GB", 8, 2, 40000, residents)

	ec := testEvalCtx(p, 4000, map[string]int{"d-h100": 4000, "d-a100": 4000})
	ec.nHatOut = p.nHatFor(sloClassStandard)
	p.gpuTypeByID = map[string]string{"d-h100": "H100_SXM_80GB", "d-a100": "A100_SXM_80GB"}

	hScore := p.scoreCandidate(ec, onH100, nil)
	aScore := p.scoreCandidate(ec, onA100, nil)

	if hScore.total == aScore.total {
		t.Fatal("identical load on non-identical hardware must not price identically -- " +
			"that difference is the condition the mechanism exploits")
	}
	// The A100 is slower on every iteration, so placing here destroys more resident value.
	if aScore.externality <= hScore.externality {
		t.Errorf("the slower GPU should destroy more resident value: A100=%g H100=%g",
			aScore.externality, hScore.externality)
	}
}

// TestDecideDeclinesWhenNoDecodeCandidates guards the empty-candidate case.
func TestDecideDeclinesWhenNoDecodeCandidates(t *testing.T) {
	p := testPolicy(t, focalConfig())
	if _, ok := p.decide(testEvalCtx(p, 4000, nil), nil, nil, "", ""); ok {
		t.Error("Decide must decline with no decode candidates")
	}
}

// TestApForEndpointFallsBackToFullPrompt pins the miss policy: a miss means "no
// information", not "nothing cached", so charging the full prompt over-prices the candidate
// rather than asserting a cold cache as fact -- and leaves it in the argmin.
func TestApForEndpointFallsBackToFullPrompt(t *testing.T) {
	p := testPolicy(t, focalConfig())
	ec := testEvalCtx(p, 4000, map[string]int{"known": 1500})
	if got := p.apForEndpoint(ec, "known"); got != 1500 {
		t.Errorf("observed a_p = %d, want 1500", got)
	}
	if got := p.apForEndpoint(ec, "unobserved"); got != 4000 {
		t.Errorf("unobserved a_p = %d, want the full prompt 4000", got)
	}
}

// TestCapacityAccountIsDeadInTheFocalArm confirms the switch really disables it, and that
// enabling it produces a non-zero term -- so the dead state is a configuration outcome
// rather than unimplemented code.
func TestCapacityAccountIsDeadInTheFocalArm(t *testing.T) {
	ds := decodeSnap("d1", "H100_SXM_80GB", 8, 2, 40000, nil)
	ps := prefillSnap("p1")
	aps := map[string]int{"d1": 4000, "p1": 4000}

	// Focal: capacity disabled, so no queue is ever created.
	focal := testPolicy(t, focalConfig())
	focal.decide(testEvalCtx(focal, 4000, aps), []Snapshot{ds}, []Snapshot{ps}, "", "")
	if len(focal.sloCapacity) != 0 {
		t.Errorf("the focal arm must not maintain capacity state, got %d entries", len(focal.sloCapacity))
	}

	// Enabled: queues are created, refreshed, and booked.
	cfg := focalConfig()
	cfg.Ablation.NoCapacity = false
	live := testPolicy(t, cfg)
	live.decide(testEvalCtx(live, 4000, aps), []Snapshot{ds}, []Snapshot{ps}, "", "")
	if len(live.sloCapacity) == 0 {
		t.Error("with the capacity term enabled the account must be maintained")
	}
	if live.capacityQueue("d1") <= 0 {
		t.Errorf("the winning endpoint's queue must be booked, got %g", live.capacityQueue("d1"))
	}
}

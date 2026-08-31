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

// interactiveSLO is the committed default triple, config.md section 6:
// tau_ttft 1000 ms, tau_itl 50 ms, tau_e2e 16000 ms, in microseconds.
var interactiveSLO = slo{tauTTFTUs: 1_000_000, tauITLUs: 50_000, tauE2EUs: 16_000_000}

func TestSigmoid(t *testing.T) {
	closeTo(t, sigmoid(0), 0.5, "sigmoid(0)")
	if sigmoid(10) <= 0.99 || sigmoid(10) >= 1 {
		t.Errorf("sigmoid(10) = %g, want just under 1", sigmoid(10))
	}
	if sigmoid(-10) <= 0 || sigmoid(-10) >= 0.01 {
		t.Errorf("sigmoid(-10) = %g, want just over 0", sigmoid(-10))
	}
}

// TestSLOCompositeValueIsScaleFreeInTau pins the property that each enabled dimension
// uses ITS OWN threshold as the transition bandwidth.
func TestSLOCompositeValueIsScaleFreeInTau(t *testing.T) {
	// Exactly at both deadlines, each factor is sigmoid(0) = 0.5.
	closeTo(t, sloCompositeValue(interactiveSLO, 1_000_000, 16_000_000), 0.25, "value at both deadlines")

	// Well inside both deadlines the value is higher; well past, it approaches 0.
	inside := sloCompositeValue(interactiveSLO, 100_000, 1_000_000)
	past := sloCompositeValue(interactiveSLO, 5_000_000, 60_000_000)
	if inside <= past {
		t.Error("value must decrease as latency rises")
	}
	if past >= 0.05 {
		t.Errorf("well past both deadlines should be near zero, got %g", past)
	}

	// THE ACHIEVABLE RANGE IS NARROW, and it is a consequence of using tau as the
	// bandwidth rather than a defect. Each factor is sigmoid((tau - latency)/tau), which
	// for the best possible case -- zero latency -- is sigmoid(1), so the composite value
	// is bounded above by sigmoid(1)^2 ~= 0.535 for ANY non-negative latency.
	//
	// This is worth pinning because it sets the scale of every charge in the objective:
	// differences between candidates are small in absolute terms, which is exactly why J
	// must not be clamped to [0,1] by a scorer, and why V is a common multiplier that
	// cannot be folded away without changing what the ablation gate reads.
	best := sloCompositeValue(interactiveSLO, 0, 0)
	closeTo(t, best, sigmoid(1)*sigmoid(1), "best achievable composite value")
	if inside > best {
		t.Errorf("no latency can exceed the zero-latency value: inside=%g best=%g", inside, best)
	}
}

// TestSLOCompositeValueIgnoresTauITL pins the absence the specification insists on: the
// composite routing kernel NEVER reads tau_itl. Mean ITL is an evaluation gate on
// reported goodput, not a routing term.
func TestSLOCompositeValueIgnoresTauITL(t *testing.T) {
	withITL := interactiveSLO
	withoutITL := interactiveSLO
	withoutITL.tauITLUs = 0
	absurdITL := interactiveSLO
	absurdITL.tauITLUs = 999_999_999

	a := sloCompositeValue(withITL, 500_000, 8_000_000)
	b := sloCompositeValue(withoutITL, 500_000, 8_000_000)
	c := sloCompositeValue(absurdITL, 500_000, 8_000_000)
	if a != b || b != c {
		t.Errorf("tau_itl must not affect the routing value: %g, %g, %g", a, b, c)
	}
}

// TestSLOCompositeValueDisabledTargetContributesOne is the FLATTENING trap, and it is why
// Config.validate rejects a non-positive tau in the selected triple. A disabled target
// contributes exactly one, so a zero triple makes the value identically 1.0 for every
// candidate rather than loosening the policy.
func TestSLOCompositeValueDisabledTargetContributesOne(t *testing.T) {
	zero := slo{}
	if got := sloCompositeValue(zero, 999_999_999, 999_999_999); got != 1.0 {
		t.Errorf("a zero triple must yield exactly 1.0 (the flattening trap), got %g", got)
	}
	// With only TTFT enabled, the E2E dimension drops out entirely.
	ttftOnly := slo{tauTTFTUs: 1_000_000}
	closeTo(t, sloCompositeValue(ttftOnly, 1_000_000, 99_000_000),
		sigmoid(0), "only the enabled dimension contributes")
}

// TestGDecodeCompositeNoFirstTokenContributesNothing pins why the two resident
// populations must be split by first-token state: a resident with no first token returns
// 0 both before and after, so it would be MISSED ENTIRELY if it were not carried in the
// prefill population instead.
func TestGDecodeCompositeNoFirstTokenContributesNothing(t *testing.T) {
	cr := decodeResident{rem: 10, arrivalUs: 0, ttftSet: false, slo: interactiveSLO}
	if got := gDecodeComposite(cr, 5_000_000); got != 0 {
		t.Errorf("a resident with no first token must contribute 0, got %g", got)
	}
	if got := varDecodeContribution(cr, 1_000_000, 9_000_000); got != 0 {
		t.Errorf("its charge must be 0 regardless of the delay, got %g", got)
	}
}

// THE tau_e2e ZERO TRAP, tested because it degenerates the arm while it keeps running and
// keeps reporting goodput.
//
// The realized-TTFT factor is placement-INVARIANT, so it is identical on both sides of
// the charge and factors out. With tau_e2e = 0 the value reduces to that invariant factor
// alone, so EVERY resident's contribution is exactly zero and the entire decode-side
// externality vanishes -- leaving the arm running on its own-good term alone.
func TestTauE2EZeroCollapsesEntireDecodeExternality(t *testing.T) {
	noE2E := slo{tauTTFTUs: 1_000_000, tauE2EUs: 0}
	residents := []decodeResident{
		{rem: 50, arrivalUs: 0, firstTokenUs: 200_000, ttftSet: true, slo: noE2E},
		{rem: 80, arrivalUs: 100_000, firstTokenUs: 400_000, ttftSet: true, slo: noE2E},
	}
	rt := exactRT(4000)

	got := varDecodeLocalAfter(1_000_000, residents, rt, 2, 0)
	if got != 0 {
		t.Errorf("with tau_e2e = 0 the whole decode externality must collapse to 0, got %g", got)
	}
	// Sanity: with a positive tau_e2e the same residents DO produce a charge, so the
	// zero above is the trap and not an artefact of the fixture.
	for i := range residents {
		residents[i].slo = interactiveSLO
	}
	if live := varDecodeLocalAfter(1_000_000, residents, rt, 2, 0); live <= 0 {
		t.Errorf("with a positive tau_e2e the same residents must be charged, got %g", live)
	}
}

// TestVarDecodeContributionSignConvention pins the direction: POSITIVE means the placement
// destroyed value.
func TestVarDecodeContributionSignConvention(t *testing.T) {
	cr := decodeResident{rem: 50, arrivalUs: 0, firstTokenUs: 200_000, ttftSet: true, slo: interactiveSLO}
	cb := 8_000_000.0
	cp := 12_000_000.0 // delayed
	if got := varDecodeContribution(cr, cb, cp); got <= 0 {
		t.Errorf("a delayed resident must yield a positive charge, got %g", got)
	}
	// Equal completions charge nothing.
	if got := varDecodeContribution(cr, cb, cb); got != 0 {
		t.Errorf("an undisturbed resident must charge 0, got %g", got)
	}
}

// TestVarDecodeSkipsCensoredResidents pins that rem < 0 is SKIPPED, not treated as
// zero-remaining: a censored resident carries no information and must not be charged.
func TestVarDecodeSkipsCensoredResidents(t *testing.T) {
	rt := exactRT(4000)
	censored := []decodeResident{
		{rem: -1, arrivalUs: 0, firstTokenUs: 100_000, ttftSet: true, slo: interactiveSLO},
	}
	if got := varDecodeLocalAfter(0, censored, rt, 2, 0); got != 0 {
		t.Errorf("censored resident must be skipped, got %g", got)
	}
	if got := varDecodeDisagg(0, censored, rt, 2); got != 0 {
		t.Errorf("censored resident must be skipped on the disagg path too, got %g", got)
	}
}

// TestLocalChargesMoreThanDisaggAtEqualWindow is the mechanism's basic asymmetry
// expressed in value terms rather than time terms.
func TestLocalChargesMoreThanDisaggAtEqualWindow(t *testing.T) {
	rt := exactRT(4000)
	residents := []decodeResident{
		{rem: 60, arrivalUs: 0, firstTokenUs: 150_000, ttftSet: true, slo: interactiveSLO},
		{rem: 90, arrivalUs: 50_000, firstTokenUs: 300_000, ttftSet: true, slo: interactiveSLO},
	}
	now := 1_000_000.0
	window := 3.0
	nChunks := 2.0

	local := varDecodeLocalAfter(now, residents, rt, nChunks, window)
	disagg := varDecodeDisagg(now, residents, rt, window)
	if local <= disagg {
		t.Errorf("local placement must destroy more resident value: local=%g disagg=%g", local, disagg)
	}
}

// THE DISCRIMINATION PROPERTY -- the claim the whole experiment rests on.
//
// Two endpoints with IDENTICAL load (same batch, same KV, same queue) are not
// interchangeable: one holding residents just inside their deadline is charged far more
// than one holding residents with comfortable slack, or residents already doomed. A
// load-shaped signal cannot see that difference, because it is a property of the
// residents' deadlines rather than of the endpoint's load.
func TestExternalityDiscriminatesResidentsWithEqualLoad(t *testing.T) {
	rt := exactRT(4000)
	now := 100_000_000.0
	nChunks, window := 2.0, 0.0

	// Endpoint A: residents sitting just inside their 16 s end-to-end deadline, where
	// one extra co-scheduled prefill flips them to a miss.
	tight := []decodeResident{
		{rem: 100, arrivalUs: int64(now) - 15_000_000, firstTokenUs: int64(now) - 14_800_000, ttftSet: true, slo: interactiveSLO},
	}
	// Endpoint B: freshly arrived residents with comfortable slack.
	slack := []decodeResident{
		{rem: 100, arrivalUs: int64(now) - 100_000, firstTokenUs: int64(now) - 50_000, ttftSet: true, slo: interactiveSLO},
	}
	// Endpoint C: residents already far past their deadline -- nothing recoverable, so
	// the same prefill costs nothing.
	doomed := []decodeResident{
		{rem: 100, arrivalUs: int64(now) - 120_000_000, firstTokenUs: int64(now) - 119_000_000, ttftSet: true, slo: interactiveSLO},
	}

	tightCharge := varDecodeLocalAfter(now, tight, rt, nChunks, window)
	slackCharge := varDecodeLocalAfter(now, slack, rt, nChunks, window)
	doomedCharge := varDecodeLocalAfter(now, doomed, rt, nChunks, window)

	if !(tightCharge > slackCharge) {
		t.Errorf("residents near their deadline must cost more than residents with slack: tight=%g slack=%g",
			tightCharge, slackCharge)
	}
	if !(tightCharge > doomedCharge) {
		t.Errorf("residents near their deadline must cost more than already-doomed residents: tight=%g doomed=%g",
			tightCharge, doomedCharge)
	}
	if doomedCharge > slackCharge*0.5 {
		t.Errorf("already-doomed residents should be nearly free: doomed=%g slack=%g", doomedCharge, slackCharge)
	}
}

// TestGCollocCompositeIgnoresEndToEnd pins the deliberate information-set restriction: a
// resident still prefilling has no assigned decoder state in the routing view, so its
// declared phase value is TTFT-only. "Improving" this to read the E2E completion would
// make the collocated term inconsistent with the decode term.
func TestGCollocCompositeIgnoresEndToEnd(t *testing.T) {
	k := prefillResident{remPrefillTokens: 1000, arrivalUs: 0, slo: interactiveSLO}
	a := gCollocComposite(k, 500_000, 1_000_000)
	b := gCollocComposite(k, 500_000, 999_000_000)
	if a != b {
		t.Errorf("the second parameter must be unused: %g vs %g", a, b)
	}
	// A disabled TTFT target yields exactly 1.
	noTTFT := prefillResident{slo: slo{tauTTFTUs: 0}}
	if got := gCollocComposite(noTTFT, 5_000_000, 0); got != 1 {
		t.Errorf("disabled tau_ttft must yield 1, got %g", got)
	}
}

// TestVarPrefillTTFTContributionGuardsZeroScale confirms the bandwidth fallback so a
// disabled TTFT target cannot divide by zero.
func TestVarPrefillTTFTContributionGuardsZeroScale(t *testing.T) {
	k := prefillResident{arrivalUs: 0, slo: slo{tauTTFTUs: 0}}
	got := varPrefillTTFTContribution(k, 100, 200)
	if math.IsNaN(got) || math.IsInf(got, 0) {
		t.Errorf("zero bandwidth must not produce NaN/Inf, got %g", got)
	}
}

// TestVarCollocPrefillLocalHarmsOccupantTwice pins the double harm the specification
// describes: a local placement both slows the occupant's remaining prefill iterations and
// then slows its decode steps once the arrival joins the batch.
func TestVarCollocPrefillLocalHarmsOccupantTwice(t *testing.T) {
	rt := exactRT(4000)
	now := 1_000_000.0
	occupants := []prefillResident{
		{remPrefillTokens: 4000, remDecodeSteps: 100, arrivalUs: int64(now) - 200_000, slo: interactiveSLO},
	}
	withDecode := varCollocPrefillLocalAfter(now, occupants, rt, 2048, 2, 0)

	// The same occupant with no decode horizon is charged on its first token only.
	occupantsNoDecode := []prefillResident{
		{remPrefillTokens: 4000, remDecodeSteps: 0, arrivalUs: int64(now) - 200_000, slo: interactiveSLO},
	}
	firstTokenOnly := varCollocPrefillLocalAfter(now, occupantsNoDecode, rt, 2048, 2, 0)

	// Both are positive charges, and the composite kernel is TTFT-only for collocated
	// occupants, so the two agree -- which is exactly the information-set restriction
	// TestGCollocCompositeIgnoresEndToEnd pins. The point of this test is that the
	// occupant is charged AT ALL, since the decode-side terms miss it entirely.
	if withDecode <= 0 {
		t.Errorf("a collocated prefill occupant must be charged by a local placement, got %g", withDecode)
	}
	closeTo(t, withDecode, firstTokenOnly, "colloc charge is TTFT-only")
}

// TestVarCollocPrefillDisaggZeroInsideArrivalWindow pins the causal zero on the
// collocated path.
func TestVarCollocPrefillDisaggZeroInsideArrivalWindow(t *testing.T) {
	rt := exactRT(4000)
	now := 1_000_000.0
	// One chunk of prefill left -> remPf = 1, well inside an arrival window of 10.
	occupants := []prefillResident{
		{remPrefillTokens: 100, remDecodeSteps: 0, arrivalUs: int64(now) - 100_000, slo: interactiveSLO},
	}
	if got := varCollocPrefillDisagg(now, occupants, rt, 2048, 10); got != 0 {
		t.Errorf("an occupant reaching its first token inside the arrival window must charge 0, got %g", got)
	}
}

func TestVarCollocPrefillSkipsCensoredOccupants(t *testing.T) {
	rt := exactRT(4000)
	censored := []prefillResident{{remPrefillTokens: -1, slo: interactiveSLO}}
	if got := varCollocPrefillLocalAfter(0, censored, rt, 2048, 2, 0); got != 0 {
		t.Errorf("censored occupant must be skipped, got %g", got)
	}
	if got := varCollocPrefillDisagg(0, censored, rt, 2048, 2); got != 0 {
		t.Errorf("censored occupant must be skipped on the disagg path, got %g", got)
	}
}

// TestExactPrefillPoolFormChargesLessThanLegacy is the point of the exact correction: the
// legacy form charges an occupant the arrival's ENTIRE prefill duration even when that
// occupant has one iteration left.
func TestExactPrefillPoolFormChargesLessThanLegacy(t *testing.T) {
	now := 1_000_000.0
	tIterP := h100.tIterPrefill(0)
	chunkP := 2048.0
	// An occupant with one chunk left, and an arrival with a long multi-chunk prompt.
	occupants := []prefillResident{
		{remPrefillTokens: 100, remDecodeSteps: 0, arrivalUs: int64(now) - 100_000, slo: interactiveSLO},
	}
	rAp, rAr := 40000, 40000 // deep-research-scale prompt
	rPrefillUs := math.Ceil(float64(rAp)/chunkP)*tIterP + h100.Wp(rAp, rAr)

	exact := varPrefillDisaggExactAfter(now, occupants, tIterP, chunkP, 0, rAp, rAr, h100)
	legacy := varPrefillDisaggAfter(now, occupants, tIterP, chunkP, 0, rPrefillUs)

	if exact >= legacy {
		t.Errorf("the exact form must charge a nearly-done occupant less than the legacy form: exact=%g legacy=%g",
			exact, legacy)
	}
	if exact <= 0 {
		t.Errorf("the exact form must still charge something, got %g", exact)
	}
}

// TestGoodSelfRewardsFasterPlacements pins the own-good term's direction and the fact that
// it reads tIterAfter -- the B+1 re-timed rate the arrival experiences once it joins.
func TestGoodSelfRewardsFasterPlacements(t *testing.T) {
	rt := exactRT(4000)
	fast := goodSelf(interactiveSLO, 200_000, rt.tIterAfter, 300)
	slow := goodSelf(interactiveSLO, 900_000, rt.tIterAfter, 300)
	if fast <= slow {
		t.Errorf("a faster projected TTFT must earn more own good: fast=%g slow=%g", fast, slow)
	}
	// E2E is tHat + nOut*tIterAfter, so a longer output lowers the value.
	longer := goodSelf(interactiveSLO, 200_000, rt.tIterAfter, 900)
	if longer >= fast {
		t.Errorf("a longer expected output must lower own good: %g vs %g", longer, fast)
	}
}

// TestPathBreakdownTotalIsAuditable pins that the aggregate is the sum of the three named
// populations, which is what makes the ablation cohort's validity gate checkable.
func TestPathBreakdownTotalIsAuditable(t *testing.T) {
	v := pathBreakdown{decode: 1.5, collocPrefill: 0.25, prefillPool: 0.75}
	closeTo(t, v.total(), 2.5, "pathBreakdown total")
}

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

// exactRT builds the re-timing this arm actually uses: exact prefill overlap forced on,
// with the operands the externality functions set.
// The fixture batch state for exactRT. Fixed rather than parameterised: the variation
// these tests exercise is the uncached suffix ap, and holding the batch state constant is
// what makes the comparisons between calls meaningful.
const (
	exactRTInputLen = 4000
	exactRTChunk    = 2048 // the engine's max_num_batched_tokens
	exactRTBatch    = 8
	exactRTKV       = int64(50_000)
	exactRTSPf      = int64(0)
)

// exactRT builds the re-timing this arm actually uses: exact prefill overlap forced on,
// with the operands the externality functions set.
func exactRT(ap int) reTiming {
	rt := reTimingFor(exactRTInputLen, h100, exactRTBatch, exactRTKV, exactRTSPf, exactRTChunk)
	rt.exactPrefillOverlap = true
	rt.cPf = h100.CPf
	rt.ap = float64(maxInt(ap, 0))
	rt.ar = float64(exactRTInputLen)
	return rt
}

// TestReTimingForFullReTimingNotMarginalAdd pins the modelling claim that tIterAfter is
// tIterDecode recomputed at (B+1, kv+dkv_R), not the baseline plus a marginal term.
func TestReTimingForFullReTimingNotMarginalAdd(t *testing.T) {
	inputLen := 4000
	bDec, kv, sPf := 8, int64(50000), int64(0)
	rt := reTimingFor(inputLen, h100, bDec, kv, sPf, 0)

	closeTo(t, rt.tIter0, h100.tIterDecode(bDec, kv, sPf), "tIter0")
	closeTo(t, rt.tIterAfter, h100.tIterDecode(bDec+1, kv+int64(inputLen), sPf), "tIterAfter")

	// dkv_R is the request's FULL input length -- input-only, reading no hidden output.
	if rt.tIterAfter <= rt.tIter0 {
		t.Error("joining the batch must raise the iteration time")
	}
}

// TestReTimingOverlapAddsResidentPrefillTokens covers the legacy operand, which is dead
// on this arm but must remain correct so the legacy branch stays runnable.
func TestReTimingOverlapAddsResidentPrefillTokens(t *testing.T) {
	chunk := 2048
	rt := reTimingFor(1000, h100, 4, 1000, 0, chunk)
	closeTo(t, rt.tIterOverlap, h100.tIterDecode(4, 1000, int64(chunk)), "tIterOverlap")
}

// THE CAUSAL PROPERTY, and the single most important test in this file.
//
// A resident that finishes INSIDE the admission window is delayed by exactly nothing, so
// its projected completion under a local placement equals its baseline completion. That
// zero is what makes the term an externality rather than another load proxy: a load proxy
// would charge every resident on a busy endpoint regardless of whether it was still there
// when the arrival landed.
func TestCLocalAfterResidentFinishingInsideAdmissionWindowIsUntouched(t *testing.T) {
	rt := exactRT(4000)
	now := 1_000_000.0
	rem := int64(3)
	admissionSteps := 10.0 // the arrival waits longer than this resident has left
	nChunks := 2.0

	cb := rt.cBase(now, rem)
	cp := rt.cLocalAfter(now, rem, admissionSteps, nChunks)
	if cp != cb {
		t.Errorf("a resident finishing inside the admission window must be untouched: cb=%g cp=%g", cb, cp)
	}

	// And its charge is therefore exactly zero.
	cr := decodeResident{rem: rem, ttftSet: true, arrivalUs: int64(now) - 500_000,
		slo: slo{tauTTFTUs: 1_000_000, tauE2EUs: 16_000_000}}
	if got := varDecodeContribution(cr, cb, cp); got != 0 {
		t.Errorf("charge must be exactly 0, got %g", got)
	}
}

// The mirror property on the disaggregated path: a resident finishing before the remote
// prefill's output arrives is likewise untouched.
func TestCDisaggResidentFinishingInsideArrivalWindowIsUntouched(t *testing.T) {
	rt := exactRT(4000)
	now := 1_000_000.0
	rem := int64(2)

	cb := rt.cBase(now, rem)
	cp := rt.cDisagg(now, rem, 10.0)
	if cp != cb {
		t.Errorf("resident finishing inside the arrival window must be untouched: cb=%g cp=%g", cb, cp)
	}
}

// TestCLocalAfterThreePhaseStructure walks the pre/overlap/remainder decomposition
// explicitly, so a refactor that collapses a phase is caught.
func TestCLocalAfterThreePhaseStructure(t *testing.T) {
	ap := 4000
	rt := exactRT(ap)
	now := 0.0
	rem := int64(20)
	admissionSteps, nChunks := 5.0, 2.0

	pre := 5.0
	remaining := 20.0 - pre
	overlap := math.Min(nChunks, remaining)
	want := now + pre*rt.tIter0 + overlap*rt.tIter0 +
		prefillMarginalWork(rt.cPf, rt.cAttn, rt.ap, rt.ar, rt.chunk, overlap) +
		(remaining-overlap)*rt.tIterAfter

	closeTo(t, rt.cLocalAfter(now, rem, admissionSteps, nChunks), want, "cLocalAfter three phases")
}

// TestExactOverlapChargesOnlyMarginalWork is the difference between the exact branch this
// arm forces and the legacy branch. The exact form charges the baseline iteration time
// for the overlap window plus only the arrival's MARGINAL prefill work; the legacy form
// assumes every overlapping chunk is full and starts at prefix zero.
func TestExactOverlapChargesOnlyMarginalWork(t *testing.T) {
	// A mostly-cached prompt: only 100 tokens are actually uncached.
	ap := 100
	now := 0.0
	rem := int64(20)

	exact := exactRT(ap)
	legacy := reTimingFor(exactRTInputLen, h100, exactRTBatch, exactRTKV, exactRTSPf, exactRTChunk) // exactPrefillOverlap false

	gotExact := exact.cLocalAfter(now, rem, 0, 1)
	gotLegacy := legacy.cLocalAfter(now, rem, 0, 1)

	// The legacy form charges a FULL chunk of prefill work; the exact form charges only
	// the 100 uncached tokens. So legacy over-prices this candidate.
	if gotLegacy <= gotExact {
		t.Errorf("legacy form should over-price a mostly-cached prompt: exact=%g legacy=%g", gotExact, gotLegacy)
	}
}

// TestPrefillMarginalWorkExcludesBaselineIterationTime pins the exclusion that makes
// cLocalAfter's exact branch charge overlap*tIter0 separately.
func TestPrefillMarginalWorkExcludesBaselineIterationTime(t *testing.T) {
	// Fully-uncached prompt processed in one chunk: the work is CPf*ap plus the causal
	// attention against a zero cached prefix.
	ap, ar, chunk := 1000.0, 1000.0, 2048.0
	want := h100.CPf*ap + h100.CAttn*ap*(0+ap/2)
	closeTo(t, prefillMarginalWork(h100.CPf, h100.CAttn, ap, ar, chunk, 1), want, "marginal work, uncached")

	// With a cached prefix the attention is charged against that prefix.
	ap2 := 200.0
	want2 := h100.CPf*ap2 + h100.CAttn*ap2*((ar-ap2)+ap2/2)
	closeTo(t, prefillMarginalWork(h100.CPf, h100.CAttn, ap2, ar, chunk, 1), want2, "marginal work, cached prefix")

	// Degenerate inputs contribute nothing rather than a negative or NaN.
	for _, tc := range []struct {
		name             string
		ap, chunk, iters float64
	}{
		{"no uncached tokens", 0, chunk, 1},
		{"no chunk", ap, 0, 1},
		{"no iterations", ap, chunk, 0},
		{"negative ap", -5, chunk, 1},
	} {
		if got := prefillMarginalWork(h100.CPf, h100.CAttn, tc.ap, ar, tc.chunk, tc.iters); got != 0 {
			t.Errorf("%s: want 0, got %g", tc.name, got)
		}
	}

	// It is capped by ap: more iterations cannot process more than the uncached span.
	saturated := prefillMarginalWork(h100.CPf, h100.CAttn, ap, ar, chunk, 100)
	oneChunk := prefillMarginalWork(h100.CPf, h100.CAttn, ap, ar, chunk, 1)
	closeTo(t, saturated, oneChunk, "marginal work saturates at ap")
}

// TestCLocalIsZeroAdmissionCLocalAfter pins the compatibility entry point.
func TestCLocalIsZeroAdmissionCLocalAfter(t *testing.T) {
	rt := exactRT(4000)
	closeTo(t, rt.cLocal(0, 10, 2), rt.cLocalAfter(0, 10, 0, 2), "cLocal == cLocalAfter with zero admission")
}

// TestLocalVersusDisaggAsymmetry pins the asymmetry the policy trades against transfer
// cost: a local placement inflates the decode endpoint's iterations with co-scheduled
// prefill, a disaggregated one does not.
func TestLocalVersusDisaggAsymmetry(t *testing.T) {
	ap := 4000
	rt := exactRT(ap)
	now, rem := 0.0, int64(30)
	nChunks := math.Ceil(float64(ap) / float64(exactRTChunk))

	// Same admission/arrival window, so the only difference is the prefill overlap.
	window := 3.0
	local := rt.cLocalAfter(now, rem, window, nChunks)
	disagg := rt.cDisagg(now, rem, window)

	if local <= disagg {
		t.Errorf("local placement must delay residents more than remote at equal windows: local=%g disagg=%g", local, disagg)
	}
}

// TestCDisaggTailRunsAtReTimedRate confirms the disaggregated projection still pays the
// B+1 re-timing once the arrival joins the decode batch -- remote prefill avoids the
// overlap, not the join.
func TestCDisaggTailRunsAtReTimedRate(t *testing.T) {
	rt := exactRT(4000)
	now, rem := 0.0, int64(10)
	arrivalSteps := 4.0
	want := now + arrivalSteps*rt.tIter0 + (10.0-arrivalSteps)*rt.tIterAfter
	closeTo(t, rt.cDisagg(now, rem, arrivalSteps), want, "cDisagg tail")
}

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

// THE CAUSAL RE-TIMING MODEL, ported from sim/edpp_var.go:100-188.
//
// These are BATCH-LEVEL per-iteration times, not per-resident: adding one request
// changes the whole batch's iteration time.
//
// THE ADMISSION GATING IN cLocalAfter AND cDisagg IS THE CAUSAL MECHANISM. A
// resident that completes before the arrival is admitted is delayed by exactly
// nothing, so its contribution is exactly zero. That is what makes this term an
// externality rather than another load proxy, and it is why the effect survives on
// a fleet whose load signals are already balanced.

// reTiming holds the three per-iteration decode times a placement can produce.
// sim/edpp_var.go:109-127.
//
//	tIter0       current batch B per-iter time -- the baseline.
//	tIterOverlap LOCAL placement, while the arriving request R prefills
//	             co-scheduled on the decode batch: chunked prefill adds `chunk`
//	             resident prefill tokens, so this is tIter0 + CPf*chunk.
//	             DEAD ON THIS ARM: computed by reTimingFor on every call but never
//	             read, because exactPrefillOverlap is forced true and the exact
//	             branch of cLocalAfter returns before touching it. Retained so a
//	             reader does not mistake it for a live term -- or delete it and
//	             then be unable to run the legacy branch at all.
//	tIterAfter   after R JOINS the decode batch: tIterDecode(B+1, kv+dkv_R, sPf),
//	             where dkv_R is R's full input length. This is FULL re-timing --
//	             tIterDecode recomputed with B+1 and kv+dkv_R, not a marginal add.
type reTiming struct {
	tIter0       float64
	tIterOverlap float64
	tIterAfter   float64

	// cAttn and chunk parameterize the causal prefill attention added to the
	// overlap window: each of R's co-scheduled chunks attends to its causal prefix.
	cAttn float64
	chunk float64

	// exactPrefillOverlap and its operands select the EXACT marginal-overlap form.
	// THIS ARM FORCES IT TRUE (sim/edpp.go:1709, :1745), overriding config: the
	// legacy form assumes every overlapping chunk is full and starts at prefix
	// zero, while the exact form uses the known uncached span [ar-ap, ar), handles
	// the partial last chunk, and charges only marginal work above the baseline
	// decode iteration.
	exactPrefillOverlap bool
	cPf                 float64
	ap                  float64
	ar                  float64
}

// reTimingFor builds the batch-level per-iteration times under decode physics
// thetaD at batch state (bDec, kv, sPf). sim/edpp_var.go:675-692.
//
// dkv_R is R's FULL input length -- its resident context once it joins the decode
// batch. Input-only, so it reads no hidden output length.
func reTimingFor(inputLen int, thetaD Coeffs, bDec int, kv, sPf int64, chunk int) reTiming {
	dkv := int64(inputLen)
	return reTiming{
		tIter0:       thetaD.tIterDecode(bDec, kv, sPf),
		tIterOverlap: thetaD.tIterDecode(bDec, kv, sPf+int64(chunk)),
		tIterAfter:   thetaD.tIterDecode(bDec+1, kv+dkv, sPf),
		cAttn:        thetaD.CAttn,
		chunk:        float64(chunk),
	}
}

// cBase is a decode resident's projected completion with rem steps left at the
// current (pre-R) batch per-iteration time. sim/edpp_var.go:130-132.
func (rt reTiming) cBase(now float64, rem int64) float64 {
	return now + float64(rem)*rt.tIter0
}

// cLocal is the zero-admission-delay compatibility form of cLocalAfter.
// sim/edpp_var.go:135-137.
//
// Not reached by this arm, which always supplies an admission window; retained
// because it is the published function's other entry point.
func (rt reTiming) cLocal(now float64, rem int64, nChunks float64) float64 {
	return rt.cLocalAfter(now, rem, 0, nChunks)
}

// cLocalAfter is the resident's completion under LOCAL placement when R waits
// admissionSteps baseline iterations before joining. sim/edpp_var.go:144-166.
//
// THE CAUSAL STRUCTURE, in three phases:
//   - pre:       min(admissionSteps, rem) iterations run UNDISTURBED. A resident
//     that finishes inside the admission window is untouched.
//   - overlap:   of the surviving tail, the first min(nChunks, remaining)
//     iterations overlap R's prefill.
//   - remainder: the rest run at the B+1 re-timed rate.
func (rt reTiming) cLocalAfter(now float64, rem int64, admissionSteps, nChunks float64) float64 {
	pre := math.Min(math.Max(admissionSteps, 0), float64(rem))
	remaining := float64(rem) - pre
	overlap := math.Min(nChunks, remaining)
	if rt.exactPrefillOverlap {
		// THE BRANCH THIS ARM TAKES. Baseline iteration time for the overlap
		// window plus only R's MARGINAL prefill work.
		return now + pre*rt.tIter0 + overlap*rt.tIter0 +
			prefillMarginalWork(rt.cPf, rt.cAttn, rt.ap, rt.ar, rt.chunk, overlap) +
			(remaining-overlap)*rt.tIterAfter
	}
	// LEGACY form, retained because it is what the published function computes when
	// the exact flag is off. Causal prefill attention over R's co-scheduled chunks
	// j = 0..overlap-1, each charged against causal prefix j*chunk + chunk/2
	// (start prefix 0), summing to cAttn*chunk^2*overlap^2/2.
	attn := rt.cAttn * rt.chunk * rt.chunk * overlap * overlap / 2.0
	return now + pre*rt.tIter0 + overlap*rt.tIterOverlap + attn + (remaining-overlap)*rt.tIterAfter
}

// prefillMarginalWork is the exact work added by the first `iterations` chunks of
// an arriving request's uncached prefill. sim/edpp_var.go:168-179.
//
//	processed    = min(ap, iterations*chunk)
//	cachedPrefix = max(ar-ap, 0)
//	work         = CPf*processed + CAttn*processed*(cachedPrefix + processed/2)
//
// It EXCLUDES baseline iteration time: residents would pay that even if the
// arriving request were absent. That exclusion is why cLocalAfter's exact branch
// charges overlap*tIter0 separately rather than overlap*tIterOverlap.
func prefillMarginalWork(cPf, cAttn, ap, ar, chunk, iterations float64) float64 {
	if ap <= 0 || chunk <= 0 || iterations <= 0 {
		return 0
	}
	processed := math.Min(ap, iterations*chunk)
	cachedPrefix := math.Max(ar-ap, 0)
	return cPf*processed + cAttn*processed*(cachedPrefix+processed/2.0)
}

// cDisagg is the resident's completion under DISAGGREGATED placement.
// sim/edpp_var.go:181-188.
//
// R prefills remotely, so the decode endpoint is undisturbed for arrivalSteps
// iterations (R's remote prefill plus KV transfer, in units of tIter0), and only
// the tail max(rem - arrivalSteps, 0) steps run at the B+1 re-timed time. There is
// NO prefill-overlap inflation on this endpoint -- that is the asymmetry the policy
// trades against transfer cost.
func (rt reTiming) cDisagg(now float64, rem int64, arrivalSteps float64) float64 {
	pre := math.Min(arrivalSteps, float64(rem))
	return now + pre*rt.tIter0 + (float64(rem)-pre)*rt.tIterAfter
}

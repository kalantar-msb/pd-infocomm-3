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

// SLO VALUE KERNELS.
//
// THIS ARM USES THE `composite` KERNEL EXCLUSIVELY -- forced at sim/edpp.go:1712
// and :1748, not read from config. The other three kernels (flip, util, hazard)
// are deliberately NOT ported: they are selected by a different rule (Rule=="var")
// that this arm does not use, and no switch in this arm's configuration reaches
// them. Porting them would imply a reachable consumer that does not exist.

// slo is one resident's resolved SLO thresholds in microseconds.
// sim/edpp_var.go:597-608.
type slo struct {
	tauTTFTUs float64
	tauITLUs  float64
	tauE2EUs  float64
}

// sigmoid is the standard logistic. sim/edpp_var.go:992.
func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

// sloCompositeValue is the bounded routing value: a smooth TTFT x E2E surrogate.
// sim/edpp_var.go:242-251.
//
// Each ENABLED dimension uses ITS OWN threshold as the transition bandwidth, so
// the value is scale-free in tau. A DISABLED target (tau <= 0) contributes exactly
// one -- which is why a zero triple FLATTENS the policy to a constant rather than
// loosening it. Config validation rejects a non-positive tau_ttft or tau_e2e in the
// selected triple at startup for exactly this reason.
//
// NOTE WHAT IS ABSENT: tau_itl. The composite kernel never reads it. Mean ITL is an
// evaluation gate on reported goodput, not a routing term.
func sloCompositeValue(s slo, ttftUs, e2eUs float64) float64 {
	u := 1.0
	if s.tauTTFTUs > 0 {
		u *= sigmoid((s.tauTTFTUs - ttftUs) / s.tauTTFTUs)
	}
	if s.tauE2EUs > 0 {
		u *= sigmoid((s.tauE2EUs - e2eUs) / s.tauE2EUs)
	}
	return u
}

// decodeResident is one already-decoding resident's value inputs.
// sim/edpp_var.go, assembled at :615-641.
type decodeResident struct {
	rem          int64 // remaining decode steps; < 0 means SKIP
	arrivalUs    int64
	firstTokenUs int64
	ttftSet      bool
	slo          slo
}

// gDecodeComposite is a decoding resident's composite good at completion cUs.
// sim/edpp_var.go:253-260.
//
// Its realized TTFT is FIXED by history, so the placement cannot move it: it enters
// as a constant factor. THAT IS WHY DEGRADATION D2c HAS THE DIRECTION IT DOES -- a
// late-recorded first token shrinks sigmoid((tau_ttft - ttft)/tau_ttft), a COMMON
// POSITIVE MULTIPLIER on both terms of the charge, scaling the whole difference
// down and biasing toward local.
//
// A resident with no first token yet returns 0 both before and after, so it
// contributes nothing here. Such requests are NOT lost: they are carried in the
// prefill-occupant population instead, where their first token is still live.
//
// THE tau_e2e ZERO TRAP, stated where it bites. Because the TTFT factor is REALIZED
// and therefore placement-invariant, it is identical on both sides of the charge and
// factors out. So if tau_e2e resolves to 0 for a resident, this function reduces to
// that invariant factor alone and the resident's contribution is EXACTLY ZERO -- and
// if tau_e2e is 0 for every resident, THE ENTIRE DECODE-SIDE EXTERNALITY IS
// IDENTICALLY ZERO and the arm degenerates to its own-good term while still running
// and still reporting goodput. A positive E2E target is a correctness requirement,
// not a tuning choice, and Config.validate enforces it.
func gDecodeComposite(cr decodeResident, cUs float64) float64 {
	if !cr.ttftSet {
		return 0
	}
	ttft := float64(cr.firstTokenUs - cr.arrivalUs)
	return sloCompositeValue(cr.slo, ttft, cUs-float64(cr.arrivalUs))
}

// varDecodeContribution is one decoding resident's charge: the good it HAD at its
// baseline completion cb minus the good it HAS at its placed completion cp.
// sim/edpp_var.go:319-338 (composite branch).
//
// Positive means the placement destroyed value. When cp == cb -- the resident
// finished inside the admission window -- this is exactly zero.
func varDecodeContribution(cr decodeResident, cb, cp float64) float64 {
	return gDecodeComposite(cr, cb) - gDecodeComposite(cr, cp)
}

// varDecodeLocalAfter sums the decode-side externality of a LOCAL placement.
// sim/edpp_var.go:288-301.
//
// Residents with rem < 0 are SKIPPED, not treated as zero-remaining: a censored
// resident carries no information and must not be charged.
func varDecodeLocalAfter(now float64, crs []decodeResident, rt reTiming, nChunks, admissionSteps float64) float64 {
	var sum float64
	for _, cr := range crs {
		if cr.rem < 0 {
			continue
		}
		cb := rt.cBase(now, cr.rem)
		cp := rt.cLocalAfter(now, cr.rem, admissionSteps, nChunks)
		sum += varDecodeContribution(cr, cb, cp)
	}
	return sum
}

// varDecodeDisagg is the disaggregated mirror. sim/edpp_var.go:303-317.
func varDecodeDisagg(now float64, crs []decodeResident, rt reTiming, arrivalSteps float64) float64 {
	var sum float64
	for _, cr := range crs {
		if cr.rem < 0 {
			continue
		}
		cb := rt.cBase(now, cr.rem)
		cp := rt.cDisagg(now, cr.rem, arrivalSteps)
		sum += varDecodeContribution(cr, cb, cp)
	}
	return sum
}

// prefillResident is one pre-first-token occupant's value inputs.
// sim/edpp_var.go, assembled at :643-673.
type prefillResident struct {
	remPrefillTokens int64 // remaining PROMPT tokens; < 0 means SKIP
	remDecodeSteps   int64 // decode horizon once it reaches its first token
	arrivalUs        int64
	slo              slo
}

// varPrefillTTFTContribution scores one occupant's FIRST-TOKEN value at risk.
// sim/edpp_var.go:411-441 (composite branch, which equals the util branch).
//
// The deadline is arrival + tau_ttft and the bandwidth is tau_ttft. Shared by the
// remote prefill-pool term and the collocated decode-endpoint term so both score
// first-token risk with identical arithmetic.
func varPrefillTTFTContribution(k prefillResident, cb, cp float64) float64 {
	deadline := float64(k.arrivalUs) + k.slo.tauTTFTUs
	scale := k.slo.tauTTFTUs
	if scale <= 0 {
		scale = 1
	}
	return sigmoid((deadline-cb)/scale) - sigmoid((deadline-cp)/scale)
}

// gCollocComposite is a collocated occupant's composite good when it reaches its
// first token at ftUs. sim/edpp_var.go:560-571.
//
// IT DELIBERATELY IGNORES THE END-TO-END COMPLETION, and the second parameter is
// unused for that reason. A resident still prefilling has no assigned decoder state
// in the routing view, so its declared phase value is TTFT-only; adding a synthetic
// decode horizon here would make the one-step potential depend on state the router
// does not observe. Do not "improve" this to read eUs -- it would make the
// collocated term inconsistent with the decode term's information set.
func gCollocComposite(k prefillResident, ftUs, _ float64) float64 {
	if k.slo.tauTTFTUs <= 0 {
		return 1
	}
	return sigmoid((k.slo.tauTTFTUs - (ftUs - float64(k.arrivalUs))) / k.slo.tauTTFTUs)
}

// varCollocContribution is one collocated occupant's charge under the composite
// kernel. sim/edpp_var.go:505-521 (composite branch).
func varCollocContribution(k prefillResident, ftB, ftP, eB, eP float64) float64 {
	return gCollocComposite(k, ftB, eB) - gCollocComposite(k, ftP, eP)
}

// varCollocPrefillLocalAfter sums the value at risk imposed on the DECODE
// endpoint's collocated prefill occupants by a LOCAL placement.
// sim/edpp_var.go:450-475.
//
// Such an occupant was placed here by a prior decision and has not produced a first
// token, so the decode-side terms miss it entirely. A local placement harms it
// TWICE: it needs ceil(remPrefillTokens/chunk) more batch iterations to reach its
// first token and R's co-scheduled prefill slows those, and then R joins the decode
// batch (B -> B+1) and slows its decode steps too. Both projections use the same
// cBase -> cLocalAfter model.
func varCollocPrefillLocalAfter(now float64, ks []prefillResident, rt reTiming, chunk, nChunks, admissionSteps float64) float64 {
	if chunk < 1 {
		chunk = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remPf := int64(math.Ceil(float64(k.remPrefillTokens) / chunk))
		ftB := rt.cBase(now, remPf)
		ftP := rt.cLocalAfter(now, remPf, admissionSteps, nChunks)
		eB, eP := ftB, ftP
		if k.remDecodeSteps > 0 {
			total := remPf + k.remDecodeSteps
			eB = rt.cBase(now, total)
			eP = rt.cLocalAfter(now, total, admissionSteps, nChunks)
		}
		sum += varCollocContribution(k, ftB, ftP, eB, eP)
	}
	return sum
}

// varCollocPrefillDisagg is the disaggregated mirror. sim/edpp_var.go:477-503.
//
// An occupant that reaches its first token inside the arrival window
// (remPf <= arrivalSteps) sees cDisagg == cBase and contributes exactly zero:
// remote placement does not delay it.
func varCollocPrefillDisagg(now float64, ks []prefillResident, rt reTiming, chunk, arrivalSteps float64) float64 {
	if chunk < 1 {
		chunk = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remPf := int64(math.Ceil(float64(k.remPrefillTokens) / chunk))
		ftB := rt.cBase(now, remPf)
		ftP := rt.cDisagg(now, remPf, arrivalSteps)
		eB, eP := ftB, ftP
		if k.remDecodeSteps > 0 {
			total := remPf + k.remDecodeSteps
			eB = rt.cBase(now, total)
			eP = rt.cDisagg(now, total, arrivalSteps)
		}
		sum += varCollocContribution(k, ftB, ftP, eB, eP)
	}
	return sum
}

// varPrefillDisaggAfter is the LEGACY remote prefill-pool externality.
// sim/edpp_var.go:347-366.
//
// NOT the branch this arm takes -- it forces the exact form below. Retained because
// it is what the published function computes when the exact flag is off, and because
// the difference between the two is the point of the exact correction: this form
// charges an occupant R's ENTIRE prefill duration rPrefillUs, even an occupant with
// one iteration left.
func varPrefillDisaggAfter(now float64, ks []prefillResident, tIterP, chunkP, admissionSteps, rPrefillUs float64) float64 {
	if chunkP < 1 {
		chunkP = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remIters := math.Ceil(float64(k.remPrefillTokens) / chunkP)
		cb := now + remIters*tIterP
		cp := cb
		if remIters > admissionSteps {
			cp += rPrefillUs
		}
		sum += varPrefillTTFTContribution(k, cb, cp)
	}
	return sum
}

// varPrefillDisaggExactAfter is the marginal-overlap correction, and THE BRANCH
// THIS ARM TAKES. sim/edpp_var.go:372-404.
//
// After R's admission, an occupant is delayed only by R's prefill chunks that
// execute BEFORE that occupant reaches its first token. Baseline prefill iterations
// are not charged again, and an occupant with one remaining iteration is not charged
// R's entire multi-chunk prompt.
func varPrefillDisaggExactAfter(
	now float64,
	ks []prefillResident,
	tIterP, chunkP, admissionSteps float64,
	rAp, rAr int,
	coeffs Coeffs,
) float64 {
	if chunkP < 1 {
		chunkP = 1
	}
	rChunks := math.Ceil(float64(maxInt(rAp, 0)) / chunkP)
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remIters := math.Ceil(float64(k.remPrefillTokens) / chunkP)
		remainingAfterAdmission := math.Max(remIters-admissionSteps, 0)
		overlap := math.Min(rChunks, remainingAfterAdmission)
		cb := now + remIters*tIterP
		cp := cb + prefillMarginalWork(
			coeffs.CPf, coeffs.CAttn,
			float64(rAp), float64(rAr),
			chunkP, overlap,
		)
		sum += varPrefillTTFTContribution(k, cb, cp)
	}
	return sum
}

// goodSelf scores the ARRIVING request's OWN smoothed goodput under a candidate.
// sim/edpp_var.go:958-990 (composite branch).
//
// R arrives at the decision instant, so its TTFT measured from arrival IS
// tHatFromNow. It then decodes nOut steps at tIterAfter -- the B+1 re-timed
// per-iteration time it experiences once it joins the batch -- so its end-to-end
// latency from arrival is tHatFromNow + nOut*tIterAfter.
//
// The rule charges externality - ownGood: the value this placement DESTROYS among
// residents minus the value it EARNS for R.
func goodSelf(s slo, tHatFromNow, tIterAfter, nOut float64) float64 {
	ttft := tHatFromNow
	e2e := tHatFromNow + nOut*tIterAfter
	return sloCompositeValue(s, ttft, e2e)
}

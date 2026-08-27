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
	"fmt"
	"math"
)

// Coeffs is the calibrated latency law theta for one GPU type.
//
// Ported from sim/edpp_coeffs.go. Units: AlphaD/AlphaP in us, C0 in us/req,
// C1/CPf in us/token, CAttn in us/token^2. Values for both GPU types in the
// campaign fleet are in config.md section 4 and are supplied through plugin
// configuration keyed by pod-label value -- see Config.CoeffsByGPUType.
type Coeffs struct {
	AlphaD float64 `json:"alphaD"` // alpha:   decode per-iteration fixed cost
	AlphaP float64 `json:"alphaP"` // alpha_p: prefill per-iteration fixed cost (~= AlphaD)
	C0     float64 `json:"c0"`     // decode per-request overhead
	C1     float64 `json:"c1"`     // decode KV-read per resident token
	CPf    float64 `json:"cPf"`    // exposed prefill compute per token
	CAttn  float64 `json:"cAttn"`  // prefill attention term
}

// minMu floors every drain rate so predictor denominators never collapse.
// Upstream: `const edppMinMu = 1e-3` (sim/edpp.go:341).
//
// THE VALUE IS 1e-3, NOT 1e-6. Under this arm's configuration the difference is
// latent rather than active -- Mu reaches AdmissionContext but rollforward never
// reads it -- yet it becomes live the moment the capacity account is enabled
// (muDNom and muPNom set the scale) or the estimator is switched to `waiting`.
const minMu = 1e-3

func clampMu(mu float64) float64 {
	if mu < minMu {
		return minMu
	}
	if mu > 1.0 {
		return 1.0
	}
	return mu
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minI64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func maxI64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// Wp is the prefill demand of ap uncached tokens for a prompt of full length ar,
// in us. sim/edpp_coeffs.go:66-71.
//
// It is the trajectory sum of the causal per-step prefill charge
// CPf*s + CAttn*s*(prefix + s/2), integrated over the prefill from prefix ar-ap
// to ar -- hence the (ar - ap/2) form. At ap == ar (no cache) this is
// CPf*ar + 0.5*CAttn*ar^2.
func (c Coeffs) Wp(ap, ar int) float64 {
	a := float64(ap)
	r := float64(ar)
	return c.CPf*a + c.CAttn*a*(r-a/2.0)
}

// Wd is the decode demand for a prompt of length ar generating o output tokens,
// in us. sim/edpp_coeffs.go:76-83.
//
// The exact discrete per-step sum Sum_{k=0}^{o-1}(C0 + C1*(ar+k))
// = C0*o + C1*o*(ar + (o-1)/2), matching the per-decode-step charge where
// context = ar + k. o is the N_out estimate at routing time.
func (c Coeffs) Wd(ar int, o float64) float64 {
	if o <= 0 {
		return 0
	}
	r := float64(ar)
	return c.C0*o + c.C1*o*(r+(o-1)/2.0)
}

// tIterDecode is the decode iteration time (us) at the given batch state.
// sim/edpp_coeffs.go:85-87.
//
// NOTE ON bDec: upstream's BatchSize() INCLUDES prefilling requests, and so does
// the target's Metrics.RunningRequestsSize. The mapping is EXACT, so this port
// preserves it and deliberately does not rename it or "fix" it to a decode-only
// count. Upstream is internally inconsistent here -- its calibration fits C0
// against a decode-only column while the policy consumes the running total -- and
// this port INHERITS that inconsistency rather than introducing a different one.
// Cost: C0 is ~5.3-5.9 us/req (config.md section 4), so miscounting four
// prefilling requests is ~21 us against a 17000-27000 us iteration, under 0.1%.
func (c Coeffs) tIterDecode(bDec int, kv, sPf int64) float64 {
	return c.AlphaD + c.C0*float64(bDec) + c.C1*float64(kv) + c.CPf*float64(sPf)
}

// tIterPrefill is the prefill iteration time (us). A dedicated prefill server
// runs no decode work, so B_dec = 0 and KV = 0. sim/edpp_coeffs.go:91-93.
func (c Coeffs) tIterPrefill(sPf int64) float64 {
	return c.AlphaP + c.CPf*float64(sPf)
}

// muDecode and muPrefill are the live drain rates mu = 1 - alpha/T_iter, clamped.
// sim/edpp_coeffs.go:97-105.
func (c Coeffs) muDecode(bDec int, kv, sPf int64) float64 {
	return clampMu(1.0 - c.AlphaD/c.tIterDecode(bDec, kv, sPf))
}

func (c Coeffs) muPrefill(sPf int64) float64 {
	return clampMu(1.0 - c.AlphaP/c.tIterPrefill(sPf))
}

// muDNom is the fixed nominal decode drain rate at the SLO-critical batch, where
// T_iter == tau_itl. sim/edpp_coeffs.go:107-110. Caller guarantees
// tauITLUs > AlphaD.
func (c Coeffs) muDNom(tauITLUs float64) float64 {
	return clampMu(1.0 - c.AlphaD/tauITLUs)
}

// muPNom is the fixed nominal prefill drain rate at the nominal operating chunk.
// sim/edpp_coeffs.go:113-116.
func (c Coeffs) muPNom(sPfNom int) float64 {
	return clampMu(1.0 - c.AlphaP/(c.AlphaP+c.CPf*float64(sPfNom)))
}

// deltaBarDecode is the marginal decode work per step at context length ctxLen.
// sim/edpp_coeffs.go:119-121.
//
// Read by the comparator arm's ITL term, not by this arm's composite objective.
// It is stated here so both arms share one law: config.md section 9.3 requires
// the comparator's routing view to be a verbatim copy of this one, package clause
// aside, because a re-derived-but-slightly-different estimator would silently
// destroy the attribution argument while every test still passed.
func (c Coeffs) deltaBarDecode(ctxLen float64) float64 {
	return c.C0 + c.C1*ctxLen
}

// validate mirrors sim/edpp_coeffs.go:127-146.
//
// The alpha ~= alpha_p bound is not cosmetic: a divergence over 10% means the
// coefficients were fit on mismatched hardware or regimes. Both files in
// config.md section 4 are within 0.03%.
func (c Coeffs) validate() error {
	switch {
	case c.AlphaD <= 0:
		return fmt.Errorf("alphaD must be > 0, got %g", c.AlphaD)
	case c.AlphaP <= 0:
		return fmt.Errorf("alphaP must be > 0, got %g", c.AlphaP)
	case c.C0 < 0:
		return fmt.Errorf("c0 must be >= 0, got %g", c.C0)
	case c.C1 < 0:
		return fmt.Errorf("c1 must be >= 0, got %g", c.C1)
	case c.CPf <= 0:
		return fmt.Errorf("cPf must be > 0, got %g", c.CPf)
	case c.CAttn < 0:
		return fmt.Errorf("cAttn must be >= 0, got %g", c.CAttn)
	}
	if rel := math.Abs(c.AlphaD-c.AlphaP) / c.AlphaD; rel > 0.10 {
		return fmt.Errorf("alphaD (%g) and alphaP (%g) diverge by %.1f%%, more than the 10%% bound: "+
			"the coefficients were probably fit on mismatched hardware or regimes", c.AlphaD, c.AlphaP, rel*100)
	}
	return nil
}

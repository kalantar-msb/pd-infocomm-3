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
)

// h100 and a100 are the campaign fleet's coefficients, copied verbatim from
// inputs/coeffs-llama70b-h100-tp4.json and inputs/coeffs-llama70b-a100real-tp4.json
// (config.md section 4).
//
// They are the real values rather than round test numbers on purpose: the tests below
// assert on the HETEROGENEITY these two carry, which is the condition the mechanism
// exploits, and round numbers would not exercise it.
var (
	h100 = Coeffs{
		AlphaD: 16613.537554540144,
		AlphaP: 16617.85321583337,
		C0:     5.347316038602452,
		C1:     0.04761401141756073,
		CPf:    6.144687138665833,
		CAttn:  0.00010075247918809842,
	}
	a100 = Coeffs{
		AlphaD: 25563.819286862163,
		AlphaP: 25568.34953836831,
		C0:     5.945331876073271,
		C1:     0.07822809856114352,
		CPf:    9.794219053944662,
		CAttn:  0.00015977670754642687,
	}
)

const eps = 1e-9

func closeTo(t *testing.T, got, want float64, what string) {
	t.Helper()
	if math.Abs(got-want) > math.Abs(want)*1e-9+eps {
		t.Errorf("%s: got %.12g, want %.12g", what, got, want)
	}
}

func TestCoeffsValidateAcceptsBothFleetGPUTypes(t *testing.T) {
	for name, c := range map[string]Coeffs{"H100": h100, "A100": a100} {
		if err := c.validate(); err != nil {
			t.Errorf("%s coefficients from config.md section 4 must validate, got %v", name, err)
		}
	}
}

// TestCoeffsValidateRejectsDivergentAlpha pins the 10% bound. A divergence that large
// means the coefficients were fit on mismatched hardware or regimes, so accepting it
// would price an endpoint under physics that never described it.
func TestCoeffsValidateRejectsDivergentAlpha(t *testing.T) {
	c := h100
	c.AlphaP = c.AlphaD * 1.11
	if err := c.validate(); err == nil {
		t.Fatal("expected rejection of an 11% alpha/alpha_p divergence")
	}

	// And confirm the real files sit far inside the bound -- config.md claims 0.03%.
	for name, real := range map[string]Coeffs{"H100": h100, "A100": a100} {
		rel := math.Abs(real.AlphaD-real.AlphaP) / real.AlphaD
		if rel > 0.0003 {
			t.Errorf("%s alpha divergence %.6f%% exceeds the 0.03%% config.md states", name, rel*100)
		}
	}
}

func TestCoeffsValidateRejectsNonPositiveAndNegative(t *testing.T) {
	for _, tc := range []struct {
		name string
		mut  func(*Coeffs)
	}{
		{"alphaD zero", func(c *Coeffs) { c.AlphaD = 0 }},
		{"alphaP zero", func(c *Coeffs) { c.AlphaP = 0 }},
		{"cPf zero", func(c *Coeffs) { c.CPf = 0 }},
		{"c0 negative", func(c *Coeffs) { c.C0 = -1 }},
		{"c1 negative", func(c *Coeffs) { c.C1 = -1 }},
		{"cAttn negative", func(c *Coeffs) { c.CAttn = -1 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := h100
			tc.mut(&c)
			if err := c.validate(); err == nil {
				t.Errorf("expected rejection for %s", tc.name)
			}
		})
	}
}

// TestWpMatchesClosedForm checks the trajectory-sum form against the two closed forms
// the specification states: the fully-uncached case and the general (ar - ap/2) shape.
func TestWpMatchesClosedForm(t *testing.T) {
	// At ap == ar (no cache) Wp = CPf*ar + 0.5*CAttn*ar^2.
	ar := 4000
	closeTo(t, h100.Wp(ar, ar),
		h100.CPf*float64(ar)+0.5*h100.CAttn*float64(ar)*float64(ar),
		"Wp fully uncached")

	// With a cached prefix the exposed work is smaller, and the attention term is
	// charged against the causal prefix rather than from zero.
	ap := 1000
	closeTo(t, h100.Wp(ap, ar),
		h100.CPf*float64(ap)+h100.CAttn*float64(ap)*(float64(ar)-float64(ap)/2),
		"Wp with cached prefix")

	if h100.Wp(ap, ar) >= h100.Wp(ar, ar) {
		t.Error("a cached prefix must reduce exposed prefill work")
	}
}

// TestWdMatchesDiscreteSum verifies Wd against the explicit per-step sum it claims to
// equal, which is the property that makes it the decode demand rather than an
// approximation of it.
func TestWdMatchesDiscreteSum(t *testing.T) {
	ar, o := 512, 40
	var want float64
	for k := 0; k < o; k++ {
		want += h100.C0 + h100.C1*float64(ar+k)
	}
	closeTo(t, h100.Wd(ar, float64(o)), want, "Wd vs discrete sum")

	if got := h100.Wd(ar, 0); got != 0 {
		t.Errorf("Wd with no output must be 0, got %g", got)
	}
	if got := h100.Wd(ar, -5); got != 0 {
		t.Errorf("Wd with negative output must be 0, got %g", got)
	}
}

// TestIterationIntercepptCarriesHeterogeneity is the load-bearing one for this
// experiment: the A100/H100 gap is present on EVERY iteration regardless of KV state,
// which is why a defaulted GPU label is wrong on every decision rather than only under
// load. config.md section 2 states the factor as 1.539.
func TestIterationInterceptCarriesHeterogeneity(t *testing.T) {
	ratio := a100.AlphaD / h100.AlphaD
	if math.Abs(ratio-1.539) > 0.001 {
		t.Errorf("A100/H100 decode intercept ratio: got %.4f, want ~1.539", ratio)
	}

	// At an idle endpoint -- no batch, no KV, no resident prefill -- the iteration times
	// still differ by the full intercept ratio.
	idleH := h100.tIterDecode(0, 0, 0)
	idleA := a100.tIterDecode(0, 0, 0)
	closeTo(t, idleH, h100.AlphaD, "idle H100 iteration is the intercept")
	closeTo(t, idleA, a100.AlphaD, "idle A100 iteration is the intercept")
	if idleA <= idleH {
		t.Error("A100 must be slower than H100 even at idle")
	}
}

func TestTIterDecodeIncludesEveryTerm(t *testing.T) {
	bDec, kv, sPf := 8, int64(4096), int64(512)
	closeTo(t, h100.tIterDecode(bDec, kv, sPf),
		h100.AlphaD+h100.C0*8+h100.C1*4096+h100.CPf*512,
		"tIterDecode")
}

// TestTIterPrefillHasNoDecodeTerms pins the modelling claim that a dedicated prefill
// server runs no decode work, so B_dec = 0 and KV = 0 and only alpha_p and CPf appear.
func TestTIterPrefillHasNoDecodeTerms(t *testing.T) {
	sPf := int64(2048)
	closeTo(t, h100.tIterPrefill(sPf), h100.AlphaP+h100.CPf*2048, "tIterPrefill")
}

// TestClampMuFloorsAtOneEMinus3 pins the constant. The specification is explicit that
// the value is 1e-3 and not 1e-6, and that the difference becomes live the moment the
// capacity account is enabled or the estimator is switched.
func TestClampMuFloorsAtOneEMinus3(t *testing.T) {
	if got := clampMu(0); got != 1e-3 {
		t.Errorf("clampMu(0) = %g, want 1e-3", got)
	}
	if got := clampMu(-5); got != 1e-3 {
		t.Errorf("clampMu(-5) = %g, want 1e-3", got)
	}
	if got := clampMu(2); got != 1.0 {
		t.Errorf("clampMu(2) = %g, want 1.0", got)
	}
	if got := clampMu(0.5); got != 0.5 {
		t.Errorf("clampMu(0.5) = %g, want 0.5", got)
	}
}

func TestMuDecodeAndPrefillAreClamped(t *testing.T) {
	// mu = 1 - alpha/T_iter, and at idle T_iter == alpha so mu == 0 and clamps to the
	// floor rather than collapsing a downstream denominator.
	if got := h100.muDecode(0, 0, 0); got != minMu {
		t.Errorf("idle muDecode = %g, want the %g floor", got, minMu)
	}
	if got := h100.muPrefill(0); got != minMu {
		t.Errorf("idle muPrefill = %g, want the %g floor", got, minMu)
	}
	// Under load mu rises above the floor.
	if got := h100.muDecode(64, 100000, 2048); got <= minMu {
		t.Errorf("loaded muDecode = %g, expected above the floor", got)
	}
}

func TestDeltaBarDecode(t *testing.T) {
	// Shared with the comparator arm, which is why it is ported here at all: config.md
	// section 9.3 requires both arms to read one law.
	closeTo(t, h100.deltaBarDecode(2048), h100.C0+h100.C1*2048, "deltaBarDecode")
}

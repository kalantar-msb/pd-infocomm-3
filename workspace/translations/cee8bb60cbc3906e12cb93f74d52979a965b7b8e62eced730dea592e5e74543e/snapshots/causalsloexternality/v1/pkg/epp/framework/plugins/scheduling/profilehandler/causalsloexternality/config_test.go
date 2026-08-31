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
	"strings"
	"testing"
)

// focalConfig is the focal arm exactly as the generated treatment overlay declares it:
// causal_externality_no_capacity_v8, config.md sections 9.1 and 9.2.
//
// Tests mutate a copy to check that each invariant is enforced, so this doubles as an
// executable statement of the shipped configuration.
func focalConfig() Config {
	return Config{
		V: 8,
		Ablation: Ablation{
			NoCapacity:        true,
			NoExternality:     false,
			NoOwnGood:         false,
			OccupancyCapacity: false,
			Decomposed:        false,
		},
		ActiveWorkload: workloadInteractive,
		WorkloadTargets: map[string]SLOTargets{
			workloadInteractive:  {TauTTFTUs: 1_000_000, TauITLUs: 50_000, TauE2EUs: 16_000_000},
			workloadReasoning:    {TauTTFTUs: 2_000_000, TauITLUs: 100_000, TauE2EUs: 802_000_000},
			workloadDeepResearch: {TauTTFTUs: 10_000_000, TauITLUs: 100_000, TauE2EUs: 40_000_000},
		},
		Engine: EngineAgreement{ChunkTokens: 2048, BlockSize: 16, MaxBatchSize: 256},
		Signals: Signals{
			GPUTypeLabelKey:             "pd-infocomm.io/gpu-type",
			PrefixMatchInfoProducerName: "approx-prefix-cache-producer",
			TokenizedPromptProducerName: "token-producer",
		},
		AdmissionEstimator: estimatorRollforward,
		Transfer: Transfer{
			SizeAware:             true,
			XferBaseUs:            50.0,
			XferBandwidthGBps:     25.0,
			KVBytesPerTokenPerGPU: 81920,
			FlatCXferUs:           5000,
		},
		CoeffsByGPUType: map[string]Coeffs{
			"H100_SXM_80GB": h100,
			"A100_SXM_80GB": a100,
		},
		ShadowTable: ShadowTable{
			EntryTTLSeconds:          900,
			SweepIntervalSeconds:     30,
			ResidentPrefillTokensCap: 2048,
		},
		Capacity:                Capacity{TauRefUs: 1_000_000, NomPrefillTokens: 512, ReferenceBatch: 256},
		OutputTokenProcessingUs: 0,
	}
}

func TestFocalConfigValidates(t *testing.T) {
	cfg := focalConfig()
	if err := cfg.validate(); err != nil {
		t.Fatalf("the shipped focal configuration must validate, got %v", err)
	}
}

// TestActiveTargetsSelectsInteractive pins the committed default and the exact
// microsecond values from config.md section 6.
func TestActiveTargetsSelectsInteractive(t *testing.T) {
	cfg := focalConfig()
	got := cfg.activeTargets()
	if got.TauTTFTUs != 1_000_000 || got.TauITLUs != 50_000 || got.TauE2EUs != 16_000_000 {
		t.Errorf("interactive triple: got %+v, want 1000ms/50ms/16000ms in us", got)
	}

	// Switching workloads selects a different triple with no other change, which is what
	// makes one assemble per workload the intended operation.
	cfg.ActiveWorkload = workloadReasoning
	if err := cfg.validate(); err != nil {
		t.Fatalf("reasoning must also validate: %v", err)
	}
	if cfg.activeTargets().TauE2EUs != 802_000_000 {
		t.Errorf("reasoning tau_e2e: got %d, want 802000000", cfg.activeTargets().TauE2EUs)
	}
}

// TestReasoningSaturatesE2ESigmoid is the reason interactive is the committed default:
// reasoning's 802 s deadline saturates the E2E sigmoid, so the externality stops
// separating candidates. config.md section 6 states it as ~50x less discrimination.
func TestReasoningSaturatesE2ESigmoid(t *testing.T) {
	interactive := slo{tauTTFTUs: 1_000_000, tauE2EUs: 16_000_000}
	reasoning := slo{tauTTFTUs: 2_000_000, tauE2EUs: 802_000_000}

	// The same absolute delay moves the interactive value far more than the reasoning
	// value, because a resident's charge scales roughly as 1/tau_e2e.
	base, delayed := 4_000_000.0, 8_000_000.0
	interactiveGap := sloCompositeValue(interactive, 300_000, base) - sloCompositeValue(interactive, 300_000, delayed)
	reasoningGap := sloCompositeValue(reasoning, 300_000, base) - sloCompositeValue(reasoning, 300_000, delayed)

	if interactiveGap <= reasoningGap {
		t.Fatalf("interactive must discriminate more than reasoning: %g vs %g", interactiveGap, reasoningGap)
	}
	if ratio := interactiveGap / reasoningGap; ratio < 10 {
		t.Errorf("expected roughly an order of magnitude more discrimination, got %.1fx", ratio)
	}
}

// TestValidateRejectsUnresolvableWorkload covers the FLATTENING failure: an unresolvable
// selection would leave a zero triple, which does not loosen the policy but flattens it.
func TestValidateRejectsUnresolvableWorkload(t *testing.T) {
	cfg := focalConfig()
	cfg.ActiveWorkload = "intercative" // a plausible typo
	err := cfg.validate()
	if err == nil {
		t.Fatal("expected rejection of an unresolvable activeWorkload")
	}
	// The message must name the available keys, so the operator can see the typo.
	for _, want := range []string{"interactive", "reasoning", "deepResearch"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error should list available workloads, missing %q: %v", want, err)
		}
	}

	cfg2 := focalConfig()
	cfg2.ActiveWorkload = ""
	if cfg2.validate() == nil {
		t.Error("expected rejection of an empty activeWorkload")
	}

	cfg3 := focalConfig()
	cfg3.WorkloadTargets = nil
	if cfg3.validate() == nil {
		t.Error("expected rejection of absent workloadTargets")
	}
}

// TestValidateRejectsNonPositiveTauInSelectedTriple is the correctness requirement, not a
// tuning choice: a zero tau_e2e makes the entire decode-side externality identically zero
// while the arm keeps running and keeps reporting goodput.
func TestValidateRejectsNonPositiveTauInSelectedTriple(t *testing.T) {
	cfg := focalConfig()
	cfg.WorkloadTargets[workloadInteractive] = SLOTargets{TauTTFTUs: 1_000_000, TauE2EUs: 0}
	err := cfg.validate()
	if err == nil {
		t.Fatal("expected rejection of tau_e2e = 0 in the selected triple")
	}
	if !strings.Contains(err.Error(), "tauE2eUs") {
		t.Errorf("error should name the field: %v", err)
	}

	cfg2 := focalConfig()
	cfg2.WorkloadTargets[workloadInteractive] = SLOTargets{TauTTFTUs: 0, TauE2EUs: 16_000_000}
	if cfg2.validate() == nil {
		t.Error("expected rejection of tau_ttft = 0 in the selected triple")
	}
}

// TestValidateIgnoresUnselectedTriples confirms only the SELECTED triple is gated, so a
// workload the operator is not running cannot block startup.
func TestValidateIgnoresUnselectedTriples(t *testing.T) {
	cfg := focalConfig()
	cfg.WorkloadTargets[workloadDeepResearch] = SLOTargets{} // not selected
	if err := cfg.validate(); err != nil {
		t.Errorf("an unselected triple must not gate startup, got %v", err)
	}
}

// TestValidateRejectsNonRollforwardEstimator pins D1's substitution as the only
// selectable estimator, and the error must say why each alternative is unreachable.
func TestValidateRejectsNonRollforwardEstimator(t *testing.T) {
	for _, name := range []string{"waiting", "little", "fluid", "scheduler-rollout", ""} {
		cfg := focalConfig()
		cfg.AdmissionEstimator = name
		if cfg.validate() == nil {
			t.Errorf("estimator %q must be rejected at this pin", name)
		}
	}
	cfg := focalConfig()
	cfg.AdmissionEstimator = estimatorRollforward
	if err := cfg.validate(); err != nil {
		t.Errorf("rollforward must be accepted, got %v", err)
	}
}

func TestValidateRejectsEngineDisagreement(t *testing.T) {
	for _, tc := range []struct {
		name string
		mut  func(*Config)
	}{
		{"chunkTokens zero", func(c *Config) { c.Engine.ChunkTokens = 0 }},
		{"blockSize zero", func(c *Config) { c.Engine.BlockSize = 0 }},
		{"maxBatchSize zero", func(c *Config) { c.Engine.MaxBatchSize = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := focalConfig()
			tc.mut(&cfg)
			if cfg.validate() == nil {
				t.Errorf("expected rejection for %s", tc.name)
			}
		})
	}
}

// TestValidateRejectsAbsentGPUTypeLabelKey guards signal 8: a defaulted GPU type is wrong
// on every decision, because heterogeneity rides the per-iteration intercept.
func TestValidateRejectsAbsentGPUTypeLabelKey(t *testing.T) {
	cfg := focalConfig()
	cfg.Signals.GPUTypeLabelKey = ""
	if cfg.validate() == nil {
		t.Error("expected rejection of an absent gpuTypeLabelKey")
	}
}

// TestValidateRejectsEmptyCoeffTableAndBadEntries pins that there is deliberately NO
// fallback theta.
func TestValidateRejectsEmptyCoeffTableAndBadEntries(t *testing.T) {
	cfg := focalConfig()
	cfg.CoeffsByGPUType = map[string]Coeffs{}
	if cfg.validate() == nil {
		t.Error("expected rejection of an empty coeffsByGpuType")
	}

	cfg2 := focalConfig()
	bad := h100
	bad.AlphaP = bad.AlphaD * 2 // divergent
	cfg2.CoeffsByGPUType["BROKEN"] = bad
	err := cfg2.validate()
	if err == nil {
		t.Fatal("expected rejection of a divergent coefficient entry")
	}
	if !strings.Contains(err.Error(), "BROKEN") {
		t.Errorf("error should name the offending GPU type: %v", err)
	}
}

// TestValidateRejectsSizeAwareTransferWithoutOperands covers D5's severe case: leaving
// kvBytesPerTokenPerGpu at 0 makes transferBytes 0, so every request is charged a flat
// base -- roughly 270x under-priced for a 4k prompt.
func TestValidateRejectsSizeAwareTransferWithoutOperands(t *testing.T) {
	cfg := focalConfig()
	cfg.Transfer.KVBytesPerTokenPerGPU = 0
	err := cfg.validate()
	if err == nil {
		t.Fatal("expected rejection of size-aware transfer with no bytes-per-token")
	}
	if !strings.Contains(err.Error(), "270x") {
		t.Errorf("error should state the magnitude of the mis-pricing: %v", err)
	}

	cfg2 := focalConfig()
	cfg2.Transfer.XferBandwidthGBps = 0
	if cfg2.validate() == nil {
		t.Error("expected rejection of size-aware transfer with no bandwidth")
	}

	// With sizeAware off, the flat field is what matters and the size-aware operands are
	// not required -- the flat path must stay representable.
	cfg3 := focalConfig()
	cfg3.Transfer = Transfer{SizeAware: false, FlatCXferUs: 5000}
	if err := cfg3.validate(); err != nil {
		t.Errorf("the flat transfer path must validate on its own, got %v", err)
	}
}

// TestValidateRejectsDisabledShadowSweep guards the requirement that the sweep exists:
// without it, requests that terminate without a final chunk are charged as residents
// forever.
func TestValidateRejectsDisabledShadowSweep(t *testing.T) {
	for _, tc := range []struct {
		name string
		mut  func(*Config)
	}{
		{"ttl zero", func(c *Config) { c.ShadowTable.EntryTTLSeconds = 0 }},
		{"sweep interval zero", func(c *Config) { c.ShadowTable.SweepIntervalSeconds = 0 }},
		{"prefill cap zero", func(c *Config) { c.ShadowTable.ResidentPrefillTokensCap = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := focalConfig()
			tc.mut(&cfg)
			if cfg.validate() == nil {
				t.Errorf("expected rejection for %s", tc.name)
			}
		})
	}
}

// TestCapacityOperandsGatedOnlyWhenLive is the dead-state property: the focal arm disables
// the capacity term, so its operands must not gate startup -- but they must be checked the
// moment the switch is flipped.
func TestCapacityOperandsGatedOnlyWhenLive(t *testing.T) {
	// Dead: a zero tauRefUs is tolerated.
	cfg := focalConfig()
	cfg.Capacity.TauRefUs = 0
	if err := cfg.validate(); err != nil {
		t.Errorf("capacity operands must not gate startup while noCapacity is true, got %v", err)
	}

	// Live: the same zero is now rejected.
	cfg.Ablation.NoCapacity = false
	if cfg.validate() == nil {
		t.Error("expected rejection of tauRefUs = 0 once the capacity term is enabled")
	}

	// And referenceBatch is gated only in occupancy mode -- doubly dead otherwise.
	cfg2 := focalConfig()
	cfg2.Ablation.NoCapacity = false
	cfg2.Capacity.ReferenceBatch = 0
	if err := cfg2.validate(); err != nil {
		t.Errorf("referenceBatch is only needed in occupancy mode, got %v", err)
	}
	cfg2.Ablation.OccupancyCapacity = true
	if cfg2.validate() == nil {
		t.Error("expected rejection of referenceBatch = 0 in occupancy mode")
	}
}

// TestCapacityRequiresTauITLAboveIntercept pins the muDNom precondition, which only
// matters once the capacity account is live.
func TestCapacityRequiresTauITLAboveIntercept(t *testing.T) {
	cfg := focalConfig()
	cfg.Ablation.NoCapacity = false
	// interactive tau_itl is 50 ms = 50000 us, which is BELOW the A100 intercept of
	// 25563 us? No -- it is above. Force it below to exercise the check.
	cfg.WorkloadTargets[workloadInteractive] = SLOTargets{
		TauTTFTUs: 1_000_000, TauITLUs: 1000, TauE2EUs: 16_000_000,
	}
	if cfg.validate() == nil {
		t.Error("expected rejection of a tau_itl below the per-iteration intercept")
	}

	// The shipped interactive tau_itl (50 ms) clears both fleet intercepts, so enabling
	// capacity on the shipped config is not blocked by this check.
	ok := focalConfig()
	ok.Ablation.NoCapacity = false
	if err := ok.validate(); err != nil {
		t.Errorf("the shipped tau_itl must clear both intercepts, got %v", err)
	}
}

func TestValidateRejectsNonPositiveV(t *testing.T) {
	cfg := focalConfig()
	cfg.V = 0
	if cfg.validate() == nil {
		t.Error("expected rejection of V = 0")
	}
}

func TestValidateRejectsNegativeOutputTokenProcessing(t *testing.T) {
	cfg := focalConfig()
	cfg.OutputTokenProcessingUs = -1
	if cfg.validate() == nil {
		t.Error("expected rejection of a negative outputTokenProcessingUs")
	}
}

// TestSortedKeysIsDeterministic guards the property that validation order and error
// messages do not vary between runs.
func TestSortedKeysIsDeterministic(t *testing.T) {
	m := map[string]int{"z": 1, "a": 2, "m": 3}
	first := sortedKeys(m)
	for i := 0; i < 20; i++ {
		got := sortedKeys(m)
		for j := range got {
			if got[j] != first[j] {
				t.Fatalf("non-deterministic key order: %v then %v", first, got)
			}
		}
	}
	if first[0] != "a" || first[2] != "z" {
		t.Errorf("keys must be sorted ascending, got %v", first)
	}
}

// TestMaxCoeffAlphaD picks the largest intercept across configured GPU types, which is the
// A100's in this fleet.
func TestMaxCoeffAlphaD(t *testing.T) {
	closeTo(t, maxCoeffAlphaD(focalConfig().CoeffsByGPUType), a100.AlphaD, "max alphaD")
}

// TestValidateRequiresBothProducerNames pins the fix for a SILENT failure.
//
// DataKey.WithNonEmptyProducerName("") returns the key unchanged, leaving producerName at
// the plugin type string. A WRONG explicit name errors at startup; an EMPTY one silently
// resolves to the type string, and if the operator declared their producer under any other
// instance name the required key looks unproduced -- so CreateMissingDataProducers
// auto-instantiates a SECOND, nil-parameter producer which the arm then reads. Because
// PluginSpec.Name defaults to Type, an empty binding accidentally works in the shipped
// overlay, which is exactly what would let this pass a smoke test and break on a rename.
func TestValidateRequiresBothProducerNames(t *testing.T) {
	cfg := focalConfig()
	cfg.Signals.PrefixMatchInfoProducerName = ""
	err := cfg.validate()
	if err == nil {
		t.Fatal("expected rejection of an empty prefixMatchInfoProducerName")
	}
	if !strings.Contains(err.Error(), "second") {
		t.Errorf("error should explain the silent-duplication hazard: %v", err)
	}

	cfg2 := focalConfig()
	cfg2.Signals.TokenizedPromptProducerName = ""
	if cfg2.validate() == nil {
		t.Error("expected rejection of an empty tokenizedPromptProducerName")
	}
}

// TestValidateRequiresPrefillCapToEqualChunkTokens pins config.md signal 14: S_pf is what is
// being prefilled in ONE engine step, so its cap is the engine's max_num_batched_tokens. The
// two settings are one fact expressed twice and must agree.
func TestValidateRequiresPrefillCapToEqualChunkTokens(t *testing.T) {
	cfg := focalConfig()
	cfg.ShadowTable.ResidentPrefillTokensCap = 4096 // larger than chunkTokens 2048
	err := cfg.validate()
	if err == nil {
		t.Fatal("expected rejection of a cap that disagrees with engine.chunkTokens")
	}
	if !strings.Contains(err.Error(), "chunkTokens") {
		t.Errorf("error should name both settings: %v", err)
	}

	// Changing the engine budget requires changing the cap with it.
	cfg2 := focalConfig()
	cfg2.Engine.ChunkTokens = 4096
	cfg2.ShadowTable.ResidentPrefillTokensCap = 4096
	if err := cfg2.validate(); err != nil {
		t.Errorf("agreeing values must validate, got %v", err)
	}
}

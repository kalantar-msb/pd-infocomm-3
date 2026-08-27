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
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// armConfig is the configuration this arm ships, matching the generated treatment
// overlay. Every value is the focal arm's value for the same key (config.md section 9.1);
// contract_test.go asserts the shape agreement mechanically.
func armConfig() Config {
	return Config{
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
		OutputTokenProcessingUs: 0,
	}
}

func TestArmConfigValidates(t *testing.T) {
	cfg := armConfig()
	if err := cfg.validate(); err != nil {
		t.Fatalf("the shipped comparator configuration must validate, got %v", err)
	}
}

// TestValidateRejectsNonRollforwardEstimator pins degradation D1's disclosure: the
// substitution is acceptable, a SILENT switch is not.
func TestValidateRejectsNonRollforwardEstimator(t *testing.T) {
	for _, estimator := range []string{"", "waiting", "little", "scheduler-rollout", "fluid"} {
		cfg := armConfig()
		cfg.AdmissionEstimator = estimator
		err := cfg.validate()
		if err == nil {
			t.Errorf("estimator %q must be rejected: only rollforward is reachable at this pin", estimator)
			continue
		}
		if !strings.Contains(err.Error(), "admissionEstimator") {
			t.Errorf("estimator %q: error should name the field, got %v", estimator, err)
		}
	}
}

func TestValidateRejectsEngineDisagreement(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Config)
		field  string
	}{
		{"chunkTokens zero", func(c *Config) { c.Engine.ChunkTokens = 0; c.ShadowTable.ResidentPrefillTokensCap = 0 }, "engine.chunkTokens"},
		{"blockSize zero", func(c *Config) { c.Engine.BlockSize = 0 }, "engine.blockSize"},
		{"maxBatchSize zero", func(c *Config) { c.Engine.MaxBatchSize = 0 }, "engine.maxBatchSize"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := armConfig()
			tc.mutate(&cfg)
			err := cfg.validate()
			if err == nil {
				t.Fatalf("%s must be rejected", tc.field)
			}
			if !strings.Contains(err.Error(), tc.field) {
				t.Errorf("error should name %s, got %v", tc.field, err)
			}
		})
	}
}

// TestValidateRejectsAbsentGPUTypeLabelKey pins that theta is never defaulted. A wrong or
// defaulted GPU label is wrong on EVERY decision, because heterogeneity rides the
// per-iteration intercept.
func TestValidateRejectsAbsentGPUTypeLabelKey(t *testing.T) {
	cfg := armConfig()
	cfg.Signals.GPUTypeLabelKey = ""
	if err := cfg.validate(); err == nil {
		t.Fatal("an absent gpuTypeLabelKey must be rejected, never defaulted")
	}
}

func TestValidateRejectsEmptyCoeffTableAndBadEntries(t *testing.T) {
	cfg := armConfig()
	cfg.CoeffsByGPUType = nil
	if err := cfg.validate(); err == nil {
		t.Fatal("an empty coeffsByGpuType must be rejected: there is deliberately no fallback entry")
	}

	cfg = armConfig()
	broken := h100
	broken.AlphaP = broken.AlphaD * 2 // 100% divergence, far past the 10% bound
	cfg.CoeffsByGPUType = map[string]Coeffs{"H100_SXM_80GB": broken}
	err := cfg.validate()
	if err == nil {
		t.Fatal("an alphaD/alphaP divergence past 10% must be rejected")
	}
	if !strings.Contains(err.Error(), "H100_SXM_80GB") {
		t.Errorf("error should name the offending GPU type, got %v", err)
	}
}

// TestValidateRejectsSizeAwareTransferWithoutOperands guards degradation D5, which matters
// MORE in this arm than in the focal one: there is no externality term to partially offset
// a mis-priced transfer.
func TestValidateRejectsSizeAwareTransferWithoutOperands(t *testing.T) {
	cfg := armConfig()
	cfg.Transfer.XferBandwidthGBps = 0
	if err := cfg.validate(); err == nil {
		t.Fatal("sizeAware transfer with zero bandwidth must be rejected")
	}

	cfg = armConfig()
	cfg.Transfer.KVBytesPerTokenPerGPU = 0
	err := cfg.validate()
	if err == nil {
		t.Fatal("sizeAware transfer with zero kvBytesPerTokenPerGpu must be rejected: " +
			"transferBytes becomes 0 and every request is charged a flat base")
	}
	if !strings.Contains(err.Error(), "kvBytesPerTokenPerGpu") {
		t.Errorf("error should name the field, got %v", err)
	}

	// The flat path stays representable when sizeAware is off.
	cfg = armConfig()
	cfg.Transfer.SizeAware = false
	cfg.Transfer.XferBandwidthGBps = 0
	cfg.Transfer.KVBytesPerTokenPerGPU = 0
	if err := cfg.validate(); err != nil {
		t.Errorf("the flat transfer path must remain valid without size-aware operands, got %v", err)
	}
}

// TestValidateRejectsDisabledShadowSweep pins that the sweep cannot be turned off. Without
// it, requests terminating without a final chunk are charged as residents forever, which in
// this arm permanently inflates S_pf and both remaining-steps estimates.
func TestValidateRejectsDisabledShadowSweep(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Config)
	}{
		{"zero TTL", func(c *Config) { c.ShadowTable.EntryTTLSeconds = 0 }},
		{"zero sweep interval", func(c *Config) { c.ShadowTable.SweepIntervalSeconds = 0 }},
		{"zero prefill cap", func(c *Config) { c.ShadowTable.ResidentPrefillTokensCap = 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := armConfig()
			tc.mutate(&cfg)
			if err := cfg.validate(); err == nil {
				t.Fatalf("%s must be rejected", tc.name)
			}
		})
	}
}

// TestValidateRequiresPrefillCapToEqualChunkTokens pins config.md signal 14: S_pf is what
// is prefilled in ONE engine step, so its cap IS max_num_batched_tokens. The two settings
// are one fact expressed twice.
func TestValidateRequiresPrefillCapToEqualChunkTokens(t *testing.T) {
	cfg := armConfig()
	cfg.ShadowTable.ResidentPrefillTokensCap = 4096
	err := cfg.validate()
	if err == nil {
		t.Fatal("a prefill cap larger than engine.chunkTokens must be rejected")
	}
	if !strings.Contains(err.Error(), "chunkTokens") {
		t.Errorf("error should relate the two settings, got %v", err)
	}
}

// TestValidateRequiresBothProducerNames is the cross-arm binding guard. An EMPTY name is
// worse than a wrong one: a wrong name errors at startup, an empty one silently resolves to
// the type string and can auto-create a second producer -- and because each arm resolves
// its binding independently, the two arms can end up reading different signals.
func TestValidateRequiresBothProducerNames(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*Config)
		field  string
	}{
		{"prefix", func(c *Config) { c.Signals.PrefixMatchInfoProducerName = "" }, "prefixMatchInfoProducerName"},
		{"token", func(c *Config) { c.Signals.TokenizedPromptProducerName = "" }, "tokenizedPromptProducerName"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := armConfig()
			tc.mutate(&cfg)
			err := cfg.validate()
			if err == nil {
				t.Fatalf("an empty %s must be rejected", tc.field)
			}
			if !strings.Contains(err.Error(), tc.field) {
				t.Errorf("error should name %s, got %v", tc.field, err)
			}
		})
	}
}

func TestValidateRejectsNegativeOutputTokenProcessing(t *testing.T) {
	cfg := armConfig()
	cfg.OutputTokenProcessingUs = -1
	if err := cfg.validate(); err == nil {
		t.Fatal("a negative outputTokenProcessingUs must be rejected")
	}
}

func TestSortedKeysIsDeterministic(t *testing.T) {
	m := map[string]Coeffs{"c": {}, "a": {}, "b": {}}
	for i := 0; i < 50; i++ {
		got := sortedKeys(m)
		if len(got) != 3 || got[0] != "a" || got[1] != "b" || got[2] != "c" {
			t.Fatalf("sortedKeys = %v, want [a b c] on every iteration", got)
		}
	}
}

// TestFocalArmOnlyKeysAreRejected is the decisive test for the "strict subset" claim, and
// it tests the failure mode that matters operationally: pasting the FOCAL arm's parameters
// block into this arm's plugin config.
//
// Because the loader decodes with DisallowUnknownFields, each of these keys must be a
// STARTUP ERROR rather than a silently ignored one. If any were accepted, an operator could
// configure a tau triple or an ablation switch on this arm and reasonably believe it took
// effect -- and nothing would read it.
func TestFocalArmOnlyKeysAreRejected(t *testing.T) {
	for _, body := range []string{
		`{"v": 8}`,
		`{"ablation": {"noCapacity": true}}`,
		`{"ablation": {"decomposed": true}}`,
		`{"activeWorkload": "interactive"}`,
		`{"workloadTargets": {"interactive": {"tauTtftUs": 1000000}}}`,
		`{"capacity": {"tauRefUs": 1000000}}`,
		`{"capacity": {"nomPrefillTokens": 512}}`,
	} {
		decoder := json.NewDecoder(bytes.NewReader([]byte(body)))
		decoder.DisallowUnknownFields()
		var cfg Config
		if err := decoder.Decode(&cfg); err == nil {
			t.Errorf("%s decoded without error; this arm reads none of these keys, so a strict "+
				"decode must reject them rather than let an operator believe they took effect", body)
		}
	}
}

// TestShippedConfigDecodesFromItsOwnKeys is the positive half: the keys this arm DOES
// declare must round-trip through the same strict decoder the loader uses.
func TestShippedConfigDecodesFromItsOwnKeys(t *testing.T) {
	raw, err := json.Marshal(armConfig())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	var got Config
	if err := decoder.Decode(&got); err != nil {
		t.Fatalf("the shipped config must decode under DisallowUnknownFields, got %v", err)
	}
	if err := got.validate(); err != nil {
		t.Fatalf("the round-tripped config must validate, got %v", err)
	}
	if got.CoeffsByGPUType["A100_SXM_80GB"].AlphaD != a100.AlphaD {
		t.Errorf("A100 alphaD did not survive the round trip: got %g, want %g",
			got.CoeffsByGPUType["A100_SXM_80GB"].AlphaD, a100.AlphaD)
	}
}

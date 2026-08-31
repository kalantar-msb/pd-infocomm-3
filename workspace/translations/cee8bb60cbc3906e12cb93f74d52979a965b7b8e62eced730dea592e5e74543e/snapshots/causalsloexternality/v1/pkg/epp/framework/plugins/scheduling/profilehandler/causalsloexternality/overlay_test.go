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
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"sigs.k8s.io/yaml"
)

// The generated treatment overlay is the file that actually reaches the cluster, and it is
// the one artifact no other test covers: the Go tests build Config values in Go, so a
// mismatch between the shipped YAML and the struct tags would pass every one of them and
// then fail at pod startup.
//
// This test decodes the real overlay through the SAME strict path the config loader uses --
// json tags, DisallowUnknownFields -- and runs the real validation over the result.
//
// The overlay lives outside the Go module, so it is not hashed into the test cache key.
// Always run this package with `-count=1`, or a cached PASS can coexist with a known-bad
// overlay.

// overlayPaths locates EVERY generated treatment overlay for this arm, relative to this
// package.
//
// The paths are resolved rather than hardcoded absolute so the test is portable, and the
// helper SKIPS rather than fails when no overlay is present, because the plugin is
// buildable and testable independently of any one experiment bundle.
//
// ALL matches are returned, and each caller checks all of them. One bundle holds one
// translation directory per translation hash, so several overlays for this arm coexist as
// soon as a translation is redone against a new target pin. Picking one of them -- by glob
// order, by mtime, by anything -- would silently check an overlay that is not the one being
// shipped. Checking every candidate is strictly stronger and needs no tie-break: each
// overlay claims to configure this plugin, so each must decode and validate against it.
func overlayPaths(t *testing.T) []string {
	t.Helper()
	// pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality -> repo root
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", "..", "..", "..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
	// The experiment bundle contains llm-d-router as a submodule.
	bundleRoot := filepath.Dir(repoRoot)
	matches, err := filepath.Glob(filepath.Join(bundleRoot,
		"workspace", "translations", "*", "generated", "causalsloexternality",
		"causalsloexternality_config.yaml"))
	if err != nil {
		t.Fatalf("glob overlay: %v", err)
	}
	if len(matches) == 0 {
		t.Skip("no generated treatment overlay found; skipping overlay agreement check")
	}
	sort.Strings(matches)
	return matches
}

// forEachOverlay runs check against every candidate overlay, as its own subtest named by the
// translation hash, so a failure names the overlay that carries it.
func forEachOverlay(t *testing.T, check func(t *testing.T, plugins pluginsConfig)) {
	t.Helper()
	for _, path := range overlayPaths(t) {
		// .../translations/<hash>/generated/causalsloexternality/causalsloexternality_config.yaml
		hash := filepath.Base(filepath.Dir(filepath.Dir(filepath.Dir(path))))
		t.Run(hash, func(t *testing.T) {
			check(t, loadOverlayPluginsAt(t, path))
		})
	}
}

type overlayFile struct {
	Scenario []struct {
		Name   string `json:"name"`
		Router struct {
			EPP struct {
				PluginsCustomConfig map[string]string `json:"pluginsCustomConfig"`
			} `json:"epp"`
		} `json:"router"`
	} `json:"scenario"`
}

type pluginsConfig struct {
	APIVersion string `json:"apiVersion"`
	Kind       string `json:"kind"`
	Plugins    []struct {
		Type       string          `json:"type"`
		Name       string          `json:"name"`
		Parameters json.RawMessage `json:"parameters"`
	} `json:"plugins"`
	SchedulingProfiles []struct {
		Name    string `json:"name"`
		Plugins []struct {
			PluginRef string `json:"pluginRef"`
		} `json:"plugins"`
	} `json:"schedulingProfiles"`
}

func loadOverlayPluginsAt(t *testing.T, path string) pluginsConfig {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read overlay: %v", err)
	}
	var overlay overlayFile
	if err := yaml.Unmarshal(raw, &overlay); err != nil {
		t.Fatalf("parse overlay: %v", err)
	}
	if len(overlay.Scenario) != 1 {
		t.Fatalf("expected exactly one scenario entry, got %d", len(overlay.Scenario))
	}
	inner, ok := overlay.Scenario[0].Router.EPP.PluginsCustomConfig["custom-plugins.yaml"]
	if !ok {
		t.Fatal("overlay has no router.epp.pluginsCustomConfig[\"custom-plugins.yaml\"]")
	}
	var plugins pluginsConfig
	if err := yaml.Unmarshal([]byte(inner), &plugins); err != nil {
		t.Fatalf("parse inner plugin config: %v", err)
	}
	return plugins
}

// TestOverlayHandlerParametersDecodeAndValidate is the decisive agreement check: the shipped
// parameters must decode under DisallowUnknownFields and pass the real validation. A key the
// struct does not declare is a POD STARTUP FAILURE, not a warning.
func TestOverlayHandlerParametersDecodeAndValidate(t *testing.T) {
	forEachOverlay(t, func(t *testing.T, plugins pluginsConfig) {
		var params json.RawMessage
		for _, p := range plugins.Plugins {
			if p.Type == HandlerPluginType {
				params = p.Parameters
			}
		}
		if params == nil {
			t.Fatalf("the overlay declares no %s plugin", HandlerPluginType)
		}

		cfg := Config{}
		dec := strictDecoderForTest(params)
		if err := dec.Decode(&cfg); err != nil {
			t.Fatalf("the shipped overlay parameters do not decode against Config: %v\n"+
				"this would be a pod startup failure, since the config loader uses DisallowUnknownFields", err)
		}
		if err := cfg.validate(); err != nil {
			t.Fatalf("the shipped overlay parameters do not pass validation: %v", err)
		}

		// Spot-check the values the objective actually depends on, so a silent edit to the
		// overlay cannot change the arm without failing here.
		if cfg.V != 8 {
			t.Errorf("v = %g, want 8", cfg.V)
		}
		if !cfg.Ablation.NoCapacity {
			t.Error("the focal arm disables the capacity term")
		}
		if cfg.Ablation.NoExternality {
			t.Error("the focal arm keeps the externality -- it is the mechanism")
		}
		if cfg.Ablation.NoOwnGood {
			t.Error("the focal arm keeps the own-good term despite no measured benefit")
		}
		if cfg.Ablation.Decomposed {
			t.Error("the focal arm is the JOINT shape, not the decode-first control")
		}
		if cfg.ActiveWorkload != workloadInteractive {
			t.Errorf("activeWorkload = %q, want the committed default %q", cfg.ActiveWorkload, workloadInteractive)
		}
		if cfg.AdmissionEstimator != estimatorRollforward {
			t.Errorf("admissionEstimator = %q, want %q", cfg.AdmissionEstimator, estimatorRollforward)
		}
		if cfg.Engine.ChunkTokens != 2048 || cfg.Engine.BlockSize != 16 || cfg.Engine.MaxBatchSize != 256 {
			t.Errorf("engine agreement drifted: %+v", cfg.Engine)
		}
	})
}

// TestOverlayCarriesAllThreeWorkloadTriples pins config.md section 6: all three triples are
// carried and one is selected, so the thresholds are never retyped between arms.
func TestOverlayCarriesAllThreeWorkloadTriples(t *testing.T) {
	forEachOverlay(t, func(t *testing.T, plugins pluginsConfig) {
		var params json.RawMessage
		for _, p := range plugins.Plugins {
			if p.Type == HandlerPluginType {
				params = p.Parameters
			}
		}
		cfg := Config{}
		if err := strictDecoderForTest(params).Decode(&cfg); err != nil {
			t.Fatalf("decode: %v", err)
		}

		want := map[string]SLOTargets{
			workloadInteractive:  {TauTTFTUs: 1_000_000, TauITLUs: 50_000, TauE2EUs: 16_000_000},
			workloadReasoning:    {TauTTFTUs: 2_000_000, TauITLUs: 100_000, TauE2EUs: 802_000_000},
			workloadDeepResearch: {TauTTFTUs: 10_000_000, TauITLUs: 100_000, TauE2EUs: 40_000_000},
		}
		for name, expect := range want {
			got, ok := cfg.WorkloadTargets[name]
			if !ok {
				t.Errorf("workloadTargets is missing %q", name)
				continue
			}
			if got != expect {
				t.Errorf("workloadTargets[%q] = %+v, want %+v", name, got, expect)
			}
		}
	})
}

// TestOverlayCarriesBothFleetCoefficientSets pins config.md sections 2 and 4: both GPU types
// are transcribed, keyed by pod-label value, so the A100 decode instance is priced correctly
// whether or not the generated scenario names it.
func TestOverlayCarriesBothFleetCoefficientSets(t *testing.T) {
	forEachOverlay(t, func(t *testing.T, plugins pluginsConfig) {
		var params json.RawMessage
		for _, p := range plugins.Plugins {
			if p.Type == HandlerPluginType {
				params = p.Parameters
			}
		}
		cfg := Config{}
		if err := strictDecoderForTest(params).Decode(&cfg); err != nil {
			t.Fatalf("decode: %v", err)
		}

		if len(cfg.CoeffsByGPUType) != 2 {
			t.Fatalf("expected both fleet GPU types, got %v", sortedKeys(cfg.CoeffsByGPUType))
		}
		for label, want := range map[string]Coeffs{"H100_SXM_80GB": h100, "A100_SXM_80GB": a100} {
			got, ok := cfg.CoeffsByGPUType[label]
			if !ok {
				t.Errorf("coeffsByGpuType is missing %q", label)
				continue
			}
			if got != want {
				t.Errorf("coeffsByGpuType[%q] does not match inputs/coeffs-*.json:\n got %+v\nwant %+v", label, got, want)
			}
		}
	})
}

// TestOverlayDeclaresBothHalvesAndBindsThem pins the structural facts a reviewer would
// otherwise have to check by eye: both halves are declared, the picker references the handler
// by name so the plugin DAG can order them, no second ProfileHandler is present, and exactly
// one profile exists naming the picker and no scorers.
func TestOverlayDeclaresBothHalvesAndBindsThem(t *testing.T) {
	forEachOverlay(t, func(t *testing.T, plugins pluginsConfig) {
		handlerIdx, pickerIdx := -1, -1
		var pickerParams json.RawMessage
		var handlerName string
		for i, p := range plugins.Plugins {
			switch p.Type {
			case HandlerPluginType:
				handlerIdx = i
				handlerName = p.Name
				if handlerName == "" {
					handlerName = p.Type // the loader defaults Name to Type
				}
			case PickerPluginType:
				pickerIdx = i
				pickerParams = p.Parameters
			}
			// EXACTLY ONE ProfileHandler is permitted across all plugins, so a stock profile
			// handler alongside this arm is a startup error.
			if p.Type == "disagg-profile-handler" || p.Type == "pd-profile-handler" || p.Type == "single-profile-handler" {
				t.Errorf("the overlay declares %q alongside this arm's handler; only one "+
					"ProfileHandler is permitted per configuration", p.Type)
			}
		}
		if handlerIdx < 0 || pickerIdx < 0 {
			t.Fatal("the overlay must declare both halves of the arm")
		}
		// Position deliberately NOT asserted. Construction order comes from the plugin DAG,
		// whose edge is the pluginRef-tagged handlerPluginName checked just below
		// (see dependency_test.go), so either declaration order loads. Asserting position
		// here would pin a property the loader does not have and imply the reference is
		// optional.
		_, _ = handlerIdx, pickerIdx

		var pp pickerParameters
		if err := strictDecoderForTest(pickerParams).Decode(&pp); err != nil {
			t.Fatalf("picker parameters do not decode: %v", err)
		}
		if pp.HandlerPluginName != handlerName {
			t.Errorf("picker handlerPluginName = %q, want the declared handler %q", pp.HandlerPluginName, handlerName)
		}

		if len(plugins.SchedulingProfiles) != 1 {
			t.Fatalf("expected exactly one role-blind profile, got %d", len(plugins.SchedulingProfiles))
		}
		profile := plugins.SchedulingProfiles[0]
		refs := make([]string, 0, len(profile.Plugins))
		for _, r := range profile.Plugins {
			refs = append(refs, r.PluginRef)
		}
		if len(refs) != 1 || refs[0] != PickerPluginType {
			t.Errorf("the profile must reference the picker and nothing else, got %v; any scorer "+
				"contribution passes through the [0,1] clamp and cannot be combined with a signed J", refs)
		}
	})
}

// TestOverlayDeclaresTheProducersTheArmBinds guards the signal plumbing: the arm binds both
// reads by producer name, so the named producers must actually be declared.
func TestOverlayDeclaresTheProducersTheArmBinds(t *testing.T) {
	forEachOverlay(t, func(t *testing.T, plugins pluginsConfig) {
		declared := map[string]bool{}
		var params json.RawMessage
		for _, p := range plugins.Plugins {
			name := p.Name
			if name == "" {
				name = p.Type
			}
			declared[name] = true
			if p.Type == HandlerPluginType {
				params = p.Parameters
			}
		}
		cfg := Config{}
		if err := strictDecoderForTest(params).Decode(&cfg); err != nil {
			t.Fatalf("decode: %v", err)
		}

		// NO `n != ""` GUARD. An empty binding is the failure this test exists to catch: it
		// leaves the data key at its type-string default, so a producer declared under any
		// other instance name causes a second, nil-parameter producer to be auto-created and
		// read instead. Guarding on non-empty would let exactly that case pass.
		if n := cfg.Signals.PrefixMatchInfoProducerName; !declared[n] {
			t.Errorf("signals.prefixMatchInfoProducerName = %q, which the overlay does not declare", n)
		}
		if n := cfg.Signals.TokenizedPromptProducerName; !declared[n] {
			t.Errorf("signals.tokenizedPromptProducerName = %q, which the overlay does not declare", n)
		}
	})
}

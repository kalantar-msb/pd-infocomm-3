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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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

// overlayPath locates the generated treatment overlay relative to this package.
//
// The path is resolved rather than hardcoded absolute so the test is portable; it SKIPS
// rather than fails when the overlay is absent, because the plugin is buildable and
// testable independently of any one experiment bundle.
func overlayPath(t *testing.T) string {
	t.Helper()
	return overlayPathFor(t, "leastttftjoint", "leastttftjoint_config.yaml")
}

// overlayPathFor locates any arm's generated overlay, so the cross-arm agreement test can
// read the focal arm's alongside this one's.
func overlayPathFor(t *testing.T, dir, file string) string {
	t.Helper()
	// pkg/epp/framework/plugins/scheduling/profilehandler/leastttftjoint -> repo root
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", "..", "..", "..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
	// The experiment bundle contains llm-d-router as a submodule.
	bundleRoot := filepath.Dir(repoRoot)
	matches, err := filepath.Glob(filepath.Join(bundleRoot,
		"workspace", "translations", "*", "generated", dir, file))
	if err != nil {
		t.Fatalf("glob overlay: %v", err)
	}
	if len(matches) == 0 {
		t.Skip("no generated treatment overlay found; skipping overlay agreement check")
	}
	if len(matches) > 1 {
		t.Fatalf("multiple overlays found, cannot pick one: %v", matches)
	}
	return matches[0]
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

func loadOverlayPlugins(t *testing.T) pluginsConfig {
	t.Helper()
	return loadOverlayPluginsFrom(t, overlayPath(t))
}

func loadOverlayPluginsFrom(t *testing.T, path string) pluginsConfig {
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

// handlerParams returns the handler plugin's raw parameters from a loaded plugin config.
func handlerParams(t *testing.T, plugins pluginsConfig, handlerType string) json.RawMessage {
	t.Helper()
	for _, p := range plugins.Plugins {
		if p.Type == handlerType {
			return p.Parameters
		}
	}
	t.Fatalf("the overlay declares no %s plugin", handlerType)
	return nil
}

// TestOverlayHandlerParametersDecodeAndValidate is the decisive agreement check: the shipped
// parameters must decode under DisallowUnknownFields and pass the real validation. A key the
// struct does not declare is a POD STARTUP FAILURE, not a warning.
func TestOverlayHandlerParametersDecodeAndValidate(t *testing.T) {
	params := handlerParams(t, loadOverlayPlugins(t), HandlerPluginType)

	cfg := Config{}
	if err := strictDecoderForTest(params).Decode(&cfg); err != nil {
		t.Fatalf("the shipped overlay parameters do not decode against Config: %v\n"+
			"this would be a pod startup failure, since the config loader uses DisallowUnknownFields", err)
	}
	if err := cfg.validate(); err != nil {
		t.Fatalf("the shipped overlay parameters do not pass validation: %v", err)
	}

	// Spot-check the values the objective depends on, so a silent edit to the overlay cannot
	// change the arm without failing here.
	if cfg.AdmissionEstimator != estimatorRollforward {
		t.Errorf("admissionEstimator = %q, want %q", cfg.AdmissionEstimator, estimatorRollforward)
	}
	if cfg.Engine.ChunkTokens != 2048 || cfg.Engine.BlockSize != 16 || cfg.Engine.MaxBatchSize != 256 {
		t.Errorf("engine agreement drifted: %+v", cfg.Engine)
	}
	if !cfg.Transfer.SizeAware {
		t.Error("transfer.sizeAware must be true (config.md section 9.1): c_xfer is the only " +
			"size-dependent remote price and this arm has no externality term to offset it")
	}
	if cfg.ShadowTable.ResidentPrefillTokensCap != int64(cfg.Engine.ChunkTokens) {
		t.Errorf("residentPrefillTokensCap %d must equal chunkTokens %d",
			cfg.ShadowTable.ResidentPrefillTokensCap, cfg.Engine.ChunkTokens)
	}
}

// TestOverlayMatchesTheConfigTheTestsExercise closes the last gap between the shipped YAML
// and the Go tests. Every other test in this package builds armConfig() in Go; if the overlay
// and armConfig disagreed, the tests would all pass against a configuration the cluster never
// receives.
func TestOverlayMatchesTheConfigTheTestsExercise(t *testing.T) {
	params := handlerParams(t, loadOverlayPlugins(t), HandlerPluginType)
	shipped := Config{}
	if err := strictDecoderForTest(params).Decode(&shipped); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if diff := configDiff(shipped, armConfig()); diff != "" {
		t.Errorf("the shipped overlay and armConfig() disagree:\n%s\n"+
			"every other test in this package exercises armConfig(), so a disagreement means the "+
			"tests validate a configuration the cluster never receives", diff)
	}
}

// TestOverlayCarriesNoFocalArmOnlyKeys is the strict-subset claim checked on the ARTIFACT
// rather than on the struct.
//
// It is the failure mode that would actually happen: someone copies the focal arm's overlay
// to make this one and leaves a tau triple or an ablation switch behind. Because the loader
// is strict that is a startup error rather than a silent no-op -- but catching it here costs
// nothing and names the offending key.
func TestOverlayCarriesNoFocalArmOnlyKeys(t *testing.T) {
	params := handlerParams(t, loadOverlayPlugins(t), HandlerPluginType)
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(params, &keys); err != nil {
		t.Fatalf("unmarshal parameters: %v", err)
	}
	for _, key := range []string{"v", "ablation", "activeWorkload", "workloadTargets", "capacity"} {
		if _, present := keys[key]; present {
			t.Errorf("the overlay carries %q, which this arm reads nowhere. Its presence would be a "+
				"pod startup failure, and it would suggest to an operator that the setting took "+
				"effect. This arm has no tau, no V, no ablation switches, and no capacity account.", key)
		}
	}
}

// TestOverlayCarriesBothFleetCoefficientSets pins config.md sections 2 and 4: both files are
// transcribed, keyed by pod-label value, so the A100 decode instance is priced correctly
// whether or not the generated scenario names it.
func TestOverlayCarriesBothFleetCoefficientSets(t *testing.T) {
	params := handlerParams(t, loadOverlayPlugins(t), HandlerPluginType)
	cfg := Config{}
	if err := strictDecoderForTest(params).Decode(&cfg); err != nil {
		t.Fatalf("decode: %v", err)
	}
	for gpuType, want := range map[string]Coeffs{"H100_SXM_80GB": h100, "A100_SXM_80GB": a100} {
		got, ok := cfg.CoeffsByGPUType[gpuType]
		if !ok {
			t.Errorf("the overlay carries no theta for %q; an unmapped label rejects the endpoint, "+
				"so the A100 decode pod would leave the candidate set entirely", gpuType)
			continue
		}
		if got != want {
			t.Errorf("theta for %q drifted from inputs/: got %+v, want %+v", gpuType, got, want)
		}
	}
	if len(cfg.CoeffsByGPUType) != 2 {
		t.Errorf("expected exactly the two fleet GPU types, got %v", sortedKeys(cfg.CoeffsByGPUType))
	}
}

// TestOverlayDeclaresBothHalvesInTheRequiredOrder pins the declaration order, which is a
// correctness requirement rather than a style one, and the one-ProfileHandler rule.
func TestOverlayDeclaresBothHalvesInTheRequiredOrder(t *testing.T) {
	plugins := loadOverlayPlugins(t)

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
		// EXACTLY ONE ProfileHandler is permitted across all plugins. That covers the stock
		// handlers AND the focal arm's, which is why the two arms are two scenarios rather
		// than one plugin list.
		switch p.Type {
		case "disagg-profile-handler", "pd-profile-handler", "single-profile-handler",
			"causal-slo-externality-handler":
			t.Errorf("the overlay declares %q alongside this arm's handler; only one "+
				"ProfileHandler is permitted per configuration, so this is a startup error", p.Type)
		}
	}
	if handlerIdx < 0 || pickerIdx < 0 {
		t.Fatal("the overlay must declare both halves of the arm")
	}
	if handlerIdx > pickerIdx {
		t.Error("the handler must be declared BEFORE the picker: by-name references resolve " +
			"backward only, so a forward reference yields nil at factory time")
	}

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
	refs := make([]string, 0, len(plugins.SchedulingProfiles[0].Plugins))
	for _, r := range plugins.SchedulingProfiles[0].Plugins {
		refs = append(refs, r.PluginRef)
	}
	if len(refs) != 1 || refs[0] != PickerPluginType {
		t.Errorf("the profile must reference the picker and nothing else, got %v; any scorer "+
			"contribution passes through the [0,1] clamp and cannot be combined with an objective "+
			"measured in microseconds", refs)
	}
}

// TestOverlayDeclaresTheProducersTheArmBinds guards the signal plumbing: the arm binds both
// reads by producer name, so the named producers must actually be declared.
func TestOverlayDeclaresTheProducersTheArmBinds(t *testing.T) {
	plugins := loadOverlayPlugins(t)

	declared := map[string]bool{}
	for _, p := range plugins.Plugins {
		name := p.Name
		if name == "" {
			name = p.Type
		}
		declared[name] = true
	}
	cfg := Config{}
	if err := strictDecoderForTest(handlerParams(t, plugins, HandlerPluginType)).Decode(&cfg); err != nil {
		t.Fatalf("decode: %v", err)
	}

	// NO `n != ""` GUARD. An empty binding is the failure this test exists to catch: it
	// leaves the data key at its type-string default, so a producer declared under any other
	// instance name causes a second, nil-parameter producer to be auto-created and read
	// instead. Guarding on non-empty would let exactly that case pass.
	if n := cfg.Signals.PrefixMatchInfoProducerName; !declared[n] {
		t.Errorf("signals.prefixMatchInfoProducerName = %q, which the overlay does not declare", n)
	}
	if n := cfg.Signals.TokenizedPromptProducerName; !declared[n] {
		t.Errorf("signals.tokenizedPromptProducerName = %q, which the overlay does not declare", n)
	}
}

// TestTheTwoOverlaysAgreeOnEverySharedKey is the strongest form of the config.md section 9.1
// agreement, because it reads BOTH SHIPPED ARTIFACTS rather than two Go fixtures.
//
// The arms run in separate EPP processes, so nothing at runtime can detect a divergence:
// these two YAML files are the only place the equality lives. A drift in theta, in the engine
// agreement, in the transfer model, or in either producer binding would mean the two arms
// differ in the PHYSICS as well as the objective -- and then no measurement could separate the
// mechanism from the machinery, which is the entire reason this arm exists.
//
// It SKIPS when the focal arm's overlay is absent, so this package stays testable on its own.
func TestTheTwoOverlaysAgreeOnEverySharedKey(t *testing.T) {
	mineCfg := Config{}
	if err := strictDecoderForTest(handlerParams(t, loadOverlayPlugins(t), HandlerPluginType)).Decode(&mineCfg); err != nil {
		t.Fatalf("decode this arm's overlay: %v", err)
	}

	focalPath := overlayPathFor(t, "causalsloexternality", "causalsloexternality_config.yaml")
	focalPlugins := loadOverlayPluginsFrom(t, focalPath)
	focalRaw := handlerParams(t, focalPlugins, "causal-slo-externality-handler")

	// Decode the focal arm's parameters into THIS arm's Config, ignoring the keys this arm
	// does not declare. That is exactly the shared-key projection under test.
	var focalKeys map[string]json.RawMessage
	if err := json.Unmarshal(focalRaw, &focalKeys); err != nil {
		t.Fatalf("unmarshal the focal arm's parameters: %v", err)
	}
	for _, focalOnly := range []string{"v", "ablation", "activeWorkload", "workloadTargets", "capacity"} {
		delete(focalKeys, focalOnly)
	}
	projected, err := json.Marshal(focalKeys)
	if err != nil {
		t.Fatalf("re-marshal: %v", err)
	}
	focalShared := Config{}
	if err := strictDecoderForTest(projected).Decode(&focalShared); err != nil {
		t.Fatalf("the focal arm's SHARED keys do not decode against this arm's Config: %v\n"+
			"that means the two arms no longer agree on the shape of a shared setting", err)
	}

	if diff := configDiff(mineCfg, focalShared); diff != "" {
		t.Errorf("the two arms' overlays disagree on shared keys:\n%s\n"+
			"config.md section 9.3 requires this arm to differ from the focal arm IN THE "+
			"OBJECTIVE ONLY. A shared value that drifts makes the arms differ in the physics too, "+
			"and the measured difference stops attributing to the mechanism.", diff)
	}
}

// configDiff reports the shared-key differences between two configurations, one per line.
func configDiff(a, b Config) string {
	var out []string
	add := func(format string, args ...any) { out = append(out, fmt.Sprintf(format, args...)) }

	if a.Engine != b.Engine {
		add("  engine: %+v against %+v", a.Engine, b.Engine)
	}
	if a.Signals != b.Signals {
		add("  signals: %+v against %+v", a.Signals, b.Signals)
	}
	if a.AdmissionEstimator != b.AdmissionEstimator {
		add("  admissionEstimator: %q against %q", a.AdmissionEstimator, b.AdmissionEstimator)
	}
	if a.Transfer != b.Transfer {
		add("  transfer: %+v against %+v", a.Transfer, b.Transfer)
	}
	if a.ShadowTable != b.ShadowTable {
		add("  shadowTable: %+v against %+v", a.ShadowTable, b.ShadowTable)
	}
	if a.OutputTokenProcessingUs != b.OutputTokenProcessingUs {
		add("  outputTokenProcessingUs: %g against %g", a.OutputTokenProcessingUs, b.OutputTokenProcessingUs)
	}
	for _, gpuType := range sortedKeys(a.CoeffsByGPUType) {
		peer, ok := b.CoeffsByGPUType[gpuType]
		if !ok {
			add("  coeffsByGpuType[%q]: present on one side only", gpuType)
			continue
		}
		if a.CoeffsByGPUType[gpuType] != peer {
			add("  coeffsByGpuType[%q]: %+v against %+v", gpuType, a.CoeffsByGPUType[gpuType], peer)
		}
	}
	for _, gpuType := range sortedKeys(b.CoeffsByGPUType) {
		if _, ok := a.CoeffsByGPUType[gpuType]; !ok {
			add("  coeffsByGpuType[%q]: present on one side only", gpuType)
		}
	}
	return strings.Join(out, "\n")
}

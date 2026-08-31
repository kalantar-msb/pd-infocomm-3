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
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"sigs.k8s.io/yaml"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/config/loader"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	reqdataprodprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// The generated treatment overlay is the file that actually reaches the cluster, and it is
// the one artifact no other test covers: every other test in this package builds Config
// values in Go, so a mismatch between the shipped YAML and the struct tags would pass all
// of them and then fail at pod startup.
//
// THE OVERLAY LIVES OUTSIDE THE GO MODULE, so it is not hashed into the test cache key.
// Always run this package with `-count=1`, or a cached PASS can coexist with a known-bad
// overlay. transfer.yaml's build command list says so for this reason.

// overlayPaths locates EVERY generated treatment overlay for this arm.
//
// ALL matches are checked. One bundle holds one translation directory per translation hash,
// so several overlays for this arm coexist as soon as a translation is redone against a new
// target pin. Picking one of them -- by glob order, by mtime, by anything -- would silently
// check an overlay that is not the one being shipped. Each overlay claims to configure this
// plugin, so each must decode and validate against it.
//
// The helper SKIPS rather than fails when no overlay is present, because the plugin is
// buildable and testable independently of any one experiment bundle.
func overlayPaths(t *testing.T) []string {
	t.Helper()
	// pkg/epp/framework/plugins/scheduling/profilehandler/leastttftjoint -> repo root
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", "..", "..", "..", "..", ".."))
	if err != nil {
		t.Fatalf("resolve repo root: %v", err)
	}
	// The experiment bundle contains llm-d-router as a submodule.
	bundleRoot := filepath.Dir(repoRoot)
	matches, err := filepath.Glob(filepath.Join(bundleRoot,
		"workspace", "translations", "*", "generated", "leastttftjoint",
		"leastttftjoint_config.yaml"))
	if err != nil {
		t.Fatalf("glob overlay: %v", err)
	}
	if len(matches) == 0 {
		t.Skip("no generated treatment overlay found; skipping overlay agreement check")
	}
	sort.Strings(matches)
	return matches
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

// loadOverlayAt returns the inner plugin config AND the scenario name. The name is a
// separate requirement: llmdbenchmark deep-merges by scenario name, so a name that does not
// match the experiment's baseline deploys a SECOND scenario instead of merging into it --
// both arms then run, neither as configured, and nothing errors.
func loadOverlayAt(t *testing.T, path string) (pluginsConfig, string) {
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
		t.Fatal(`overlay has no router.epp.pluginsCustomConfig["custom-plugins.yaml"]`)
	}
	var plugins pluginsConfig
	if err := yaml.Unmarshal([]byte(inner), &plugins); err != nil {
		t.Fatalf("parse inner plugin config: %v", err)
	}
	return plugins, overlay.Scenario[0].Name
}

func forEachOverlay(t *testing.T, check func(t *testing.T, plugins pluginsConfig, scenario string)) {
	t.Helper()
	for _, path := range overlayPaths(t) {
		hash := filepath.Base(filepath.Dir(filepath.Dir(filepath.Dir(path))))
		t.Run(hash, func(t *testing.T) {
			plugins, scenario := loadOverlayAt(t, path)
			check(t, plugins, scenario)
		})
	}
}

func overlayParameters(t *testing.T, plugins pluginsConfig, pluginType string) json.RawMessage {
	t.Helper()
	for _, p := range plugins.Plugins {
		if p.Type == pluginType {
			return p.Parameters
		}
	}
	t.Fatalf("the overlay declares no %s plugin", pluginType)
	return nil
}

// TestOverlayHandlerParametersDecodeAndValidate is the decisive agreement check: the shipped
// parameters must decode under DisallowUnknownFields and pass the real arm-aware validation.
// A key the struct does not declare is a POD STARTUP FAILURE, not a warning.
func TestOverlayHandlerParametersDecodeAndValidate(t *testing.T) {
	forEachOverlay(t, func(t *testing.T, plugins pluginsConfig, _ string) {
		params := overlayParameters(t, plugins, HandlerPluginType)

		cfg := causalsloexternality.Config{}
		if err := fwkplugin.StrictDecoder(params).Decode(&cfg); err != nil {
			t.Fatalf("the shipped overlay parameters do not decode against Config: %v\n"+
				"this would be a pod startup failure, since the config loader uses DisallowUnknownFields", err)
		}

		// The real gate, through the real factory, so the arm-aware validation runs.
		if _, err := HandlerFactory(handlerName, fwkplugin.StrictDecoder(params), newFakeHandle()); err != nil {
			t.Fatalf("the shipped overlay parameters do not pass validation: %v", err)
		}

		if cfg.AdmissionEstimator != "rollforward" {
			t.Errorf("admissionEstimator = %q, want rollforward -- the D1 substitution", cfg.AdmissionEstimator)
		}
		if cfg.Engine.ChunkTokens != 2048 || cfg.Engine.BlockSize != 16 || cfg.Engine.MaxBatchSize != 256 {
			t.Errorf("engine agreement drifted: %+v", cfg.Engine)
		}
		if !cfg.Transfer.SizeAware {
			t.Error("transfer.sizeAware must be true -- config.md section 9.1")
		}
	})
}

// TestOverlayCarriesNoSLOValueConfig is the reader-trap check on the SHIPPED FILE, and it is
// the one this arm's specification is most explicit about.
//
// These five keys are known to the shared Config struct, so strict decoding accepts them and
// only an arm-aware check can refuse them. The check that matters is on the file, not on a
// Go value a test constructed: the realistic way the trap is laid is a human copying the
// focal arm's overlay to make this one.
func TestOverlayCarriesNoSLOValueConfig(t *testing.T) {
	forEachOverlay(t, func(t *testing.T, plugins pluginsConfig, _ string) {
		params := overlayParameters(t, plugins, HandlerPluginType)

		var raw map[string]any
		if err := json.Unmarshal(params, &raw); err != nil {
			t.Fatalf("decode parameters as a generic map: %v", err)
		}
		for _, key := range []string{"v", "ablation", "activeWorkload", "workloadTargets", "capacity"} {
			if _, present := raw[key]; present {
				t.Errorf("the overlay sets %q, which this arm's objective never reads: a "+
					"populated-but-never-read field implies a consumer that does not exist, and an "+
					"operator tuning it here would be tuning nothing", key)
			}
		}

		cfg := causalsloexternality.Config{}
		if err := fwkplugin.StrictDecoder(params).Decode(&cfg); err != nil {
			t.Fatalf("decode: %v", err)
		}
		// Belt and braces: the same absence expressed as decoded values, so a future key
		// rename cannot make the string check above vacuous.
		if cfg.V != 0 || cfg.ActiveWorkload != "" || len(cfg.WorkloadTargets) != 0 {
			t.Errorf("no tau and no V may reach this arm: v=%g activeWorkload=%q workloadTargets=%d",
				cfg.V, cfg.ActiveWorkload, len(cfg.WorkloadTargets))
		}
	})
}

// TestOverlayCarriesBothCoefficientSets pins config.md section 4 and the fleet capability gap
// of section 2 together.
//
// The generated scenario names H100 for both roles and cannot express the A100 decode
// instance, so that pod is added outside the scenario. Both coefficient sets must be here
// anyway, keyed by pod-label value, or the hand-added pod is rejected on arrival rather than
// priced -- and heterogeneity is the condition the whole experiment turns on.
func TestOverlayCarriesBothCoefficientSets(t *testing.T) {
	forEachOverlay(t, func(t *testing.T, plugins pluginsConfig, _ string) {
		cfg := causalsloexternality.Config{}
		if err := fwkplugin.StrictDecoder(overlayParameters(t, plugins, HandlerPluginType)).Decode(&cfg); err != nil {
			t.Fatalf("decode: %v", err)
		}

		want := map[string]causalsloexternality.Coeffs{gpuH100: h100Coeffs(), gpuA100: a100Coeffs()}
		if !reflect.DeepEqual(cfg.CoeffsByGPUType, want) {
			t.Errorf("coeffsByGpuType does not match config.md section 4:\n got %+v\nwant %+v",
				cfg.CoeffsByGPUType, want)
		}
	})
}

// TestOverlayIsCommonModeWithTheFocalArmOutsideTheObjective is the ATTRIBUTION check at the
// level of the two shipped files.
//
// The arms differ in the objective only, so everything else in their parameter blocks must
// agree. A drift here -- a retyped coefficient, a different transfer price, a different
// shadow-table TTL, a differently-named producer -- would make every measured difference
// between the arms uninterpretable, and no test inside either arm would notice.
//
// It SKIPS when the focal arm's overlay is absent, since this arm ships independently.
func TestOverlayIsCommonModeWithTheFocalArmOutsideTheObjective(t *testing.T) {
	for _, path := range overlayPaths(t) {
		translation := filepath.Dir(filepath.Dir(path))
		focalPath := filepath.Join(translation, "causalsloexternality", "causalsloexternality_config.yaml")
		if _, err := os.Stat(focalPath); err != nil {
			t.Skip("no focal-arm overlay alongside this one; skipping the common-mode check")
		}

		t.Run(filepath.Base(filepath.Dir(translation)), func(t *testing.T) {
			mine, _ := loadOverlayAt(t, path)
			theirs, _ := loadOverlayAt(t, focalPath)

			decode := func(plugins pluginsConfig, pluginType string) causalsloexternality.Config {
				cfg := causalsloexternality.Config{}
				if err := fwkplugin.StrictDecoder(overlayParameters(t, plugins, pluginType)).Decode(&cfg); err != nil {
					t.Fatalf("decode %s parameters: %v", pluginType, err)
				}
				return cfg
			}
			comparator := decode(mine, HandlerPluginType)
			focal := decode(theirs, causalsloexternality.HandlerPluginType)

			if !reflect.DeepEqual(comparator.CoeffsByGPUType, focal.CoeffsByGPUType) {
				t.Error("the two shipped overlays carry different per-GPU coefficients")
			}
			if comparator.Transfer != focal.Transfer {
				t.Error("the two shipped overlays carry different transfer models")
			}
			if comparator.Engine != focal.Engine {
				t.Error("the two shipped overlays disagree with the engine differently")
			}
			if comparator.ShadowTable != focal.ShadowTable {
				t.Error("the two shipped overlays configure the shadow table differently, so the " +
					"arms would see different resident populations for the same traffic")
			}
			if comparator.Signals != focal.Signals {
				t.Error("the two shipped overlays bind different signals: a data key's identity " +
					"includes its producer name, so differently-bound arms price different prompts")
			}
			if comparator.AdmissionEstimator != focal.AdmissionEstimator {
				t.Error("the two shipped overlays run different admission estimators, so D1 is not " +
					"comparable between the arms")
			}
			if comparator.OutputTokenProcessingUs != focal.OutputTokenProcessingUs {
				t.Error("the two shipped overlays put the arms on different post-processing clocks")
			}
		})
	}
}

// TestOverlayDeclaresOneRoleBlindProfileCarryingOnlyThisPicker pins the required structural
// shape at the config level.
//
// ONE profile, because all profiles receive the same candidateEndpoints slice, so one
// unfiltered profile sees BOTH pools -- which is what lets a single picker run enumerate D
// local plus D*P disaggregated candidates in one argmin. NO SCORERS, because a contribution
// passing through the [0,1] score-range enforcement cannot be combined with a microsecond
// latency on one scale.
func TestOverlayDeclaresOneRoleBlindProfileCarryingOnlyThisPicker(t *testing.T) {
	forEachOverlay(t, func(t *testing.T, plugins pluginsConfig, _ string) {
		if len(plugins.SchedulingProfiles) != 1 {
			t.Fatalf("expected exactly one scheduling profile, got %d", len(plugins.SchedulingProfiles))
		}
		profile := plugins.SchedulingProfiles[0]
		if len(profile.Plugins) != 1 {
			t.Fatalf("the profile must reference this arm's picker and nothing else, got %d refs: %+v",
				len(profile.Plugins), profile.Plugins)
		}
		if got := profile.Plugins[0].PluginRef; got != PickerPluginType {
			t.Errorf("the profile references %q, want %q", got, PickerPluginType)
		}
	})
}

// TestOverlayDeclaresNoOtherProfileHandler pins the single-ProfileHandler rule. Exactly one
// is permitted across all instantiated plugins, so a config naming a second one fails to
// start -- including the FOCAL arm's handler, which is the mistake a copied overlay would
// make.
func TestOverlayDeclaresNoOtherProfileHandler(t *testing.T) {
	forbidden := []string{
		causalsloexternality.HandlerPluginType,
		"disagg-profile-handler",
		"single-profile-handler",
		"pd-profile-handler",
	}
	forEachOverlay(t, func(t *testing.T, plugins pluginsConfig, _ string) {
		var handlers int
		for _, p := range plugins.Plugins {
			if p.Type == HandlerPluginType {
				handlers++
			}
			for _, other := range forbidden {
				if p.Type == other {
					t.Errorf("the overlay declares %q alongside this arm's handler; exactly one "+
						"ProfileHandler is permitted across the whole config and the EPP refuses to "+
						"start otherwise", other)
				}
			}
		}
		if handlers != 1 {
			t.Errorf("expected exactly one %s, got %d", HandlerPluginType, handlers)
		}
	})
}

// TestOverlayPickerNamesThisArmsHandler pins the cross-arm binding at the config level. The
// Go guard refuses a mismatch at startup, but catching it here names the file that carries it.
func TestOverlayPickerNamesThisArmsHandler(t *testing.T) {
	forEachOverlay(t, func(t *testing.T, plugins pluginsConfig, _ string) {
		var params struct {
			HandlerPluginName string `json:"handlerPluginName"`
		}
		if err := json.Unmarshal(overlayParameters(t, plugins, PickerPluginType), &params); err != nil {
			t.Fatalf("decode picker parameters: %v", err)
		}
		if params.HandlerPluginName != HandlerPluginType {
			t.Errorf("the picker names handler %q; it must name this arm's handler %q, or the "+
				"argmin runs the other arm's objective under this arm's name",
				params.HandlerPluginName, HandlerPluginType)
		}
	})
}

// TestOverlayScenarioNameMatchesTheExperiment pins the deep-merge key. A mismatched name
// makes llmdbenchmark deploy a SECOND scenario rather than merging into the baseline, and
// nothing errors.
func TestOverlayScenarioNameMatchesTheExperiment(t *testing.T) {
	forEachOverlay(t, func(t *testing.T, _ pluginsConfig, scenario string) {
		if scenario != "pdinfocomm3" {
			t.Errorf("scenario name = %q, want %q -- llmdbenchmark merges by name", scenario, "pdinfocomm3")
		}
	})
}

// ---------------------------------------------------------------------------
// The real config loader
// ---------------------------------------------------------------------------

// registerArm registers both halves of this arm plus the two producers the shipped config
// declares, exactly as cmd/epp/runner/runner.go registers them. Registration is global, so
// the helper is idempotent and the tests using it do not run in parallel.
//
// The picker's ConfigParser is the point of the exercise: registering the factory alone
// leaves the picker out of PluginsWithPluginDependencies, the DAG gains no edge, and
// construction order becomes a coin flip.
func registerArm(t *testing.T) {
	t.Helper()
	register := func(pluginType string, reg func()) {
		if _, already := fwkplugin.Registry[pluginType]; !already {
			reg()
		}
	}
	register(HandlerPluginType, func() {
		fwkplugin.Register(HandlerPluginType, fwkplugin.StabilityBeta, HandlerFactory)
	})
	register(PickerPluginType, func() {
		fwkplugin.RegisterWithPluginDependencies(PickerPluginType, fwkplugin.StabilityBeta,
			PickerFactory, PickerConfigParser)
	})
	register(tokenizer.PluginType, func() {
		fwkplugin.RegisterAsDefaultProducer(tokenizer.PluginType, fwkplugin.StabilityBeta,
			tokenizer.PluginFactory, tokenizer.TokenizedPromptDataKey)
	})
	register(reqdataprodprefix.ApproxPrefixCachePluginType, func() {
		fwkplugin.RegisterAsDefaultProducer(reqdataprodprefix.ApproxPrefixCachePluginType,
			fwkplugin.StabilityBeta, reqdataprodprefix.ApproxPrefixCacheFactory,
			attrprefix.PrefixCacheMatchInfoDataKey)
	})

	// The flow-control policies the config does NOT declare. applySystemDefaults
	// instantiates these on every load regardless of what the document asks for, so
	// without them the load fails for a reason that has nothing to do with this arm.
	register(fcfs.FCFSOrderingPolicyType, func() {
		fwkplugin.Register(fcfs.FCFSOrderingPolicyType, fwkplugin.StabilityBeta, fcfs.FCFSOrderingPolicyFactory)
	})
	register(globalstrict.GlobalStrictFairnessPolicyType, func() {
		fwkplugin.Register(globalstrict.GlobalStrictFairnessPolicyType, fwkplugin.StabilityBeta,
			globalstrict.GlobalStrictFairnessPolicyFactory)
	})
	register(usagelimits.StaticUsageLimitPolicyType, func() {
		fwkplugin.Register(usagelimits.StaticUsageLimitPolicyType, fwkplugin.StabilityBeta,
			usagelimits.StaticPolicyFactory)
	})
}

// innerPluginConfig extracts the EndpointPickerConfig document the overlay carries as a
// YAML-in-YAML string, which is the exact bytes the EPP pod reads from its mounted file.
func innerPluginConfig(t *testing.T, overlay string) []byte {
	t.Helper()
	raw, err := os.ReadFile(overlay)
	if err != nil {
		t.Fatalf("read overlay: %v", err)
	}
	var parsed overlayFile
	if err := yaml.Unmarshal(raw, &parsed); err != nil {
		t.Fatalf("parse overlay: %v", err)
	}
	if len(parsed.Scenario) != 1 {
		t.Fatalf("expected exactly one scenario entry, got %d", len(parsed.Scenario))
	}
	inner, ok := parsed.Scenario[0].Router.EPP.PluginsCustomConfig["custom-plugins.yaml"]
	if !ok {
		t.Fatal(`overlay has no router.epp.pluginsCustomConfig["custom-plugins.yaml"]`)
	}
	return []byte(inner)
}

// isArmLoadFailure separates a failure in this arm's plugins from one in a later default
// layer. After building the plugins the loader applies system defaults across five layers,
// instantiating plugins the config never mentions; registering that whole set here would
// mirror cmd/epp/runner/runner.go inside a plugin package and would break every time
// upstream adds a default. The assertions after the load are what make tolerating a
// later-layer error safe -- they fail if instantiation did not actually complete.
func isArmLoadFailure(err error) bool {
	return !strings.Contains(err.Error(), "system default application failed")
}

// TestShippedTreatmentConfigLoads is the startup rehearsal. A failure here is a crashlooping
// EPP on the cluster, and this arm producing no data at all.
func TestShippedTreatmentConfigLoads(t *testing.T) {
	registerArm(t)

	for _, overlay := range overlayPaths(t) {
		hash := filepath.Base(filepath.Dir(filepath.Dir(filepath.Dir(overlay))))
		t.Run(hash, func(t *testing.T) {
			logger := logging.NewTestLogger()

			rawConfig, _, err := loader.LoadRawConfig(innerPluginConfig(t, overlay), logger)
			if err != nil {
				t.Fatalf("the shipped plugin config does not parse: %v", err)
			}

			handle := testutils.NewTestHandle(context.Background())
			if _, err := loader.InstantiateAndConfigure(rawConfig, handle, logger); err != nil && isArmLoadFailure(err) {
				t.Fatalf("the shipped plugin config does not load, so the EPP would crashloop "+
					"and this arm would produce no data: %v", err)
			}

			// Both halves exist as instances, and the picker is bound to the handler the
			// loader built. Two divergent copies of the state would each see a fraction of
			// the residents while both plugins looked healthy.
			handler := handle.Plugin(HandlerPluginType)
			if handler == nil {
				t.Fatalf("no %s instance after loading", HandlerPluginType)
			}
			picker := handle.Plugin(PickerPluginType)
			if picker == nil {
				t.Fatalf("no %s instance after loading", PickerPluginType)
			}
			if got := handler.TypedName().Type; got != HandlerPluginType {
				t.Errorf("the instantiated handler reports type %q, want %q", got, HandlerPluginType)
			}
			if got := picker.TypedName().Type; got != PickerPluginType {
				t.Errorf("the instantiated picker reports type %q, want %q", got, PickerPluginType)
			}
		})
	}
}

// TestShippedTreatmentConfigLoadsWithPickerDeclaredFirst is the regression test for the
// construction-order hazard, driven through the real loader rather than by reasoning about
// the sort.
//
// It moves the picker ahead of the handler in the plugins list. With the pluginRef
// dependency in place the DAG still constructs the handler first, so this loads exactly like
// the shipped order; without it, the load depends on Go map iteration order and fails most
// of the time. The repeat count makes an accidental pass unlikely rather than possible.
func TestShippedTreatmentConfigLoadsWithPickerDeclaredFirst(t *testing.T) {
	registerArm(t)

	inner := string(innerPluginConfig(t, overlayPaths(t)[0]))

	// Reorder through the generic YAML shape, so the edit cannot depend on formatting.
	var doc map[string]any
	if err := yaml.Unmarshal([]byte(inner), &doc); err != nil {
		t.Fatalf("parse inner plugin config: %v", err)
	}
	entries, ok := doc["plugins"].([]any)
	if !ok {
		t.Fatal("inner config has no plugins list")
	}
	var armEntries, others []any
	for _, e := range entries {
		m, isMap := e.(map[string]any)
		if !isMap {
			t.Fatalf("unexpected plugins entry %T", e)
		}
		switch m["type"] {
		case PickerPluginType, HandlerPluginType:
			armEntries = append(armEntries, e)
		default:
			others = append(others, e)
		}
	}
	if len(armEntries) != 2 {
		t.Fatalf("expected both arm halves in the plugins list, got %d", len(armEntries))
	}
	if armEntries[0].(map[string]any)["type"] != PickerPluginType {
		armEntries[0], armEntries[1] = armEntries[1], armEntries[0]
	}
	doc["plugins"] = append(append([]any{}, armEntries...), others...)

	reordered, err := yaml.Marshal(doc)
	if err != nil {
		t.Fatalf("re-marshal inner plugin config: %v", err)
	}

	for attempt := range 12 {
		logger := logging.NewTestLogger()
		rawConfig, _, err := loader.LoadRawConfig(reordered, logger)
		if err != nil {
			t.Fatalf("attempt %d: reordered config does not parse: %v", attempt, err)
		}
		handle := testutils.NewTestHandle(context.Background())
		if _, err := loader.InstantiateAndConfigure(rawConfig, handle, logger); err != nil && isArmLoadFailure(err) {
			t.Fatalf("attempt %d: declaring the picker before the handler broke the load, so the "+
				"pluginRef dependency edge is not carrying construction order: %v", attempt, err)
		}
		if handle.Plugin(PickerPluginType) == nil {
			t.Fatalf("attempt %d: no picker instance after loading", attempt)
		}
	}
}

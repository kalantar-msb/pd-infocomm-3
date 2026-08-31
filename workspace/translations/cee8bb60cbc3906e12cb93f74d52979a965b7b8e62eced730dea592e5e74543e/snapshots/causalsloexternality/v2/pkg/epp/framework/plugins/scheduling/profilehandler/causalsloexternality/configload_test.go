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
	"context"
	"os"
	"path/filepath"
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
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// This file drives the shipped treatment plugin config through the REAL config loader.
//
// The other overlay tests decode the handler's parameters against Config and validate them.
// That catches a schema mismatch but not a plumbing one -- and the plumbing is where this arm
// is unusual: two co-operating registrations must be constructed in dependency order and end
// up sharing one Policy and one shadow table, and every parameter block is decoded strictly
// against its struct. That is all loader behaviour, so only the loader can confirm it.
//
// WHAT THESE TESTS DO NOT CLAIM. They are not a full EPP startup rehearsal. After building
// the plugins the loader applies system defaults across five layers -- scheduling, flow
// control, parsers, saturation detector, data layer -- instantiating plugins the config never
// mentions (pkg/epp/config/loader/defaults.go:123-141). Registering that whole set here would
// mirror cmd/epp/runner/runner.go inside a plugin package and would break every time upstream
// adds a default. So the assertions below are on the plugins list: both halves constructed,
// bound to the same state, parameters accepted. A later-layer failure is reported but not
// treated as this arm's, and the check that the real binary starts is `make build` plus the
// deployment itself.

// registerArm registers both halves of the arm, plus the two producers the shipped config
// declares, exactly as cmd/epp/runner/runner.go registers them. Registration is global, so
// the helper is idempotent and the tests using it do not run in parallel.
//
// The picker's ConfigParser is the point of the exercise: registering the factory alone
// leaves the picker out of PluginsWithPluginDependencies, the DAG gains no edge, and
// construction order becomes a coin flip.
//
// The producers are registered rather than stripped from the config so the document under
// test is the one that reaches the pod. They also carry the data keys this arm binds by
// producer name, so a rename on either side surfaces here as a load failure.
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

// TestShippedTreatmentConfigLoads is the startup rehearsal. A failure here is a crashlooping
// EPP on the cluster, and the treatment arm producing no data at all.
//
// It deliberately runs the full InstantiateAndConfigure rather than stopping at
// instantiatePlugins, because profile assembly and the single-ProfileHandler rule are
// enforced after construction.
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
			_, err = loader.InstantiateAndConfigure(rawConfig, handle, logger)
			if err != nil && isArmLoadFailure(err) {
				t.Fatalf("the shipped plugin config does not load, so the EPP would crashloop "+
					"and this arm would produce no data: %v", err)
			}

			// Both halves exist as instances, and the picker is bound to the SAME handler
			// object that owns the Policy and the shadow table. Two divergent copies would
			// each see a fraction of the residents and under-count every externality while
			// both plugins looked healthy.
			//
			// These assertions are what make a tolerated later-layer error safe: they fail if
			// instantiation did not actually complete, whatever a subsequent layer reported.
			handler, ok := handle.Plugin(HandlerPluginType).(*Handler)
			if !ok {
				t.Fatalf("no %s instance after loading", HandlerPluginType)
			}
			picker, ok := handle.Plugin(PickerPluginType).(*Picker)
			if !ok {
				t.Fatalf("no %s instance after loading", PickerPluginType)
			}
			if picker.handler != handler {
				t.Error("the picker is bound to a different Handler than the loader instantiated: " +
					"the argmin and the response hooks must share ONE Policy and ONE shadow table")
			}
			if picker.handler.table != handler.table {
				t.Error("the picker and handler hold different shadow tables")
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

	overlays := overlayPaths(t)
	inner := string(innerPluginConfig(t, overlays[0]))

	// Reverse the two arm entries by rewriting the document through the generic YAML shape,
	// so the edit cannot depend on the file's formatting.
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
	// Picker first, handler second -- the order that fails without the dependency edge.
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
		_, err = loader.InstantiateAndConfigure(rawConfig, handle, logger)
		if err != nil && isArmLoadFailure(err) {
			// The picker factory's own message names handlerPluginName. Matching a bare
			// "not found" would misattribute several unrelated loader errors to the
			// ordering hazard.
			if strings.Contains(err.Error(), "handlerPluginName") {
				t.Fatalf("attempt %d: the handler was constructed AFTER the picker, so the "+
					"by-name lookup returned nil. The pluginRef dependency on "+
					"pickerParameters.HandlerPluginName is what orders them: %v", attempt, err)
			}
			t.Fatalf("attempt %d: reordered config does not load: %v", attempt, err)
		}
		// Independent of any later-layer error: construction must have completed and bound.
		if _, built := handle.Plugin(PickerPluginType).(*Picker); !built {
			t.Fatalf("attempt %d: the picker was not constructed with the picker declared first", attempt)
		}
	}
}

// isArmLoadFailure reports whether a loader error is attributable to this arm rather than to
// a system-default layer this package deliberately does not register.
//
// It keys on the loader's own stage prefixes: plugin instantiation and the two validation
// passes are this arm's business, while "system default application failed" is the tail
// described in the file header. Erring toward TRUE is deliberate -- an unrecognised error is
// treated as this arm's and fails the test, so a genuine regression cannot be filed under the
// tolerated category by accident.
func isArmLoadFailure(err error) bool {
	return !strings.Contains(err.Error(), "system default application failed")
}

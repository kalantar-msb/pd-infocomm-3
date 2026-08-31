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
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// The tests in this file guard the CONSTRUCTION-ORDER contract between this arm's two
// plugin registrations.
//
// The picker resolves the handler that owns their shared Policy and shadow table with
// handle.Plugin(handlerPluginName), so the handler must already be constructed. Order
// comes from a TOPOLOGICAL SORT over the plugin DAG
// (pkg/epp/config/loader/configloader.go:244-250), not from position in the plugins
// list: two plugins with no declared dependency between them both sit at in-degree 0,
// and their relative order is taken from ranging a Go map
// (pkg/epp/util/utils.go:42-46), which is randomized per process. Measured at this pin,
// an undeclared dependency puts the picker first in roughly 5 starts out of 6, and on
// those starts the lookup returns nil and the factory errors, so the EPP crashloops
// nondeterministically.
//
// The ordering guarantee is therefore carried entirely by the `pluginRef` tag on
// pickerParameters.HandlerPluginName plus PickerConfigParser being registered through
// RegisterWithPluginDependencies. Neither shows up in behaviour until the EPP starts,
// and the failure is a nondeterministic crashloop rather than a wrong number, so these
// tests assert the mechanism directly.

// TestPickerParametersDeclaresHandlerAsPluginRef asserts the tag the loader reflects on.
//
// findPluginDependencies (configloader.go:304-336) looks up exactly the "pluginRef" tag
// key on each exported field and treats the field's string value as a dependency name.
// A renamed or dropped tag compiles, passes every other test in this package, and
// reintroduces the crashloop.
func TestPickerParametersDeclaresHandlerAsPluginRef(t *testing.T) {
	field, ok := reflect.TypeOf(pickerParameters{}).FieldByName("HandlerPluginName")
	if !ok {
		t.Fatal("pickerParameters has no HandlerPluginName field")
	}
	if _, tagged := field.Tag.Lookup("pluginRef"); !tagged {
		t.Errorf("HandlerPluginName is missing the pluginRef tag, so the config loader "+
			"cannot see the handler dependency and will construct the picker before the "+
			"handler in most EPP starts; tag is %q", field.Tag)
	}
	if !field.IsExported() {
		t.Error("HandlerPluginName must be exported: findPluginDependencies skips unexported fields")
	}
	if field.Type.Kind() != reflect.String {
		t.Errorf("HandlerPluginName must be a string for findPluginDependencies to read it, got %s", field.Type.Kind())
	}
}

// TestPickerConfigParserSurfacesHandlerName replays what the loader does with the
// parser: parse the raw parameters, then read the tagged field off the returned value.
// The parser must return a value findPluginDependencies can reflect over, which means a
// struct (or pointer to one) and not, say, a map.
func TestPickerConfigParserSurfacesHandlerName(t *testing.T) {
	decoder := json.NewDecoder(bytes.NewReader([]byte(`{"handlerPluginName":"csle-handler"}`)))
	parsed, err := PickerConfigParser(decoder, newFakeHandle())
	if err != nil {
		t.Fatalf("PickerConfigParser: %v", err)
	}

	value := reflect.ValueOf(parsed)
	if value.Kind() == reflect.Pointer {
		value = value.Elem()
	}
	if value.Kind() != reflect.Struct {
		t.Fatalf("PickerConfigParser returned %s; findPluginDependencies only walks structs", value.Kind())
	}

	// Mirror findPluginDependencies: collect every pluginRef-tagged string value.
	var dependencies []string
	for idx := range value.NumField() {
		field := value.Type().Field(idx)
		if _, tagged := field.Tag.Lookup("pluginRef"); !tagged || !field.IsExported() {
			continue
		}
		if name := value.Field(idx).String(); name != "" {
			dependencies = append(dependencies, name)
		}
	}
	if len(dependencies) != 1 || dependencies[0] != "csle-handler" {
		t.Errorf("dependencies = %v, want [csle-handler]: the DAG edge that orders the "+
			"handler before the picker is derived from exactly this list", dependencies)
	}
}

// TestPickerConfigParserRejectsUnknownFields keeps the parser and the factory agreeing on
// the schema. The loader calls the parser with a StrictDecoder, so a typo'd key must be an
// error rather than a silently defaulted handler name -- which surfaces as the "requires
// handlerPluginName" error and reads as a missing key rather than a typo.
func TestPickerConfigParserRejectsUnknownFields(t *testing.T) {
	decoder := json.NewDecoder(bytes.NewReader([]byte(`{"handlerPluginNam":"csle-handler"}`)))
	decoder.DisallowUnknownFields()
	if _, err := PickerConfigParser(decoder, newFakeHandle()); err == nil {
		t.Error("PickerConfigParser accepted an unknown field under a strict decoder")
	}
}

// TestPickerFactoryErrorPointsAtTheDependency keeps the unresolved-handler error pointing
// operators at the pluginRef dependency.
//
// Position in the plugins list carries no ordering, so an error advising the operator to
// declare the handler earlier sends them to reorder the list, watch the crashloop persist,
// and never suspect the tag.
func TestPickerFactoryErrorPointsAtTheDependency(t *testing.T) {
	decoder := json.NewDecoder(bytes.NewReader([]byte(`{"handlerPluginName":"absent-handler"}`)))
	_, err := PickerFactory("csle-picker", decoder, newFakeHandle())
	if err == nil {
		t.Fatal("PickerFactory accepted a handlerPluginName that resolves to nil")
	}
	if strings.Contains(err.Error(), "EARLIER") || strings.Contains(err.Error(), "backward") {
		t.Errorf("factory error advises config-order placement, which carries no ordering: %q", err.Error())
	}
	if !strings.Contains(err.Error(), "pluginRef") {
		t.Errorf("factory error should point at the pluginRef dependency, got %q", err.Error())
	}
}

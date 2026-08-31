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
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality"
)

// The two plugin types that switch this arm on.
//
// TWO REGISTRATIONS, NOT ONE PLUGIN WITH TWO INTERFACES, for the same structural reason
// as the focal arm: ProfileHandler.Pick and Picker.Pick share a method name with
// different signatures (pkg/epp/framework/interface/scheduling/plugins.go:47, :77), so no
// single Go type can satisfy both. The handler carries ProfileHandler (Pick +
// ProcessResults) plus PreRequest and ResponseBodyProcessor and owns the shared state;
// the picker carries Filter + Picker and runs the argmin.
//
// Filter is required rather than incidental: Picker.Pick receives only []*ScoredEndpoint
// and no request, so the arriving request's token count and per-endpoint a_p reach the
// argmin only through a per-endpoint attribute written in the Filter.
//
// BOTH HALVES ARE THE SHARED TYPES from package causalsloexternality, constructed against
// this arm's Objective. Nothing about the handler or the picker differs between the arms,
// and that is the property the comparison rests on -- see this package's doc comment on
// the verbatim-copy contract.
const (
	HandlerPluginType = "least-ttft-joint-handler"
	PickerPluginType  = "least-ttft-joint-picker"
)

// arm binds this package's Objective to the two type strings above.
//
// EXACTLY ONE ProfileHandler IS PERMITTED across all instantiated plugins
// (pkg/epp/config/loader/configloader.go:382-391 errors with "multiple profile handlers
// found"). Both arms are compiled into the same binary and both are registered, but only
// ONE can be instantiated per EPP: the treatment overlay selects an arm by naming its two
// types, and there is one overlay per arm. So the two arms are never co-resident, which
// is what makes the shared per-endpoint attribute key and the shared metric collectors
// unambiguous at runtime. The metric collectors carry plugin_name and plugin_type labels,
// which is how the two arms' D1 and D8 rates are compared -- config.md requires that
// comparison before any result is read.
var arm = causalsloexternality.Arm{
	HandlerType: HandlerPluginType,
	PickerType:  PickerPluginType,
	Objective:   Objective{},
}

// HandlerFactory builds this arm's state-owning half: the shared Handler, carrying a
// Policy that minimises this arm's Objective.
var HandlerFactory fwkplugin.FactoryFunc = causalsloexternality.NewHandlerFactory(arm)

// PickerFactory builds this arm's argmin half and binds it to the handler that owns the
// shared Policy and shadow table.
//
// The binding is checked against the ARM, not just the Go type: both arms are the same
// type, so a picker pointed at the focal handler would otherwise construct successfully
// and then run the focal objective while every log line and metric label named this arm.
// The shared factory rejects that -- see the cross-arm guard in newPicker.
var PickerFactory fwkplugin.FactoryFunc = causalsloexternality.NewPickerFactory(arm)

// PickerConfigParser exposes this picker's dependency on its handler so the config
// loader can order construction.
//
// IT MUST BE REGISTERED WITH THE FACTORY via RegisterWithPluginDependencies. The picker
// resolves its handler with handle.Plugin(name), and instantiatePlugins constructs in
// TOPOLOGICAL order over the plugin DAG (pkg/epp/config/loader/configloader.go:244-250),
// not in config order. With plain Register this type is absent from
// PluginsWithPluginDependencies, the `pluginRef` tag on the handler-name field is never
// read, both plugins sit at in-degree 0, and their relative order comes from ranging a Go
// map (pkg/epp/util/utils.go:42-46) -- measured at this pin, the picker is built first in
// roughly 5 starts out of 6, the lookup returns nil, and the EPP crashloops
// nondeterministically.
var PickerConfigParser fwkplugin.ConfigParserFunc = causalsloexternality.NewPickerConfigParser(arm)

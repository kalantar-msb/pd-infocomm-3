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
	"reflect"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const (
	gpuTypeLabel = "pd-infocomm.io/gpu-type"
	gpuH100      = "H100_SXM_80GB"
	gpuA100      = "A100_SXM_80GB"

	prefixProducer = "approx-prefix-cache-producer"
	tokenProducer  = "token-producer"

	handlerName = "arm-handler"

	// sloClassStandard is the one class every cohort of all three workloads declares, and
	// workloadInteractive is the focal arm's committed default. Neither is read by this
	// arm's objective; they appear only in the focal-arm control fixtures.
	sloClassStandard    = "standard"
	workloadInteractive = "interactive"
)

// h100Coeffs and a100Coeffs are config.md section 4, transcribed. Both GPU types are
// carried by every arm's plugin config, so a hand-added A100 decode pod is priced
// correctly whether or not the generated scenario names it.
func h100Coeffs() causalsloexternality.Coeffs {
	return causalsloexternality.Coeffs{
		AlphaD: 16613.537554540144,
		AlphaP: 16617.85321583337,
		C0:     5.347316038602452,
		C1:     0.04761401141756073,
		CPf:    6.144687138665833,
		CAttn:  0.00010075247918809842,
	}
}

func a100Coeffs() causalsloexternality.Coeffs {
	return causalsloexternality.Coeffs{
		AlphaD: 25563.819286862163,
		AlphaP: 25568.34953836831,
		C0:     5.945331876073271,
		C1:     0.07822809856114352,
		CPf:    9.794219053944662,
		CAttn:  0.00015977670754642687,
	}
}

// comparatorConfig is this arm's whole configuration surface, and its SHAPE is the
// assertion: it is a STRICT SUBSET of the focal arm's, carrying no v, no ablation, no
// activeWorkload, no workloadTargets, and no capacity, because this objective reads none
// of them. Every field present here has the same meaning AND THE SAME VALUE as the focal
// arm's -- if the two arms carried different coefficients or a different transfer model
// the comparison would be meaningless.
func comparatorConfig() causalsloexternality.Config {
	return causalsloexternality.Config{
		Engine: causalsloexternality.EngineAgreement{
			ChunkTokens:  2048,
			BlockSize:    16,
			MaxBatchSize: 256,
		},
		Signals: causalsloexternality.Signals{
			GPUTypeLabelKey:             gpuTypeLabel,
			PrefixMatchInfoProducerName: prefixProducer,
			TokenizedPromptProducerName: tokenProducer,
		},
		AdmissionEstimator: "rollforward",
		Transfer: causalsloexternality.Transfer{
			SizeAware:             true,
			XferBaseUs:            50.0,
			XferBandwidthGBps:     25.0,
			KVBytesPerTokenPerGPU: 81920,
		},
		CoeffsByGPUType: map[string]causalsloexternality.Coeffs{
			gpuH100: h100Coeffs(),
			gpuA100: a100Coeffs(),
		},
		ShadowTable: causalsloexternality.ShadowTable{
			EntryTTLSeconds:          900,
			SweepIntervalSeconds:     30,
			ResidentPrefillTokensCap: 2048,
		},
		OutputTokenProcessingUs: 0.0,
	}
}

var _ fwkplugin.Handle = &fakeHandle{}

type fakeHandle struct {
	plugins  map[string]fwkplugin.Plugin
	registry prometheus.Registerer
}

func newFakeHandle() *fakeHandle {
	return &fakeHandle{plugins: map[string]fwkplugin.Plugin{}, registry: prometheus.NewRegistry()}
}

func (h *fakeHandle) Context() context.Context                            { return context.Background() }
func (h *fakeHandle) Plugin(name string) fwkplugin.Plugin                 { return h.plugins[name] }
func (h *fakeHandle) AddPlugin(name string, p fwkplugin.Plugin)           { h.plugins[name] = p }
func (h *fakeHandle) GetAllPluginsWithNames() map[string]fwkplugin.Plugin { return h.plugins }
func (h *fakeHandle) PodList() []k8stypes.NamespacedName                  { return nil }
func (h *fakeHandle) Metrics() fwkplugin.MetricsRecorder                  { return h.registry }

func (h *fakeHandle) GetAllPlugins() []fwkplugin.Plugin {
	out := make([]fwkplugin.Plugin, 0, len(h.plugins))
	for _, p := range h.plugins {
		out = append(out, p)
	}
	return out
}

// decoderFor wraps a config value the way plugin.StrictDecoder does at startup, so
// factory tests exercise the real decode path including DisallowUnknownFields.
func decoderFor(t *testing.T, v any) *json.Decoder {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return fwkplugin.StrictDecoder(raw)
}

// buildArmWith constructs both halves of one arm the way instantiatePlugins does:
// handler first, then the picker resolving it by name.
func buildArmWith(t *testing.T, cfgArm causalsloexternality.Arm, cfg causalsloexternality.Config) (fwkplugin.Plugin, fwkplugin.Plugin) {
	t.Helper()
	handle := newFakeHandle()

	handler, err := causalsloexternality.NewHandlerFactory(cfgArm)(handlerName, decoderFor(t, cfg), handle)
	if err != nil {
		t.Fatalf("handler factory: %v", err)
	}
	handle.AddPlugin(handlerName, handler)

	picker, err := causalsloexternality.NewPickerFactory(cfgArm)("arm-picker",
		decoderFor(t, map[string]string{"handlerPluginName": handlerName}), handle)
	if err != nil {
		t.Fatalf("picker factory: %v", err)
	}
	return handler, picker
}

// buildComparatorArm is buildArmWith for this package's own arm.
func buildComparatorArm(t *testing.T, cfg causalsloexternality.Config) (fwksched.ProfileHandler, fwksched.Picker) {
	t.Helper()
	handler, picker := buildArmWith(t, arm, cfg)
	return handler.(fwksched.ProfileHandler), picker.(fwksched.Picker)
}

// testEndpoint builds an endpoint with the labels and metrics the arm reads. An empty
// batch with abundant free KV is the state in which the rollforward estimator takes its
// immediate-admission branch, so admission delay is exactly one iteration.
func testEndpoint(name, role, gpuType string, metrics *fwkdl.Metrics) fwksched.Endpoint {
	labels := map[string]string{}
	if role != "" {
		labels[bylabel.RoleLabel] = role
	}
	if gpuType != "" {
		labels[gpuTypeLabel] = gpuType
	}
	if metrics == nil {
		metrics = &fwkdl.Metrics{CacheBlockSize: 16, CacheNumBlocks: 10000}
	}
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Namespace: "default", Name: name},
			Address: "10.0.0.1",
			Port:    "8000",
			Labels:  labels,
		},
		metrics,
		fwkdl.NewAttributes(),
	)
}

// testRequest builds a request carrying a tokenized prompt of testInputLen tokens.
//
// Both the length and the request ID are fixed. The per-candidate arithmetic tests vary a_r
// through the eval fixture instead, where the value is an operand rather than a payload, and
// the shadow table keys on the request ID, so no test here needs two live requests.
func testRequest() *fwksched.InferenceRequest {
	tokens := make([]uint32, testInputLen)
	return &fwksched.InferenceRequest{
		RequestID:   "r1",
		TargetModel: "meta-llama/Llama-3.3-70B-Instruct",
		Headers:     map[string]string{metadata.OldObjectiveKey: sloClassStandard},
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{tokens}},
		},
	}
}

func scoredFrom(endpoints []fwksched.Endpoint, scores map[string]float64) []*fwksched.ScoredEndpoint {
	out := make([]*fwksched.ScoredEndpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		out = append(out, &fwksched.ScoredEndpoint{Endpoint: ep, Score: scores[ep.GetMetadata().ID.String()]})
	}
	return out
}

func endpointID(ep fwksched.Endpoint) string {
	if ep == nil || ep.GetMetadata() == nil {
		return ""
	}
	return ep.GetMetadata().ID.String()
}

// ---------------------------------------------------------------------------
// Factory construction and registration
// ---------------------------------------------------------------------------

func TestFactoriesBuildBothHalvesAndShareOneState(t *testing.T) {
	handler, picker := buildComparatorArm(t, comparatorConfig())

	hp, ok := handler.(fwkplugin.Plugin)
	if !ok {
		t.Fatal("handler must be a Plugin")
	}
	if got := hp.TypedName().Type; got != HandlerPluginType {
		t.Errorf("handler type = %q, want %q", got, HandlerPluginType)
	}
	pp, ok := picker.(fwkplugin.Plugin)
	if !ok {
		t.Fatal("picker must be a Plugin")
	}
	if got := pp.TypedName().Type; got != PickerPluginType {
		t.Errorf("picker type = %q, want %q", got, PickerPluginType)
	}

	// The picker must also be the Filter, because the arriving request's token count and
	// per-endpoint a_p reach the argmin no other way: Picker.Pick receives only
	// []*ScoredEndpoint and no request.
	if _, isFilter := picker.(fwksched.Filter); !isFilter {
		t.Error("the picker half must also implement Filter")
	}
}

// TestBothHalvesReportThisArmsTypesAndNotTheFocalArms is the guard against the cheapest
// possible failure of this whole exercise: registering the comparator by reusing the focal
// arm's type strings. The overlay selects an arm by NAMING its types, so a duplicated type
// string would make the comparator unreachable while everything still compiled and ran.
func TestBothHalvesReportThisArmsTypesAndNotTheFocalArms(t *testing.T) {
	if HandlerPluginType == causalsloexternality.HandlerPluginType {
		t.Fatal("the comparator must not share the focal arm's handler type string")
	}
	if PickerPluginType == causalsloexternality.PickerPluginType {
		t.Fatal("the comparator must not share the focal arm's picker type string")
	}
	for _, typ := range []string{HandlerPluginType, PickerPluginType} {
		if !strings.HasPrefix(typ, "least-ttft-joint-") {
			t.Errorf("type %q should name this arm", typ)
		}
	}
}

// TestPickerRejectsTheOtherArmsHandler is the CROSS-ARM BINDING GUARD, and it is the one
// test in this file whose absence would leave a silent arm swap possible.
//
// Both arms are the SAME Go type -- that is the whole point of the shared machinery -- so
// a type assertion cannot tell them apart. A comparator picker bound to a focal handler
// would construct successfully and then run the FOCAL objective, because the objective
// lives in the Policy the handler owns, while every log line, every plugin_type metric
// label, and the treatment overlay's own name reported the comparator. The reported
// arm-vs-arm comparison would then be a comparison of one arm with itself.
func TestPickerRejectsTheOtherArmsHandler(t *testing.T) {
	handle := newFakeHandle()

	focalHandler, err := causalsloexternality.HandlerFactory(handlerName, decoderFor(t, focalConfig()), handle)
	if err != nil {
		t.Fatalf("focal handler factory: %v", err)
	}
	handle.AddPlugin(handlerName, focalHandler)

	_, err = PickerFactory("arm-picker",
		decoderFor(t, map[string]string{"handlerPluginName": handlerName}), handle)
	if err == nil {
		t.Fatal("a comparator picker must refuse to bind to the focal arm's handler: it would " +
			"silently run the focal objective under this arm's name")
	}
	if !strings.Contains(err.Error(), causalsloexternality.HandlerPluginType) {
		t.Errorf("the error should name the handler's actual arm: %v", err)
	}
}

// focalConfig is the focal arm's configuration, used only to build a focal handler for
// the cross-arm and cross-objective tests. It is this arm's config PLUS the SLO-value
// half, which is exactly the subset relationship the two arms stand in.
func focalConfig() causalsloexternality.Config {
	cfg := comparatorConfig()
	cfg.V = 8
	cfg.Ablation = causalsloexternality.Ablation{NoCapacity: true}
	cfg.ActiveWorkload = workloadInteractive
	cfg.WorkloadTargets = map[string]causalsloexternality.SLOTargets{
		workloadInteractive: {TauTTFTUs: 1000000, TauITLUs: 50000, TauE2EUs: 16000000},
	}
	return cfg
}

func TestShippedComparatorConfigValidates(t *testing.T) {
	if _, _, err := (causalsloexternality.EvalFixture{
		Config:    comparatorConfig(),
		Objective: Objective{},
	}).Build(); err != nil {
		t.Fatalf("fixture over the shipped comparator configuration: %v", err)
	}
	// The real gate is the factory, which validates.
	buildComparatorArm(t, comparatorConfig())
}

// TestConfigRejectsTheSLOValueHalf pins the reader-trap guard.
//
// These five keys are KNOWN to the shared Config struct, so DisallowUnknownFields accepts
// them; only an arm-aware check can tell "known to the type" from "read by this
// objective". The realistic way the trap is laid is an operator copying the focal arm's
// overlay to make this one and tuning tau in it -- silently ignoring the value would leave
// them believing they had changed this arm's behaviour, and the arm-vs-arm comparison
// would be read as if they had.
func TestConfigRejectsTheSLOValueHalf(t *testing.T) {
	for _, tc := range []struct {
		name  string
		mutet func(*causalsloexternality.Config)
		key   string
	}{
		{"v", func(c *causalsloexternality.Config) { c.V = 8 }, "v"},
		{"ablation", func(c *causalsloexternality.Config) {
			c.Ablation = causalsloexternality.Ablation{NoCapacity: true}
		}, "ablation"},
		{"activeWorkload", func(c *causalsloexternality.Config) { c.ActiveWorkload = workloadInteractive }, "activeWorkload"},
		{"workloadTargets", func(c *causalsloexternality.Config) {
			c.WorkloadTargets = map[string]causalsloexternality.SLOTargets{
				workloadInteractive: {TauTTFTUs: 1, TauITLUs: 1, TauE2EUs: 1},
			}
		}, "workloadTargets"},
		{"capacity", func(c *causalsloexternality.Config) {
			c.Capacity = causalsloexternality.Capacity{TauRefUs: 1000000, NomPrefillTokens: 512}
		}, "capacity"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := comparatorConfig()
			tc.mutet(&cfg)
			_, err := HandlerFactory(handlerName, decoderFor(t, cfg), newFakeHandle())
			if err == nil {
				t.Fatalf("setting %s must be refused: this objective never reads it, so a value "+
					"here would be tuning nothing", tc.key)
			}
			if !strings.Contains(err.Error(), tc.key) {
				t.Errorf("the error should name %s: %v", tc.key, err)
			}
		})
	}
}

// TestHandlerFactoryRequiresParameters pins the no-defaults stance: this arm's physics is
// entirely configured, so a missing value must fail loudly rather than price candidates
// under invented coefficients.
func TestHandlerFactoryRequiresParameters(t *testing.T) {
	if _, err := HandlerFactory(handlerName, nil, newFakeHandle()); err == nil {
		t.Error("expected an error when parameters are absent")
	}
}

// TestHandlerFactoryRejectsAbsentCoefficients covers the one config error that would be
// invisible at runtime: theta is keyed by pod label and there is deliberately no fallback
// entry, because heterogeneity rides the per-iteration intercept and a defaulted label is
// wrong on EVERY decision rather than only under load.
func TestHandlerFactoryRejectsAbsentCoefficients(t *testing.T) {
	cfg := comparatorConfig()
	cfg.CoeffsByGPUType = nil
	if _, err := HandlerFactory(handlerName, decoderFor(t, cfg), newFakeHandle()); err == nil {
		t.Error("expected an empty coeffsByGpuType to be rejected")
	}
}

// TestBothArmsCarryIdenticalCoefficientsAndTransferModel is an attribution guard rather
// than a config check. The arms exist to differ in the objective ONLY; if they carried
// different physics constants, every measured difference would be uninterpretable and no
// per-arm test would notice.
func TestBothArmsCarryIdenticalCoefficientsAndTransferModel(t *testing.T) {
	comparator, focal := comparatorConfig(), focalConfig()
	if !reflect.DeepEqual(comparator.CoeffsByGPUType, focal.CoeffsByGPUType) {
		t.Error("the two arms must carry identical per-GPU coefficients")
	}
	if comparator.Transfer != focal.Transfer {
		t.Error("the two arms must carry an identical transfer model")
	}
	if comparator.Engine != focal.Engine {
		t.Error("the two arms must agree with the engine identically")
	}
	if comparator.Signals != focal.Signals {
		t.Error("the two arms must bind the same signals, by the same producer names: a data " +
			"key's identity includes its producer name, so differently-bound arms price different prompts")
	}
	if comparator.AdmissionEstimator != focal.AdmissionEstimator {
		t.Error("the two arms must run the same admission estimator, or D1 is not comparable between them")
	}
}

// ---------------------------------------------------------------------------
// The construction-order contract
// ---------------------------------------------------------------------------

// TestPickerConfigParserSurfacesTheHandlerDependency replays what the loader does with the
// parser: parse the raw parameters, then read the pluginRef-tagged field off the returned
// value. findPluginDependencies (pkg/epp/config/loader/configloader.go:304-336) looks up
// exactly that tag key on each exported field, and the name it finds becomes an edge in
// the plugin DAG that instantiatePlugins topologically sorts (:244-250).
//
// Without the edge both plugins sit at in-degree 0 and their order comes from ranging a Go
// map (pkg/epp/util/utils.go:42-46): the picker is built first in most starts, its handler
// lookup returns nil, and the EPP crashloops nondeterministically. Nothing about that
// shows up in behaviour until startup, which is why the mechanism is asserted directly.
func TestPickerConfigParserSurfacesTheHandlerDependency(t *testing.T) {
	parsed, err := PickerConfigParser(decoderFor(t, map[string]string{"handlerPluginName": handlerName}), newFakeHandle())
	if err != nil {
		t.Fatalf("PickerConfigParser: %v", err)
	}

	value := reflect.ValueOf(parsed)
	if value.Kind() == reflect.Pointer {
		value = value.Elem()
	}
	if value.Kind() != reflect.Struct {
		t.Fatalf("the parser returned %s; findPluginDependencies only walks structs", value.Kind())
	}

	var found bool
	for i := range value.NumField() {
		field := value.Type().Field(i)
		if _, tagged := field.Tag.Lookup("pluginRef"); !tagged {
			continue
		}
		found = true
		if !field.IsExported() {
			t.Errorf("%s carries pluginRef but is unexported; findPluginDependencies skips it", field.Name)
		}
		if field.Type.Kind() != reflect.String {
			t.Errorf("%s must be a string for findPluginDependencies to read it, got %s", field.Name, field.Type.Kind())
		}
		if got := value.Field(i).String(); got != handlerName {
			t.Errorf("the pluginRef field should carry the configured handler name, got %q", got)
		}
	}
	if !found {
		t.Error("no pluginRef-tagged field: the loader cannot see this picker's dependency on its " +
			"handler and will construct them in map order")
	}
}

func TestPickerRequiresTheHandlerName(t *testing.T) {
	handle := newFakeHandle()
	if _, err := PickerFactory("arm-picker", decoderFor(t, map[string]string{}), handle); err == nil {
		t.Error("expected an error when handlerPluginName is absent")
	}
	_, err := PickerFactory("arm-picker",
		decoderFor(t, map[string]string{"handlerPluginName": "no-such-handler"}), handle)
	if err == nil {
		t.Fatal("expected an error when the named handler is absent")
	}
	if !strings.Contains(err.Error(), "no-such-handler") {
		t.Errorf("the error should name the unresolved plugin: %v", err)
	}
}

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
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// ---------------------------------------------------------------------------
// Test doubles
// ---------------------------------------------------------------------------

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

type stubPlugin struct{ name fwkplugin.TypedName }

func (p *stubPlugin) TypedName() fwkplugin.TypedName { return p.name }

// decoderFor wraps a config value the way plugin.StrictDecoder does at startup, so factory
// tests exercise the real decode path including DisallowUnknownFields.
func decoderFor(t *testing.T, v any) *json.Decoder {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return fwkplugin.StrictDecoder(raw)
}

func decoderForJSON(t *testing.T, raw string) *json.Decoder {
	t.Helper()
	dec := json.NewDecoder(bytes.NewReader([]byte(raw)))
	dec.DisallowUnknownFields()
	return dec
}

// testEndpoint builds an endpoint with the labels and metrics the arm reads.
func testEndpoint(name, role, gpuType string, metrics *fwkdl.Metrics) fwksched.Endpoint {
	labels := map[string]string{}
	if role != "" {
		labels[bylabel.RoleLabel] = role
	}
	if gpuType != "" {
		labels["pd-infocomm.io/gpu-type"] = gpuType
	}
	if metrics == nil {
		metrics = &fwkdl.Metrics{CacheBlockSize: 16, CacheNumBlocks: 1000}
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

// testRequest builds a request with a tokenized prompt of the given length.
func testRequest(id string, promptTokens int) *fwksched.InferenceRequest {
	tokens := make([]uint32, promptTokens)
	return &fwksched.InferenceRequest{
		RequestID:   id,
		TargetModel: "meta-llama/Llama-3.3-70B-Instruct",
		Headers:     map[string]string{metadata.OldObjectiveKey: sloClassStandard},
		Body: &fwkrh.InferenceRequestBody{
			TokenizedPrompt: &fwkrh.TokenizedPrompt{PerPromptTokens: [][]uint32{tokens}},
		},
	}
}

// buildArm constructs both halves the way instantiatePlugins does: handler first, then the
// picker resolving it by name.
func buildArm(t *testing.T, cfg Config) (*Handler, *Picker) {
	t.Helper()
	handle := newFakeHandle()

	hp, err := HandlerFactory("arm-handler", decoderFor(t, cfg), handle)
	if err != nil {
		t.Fatalf("HandlerFactory: %v", err)
	}
	handler := hp.(*Handler)
	handle.AddPlugin("arm-handler", handler)
	t.Cleanup(handler.table.stop)

	pp, err := PickerFactory("arm-picker",
		decoderFor(t, pickerParameters{HandlerPluginName: "arm-handler"}), handle)
	if err != nil {
		t.Fatalf("PickerFactory: %v", err)
	}
	return handler, pp.(*Picker)
}

// ---------------------------------------------------------------------------
// Factory construction
// ---------------------------------------------------------------------------

func TestHandlerFactoryBuildsFromTheShippedConfig(t *testing.T) {
	handler, picker := buildArm(t, focalConfig())

	if handler.TypedName().Type != HandlerPluginType {
		t.Errorf("handler type = %q, want %q", handler.TypedName().Type, HandlerPluginType)
	}
	if picker.TypedName().Type != PickerPluginType {
		t.Errorf("picker type = %q, want %q", picker.TypedName().Type, PickerPluginType)
	}
	// THE SHARED-STATE PROPERTY: the picker must hold the handler's Policy, not a copy.
	// Two divergent shadow tables would each see a fraction of the residents and
	// under-count every externality while both plugins looked healthy.
	if picker.handler != handler {
		t.Error("the picker must share the handler's state object")
	}
	if handler.policy.table != handler.table {
		t.Error("the policy and the handler must share one shadow table")
	}
}

// TestHandlerFactoryRequiresParameters pins the no-defaults stance: a missing value must
// fail loudly rather than price candidates under invented physics.
func TestHandlerFactoryRequiresParameters(t *testing.T) {
	if _, err := HandlerFactory("h", nil, newFakeHandle()); err == nil {
		t.Error("expected an error when parameters are absent")
	}
}

func TestHandlerFactoryRejectsInvalidConfig(t *testing.T) {
	cfg := focalConfig()
	cfg.AdmissionEstimator = "waiting"
	if _, err := HandlerFactory("h", decoderFor(t, cfg), newFakeHandle()); err == nil {
		t.Error("expected the estimator validation to reject `waiting`")
	}
}

// TestHandlerFactoryRejectsUnknownParameterKeys pins the strict decoding the config loader
// applies: a misspelled key is a startup error, not a silently ignored one.
func TestHandlerFactoryRejectsUnknownParameterKeys(t *testing.T) {
	raw := `{"v": 8, "activeWorkloadTypo": "interactive"}`
	_, err := HandlerFactory("h", decoderForJSON(t, raw), newFakeHandle())
	if err == nil {
		t.Fatal("expected a misspelled parameter key to be rejected")
	}
	if !strings.Contains(err.Error(), "activeWorkloadTypo") {
		t.Errorf("error should name the unknown field: %v", err)
	}
}

// TestPickerFactoryRequiresTheHandlerName covers the forward-reference hazard: by-name
// name must fail loudly rather than construct a second copy of the state. Construction
// ORDER is guaranteed by the pluginRef dependency rather than by position in the plugins
// list -- see dependency_test.go -- so the error must not tell the operator to reorder.
func TestPickerFactoryRequiresTheHandlerName(t *testing.T) {
	handle := newFakeHandle()

	if _, err := PickerFactory("p", decoderFor(t, pickerParameters{}), handle); err == nil {
		t.Error("expected an error when handlerPluginName is absent")
	}

	// Named but absent from the handle entirely.
	_, err := PickerFactory("p",
		decoderFor(t, pickerParameters{HandlerPluginName: "no-such-handler"}), handle)
	if err == nil {
		t.Fatal("expected an error when the named handler is absent")
	}
	if !strings.Contains(err.Error(), "no-such-handler") {
		t.Errorf("error should name the unresolved plugin: %v", err)
	}

	// Named but the wrong type.
	handle.AddPlugin("not-the-handler", &stubPlugin{name: fwkplugin.TypedName{Type: "other", Name: "not-the-handler"}})
	if _, err := PickerFactory("p",
		decoderFor(t, pickerParameters{HandlerPluginName: "not-the-handler"}), handle); err == nil {
		t.Error("expected an error when the named plugin is the wrong type")
	}
}

// TestConsumesBindsSignalsByProducerName pins the by-name binding both arms depend on: a
// data key's identity is "<DataType>/<ProducerName>", so an unbound read can silently become
// a second, differently-configured signal.
func TestConsumesBindsSignalsByProducerName(t *testing.T) {
	cfg := focalConfig()
	cfg.Signals.PrefixMatchInfoProducerName = "my-prefix-producer"
	cfg.Signals.TokenizedPromptProducerName = "my-token-producer"
	handler, picker := buildArm(t, cfg)

	if got := handler.prefixDataKey.String(); !strings.HasSuffix(got, "/my-prefix-producer") {
		t.Errorf("prefix key = %q, want it bound to the configured producer name", got)
	}
	if got := handler.tokenDataKey.String(); !strings.HasSuffix(got, "/my-token-producer") {
		t.Errorf("token key = %q, want it bound to the configured producer name", got)
	}

	// Both halves must declare the SAME keys, or the DAG could order one before a producer
	// and not the other.
	hDeps, pDeps := handler.Consumes(), picker.Consumes()
	if len(hDeps.Required) != len(pDeps.Required) || len(hDeps.Optional) != len(pDeps.Optional) {
		t.Error("both halves must declare the same dependencies")
	}
	// The tokenized prompt is REQUIRED, so a missing producer is an init-time error.
	if _, ok := hDeps.Required[handler.tokenDataKey]; !ok {
		t.Error("the tokenized prompt must be a Required dependency")
	}
	// The prefix read is OPTIONAL: a miss is a legitimate runtime state the arm handles by
	// treating the prompt as fully uncached, so requiring it would refuse to start on a
	// fleet whose producer has not warmed up.
	if _, ok := hDeps.Optional[handler.prefixDataKey]; !ok {
		t.Error("the prefix match info must be an Optional dependency")
	}
}

// ---------------------------------------------------------------------------
// Filter: rejection, and the a_p computation
// ---------------------------------------------------------------------------

// TestFilterRejectsAndCountsUnpriceableEndpoints is the observability requirement: a
// mislabelled endpoint leaving the candidate set must be attributable, or it is
// indistinguishable from a routing preference.
func TestFilterRejectsAndCountsUnpriceableEndpoints(t *testing.T) {
	_, picker := buildArm(t, focalConfig())
	ctx := context.Background()
	request := testRequest("r1", 4000)

	endpoints := []fwksched.Endpoint{
		testEndpoint("good", bylabel.RoleDecode, "H100_SXM_80GB", nil),
		testEndpoint("no-gpu-label", bylabel.RoleDecode, "", nil),
		testEndpoint("unmapped-gpu", bylabel.RoleDecode, "V100_NOT_IN_TABLE", nil),
		testEndpoint("no-role", "", "H100_SXM_80GB", nil),
		testEndpoint("unknown-role", "sidecar", "H100_SXM_80GB", nil),
	}

	before := map[string]float64{}
	for _, r := range []string{
		rejectReasonGPUTypeLabelAbsent, rejectReasonGPUTypeUnmapped,
		rejectReasonRoleLabelAbsent, rejectReasonRoleUnknown,
	} {
		before[r] = readCounterWithReason(t, candidateRejectedCount, "arm-picker", r)
	}

	accepted := picker.Filter(ctx, request, endpoints)

	if len(accepted) != 1 {
		names := make([]string, 0, len(accepted))
		for _, ep := range accepted {
			names = append(names, ep.GetMetadata().ID.Name)
		}
		t.Fatalf("expected only the well-labelled endpoint, got %v", names)
	}
	if accepted[0].GetMetadata().ID.Name != "good" {
		t.Errorf("accepted %q, want \"good\"", accepted[0].GetMetadata().ID.Name)
	}

	for _, r := range []string{
		rejectReasonGPUTypeLabelAbsent, rejectReasonGPUTypeUnmapped,
		rejectReasonRoleLabelAbsent, rejectReasonRoleUnknown,
	} {
		got := readCounterWithReason(t, candidateRejectedCount, "arm-picker", r) - before[r]
		if got != 1 {
			t.Errorf("rejection reason %q counted %g times, want 1", r, got)
		}
	}
}

// TestFilterRejectsAbsentRoleLabelDeliberatelyUnlikeDecodeFilter documents the divergence
// from the stock decode-filter, which passes allowsNoLabel = true and so treats an
// unlabelled pod as decode-eligible. This arm rejects instead: an unlabelled pod would
// otherwise be priced as a decode candidate under a role it may not serve.
func TestFilterRejectsAbsentRoleLabelDeliberatelyUnlikeDecodeFilter(t *testing.T) {
	_, picker := buildArm(t, focalConfig())
	unlabelled := testEndpoint("unlabelled", "", "H100_SXM_80GB", nil)

	// The stock decode filter accepts it.
	stock := bylabel.NewDecodeRole()
	if got := stock.Filter(context.Background(), testRequest("r", 100), []fwksched.Endpoint{unlabelled}); len(got) != 1 {
		t.Fatal("fixture assumption broken: the stock decode filter should accept an unlabelled pod")
	}

	// This arm does not.
	if got := picker.Filter(context.Background(), testRequest("r", 100), []fwksched.Endpoint{unlabelled}); len(got) != 0 {
		t.Error("this arm must reject an endpoint with no role label")
	}
}

// TestFilterAcceptsDualRoleEndpoints confirms the role sets are not disjoint in general:
// prefill-decode and its relatives satisfy both pools.
func TestFilterAcceptsDualRoleEndpoints(t *testing.T) {
	_, picker := buildArm(t, focalConfig())
	for _, role := range []string{bylabel.RolePrefillDecode, bylabel.RoleBoth, bylabel.RoleEncodePrefillDecode} { //nolint:staticcheck // SA1019: RoleBoth is still accepted by the stock filters
		ep := testEndpoint("dual", role, "H100_SXM_80GB", nil)
		accepted := picker.Filter(context.Background(), testRequest("r", 100), []fwksched.Endpoint{ep})
		if len(accepted) != 1 {
			t.Errorf("role %q must be accepted", role)
			continue
		}
		in, _ := picker.readDecisionInput([]*fwksched.ScoredEndpoint{{Endpoint: accepted[0]}})
		if in == nil {
			t.Fatalf("role %q: no decision input attached", role)
		}
		if len(in.decodeIDs) != 1 || len(in.prefillIDs) != 1 {
			t.Errorf("role %q must land in both pools, got decode=%v prefill=%v", role, in.decodeIDs, in.prefillIDs)
		}
	}
}

// TestFilterRejectsBlockSizeDisagreement pins signal 23's loud failure: a config/engine
// disagreement leaves a latent unit bug in the admission test, which compares blocks
// accumulated in one unit against a need expressed in another.
func TestFilterRejectsBlockSizeDisagreement(t *testing.T) {
	_, picker := buildArm(t, focalConfig()) // configured engine block size 16
	ep := testEndpoint("mismatched", bylabel.RoleDecode, "H100_SXM_80GB",
		&fwkdl.Metrics{CacheBlockSize: 32, CacheNumBlocks: 1000})

	before := readCounter(t, blockSizeMismatchCount, "arm-picker", PickerPluginType)
	accepted := picker.Filter(context.Background(), testRequest("r", 100), []fwksched.Endpoint{ep})
	after := readCounter(t, blockSizeMismatchCount, "arm-picker", PickerPluginType)

	if len(accepted) != 0 {
		t.Error("an endpoint whose scraped block size disagrees with config must be rejected")
	}
	if after-before != 1 {
		t.Errorf("the mismatch counter moved by %g, want 1", after-before)
	}
}

// TestFilterUsesTheProducerBlockSizeForAP is the 4x trap. The engine reports 16 while the
// prefix producer clamps to a >= 64-token floor, so multiplying the producer's block COUNT
// by the ENGINE's block SIZE would understate the cached span by 4x -- inflating a_p and
// over-pricing prefill, which biases toward remote.
func TestFilterUsesTheProducerBlockSizeForAP(t *testing.T) {
	_, picker := buildArm(t, focalConfig())
	ep := testEndpoint("cached", bylabel.RoleDecode, "H100_SXM_80GB", nil)

	// 10 cached blocks of 64 tokens each: the true cached span is 640 tokens.
	info := attrprefix.NewPrefixCacheMatchInfo(10, 100, 64).WithCachedBlockCount(10)
	ep.Put(picker.prefixDataKey.String(), info)

	accepted := picker.Filter(context.Background(), testRequest("r", 4000), []fwksched.Endpoint{ep})
	if len(accepted) != 1 {
		t.Fatal("the endpoint must be accepted")
	}
	in, _ := picker.readDecisionInput([]*fwksched.ScoredEndpoint{{Endpoint: accepted[0]}})
	if in == nil {
		t.Fatal("no decision input attached")
	}
	id := accepted[0].GetMetadata().ID.String()

	// a_p = 4000 - 10*64 = 3360. Using the engine's 16 would give 4000 - 160 = 3840, which
	// is the 4x understatement of the cached span.
	if got := in.apByEndpoint[id]; got != 3360 {
		t.Errorf("a_p = %d, want 3360 (4000 - 10 blocks x 64 tokens); 3840 would mean the "+
			"engine block size was used and the cached span understated 4x", got)
	}
}

// TestFilterUsesCachedBlockCountNotWeightedMatchScore pins the other prefix trap: the
// tier-weighted match score is <= the literal count, so using it would overstate the
// uncached suffix.
func TestFilterUsesCachedBlockCountNotWeightedMatchScore(t *testing.T) {
	_, picker := buildArm(t, focalConfig())
	ep := testEndpoint("tiered", bylabel.RoleDecode, "H100_SXM_80GB", nil)

	// A RAM-tier hit: the weighted match score is 4 but 10 blocks are literally cached.
	info := attrprefix.NewPrefixCacheMatchInfo(4, 100, 64).WithCachedBlockCount(10)
	ep.Put(picker.prefixDataKey.String(), info)

	accepted := picker.Filter(context.Background(), testRequest("r", 4000), []fwksched.Endpoint{ep})
	in, _ := picker.readDecisionInput([]*fwksched.ScoredEndpoint{{Endpoint: accepted[0]}})
	id := accepted[0].GetMetadata().ID.String()

	if got := in.apByEndpoint[id]; got != 3360 {
		t.Errorf("a_p = %d, want 3360 from the unweighted count of 10; 3744 would mean the "+
			"tier-weighted score of 4 was used and the uncached suffix overstated", got)
	}
}

// TestFilterCountsPrefixMissAndChargesFullPrompt pins the miss policy: a miss means "no
// information", not "nothing cached", so the full prompt is charged -- over-pricing the
// candidate rather than asserting a cold cache as fact, and leaving it in the argmin.
func TestFilterCountsPrefixMissAndChargesFullPrompt(t *testing.T) {
	_, picker := buildArm(t, focalConfig())
	ep := testEndpoint("nocache", bylabel.RoleDecode, "H100_SXM_80GB", nil)

	before := readCounter(t, prefixUnobservedCount, "arm-picker", PickerPluginType)
	accepted := picker.Filter(context.Background(), testRequest("r", 4000), []fwksched.Endpoint{ep})
	after := readCounter(t, prefixUnobservedCount, "arm-picker", PickerPluginType)

	if after-before != 1 {
		t.Errorf("the prefix-miss counter moved by %g, want 1", after-before)
	}
	in, _ := picker.readDecisionInput([]*fwksched.ScoredEndpoint{{Endpoint: accepted[0]}})
	id := accepted[0].GetMetadata().ID.String()
	if got := in.apByEndpoint[id]; got != 4000 {
		t.Errorf("a_p on a miss = %d, want the full 4000-token prompt", got)
	}
}

// TestFilterDeclinesWhenTokenizationUnavailable is DEGRADATION D8: a_r is absent, so the
// arm makes NO DECISION and the inherited pick stands -- a silent fallback to a third
// policy, which is why it is counted per arm.
func TestFilterDeclinesWhenTokenizationUnavailable(t *testing.T) {
	_, picker := buildArm(t, focalConfig())
	ep := testEndpoint("d1", bylabel.RoleDecode, "H100_SXM_80GB", nil)

	// A nil TokenizedPrompt is the only detectable absence.
	request := &fwksched.InferenceRequest{RequestID: "r1", Body: &fwkrh.InferenceRequestBody{}}

	before := readCounter(t, tokenizationUnavailableCount, "arm-picker", PickerPluginType)
	accepted := picker.Filter(context.Background(), request, []fwksched.Endpoint{ep})
	after := readCounter(t, tokenizationUnavailableCount, "arm-picker", PickerPluginType)

	if after-before != 1 {
		t.Errorf("the D8 counter moved by %g, want 1", after-before)
	}
	// The endpoints pass through untouched, with no decision input attached.
	if len(accepted) != 1 {
		t.Errorf("endpoints must pass through so the inherited pick can stand, got %d", len(accepted))
	}
	if in, _ := picker.readDecisionInput([]*fwksched.ScoredEndpoint{{Endpoint: accepted[0]}}); in != nil {
		t.Error("no decision input must be attached when the arm declines")
	}
}

// TestPromptTokensDistinguishesNilFromEmpty covers what the D8 counter can and cannot see.
func TestPromptTokensDistinguishesNilFromEmpty(t *testing.T) {
	if _, ok := promptTokens(nil); ok {
		t.Error("a nil request must not be scorable")
	}
	if _, ok := promptTokens(&fwksched.InferenceRequest{}); ok {
		t.Error("a nil body must not be scorable")
	}
	if _, ok := promptTokens(&fwksched.InferenceRequest{Body: &fwkrh.InferenceRequestBody{}}); ok {
		t.Error("a nil TokenizedPrompt must not be scorable")
	}
	// A present-but-empty tokenization IS scorable, and reports zero tokens. The producer
	// leaves the field nil when the backend yields zero tokens, so this shape means the
	// prompt was genuinely tokenized.
	got, ok := promptTokens(&fwksched.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{TokenizedPrompt: &fwkrh.TokenizedPrompt{}},
	})
	if !ok || got != 0 {
		t.Errorf("an empty tokenization = (%d, %v), want (0, true)", got, ok)
	}
}

// TestSLOClassReadsBothObjectiveHeaderSpellings matters because the load generator emits the
// deprecated spelling while the target has introduced a newer one; honouring only one would
// silently resolve every request to the fallback class.
func TestSLOClassReadsBothObjectiveHeaderSpellings(t *testing.T) {
	for _, key := range []string{metadata.ObjectiveKey, metadata.OldObjectiveKey} {
		req := &fwksched.InferenceRequest{Headers: map[string]string{key: "premium"}}
		if got := sloClassOf(req); got != "premium" {
			t.Errorf("header %q: got %q, want \"premium\"", key, got)
		}
	}
	// Absent or empty falls back to the one class every cohort declares.
	if got := sloClassOf(&fwksched.InferenceRequest{}); got != sloClassStandard {
		t.Errorf("no header: got %q, want %q", got, sloClassStandard)
	}
	if got := sloClassOf(nil); got != sloClassStandard {
		t.Errorf("nil request: got %q, want %q", got, sloClassStandard)
	}
}

// ---------------------------------------------------------------------------
// Picker: the joint pick
// ---------------------------------------------------------------------------

// runArm drives Filter then Pick, as one profile run would.
func runArm(t *testing.T, picker *Picker, request *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint) *fwksched.ProfileRunResult {
	t.Helper()
	accepted := picker.Filter(context.Background(), request, endpoints)
	scored := make([]*fwksched.ScoredEndpoint, 0, len(accepted))
	for _, ep := range accepted {
		// Score 0 for every endpoint, which is what a profile with no scorers produces.
		scored = append(scored, &fwksched.ScoredEndpoint{Endpoint: ep, Score: 0})
	}
	return picker.Pick(context.Background(), scored)
}

// TestPickerReturnsBothPicksInOneResult pins the shape: one picker run carries both picks,
// which is legitimate because ProfileRunResult.TargetEndpoints is a slice.
func TestPickerReturnsBothPicksInOneResult(t *testing.T) {
	handler, picker := buildArm(t, focalConfig())
	handler.policy.observeCompletedOutput(sloClassStandard, 300)

	endpoints := []fwksched.Endpoint{
		testEndpoint("decode-1", bylabel.RoleDecode, "H100_SXM_80GB", nil),
		testEndpoint("decode-2", bylabel.RoleDecode, "A100_SXM_80GB", nil),
		testEndpoint("prefill-1", bylabel.RolePrefill, "H100_SXM_80GB", nil),
	}
	result := runArm(t, picker, testRequest("r1", 4000), endpoints)

	if result == nil || len(result.TargetEndpoints) == 0 {
		t.Fatal("the picker must select at least the decode endpoint")
	}
	if len(result.TargetEndpoints) > 2 {
		t.Errorf("at most two endpoints (decode, prefill), got %d", len(result.TargetEndpoints))
	}
	// TargetEndpoints[0] is always the decode pick.
	first := result.TargetEndpoints[0].GetMetadata().Labels[bylabel.RoleLabel]
	if first != bylabel.RoleDecode {
		t.Errorf("TargetEndpoints[0] must be the decode pick, got role %q", first)
	}
	if len(result.TargetEndpoints) == 2 {
		second := result.TargetEndpoints[1].GetMetadata().Labels[bylabel.RoleLabel]
		if second != bylabel.RolePrefill {
			t.Errorf("TargetEndpoints[1] must be the prefill pick, got role %q", second)
		}
	}
}

// TestPickerIgnoresInheritedScores pins that ScoredEndpoint.Score is not read as an
// objective: it carries CLAMPED scorer contributions, which cannot be combined with a
// signed, unbounded J on one scale.
func TestPickerIgnoresInheritedScores(t *testing.T) {
	handler, picker := buildArm(t, focalConfig())
	handler.policy.observeCompletedOutput(sloClassStandard, 300)

	endpoints := []fwksched.Endpoint{
		testEndpoint("decode-a", bylabel.RoleDecode, "H100_SXM_80GB",
			&fwkdl.Metrics{CacheBlockSize: 16, CacheNumBlocks: 1000, RunningRequestsSize: 200, WaitingQueueSize: 40, KVCacheUsagePercent: 0.95}),
		testEndpoint("decode-b", bylabel.RoleDecode, "H100_SXM_80GB",
			&fwkdl.Metrics{CacheBlockSize: 16, CacheNumBlocks: 1000}),
	}
	request := testRequest("r1", 4000)
	accepted := picker.Filter(context.Background(), request, endpoints)

	// Give the heavily loaded endpoint the maximum inherited score. If the picker honoured
	// Score it would pick that one; it must pick on J instead.
	scored := make([]*fwksched.ScoredEndpoint, 0, len(accepted))
	for _, ep := range accepted {
		score := 0.0
		if ep.GetMetadata().ID.Name == "decode-a" {
			score = 1.0
		}
		scored = append(scored, &fwksched.ScoredEndpoint{Endpoint: ep, Score: score})
	}
	result := picker.Pick(context.Background(), scored)
	if result == nil || len(result.TargetEndpoints) == 0 {
		t.Fatal("expected a pick")
	}
	if got := result.TargetEndpoints[0].GetMetadata().ID.Name; got != "decode-b" {
		t.Errorf("the picker must decide on J, not the inherited score; picked %q", got)
	}
}

// TestPickerFallsBackWhenTheFilterDeclined confirms the arm yields to the inherited order
// rather than inventing a decision when a_r is unavailable.
func TestPickerFallsBackWhenTheFilterDeclined(t *testing.T) {
	_, picker := buildArm(t, focalConfig())
	ep := testEndpoint("d1", bylabel.RoleDecode, "H100_SXM_80GB", nil)
	// No decision input attached, as after a D8 decline.
	scored := []*fwksched.ScoredEndpoint{{Endpoint: ep, Score: 0}}

	result := picker.Pick(context.Background(), scored)
	if result == nil || len(result.TargetEndpoints) != 1 {
		t.Fatal("expected the inherited first endpoint")
	}
}

func TestPickerHandlesEmptyInput(t *testing.T) {
	_, picker := buildArm(t, focalConfig())
	result := picker.Pick(context.Background(), nil)
	if result == nil || len(result.TargetEndpoints) != 0 {
		t.Error("an empty candidate set must yield an empty result, not a panic")
	}
}

// TestPickerRecordsPlacementAndArgminDuration pins two required counters.
func TestPickerRecordsPlacementAndArgminDuration(t *testing.T) {
	handler, picker := buildArm(t, focalConfig())
	handler.policy.observeCompletedOutput(sloClassStandard, 300)

	endpoints := []fwksched.Endpoint{
		testEndpoint("decode-1", bylabel.RoleDecode, "H100_SXM_80GB", nil),
		testEndpoint("prefill-1", bylabel.RolePrefill, "H100_SXM_80GB", nil),
	}
	beforeLocal := readCounterWithReason(t, placementChosenCount, "arm-picker", placementLocal)
	beforeDisagg := readCounterWithReason(t, placementChosenCount, "arm-picker", placementDisaggregated)

	runArm(t, picker, testRequest("r1", 4000), endpoints)

	afterLocal := readCounterWithReason(t, placementChosenCount, "arm-picker", placementLocal)
	afterDisagg := readCounterWithReason(t, placementChosenCount, "arm-picker", placementDisaggregated)
	if (afterLocal-beforeLocal)+(afterDisagg-beforeDisagg) != 1 {
		t.Error("exactly one placement must be recorded per decision")
	}
}

// ---------------------------------------------------------------------------
// Handler: profile driving, the result split, PreRequest, ResponseBody
// ---------------------------------------------------------------------------

type stubProfile struct{}

func (stubProfile) Run(context.Context, *fwksched.InferenceRequest, []fwksched.Endpoint) (*fwksched.ProfileRunResult, error) {
	return &fwksched.ProfileRunResult{}, nil
}

// TestPickRunsTheJointProfileOnceThenStops pins the loop contract: an empty map means done,
// and the handler must not re-run the profile on the second invocation.
func TestPickRunsTheJointProfileOnceThenStops(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	profiles := map[string]fwksched.SchedulerProfile{jointProfileName: stubProfile{}}

	first := handler.Pick(context.Background(), testRequest("r1", 100), profiles, map[string]*fwksched.ProfileRunResult{})
	if len(first) != 1 {
		t.Fatalf("first Pick must run one profile, got %d", len(first))
	}
	if _, ok := first[jointProfileName]; !ok {
		t.Errorf("must run the joint profile, got %v", first)
	}

	second := handler.Pick(context.Background(), testRequest("r1", 100), profiles,
		map[string]*fwksched.ProfileRunResult{jointProfileName: {}})
	if len(second) != 0 {
		t.Errorf("second Pick must return an empty map to signal done, got %d", len(second))
	}
}

// TestPickFallsBackToASingleRenamedProfile guards against a renamed profile silently
// producing zero results.
func TestPickFallsBackToASingleRenamedProfile(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	profiles := map[string]fwksched.SchedulerProfile{"renamed": stubProfile{}}
	got := handler.Pick(context.Background(), testRequest("r1", 100), profiles, map[string]*fwksched.ProfileRunResult{})
	if len(got) != 1 {
		t.Fatalf("a single renamed profile must still be run, got %d", len(got))
	}
	if _, ok := got["renamed"]; !ok {
		t.Errorf("expected the renamed profile, got %v", got)
	}
}

// THE MANDATORY SPLIT. The director turns EVERY endpoint in the primary profile's
// TargetEndpoints into a routing destination and comma-joins them, so leaving both picks in
// the primary profile would dispatch each request to BOTH pods -- a silent misroute that
// still returns 200s.
func TestProcessResultsSplitsPicksSoOnlyDecodeIsPrimary(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	decode := testEndpoint("decode-1", bylabel.RoleDecode, "H100_SXM_80GB", nil)
	prefill := testEndpoint("prefill-1", bylabel.RolePrefill, "H100_SXM_80GB", nil)

	joint := map[string]*fwksched.ProfileRunResult{
		jointProfileName: {TargetEndpoints: []fwksched.Endpoint{decode, prefill}},
	}
	result, err := handler.ProcessResults(context.Background(), testRequest("r1", 4000), joint)
	if err != nil {
		t.Fatalf("ProcessResults: %v", err)
	}

	if result.PrimaryProfileName != resultProfileDecode {
		t.Errorf("primary profile = %q, want %q", result.PrimaryProfileName, resultProfileDecode)
	}
	primary := result.ProfileResults[resultProfileDecode]
	if primary == nil || len(primary.TargetEndpoints) != 1 {
		t.Fatalf("the primary profile must carry EXACTLY ONE endpoint or the request is "+
			"dispatched to both pods; got %v", primary)
	}
	if got := primary.TargetEndpoints[0].GetMetadata().ID.Name; got != "decode-1" {
		t.Errorf("primary endpoint = %q, want the decode pick", got)
	}
	prefillResult := result.ProfileResults[resultProfilePrefill]
	if prefillResult == nil || len(prefillResult.TargetEndpoints) != 1 {
		t.Fatalf("the prefill pick must move to its own profile result, got %v", prefillResult)
	}
	if got := prefillResult.TargetEndpoints[0].GetMetadata().ID.Name; got != "prefill-1" {
		t.Errorf("prefill endpoint = %q, want the prefill pick", got)
	}
}

// TestProcessResultsLocalPlacementHasNoPrefillProfile pins that a local win produces no
// prefill entry, which is what makes PreRequest omit the header.
func TestProcessResultsLocalPlacementHasNoPrefillProfile(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	decode := testEndpoint("decode-1", bylabel.RoleDecode, "H100_SXM_80GB", nil)

	result, err := handler.ProcessResults(context.Background(), testRequest("r1", 4000),
		map[string]*fwksched.ProfileRunResult{
			jointProfileName: {TargetEndpoints: []fwksched.Endpoint{decode}},
		})
	if err != nil {
		t.Fatalf("ProcessResults: %v", err)
	}
	if _, ok := result.ProfileResults[resultProfilePrefill]; ok {
		t.Error("a local placement must not produce a prefill profile result")
	}
}

func TestProcessResultsErrorsWhenNothingSelected(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	if _, err := handler.ProcessResults(context.Background(), testRequest("r1", 100),
		map[string]*fwksched.ProfileRunResult{jointProfileName: {}}); err == nil {
		t.Error("expected an error when the argmin selected nothing")
	}
}

// TestProcessResultsTracksTheResident confirms the shadow table is populated at decision
// time, with the prefill endpoint recorded so a remotely-prefilling resident is not charged
// prefill capacity on its decode endpoint.
func TestProcessResultsTracksTheResident(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	decode := testEndpoint("decode-1", bylabel.RoleDecode, "H100_SXM_80GB", nil)
	prefill := testEndpoint("prefill-1", bylabel.RolePrefill, "H100_SXM_80GB", nil)

	if _, err := handler.ProcessResults(context.Background(), testRequest("r1", 4000),
		map[string]*fwksched.ProfileRunResult{
			jointProfileName: {TargetEndpoints: []fwksched.Endpoint{decode, prefill}},
		}); err != nil {
		t.Fatalf("ProcessResults: %v", err)
	}
	if got := handler.table.size(); got != 1 {
		t.Fatalf("shadow table size = %d, want 1", got)
	}
	decodeID := decode.GetMetadata().ID.String()
	prefillID := prefill.GetMetadata().ID.String()

	_, onDecode := handler.table.residentsFor(decodeID)
	if len(onDecode) != 0 {
		t.Error("a remotely-prefilling resident must not occupy prefill capacity on its decode endpoint")
	}
	if got := handler.table.prefillOccupantsFor(prefillID); len(got) != 1 {
		t.Errorf("the prefill pool must hold the occupant, got %d", len(got))
	}
}

// TestProcessResultsSkipsTrackingWithoutARequestID guards against an entry the response
// hooks can never find again, which would be charged as a phantom resident until the sweep.
func TestProcessResultsSkipsTrackingWithoutARequestID(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	decode := testEndpoint("decode-1", bylabel.RoleDecode, "H100_SXM_80GB", nil)
	request := testRequest("", 4000)

	if _, err := handler.ProcessResults(context.Background(), request,
		map[string]*fwksched.ProfileRunResult{
			jointProfileName: {TargetEndpoints: []fwksched.Endpoint{decode}},
		}); err != nil {
		t.Fatalf("ProcessResults: %v", err)
	}
	if got := handler.table.size(); got != 0 {
		t.Errorf("no request ID means no tracking, got %d entries", got)
	}
}

// TestPreRequestSetsPrefillHeaderOnlyWhenDisaggregated pins both halves, including the
// unconditional delete that prevents a stale header from converting a local decision into a
// disaggregated one.
func TestPreRequestSetsPrefillHeaderOnlyWhenDisaggregated(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	prefill := testEndpoint("prefill-1", bylabel.RolePrefill, "H100_SXM_80GB", nil)

	// Disaggregated: the header names the prefill pod as host:port.
	request := testRequest("r1", 4000)
	if err := handler.PreRequest(context.Background(), request, &fwksched.SchedulingResult{
		PrimaryProfileName: resultProfileDecode,
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			resultProfilePrefill: {TargetEndpoints: []fwksched.Endpoint{prefill}},
		},
	}); err != nil {
		t.Fatalf("PreRequest on a disaggregated placement: %v", err)
	}
	if got := request.Headers[routing.PrefillEndpointHeader]; got != "10.0.0.1:8000" {
		t.Errorf("prefill header = %q, want \"10.0.0.1:8000\"", got)
	}

	// Local: a STALE header from an earlier hop must be removed, or a locally-placed
	// request's prefill would be sent to a remote pod.
	stale := testRequest("r2", 4000)
	stale.Headers[routing.PrefillEndpointHeader] = "10.9.9.9:8000"
	if err := handler.PreRequest(context.Background(), stale, &fwksched.SchedulingResult{
		PrimaryProfileName: resultProfileDecode,
		ProfileResults:     map[string]*fwksched.ProfileRunResult{},
	}); err != nil {
		t.Fatalf("a local placement is not an error: %v", err)
	}
	if got, ok := stale.Headers[routing.PrefillEndpointHeader]; ok {
		t.Errorf("a local placement must leave no prefill header, got %q", got)
	}

	// A nil result still clears the stale header.
	stale2 := testRequest("r3", 4000)
	stale2.Headers[routing.PrefillEndpointHeader] = "10.9.9.9:8000"
	if err := handler.PreRequest(context.Background(), stale2, nil); err != nil {
		t.Fatalf("a nil scheduling result is not an error: %v", err)
	}
	if _, ok := stale2.Headers[routing.PrefillEndpointHeader]; ok {
		t.Error("a nil scheduling result must still clear the stale header")
	}

	// A prefill pick carrying no metadata MUST be an error: the header cannot be written,
	// so the request would run locally while every counter records a disaggregated
	// placement. Failing loudly is the whole point of the arm's silent-fallback discipline.
	noMeta := testRequest("r4", 4000)
	err := handler.PreRequest(context.Background(), noMeta, &fwksched.SchedulingResult{
		PrimaryProfileName: resultProfileDecode,
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			resultProfilePrefill: {TargetEndpoints: []fwksched.Endpoint{noMetadataEndpoint{}}},
		},
	})
	if err == nil {
		t.Error("a prefill pick with no metadata must fail rather than silently run locally")
	}
	if _, ok := noMeta.Headers[routing.PrefillEndpointHeader]; ok {
		t.Error("no prefill header should be written when the pick carries no metadata")
	}
}

// noMetadataEndpoint is an Endpoint whose GetMetadata returns nil, which is the state that
// makes the prefill header unwritable.
type noMetadataEndpoint struct{ fwksched.Endpoint }

func (noMetadataEndpoint) GetMetadata() *fwkdl.EndpointMetadata { return nil }

// TestResponseBodyAdvancesResidentAndFeedsNOut walks the whole response lifecycle: chunks
// advance StepsDone and mark the realized first token, and only the terminal chunk folds the
// realized output length into the per-class mean.
func TestResponseBodyAdvancesResidentAndFeedsNOut(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	decode := testEndpoint("decode-1", bylabel.RoleDecode, "H100_SXM_80GB", nil)
	request := testRequest("r1", 4000)

	if _, err := handler.ProcessResults(context.Background(), request,
		map[string]*fwksched.ProfileRunResult{
			jointProfileName: {TargetEndpoints: []fwksched.Endpoint{decode}},
		}); err != nil {
		t.Fatalf("ProcessResults: %v", err)
	}
	decodeID := decode.GetMetadata().ID.String()

	// Mid-stream chunks.
	handler.ResponseBody(context.Background(), request,
		&fwkrc.Response{RequestID: "r1", Usage: fwkrh.Usage{CompletionTokens: 1}}, nil)
	residents, _ := handler.table.residentsFor(decodeID)
	if len(residents) != 1 || !residents[0].TTFTSet {
		t.Fatalf("the first non-zero chunk must mark the realized first token, got %+v", residents)
	}

	handler.ResponseBody(context.Background(), request,
		&fwkrc.Response{RequestID: "r1", Usage: fwkrh.Usage{CompletionTokens: 250}}, nil)

	// Before the terminal chunk the class mean is still the seed.
	if got := handler.policy.nHatFor(sloClassStandard); got != 1 {
		t.Errorf("nHat before completion = %g, want the seed 1", got)
	}

	// Terminal chunk.
	handler.ResponseBody(context.Background(), request,
		&fwkrc.Response{RequestID: "r1", EndOfStream: true, Usage: fwkrh.Usage{CompletionTokens: 300}}, nil)

	if got := handler.table.size(); got != 0 {
		t.Errorf("the completed resident must be removed, got %d entries", got)
	}
	if got := handler.policy.nHatFor(sloClassStandard); got != 300 {
		t.Errorf("nHat after completion = %g, want 300", got)
	}
}

// TestResponseBodyZeroUsageLooksLikeAMissingEngineFlag documents the shape of the
// --enable-force-include-usage failure: every chunk carries zero, so StepsDone stays 0 for
// the request's whole lifetime while the table looks present and correct.
func TestResponseBodyZeroUsageLeavesStepsDoneAtZero(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	decode := testEndpoint("decode-1", bylabel.RoleDecode, "H100_SXM_80GB", nil)
	request := testRequest("r1", 4000)
	if _, err := handler.ProcessResults(context.Background(), request,
		map[string]*fwksched.ProfileRunResult{
			jointProfileName: {TargetEndpoints: []fwksched.Endpoint{decode}},
		}); err != nil {
		t.Fatalf("ProcessResults: %v", err)
	}

	for i := 0; i < 5; i++ {
		handler.ResponseBody(context.Background(), request,
			&fwkrc.Response{RequestID: "r1", Usage: fwkrh.Usage{CompletionTokens: 0}}, nil)
	}
	decodeID := decode.GetMetadata().ID.String()
	residents, prefill := handler.table.residentsFor(decodeID)
	if len(residents) != 0 {
		t.Error("with no usage the request never registers a first token")
	}
	// It stays in the PREFILL population, where the collocated term can still see it --
	// which is why that split exists.
	if len(prefill) != 1 {
		t.Errorf("the request must remain a pre-first-token occupant, got %d", len(prefill))
	}
}

func TestResponseBodyIgnoresUnknownAndNilInputs(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	// None of these may panic.
	handler.ResponseBody(context.Background(), nil, &fwkrc.Response{}, nil)
	handler.ResponseBody(context.Background(), testRequest("r1", 100), nil, nil)
	handler.ResponseBody(context.Background(), testRequest("", 100), &fwkrc.Response{}, nil)
	handler.ResponseBody(context.Background(), testRequest("never-placed", 100),
		&fwkrc.Response{RequestID: "never-placed", EndOfStream: true}, nil)
}

// TestEndpointIDMatchesTheInTreeIdentity pins that the shadow table keys agree with the rest
// of the EPP: EndpointMetadata.ID.String(), not the routing address.
func TestEndpointIDMatchesTheInTreeIdentity(t *testing.T) {
	ep := testEndpoint("decode-1", bylabel.RoleDecode, "H100_SXM_80GB", nil)
	want := ep.GetMetadata().ID.String()
	if got := endpointID(ep); got != want {
		t.Errorf("endpointID = %q, want %q", got, want)
	}
	if got := endpointID(nil); got != "" {
		t.Errorf("a nil endpoint must yield the empty string, got %q", got)
	}
}

// TestSnapshotForReadsTheScrapeTrapsCorrectly pins both silent traps: the usage field is a
// FRACTION despite its name, and KvCacheMaxTokenCapacity is unusable so capacity is derived
// from CacheNumBlocks * CacheBlockSize.
func TestSnapshotForReadsTheScrapeTrapsCorrectly(t *testing.T) {
	handler, _ := buildArm(t, focalConfig())
	metrics := &fwkdl.Metrics{
		RunningRequestsSize:     12,
		WaitingQueueSize:        5,
		KVCacheUsagePercent:     0.25, // a FRACTION, not 25 percent
		CacheBlockSize:          16,
		CacheNumBlocks:          1000,
		KvCacheMaxTokenCapacity: 0, // always 0 at this pin
	}
	snap := handler.snapshotFor("d1", "H100_SXM_80GB", metrics, 16)

	// KV in use = usage * blockSize * numBlocks = 0.25 * 16 * 1000 = 4000.
	if snap.KvTokensInUse != 4000 {
		t.Errorf("KvTokensInUse = %d, want 4000; 40 would mean the fraction was divided by 100",
			snap.KvTokensInUse)
	}
	// Free blocks = (1 - usage) * numBlocks = 750, a FLOOR after truncation (D7).
	if snap.FreeKVBlocks != 750 {
		t.Errorf("FreeKVBlocks = %d, want 750", snap.FreeKVBlocks)
	}
	if snap.BatchSize != 12 || snap.QueueDepth != 5 {
		t.Errorf("batch/queue = %d/%d, want 12/5", snap.BatchSize, snap.QueueDepth)
	}
	// The D1 guard is closed.
	if snap.SchedulerStateObserved {
		t.Error("SchedulerStateObserved must be false at this pin")
	}
	// MaxBatchSize comes from config, since no metric exposes it.
	if snap.MaxBatchSize != 256 {
		t.Errorf("MaxBatchSize = %d, want the configured 256", snap.MaxBatchSize)
	}
}

// TestDecisionInputCloneIsDeep matters because Attributes.Get returns value.Clone() rather
// than the stored pointer, so a shallow clone would hand the picker aliased maps.
func TestDecisionInputCloneIsDeep(t *testing.T) {
	in := &decisionInput{
		requestID:    "r1",
		sloClass:     sloClassStandard,
		inputLen:     4000,
		nowUs:        123,
		apByEndpoint: map[string]int{"d1": 100},
		decodeIDs:    []string{"d1"},
		prefillIDs:   []string{"p1"},
		snapshots:    map[string]Snapshot{"d1": {ID: "d1"}},
	}
	cloned := in.Clone().(*decisionInput)

	in.apByEndpoint["d1"] = 999
	in.decodeIDs[0] = "mutated"
	in.snapshots["d1"] = Snapshot{ID: "mutated"}

	if cloned.apByEndpoint["d1"] != 100 {
		t.Error("apByEndpoint must be deep-copied")
	}
	if cloned.decodeIDs[0] != "d1" {
		t.Error("decodeIDs must be deep-copied")
	}
	if cloned.snapshots["d1"].ID != "d1" {
		t.Error("snapshots must be deep-copied")
	}
	if (*decisionInput)(nil).Clone() != nil {
		t.Error("a nil clone must yield nil")
	}
}

// TestPreferredByScoreIsDeterministicUnderShuffledInput pins the fix for the map-ordering
// hazard directly. runPickerPlugin builds its slice by ranging a map
// (scheduler_profile.go:293-301), so this function must not depend on arrival order.
func TestPreferredByScoreIsDeterministicUnderShuffledInput(t *testing.T) {
	ids := []string{"default/dC", "default/dA", "default/dB"}
	in := &decisionInput{decodeIDs: ids, prefillIDs: []string{"default/pB", "default/pA"}}

	build := func(order []string, prefills []string) []*fwksched.ScoredEndpoint {
		out := make([]*fwksched.ScoredEndpoint, 0, len(order)+len(prefills))
		for _, id := range order {
			name := id[len("default/"):]
			out = append(out, &fwksched.ScoredEndpoint{
				Endpoint: testEndpoint(name, bylabel.RoleDecode, "H100_SXM_80GB", nil), Score: 0,
			})
		}
		for _, id := range prefills {
			name := id[len("default/"):]
			out = append(out, &fwksched.ScoredEndpoint{
				Endpoint: testEndpoint(name, bylabel.RolePrefill, "H100_SXM_80GB", nil), Score: 0,
			})
		}
		return out
	}

	// Every permutation of an all-zero score field must yield the same lowest-ID answer.
	for _, order := range [][]string{
		{"default/dA", "default/dB", "default/dC"},
		{"default/dC", "default/dB", "default/dA"},
		{"default/dB", "default/dC", "default/dA"},
	} {
		d, p := preferredByScore(build(order, []string{"default/pB", "default/pA"}), in)
		if d != "default/dA" {
			t.Errorf("order %v: decode preference = %q, want the lowest ID default/dA", order, d)
		}
		if p != "default/pA" {
			t.Errorf("order %v: prefill preference = %q, want the lowest ID default/pA", order, p)
		}
	}

	// A genuine score still wins over the ID tie-break.
	scored := build([]string{"default/dA", "default/dB", "default/dC"}, nil)
	scored[2].Score = 0.9 // dC
	if d, _ := preferredByScore(scored, in); d != "default/dC" {
		t.Errorf("a real score must beat the ID tie-break, got %q", d)
	}
}

// TestDeclinePathReturnsOnlyDecodeEligibleEndpoints is the fix for the severe D8 defect.
//
// Without it the Filter returned the UNFILTERED candidate set and the picker returned a
// map-random element of it, so in a 1P/2D deployment a decode request reached the prefill
// pod roughly one time in three. Nothing downstream catches that: the director reads
// Address/Port off whatever the primary profile returns and dispatches
// (director.go:486-496), and role eligibility is enforced only by a bylabel filter, which
// a role-blind profile does not include.
func TestDeclinePathReturnsOnlyDecodeEligibleEndpoints(t *testing.T) {
	_, picker := buildArm(t, focalConfig())
	// A request with NO tokenized prompt: the decline path.
	request := &fwksched.InferenceRequest{RequestID: "r1", Body: &fwkrh.InferenceRequestBody{}}

	endpoints := []fwksched.Endpoint{
		testEndpoint("prefill-1", bylabel.RolePrefill, "H100_SXM_80GB", nil),
		testEndpoint("decode-b", bylabel.RoleDecode, "H100_SXM_80GB", nil),
		testEndpoint("decode-a", bylabel.RoleDecode, "A100_SXM_80GB", nil),
		testEndpoint("no-gpu-label", bylabel.RoleDecode, "", nil),
		testEndpoint("no-role", "", "H100_SXM_80GB", nil),
	}

	accepted := picker.Filter(context.Background(), request, endpoints)
	for _, ep := range accepted {
		role := ep.GetMetadata().Labels[bylabel.RoleLabel]
		if !isDecodeRole(role) {
			t.Errorf("the decline path returned %q with role %q; a decode request must never "+
				"be dispatchable to a non-decode endpoint",
				ep.GetMetadata().ID.Name, role)
		}
	}
	if len(accepted) != 2 {
		names := make([]string, 0, len(accepted))
		for _, ep := range accepted {
			names = append(names, ep.GetMetadata().ID.Name)
		}
		t.Fatalf("expected the two well-labelled decode endpoints, got %v", names)
	}

	// And the pick itself is deterministic across repeated identical requests.
	seen := map[string]int{}
	for i := 0; i < 40; i++ {
		scored := make([]*fwksched.ScoredEndpoint, 0, len(accepted))
		for _, ep := range accepted {
			scored = append(scored, &fwksched.ScoredEndpoint{Endpoint: ep, Score: 0})
		}
		result := picker.Pick(context.Background(), scored)
		if result == nil || len(result.TargetEndpoints) != 1 {
			t.Fatal("the decline path must still yield exactly one endpoint")
		}
		seen[result.TargetEndpoints[0].GetMetadata().ID.Name]++
	}
	if len(seen) != 1 {
		t.Errorf("the decline fallback must be deterministic, got %v", seen)
	}
	if _, ok := seen["decode-a"]; !ok {
		t.Errorf("expected the lowest-ID decode-eligible endpoint, got %v", seen)
	}
}

// TestDeclinePathDeclinesWhenNothingIsDecodeEligible guards the empty case rather than
// dispatching to whatever happens to be present.
func TestDeclinePathDeclinesWhenNothingIsDecodeEligible(t *testing.T) {
	_, picker := buildArm(t, focalConfig())
	request := &fwksched.InferenceRequest{RequestID: "r1", Body: &fwkrh.InferenceRequestBody{}}
	prefillOnly := []fwksched.Endpoint{testEndpoint("prefill-1", bylabel.RolePrefill, "H100_SXM_80GB", nil)}

	accepted := picker.Filter(context.Background(), request, prefillOnly)
	if len(accepted) != 0 {
		t.Fatalf("a prefill-only fleet has no decode candidate, got %d", len(accepted))
	}
	scored := []*fwksched.ScoredEndpoint{{Endpoint: prefillOnly[0], Score: 0}}
	if result := picker.Pick(context.Background(), scored); result == nil || len(result.TargetEndpoints) != 0 {
		t.Error("with no decode-eligible endpoint the picker must decline rather than dispatch to prefill")
	}
}

// TestPlacementCounterRecordsWhatTheResultCarries pins issue 8's fix: a disaggregated win
// whose prefill endpoint is absent from the scored set runs LOCALLY, and must be counted
// that way.
func TestPlacementCounterRecordsWhatTheResultCarries(t *testing.T) {
	handler, picker := buildArm(t, focalConfig())
	handler.policy.observeCompletedOutput(sloClassStandard, 300)

	endpoints := []fwksched.Endpoint{
		testEndpoint("decode-1", bylabel.RoleDecode, "H100_SXM_80GB", nil),
		testEndpoint("prefill-1", bylabel.RolePrefill, "H100_SXM_80GB", nil),
	}
	accepted := picker.Filter(context.Background(), testRequest("r1", 4000), endpoints)

	// Hand the picker only the DECODE endpoint, while the attached decision input still
	// describes both -- so the argmin may choose to disaggregate onto an endpoint the
	// scored set does not contain.
	scored := []*fwksched.ScoredEndpoint{}
	for _, ep := range accepted {
		if isDecodeRole(ep.GetMetadata().Labels[bylabel.RoleLabel]) {
			scored = append(scored, &fwksched.ScoredEndpoint{Endpoint: ep, Score: 0})
		}
	}

	beforeLocal := readCounterWithReason(t, placementChosenCount, "arm-picker", placementLocal)
	beforeDisagg := readCounterWithReason(t, placementChosenCount, "arm-picker", placementDisaggregated)
	result := picker.Pick(context.Background(), scored)
	afterLocal := readCounterWithReason(t, placementChosenCount, "arm-picker", placementLocal)
	afterDisagg := readCounterWithReason(t, placementChosenCount, "arm-picker", placementDisaggregated)

	if result == nil || len(result.TargetEndpoints) != 1 {
		t.Fatalf("only the decode endpoint can be returned, got %v", result)
	}
	// Exactly one placement is counted, and it agrees with the single-endpoint result.
	if (afterLocal-beforeLocal)+(afterDisagg-beforeDisagg) != 1 {
		t.Fatal("exactly one placement must be counted")
	}
	if afterDisagg-beforeDisagg != 0 {
		t.Error("a result carrying no prefill endpoint must NOT be counted as disaggregated")
	}
}

// TestDecomposedControlIsDeterministicUnderAnAllZeroScoreField is the reviewer-requested
// case, and it is the one that would have caught a random decode placement.
//
// It drives the FULL picker path -- Filter then Pick -- with no scorers, so every
// ScoredEndpoint.Score is 0 exactly as in the shipped profile, and the endpoints reach the
// picker in map order. With Ablation.Decomposed the enumeration is restricted to ONE
// endpoint, so a nondeterministic preference would place decode at random per request.
func TestDecomposedControlIsDeterministicUnderAnAllZeroScoreField(t *testing.T) {
	cfg := focalConfig()
	cfg.Ablation.Decomposed = true
	handler, picker := buildArm(t, cfg)
	handler.policy.observeCompletedOutput(sloClassStandard, 300)

	// Several decode endpoints, so a random choice would show up quickly.
	endpoints := []fwksched.Endpoint{
		testEndpoint("decode-c", bylabel.RoleDecode, "H100_SXM_80GB", nil),
		testEndpoint("decode-a", bylabel.RoleDecode, "H100_SXM_80GB", nil),
		testEndpoint("decode-d", bylabel.RoleDecode, "A100_SXM_80GB", nil),
		testEndpoint("decode-b", bylabel.RoleDecode, "H100_SXM_80GB", nil),
		testEndpoint("prefill-1", bylabel.RolePrefill, "H100_SXM_80GB", nil),
	}

	seen := map[string]int{}
	for i := 0; i < 60; i++ {
		result := runArm(t, picker, testRequest("r1", 4000), endpoints)
		if result == nil || len(result.TargetEndpoints) == 0 {
			t.Fatal("expected a pick")
		}
		seen[result.TargetEndpoints[0].GetMetadata().ID.Name]++
	}
	if len(seen) != 1 {
		t.Fatalf("the decomposed control placed decode on %d different endpoints across "+
			"identical requests (%v); it must be deterministic or it measures a random-decode "+
			"policy rather than a decomposition", len(seen), seen)
	}
	// With an all-zero score field the preference is the lowest ID, which agrees with the
	// ID-sorted enumeration.
	if _, ok := seen["decode-a"]; !ok {
		t.Errorf("expected the lowest-ID decode endpoint, got %v", seen)
	}
}

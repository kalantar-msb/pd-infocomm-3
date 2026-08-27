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
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// readCounter reads one counter's current value for a plugin identity. Tests use it to
// probe how many candidates the argmin evaluated, since every admission estimate
// increments the substitution counter.
func readCounter(t *testing.T, vec *prometheus.CounterVec, name, typ string) float64 {
	t.Helper()
	return testutil.ToFloat64(vec.WithLabelValues(name, typ))
}

func readCounterWithReason(t *testing.T, vec *prometheus.CounterVec, name, reason string) float64 {
	t.Helper()
	return testutil.ToFloat64(vec.WithLabelValues(name, PickerPluginType, reason))
}

func readGauge(t *testing.T, vec *prometheus.GaugeVec, name, typ string) float64 {
	t.Helper()
	return testutil.ToFloat64(vec.WithLabelValues(name, typ))
}

// TestRegisterMetricsRequiresARegisterer pins the refusal: every degradation this arm
// declares is invisible in the reported latency without its counter, so a nil registerer is
// an error rather than a silent skip.
func TestRegisterMetricsRequiresARegisterer(t *testing.T) {
	if err := registerMetrics(nil); err == nil {
		t.Error("expected an error for a nil registerer")
	}
}

// TestRegisterMetricsIsIdempotent is required rather than defensive: BOTH of this arm's
// plugin registrations call it, so the second caller must not fail.
func TestRegisterMetricsIsIdempotent(t *testing.T) {
	reg := prometheus.NewRegistry()
	if err := registerMetrics(reg); err != nil {
		t.Fatalf("first registration failed: %v", err)
	}
	if err := registerMetrics(reg); err != nil {
		t.Errorf("second registration of the same collectors must succeed, got %v", err)
	}
}

// TestEveryCollectorIsRegistered guards the list: a metric declared but left out of
// collectors() would never be exported, which is the silent failure this whole file exists
// to prevent.
func TestEveryCollectorIsRegistered(t *testing.T) {
	reg := prometheus.NewRegistry()
	if err := registerMetrics(reg); err != nil {
		t.Fatalf("registration failed: %v", err)
	}
	if got := len(collectors()); got != 10 {
		t.Errorf("collectors() returns %d metrics; update this count deliberately when adding one", got)
	}
}

// TestPluginMetricsCarryPluginIdentity pins the labels that make the two arms comparable.
// config.md's D8 requires comparing the two arms' tokenization-failure and substitution
// rates before any result is read, and that is only possible if the counters are
// per-instance.
func TestPluginMetricsCarryPluginIdentity(t *testing.T) {
	armA := newPluginMetrics("arm-a", HandlerPluginType)
	armB := newPluginMetrics("arm-b", HandlerPluginType)

	beforeA := readCounter(t, tokenizationUnavailableCount, "arm-a", HandlerPluginType)
	beforeB := readCounter(t, tokenizationUnavailableCount, "arm-b", HandlerPluginType)

	armA.recordTokenizationUnavailable()
	armA.recordTokenizationUnavailable()
	armB.recordTokenizationUnavailable()

	if got := readCounter(t, tokenizationUnavailableCount, "arm-a", HandlerPluginType) - beforeA; got != 2 {
		t.Errorf("arm-a delta = %g, want 2", got)
	}
	if got := readCounter(t, tokenizationUnavailableCount, "arm-b", HandlerPluginType) - beforeB; got != 1 {
		t.Errorf("arm-b delta = %g, want 1", got)
	}
}

// TestRejectionReasonsAreDistinguishable pins that a rejection is attributable, which is
// what separates a mislabelled endpoint from a routing preference.
func TestRejectionReasonsAreDistinguishable(t *testing.T) {
	m := newPluginMetrics("reasons", PickerPluginType)
	reasons := []string{
		rejectReasonGPUTypeLabelAbsent,
		rejectReasonGPUTypeUnmapped,
		rejectReasonRoleLabelAbsent,
		rejectReasonRoleUnknown,
		rejectReasonMetricsUnavailable,
		rejectReasonBlockSizeDisagrees,
		rejectReasonEndpointMetadataNil,
	}
	before := map[string]float64{}
	for _, r := range reasons {
		before[r] = readCounterWithReason(t, candidateRejectedCount, "reasons", r)
	}
	for _, r := range reasons {
		m.recordCandidateRejected(r)
	}
	for _, r := range reasons {
		got := readCounterWithReason(t, candidateRejectedCount, "reasons", r) - before[r]
		if got != 1 {
			t.Errorf("reason %q delta = %g, want 1", r, got)
		}
	}
}

func TestPlacementCounterDistinguishesLocalFromDisaggregated(t *testing.T) {
	m := newPluginMetrics("placements", PickerPluginType)
	beforeLocal := testutil.ToFloat64(placementChosenCount.WithLabelValues("placements", PickerPluginType, placementLocal))
	beforeDisagg := testutil.ToFloat64(placementChosenCount.WithLabelValues("placements", PickerPluginType, placementDisaggregated))

	m.recordPlacement(true)
	m.recordPlacement(false)
	m.recordPlacement(false)

	gotLocal := testutil.ToFloat64(placementChosenCount.WithLabelValues("placements", PickerPluginType, placementLocal)) - beforeLocal
	gotDisagg := testutil.ToFloat64(placementChosenCount.WithLabelValues("placements", PickerPluginType, placementDisaggregated)) - beforeDisagg
	if gotLocal != 1 || gotDisagg != 2 {
		t.Errorf("placements: local %g (want 1), disaggregated %g (want 2)", gotLocal, gotDisagg)
	}
}

func TestShadowTableGaugeTracksSize(t *testing.T) {
	m := newPluginMetrics("gauge", HandlerPluginType)
	m.setShadowTableSize(17)
	if got := readGauge(t, shadowTableSizeGauge, "gauge", HandlerPluginType); got != 17 {
		t.Errorf("gauge = %g, want 17", got)
	}
	m.setShadowTableSize(0)
	if got := readGauge(t, shadowTableSizeGauge, "gauge", HandlerPluginType); got != 0 {
		t.Errorf("gauge = %g, want 0 -- near zero under load is the signal that the mechanism is inert", got)
	}
}

// strictDecoderForTest mirrors plugin.StrictDecoder for raw parameter bytes, so the overlay
// agreement test exercises the same decode strictness the config loader applies.
func strictDecoderForTest(raw json.RawMessage) *json.Decoder {
	return fwkplugin.StrictDecoder(raw)
}

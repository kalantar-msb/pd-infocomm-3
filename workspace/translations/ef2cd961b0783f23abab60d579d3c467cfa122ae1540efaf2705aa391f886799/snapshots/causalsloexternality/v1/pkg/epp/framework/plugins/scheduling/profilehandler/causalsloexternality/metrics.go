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
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// REQUIRED OBSERVABILITY, not instrumentation.
//
// Every degradation this arm declares is INVISIBLE in the goodput number, and the
// target's own fallbacks are silent. The specification layer lists these counters as a
// REQUIREMENT of the port, and each one below exists because without it a specific
// wrong-but-running state is indistinguishable from correct operation:
//
//   - estimator substitution (D1) -- otherwise the port runs a different TTFT
//     estimator than the one that produced every number in sim_results/, invisibly.
//   - tokenization unavailable (D8) -- a_r is absent and nothing is scorable; and if
//     the RATE differs between the two arms it confounds the comparison the arms
//     exist to make, because they are separate plugin instances.
//   - prefix unobserved -- a_p fell back to the full prompt, re-pricing the
//     local/remote boundary.
//   - candidate rejected, BY REASON -- a mislabelled endpoint leaving the candidate
//     set is otherwise indistinguishable from a routing PREFERENCE.
//   - block-size mismatch -- a config/engine unit disagreement, which leaves a latent
//     unit bug in the admission test.
//   - shadow-table size (gauge) -- near zero under load means the resident
//     populations are empty and every externality is 0.
//   - shadow-table entries reaped -- entries dropped without a final chunk.
//   - placement chosen, local vs disaggregated -- what the argmin actually picked.
//   - argmin duration -- EPP-side work INSIDE the TTFT being measured.
//
// All counters carry plugin_name and plugin_type labels, which is what makes the two
// arms' rates comparable: config.md's D8 requires that comparison before any result is
// read.

// Rejection reasons for candidateRejected. Kept as constants so a new reason cannot be
// introduced as an unlabelled string.
const (
	rejectReasonGPUTypeLabelAbsent  = "gpu_type_label_absent"
	rejectReasonGPUTypeUnmapped     = "gpu_type_unmapped"
	rejectReasonRoleLabelAbsent     = "role_label_absent"
	rejectReasonRoleUnknown         = "role_unknown"
	rejectReasonMetricsUnavailable  = "metrics_unavailable"
	rejectReasonBlockSizeDisagrees  = "block_size_disagrees"
	rejectReasonEndpointMetadataNil = "endpoint_metadata_nil"
)

// Placement outcomes for placementChosen.
const (
	placementLocal         = "local"
	placementDisaggregated = "disaggregated"
)

var (
	estimatorSubstitutionCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_estimator_substitution_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total admission-delay estimates served by the rollforward substitute instead of the "+
					"published scheduler rollout (degradation D1). Non-zero is expected at this pin: the "+
					"engine's wait-queue contents have no route. A zero count means the estimator never ran.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	tokenizationUnavailableCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_tokenization_unavailable_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total requests whose TokenizedPrompt was absent, so the arm made no decision and the "+
					"inherited pick stood (degradation D8). This counts ONLY a nil TokenizedPrompt; it "+
					"CANNOT detect a silently ESTIMATED token count, because the tokenizer's estimate "+
					"backend leaves no provenance marker on the struct. Compare this rate between arms "+
					"before reading any result.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	prefixUnobservedCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_prefix_unobserved_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total per-endpoint prefix reads that missed, so a_p fell back to the full prompt. A miss "+
					"means no information, not a cold cache: charging the full prompt over-prices the "+
					"candidate and re-prices the local/remote boundary.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	candidateRejectedCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_candidate_rejected_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total endpoints removed from the candidate set, by reason. Without this a mislabelled "+
					"endpoint leaving the candidate set is indistinguishable from a routing preference.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "reason"},
	)

	blockSizeMismatchCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_block_size_mismatch_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total endpoints whose scraped CacheBlockSize disagreed with the configured engine block "+
					"size. A disagreement leaves a latent unit bug in the admission test, which compares "+
					"blocks accumulated in one unit against a need expressed in another.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	shadowTableSizeGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_shadow_table_entries",
			Help: metricsutil.HelpMsgWithStability(
				"Current shadow-table entry count. Near zero under load means the resident populations are "+
					"empty and every externality is therefore 0 -- the arm is running but the mechanism is not.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	shadowTableReapedCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_shadow_table_reaped_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total shadow-table entries reaped by the TTL sweep, i.e. requests that stopped producing "+
					"chunks without completing. A rising rate means residents are being charged for requests "+
					"that are gone.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	placementChosenCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_placement_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total placements the joint argmin chose, local versus disaggregated.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "placement"},
	)

	argminDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_argmin_duration_seconds",
			Help: metricsutil.HelpMsgWithStability(
				"Wall time of the joint argmin itself. This is EPP-side work inside the TTFT being "+
					"measured, and it grows with D + D*P candidates.",
				compbasemetrics.ALPHA),
			Buckets: prometheus.ExponentialBuckets(0.00005, 2, 12), // 50us .. ~100ms
		},
		[]string{"plugin_name", "plugin_type"},
	)

	unmappedGPUTypeAtScoreCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "causal_slo_externality_unmapped_gpu_type_at_score_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total theta lookups that missed at scoring time. This should be identically zero: the "+
					"filter rejects an unmapped endpoint before scoring, so a non-zero value is a BUG in "+
					"which some endpoint was priced under zero-valued physics.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)
)

// collectors is the single list registerMetrics walks, so a new metric cannot be
// declared above and silently left unregistered.
func collectors() []prometheus.Collector {
	return []prometheus.Collector{
		estimatorSubstitutionCount,
		tokenizationUnavailableCount,
		prefixUnobservedCount,
		candidateRejectedCount,
		blockSizeMismatchCount,
		shadowTableSizeGauge,
		shadowTableReapedCount,
		placementChosenCount,
		argminDurationSeconds,
		unmappedGPUTypeAtScoreCount,
	}
}

// registerMetrics registers every collector, tolerating a repeat registration of the
// SAME collector.
//
// The idempotence matters here rather than being defensive boilerplate: both of this
// arm's plugin registrations call it, and so does a second arm instance if one is
// configured, so the first caller wins and the rest must not fail.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("metrics registerer is required: every degradation this arm declares is " +
			"invisible in the goodput number without its counter")
	}
	for _, collector := range collectors() {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register causal-slo-externality metric: %w", err)
		}
	}
	return nil
}

// pluginMetrics binds the counters to one plugin instance's identity, so the two arms
// can be told apart in the same EPP.
type pluginMetrics struct {
	name string
	typ  string
}

func newPluginMetrics(name, typ string) *pluginMetrics {
	return &pluginMetrics{name: name, typ: typ}
}

func (m *pluginMetrics) recordEstimatorSubstitution() {
	estimatorSubstitutionCount.WithLabelValues(m.name, m.typ).Inc()
}

func (m *pluginMetrics) recordTokenizationUnavailable() {
	tokenizationUnavailableCount.WithLabelValues(m.name, m.typ).Inc()
}

func (m *pluginMetrics) recordPrefixUnobserved() {
	prefixUnobservedCount.WithLabelValues(m.name, m.typ).Inc()
}

func (m *pluginMetrics) recordCandidateRejected(reason string) {
	candidateRejectedCount.WithLabelValues(m.name, m.typ, reason).Inc()
}

func (m *pluginMetrics) recordBlockSizeMismatch() {
	blockSizeMismatchCount.WithLabelValues(m.name, m.typ).Inc()
}

func (m *pluginMetrics) setShadowTableSize(n int) {
	shadowTableSizeGauge.WithLabelValues(m.name, m.typ).Set(float64(n))
}

func (m *pluginMetrics) addShadowTableReaped(n int) {
	shadowTableReapedCount.WithLabelValues(m.name, m.typ).Add(float64(n))
}

func (m *pluginMetrics) recordPlacement(local bool) {
	placement := placementDisaggregated
	if local {
		placement = placementLocal
	}
	placementChosenCount.WithLabelValues(m.name, m.typ, placement).Inc()
}

func (m *pluginMetrics) observeArgminDuration(seconds float64) {
	argminDurationSeconds.WithLabelValues(m.name, m.typ).Observe(seconds)
}

func (m *pluginMetrics) recordUnmappedGPUTypeAtScore(string) {
	unmappedGPUTypeAtScoreCount.WithLabelValues(m.name, m.typ).Inc()
}

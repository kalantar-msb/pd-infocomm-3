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
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// REQUIRED OBSERVABILITY, not instrumentation.
//
// Every degradation this arm declares is INVISIBLE in the reported latency, and the
// target's own fallbacks are silent. The specification layer lists these counters as a
// REQUIREMENT of the port, and each one exists because without it a specific
// wrong-but-running state is indistinguishable from correct operation. The two that the
// specification names explicitly, and that gate reading any result at all, are the
// estimator substitution (D1) and the tokenization decline (D8).
//
// # WHY THIS ARM DECLARES ITS OWN METRIC FAMILIES RATHER THAN SHARING THE FOCAL ARM'S
//
// The focal arm's collectors are named `causal_slo_externality_*` and are package-private
// to `causalsloexternality`. Its registerMetrics tolerates a repeat registration only of
// the IDENTICAL collector object (it compares AlreadyRegisteredError.ExistingCollector
// against its own pointer), so a second package declaring same-named collectors would
// fail that comparison and return an error from the factory -- which fails EPP startup
// outright, since both arms register into one registry.
//
// The focal arm's translation is finalized and must not be edited, and its collectors
// cannot be imported. So this arm carries a parallel family, `least_ttft_joint_*`, with
// the SAME label set and the SAME semantics.
//
// THE CONSEQUENCE IS DISCLOSED RATHER THAN GLOSSED: config.md's D8 requires comparing the
// two arms' tokenization-decline RATES before any result is read, and D1 requires
// comparing their estimator regimes. With one shared family that would be a single query
// selecting on plugin_type; with two families it is two queries, summed per arm and then
// divided. The comparison is equally available and equally exact -- it is not one
// selector -- and the metric names, label sets, help semantics, and histogram buckets are
// deliberately identical so the two families line up term for term.
//
// Every counter carries plugin_name and plugin_type labels, which is what distinguishes
// this arm's two plugin instances (handler and picker) from each other.

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
	rejectReasonPrefillPickMissing  = "prefill_pick_missing_from_scored_set"
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
			Name:      "least_ttft_joint_estimator_substitution_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total admission-delay estimates served by the rollforward substitute instead of the "+
					"published scheduler rollout (degradation D1). Non-zero is expected at this pin: the "+
					"engine's wait-queue contents have no route. A zero count means the estimator never "+
					"ran. For THIS arm the rollout was the LIVE path upstream, not a fallback, so every "+
					"number in sim_results/ for this arm was produced by an estimator that cannot run "+
					"here. Compare this against the focal arm's counter: the comparison between the two "+
					"arms is fair only where both show the same estimator regime, because D1's direction "+
					"for this objective is load- and pool-dependent rather than a single sign.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	tokenizationUnavailableCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "least_ttft_joint_tokenization_unavailable_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total requests whose TokenizedPrompt was absent, so the arm made no decision "+
					"(degradation D8). This counts ONLY a nil TokenizedPrompt; it CANNOT detect a "+
					"silently ESTIMATED token count, because the tokenizer's estimate backend leaves no "+
					"provenance marker on the struct. On this path the arm falls back to a THIRD policy "+
					"-- neither arm -- so if the RATE differs between the two arms it confounds the very "+
					"comparison the arms exist to make. It is likeliest on long prompts, exactly where "+
					"local-versus-remote is most contested. Compare this rate against the focal arm's "+
					"before reading any result.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	prefixUnobservedCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "least_ttft_joint_prefix_unobserved_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total per-endpoint prefix reads that missed, so a_p fell back to the full prompt. A miss "+
					"means no information, not a cold cache: charging the full prompt over-prices the "+
					"candidate and re-prices the local/remote boundary. Both arms must read the SAME "+
					"prefix producer instance, so a divergence in this rate is evidence the two producer "+
					"bindings differ.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	candidateRejectedCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "least_ttft_joint_candidate_rejected_total",
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
			Name:      "least_ttft_joint_block_size_mismatch_total",
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
			Name:      "least_ttft_joint_shadow_table_entries",
			Help: metricsutil.HelpMsgWithStability(
				"Current shadow-table entry count. Near zero under load means the resident populations are "+
					"empty, so S_pf is 0 on every candidate and every remaining-steps estimate collapses "+
					"to its floor of 1 -- the arm is running but it is pricing an idle fleet.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type"},
	)

	shadowTableReapedCount = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "least_ttft_joint_shadow_table_reaped_total",
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
			Name:      "least_ttft_joint_placement_total",
			Help: metricsutil.HelpMsgWithStability(
				"Total placements the joint argmin chose, local versus disaggregated. This is the split "+
					"that D1 and D5 move, so it is the primary read for both.",
				compbasemetrics.ALPHA),
		},
		[]string{"plugin_name", "plugin_type", "placement"},
	)

	argminDurationSeconds = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "least_ttft_joint_argmin_duration_seconds",
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
			Name:      "least_ttft_joint_unmapped_gpu_type_at_score_total",
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
// arm's plugin registrations call it, and so does a second instance of this arm if one is
// configured, so the first caller wins and the rest must not fail.
//
// It does NOT tolerate a DIFFERENT collector under the same name, which is exactly the
// check that would fire if this arm's metric names were ever changed to collide with the
// focal arm's -- see the family-naming note at the top of this file. That failure would
// surface as a factory error at startup rather than as silently merged series.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return errors.New("metrics registerer is required: every degradation this arm declares is " +
			"invisible in the reported latency without its counter")
	}
	for _, collector := range collectors() {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register least-ttft-joint metric: %w", err)
		}
	}
	return nil
}

// pluginMetrics binds the counters to one plugin instance's identity, so this arm's two
// registrations can be told apart in the same EPP.
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

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
	"sort"
)

// estimatorRollforward is the only admission estimator selectable at this pin.
//
// config.md section 9.1 declares it, and it is the D1 SUBSTITUTION rather than the
// published estimator. The other three named upstream are unreachable here:
// `waiting` reads a per-endpoint work account that does not exist, `little` reads an
// arrival rate that has no route, and the published `scheduler-rollout` needs the
// engine's ordered wait queue, its current grants, and its step start instant --
// against which the target exposes Metrics.WaitingQueueSize, one integer.
const estimatorRollforward = "rollforward"

// Workload names for the tau triple selector. config.md section 6.
const (
	workloadInteractive  = "interactive"
	workloadReasoning    = "reasoning"
	workloadDeepResearch = "deepResearch"
)

// SLOTargets is one workload's tau triple, in microseconds.
//
// TauITLUs is CARRIED BUT NEVER READ BY THE ROUTING KERNEL. The composite kernel
// reads only tau_ttft and tau_e2e; mean ITL is an evaluation gate on reported
// goodput, not a routing term. It is present because muDNom consumes it if the
// capacity account is re-enabled, and because omitting a threshold config.md states
// would make this config look like it disagreed with section 6.
type SLOTargets struct {
	TauTTFTUs int64 `json:"tauTtftUs"`
	TauITLUs  int64 `json:"tauItlUs"`
	TauE2EUs  int64 `json:"tauE2eUs"`
}

// Ablation holds the config switches of config.md section 10.
//
// Each is a switch rather than a compile-time constant because reproducing an
// ablation on the cluster must need no code change.
//
// The decomposition comparison is the cheapest on-cluster check that the joint shape
// survived the port -- but flipping Decomposed alone does not perform it. See that field's
// doc for what it does and does not reproduce.
type Ablation struct {
	// NoCapacity disables the capacity drift term. The focal arm sets it true --
	// the `causal_externality_no_capacity_v8` controller -- so the operative
	// objective is total = V * (externality - ownGood).
	NoCapacity bool `json:"noCapacity"`
	// NoExternality removes the resident externality. The focal arm keeps the term:
	// removing it costs +0.0154 equal-cell mean goodput, CI [+0.0112, +0.0195]. It
	// is the only term with a measured, supported effect -- it IS the mechanism.
	NoExternality bool `json:"noExternality"`
	// NoOwnGood removes the arriving request's own good. The focal arm keeps the
	// term DESPITE NO MEASURED BENEFIT: removing it GAINS 0.0016 with a CI that
	// crosses zero, [-0.0053, +0.0022].
	NoOwnGood bool `json:"noOwnGood"`
	// OccupancyCapacity selects the occupancy-time capacity account over the
	// quadratic-drift one. Doubly dead while NoCapacity is true.
	OccupancyCapacity bool `json:"occupancyCapacity"`
	// Decomposed restricts the decode enumeration to a single endpoint -- the one
	// preferredByScore names -- instead of enumerating every decode candidate.
	//
	// WHAT IT REPRODUCES DEPENDS ENTIRELY ON THE PROFILE, and as this arm ships it does
	// NOT reproduce config.md section 10's `decode-first full` ablation. That ablation
	// fixes decode to a LOAD-SHAPED scorer's choice. This arm's committed profile carries
	// NO scorers, so every ScoredEndpoint.Score is 0 and the "preference" degenerates to
	// the lowest-ID decode endpoint: a fixed, arbitrary pod, not a load-shaped decision.
	// Flipping this switch alone therefore measures "always decode on one particular pod",
	// which is a different and weaker control than the ablation it resembles.
	//
	// TO ACTUALLY REPRODUCE THE SECTION 10 ABLATION the operator must also attach the
	// baseline's load-shaped scorers to the profile (active-request-scorer,
	// prefix-cache-scorer, and the rest of the baseline's decode profile), so that
	// preferredByScore has a real preference to read. That is safe despite the [0,1] clamp
	// on scorer contributions: this picker ignores ScoredEndpoint.Score when computing J,
	// so the clamp never touches the objective -- the scores would supply only the
	// tie-break and this restriction.
	//
	// The switch is kept reachable because config.md section 10 makes the decomposition
	// comparison the cheapest on-cluster check that the joint shape survived the port
	// (+0.0485 equal-cell mean goodput in favour of the joint shape, CI [+0.0305,
	// +0.0665]). It is documented this precisely because a control that silently measured
	// something else would look like it confirmed the pre-registered expectation.
	Decomposed bool `json:"decomposed"`
}

// EngineAgreement carries the three engine settings no metric exposes usefully.
//
// Each MUST equal the engine flag of the same meaning, and BlockSize is additionally
// validated against the scraped Metrics.CacheBlockSize at decision time with a loud
// failure on disagreement: upstream has ONE block size, and a port carrying two that
// silently convert leaves a latent unit bug in the admission test, which compares
// blocks accumulated in one unit against a need expressed in another.
type EngineAgreement struct {
	ChunkTokens  int `json:"chunkTokens"`  // = max_num_batched_tokens, signal 22
	BlockSize    int `json:"blockSize"`    // = block_size, signal 23
	MaxBatchSize int `json:"maxBatchSize"` // = max_num_seqs, signal 24
}

// Signals names the label keys and producer bindings the arm reads.
type Signals struct {
	// GPUTypeLabelKey is the pod label whose value selects theta -- signal 8. Read
	// from Endpoint.GetMetadata().Labels, never from a node object: plugins see
	// endpoints, not nodes. An endpoint whose label is absent, or whose value is not
	// a key of CoeffsByGPUType, is REJECTED and the rejection counted -- never
	// defaulted.
	GPUTypeLabelKey string `json:"gpuTypeLabelKey"`
	// PrefixMatchInfoProducerName binds the a_p read BY PRODUCER NAME -- signal 11.
	// A data key's identity is "<DataType>/<ProducerName>", so an unbound read can
	// silently become a second, differently-configured signal. Both arms must be
	// given the same value so they read one signal.
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName"`
	// TokenizedPromptProducerName binds the a_r read BY PRODUCER NAME -- signal 10,
	// same reasoning.
	TokenizedPromptProducerName string `json:"tokenizedPromptProducerName"`
}

// ShadowTable configures the EPP-side resident index -- signals 12-18.
type ShadowTable struct {
	// EntryTTLSeconds bounds how long an entry with no further chunks is charged as
	// a resident. THE SWEEP IS REQUIRED, NOT HYGIENE: requests terminate without a
	// final chunk (client disconnect, upstream error) and the ResponseBodyProcessor
	// signature carries no error or termination state, so a disconnect is
	// indistinguishable from a completion. Without the sweep those entries are
	// charged forever and permanently inflate every externality on that endpoint.
	EntryTTLSeconds int `json:"entryTtlSeconds"`
	// SweepIntervalSeconds is how often the reaper runs.
	SweepIntervalSeconds int `json:"sweepIntervalSeconds"`
	// ResidentPrefillTokensCap caps S_pf both per occupant and in total --
	// degradation D4. Still an over-estimate, biasing toward remote.
	ResidentPrefillTokensCap int64 `json:"residentPrefillTokensCap"`
}

// Transfer is the KV-transfer price model -- degradation D5, config.md section 7.
//
// UNMEASURED ON THIS TARGET. c_xfer is the only size-dependent price of going
// remote and it enters the objective at exactly one place, so a wrong value
// mis-prices SYSTEMATICALLY rather than noisily.
type Transfer struct {
	SizeAware             bool    `json:"sizeAware"`
	XferBaseUs            float64 `json:"xferBaseUs"`
	XferBandwidthGBps     float64 `json:"xferBandwidthGbps"`
	KVBytesPerTokenPerGPU float64 `json:"kvBytesPerTokenPerGpu"`
	// FlatCXferUs is the flat cost used when SizeAware is false. A DISTINCT field
	// from XferBaseUs, which is only the additive base of the size-aware form.
	// Conflating them makes the flat path unrepresentable and silently reprices
	// every disaggregated candidate if SizeAware is ever turned off.
	FlatCXferUs int64 `json:"flatCXferUs"`
}

// Capacity is the capacity account's configuration. DEAD STATE while
// Ablation.NoCapacity is true, which is the focal arm's setting.
//
// Declared rather than omitted because a HALF-specified subsystem is worse than an
// absent one: the read side would look complete. A port re-enabling the capacity
// term must additionally supply a monotonic clock and per-endpoint drain timestamps.
type Capacity struct {
	TauRefUs         int64 `json:"tauRefUs"`
	NomPrefillTokens int   `json:"nomPrefillTokens"`
	ReferenceBatch   int   `json:"referenceBatch"`
}

// Config is the arm's full configuration. Provenance for every value is in
// config.md sections 9.1 (shared) and 9.2 (focal).
//
// Decoded with plugin.StrictDecoder, so DisallowUnknownFields is set and a
// misspelled key is a STARTUP ERROR rather than a silently ignored one. There are
// deliberately NO defaults for the physics: every field that changes a decision must
// be stated in the scenario overlay, so a missing value fails loudly instead of
// quietly pricing candidates under invented physics.
type Config struct {
	// V is Neely's penalty/stability tradeoff knob. config.md section 9.2: 8.
	//
	// KEPT, NOT FOLDED AWAY, even though with capacity disabled it is a common
	// positive multiplier that cannot change the argmin: the ablation cohort's
	// validity gate asserts score = 8 * (externality - ownGood) EXACTLY, and a port
	// that folds V away cannot reproduce that check.
	V float64 `json:"v"`

	Ablation Ablation `json:"ablation"`

	// ActiveWorkload selects one triple from WorkloadTargets.
	//
	// Every cohort in all three workloads declares slo_class: standard, so the class
	// is ONE CONSTANT STRING across the whole grid and a per-request lookup by SLO
	// class CANNOT select the triple -- the simulation varies tau per invocation.
	// All three are carried and one is selected; run one assemble per workload.
	ActiveWorkload  string                `json:"activeWorkload"`
	WorkloadTargets map[string]SLOTargets `json:"workloadTargets"`

	Engine EngineAgreement `json:"engine"`

	Signals Signals `json:"signals"`

	// AdmissionEstimator must be "rollforward" -- see estimatorRollforward.
	AdmissionEstimator string `json:"admissionEstimator"`

	Transfer Transfer `json:"transfer"`

	// CoeffsByGPUType is theta keyed by the Signals.GPUTypeLabelKey VALUE.
	//
	// THERE IS NO FALLBACK ENTRY, ON PURPOSE. An unmapped label must have already
	// caused the endpoint to be rejected; a silent default here would price the
	// endpoint under the wrong physics on every decision. The two GPU types in the
	// campaign fleet differ by a factor of 1.539 in the per-iteration intercept, on
	// every iteration regardless of KV state.
	CoeffsByGPUType map[string]Coeffs `json:"coeffsByGpuType"`

	ShadowTable ShadowTable `json:"shadowTable"`

	Capacity Capacity `json:"capacity"`

	// OutputTokenProcessingUs is the client-visible per-token post-processing
	// latency (streaming detokenization), added explicitly to every TTFT projection
	// because it sits outside the calibrated theta -- config.md section 5 signal 25.
	//
	// config.md gives no value for it. It is NOT decision-neutral despite being
	// added to both the local and disaggregated projections: both feed tHat, which
	// enters ownGood through a sigmoid, so a common additive constant still moves
	// the two candidates' values by different amounts.
	OutputTokenProcessingUs float64 `json:"outputTokenProcessingUs"`
}

// activeTargets returns the selected tau triple.
//
// Callers may assume the key exists: validate rejects an unresolvable selection at
// startup, so this cannot silently yield a zero triple.
func (c *Config) activeTargets() SLOTargets {
	return c.WorkloadTargets[c.ActiveWorkload]
}

// validate enforces every invariant whose violation would otherwise be SILENT.
//
// Each check below corresponds to a failure mode named in config.md or in the
// specification layer, where the consequence is a policy that keeps running and
// keeps reporting goodput while computing something other than the published
// objective.
func (c *Config) validate() error {
	if c.V <= 0 {
		return fmt.Errorf("v must be > 0, got %g: it is a common positive multiplier on the "+
			"objective and the ablation validity gate asserts score = V*(externality - ownGood)", c.V)
	}

	// --- The tau triple. An unresolvable selection FLATTENS the policy rather
	// than loosening it, so it must fail here.
	if c.ActiveWorkload == "" {
		return fmt.Errorf("activeWorkload is required: the SLO class is the constant string "+
			"%q across every cohort of all three workloads, so a per-request lookup cannot "+
			"select the tau triple and one workload must be named in config", sloClassStandard)
	}
	if len(c.WorkloadTargets) == 0 {
		return errors.New("workloadTargets is required and must carry the tau triples from config.md section 6")
	}
	targets, ok := c.WorkloadTargets[c.ActiveWorkload]
	if !ok {
		return fmt.Errorf("activeWorkload %q is not a key of workloadTargets (have %v): an "+
			"unresolvable selection would leave a zero triple, which does not loosen the policy "+
			"but flattens it -- every resident charge becomes 1.0 - 1.0 = 0, the externality is 0 "+
			"on every candidate, and the argmin degenerates to enumeration order",
			c.ActiveWorkload, sortedKeys(c.WorkloadTargets))
	}
	if targets.TauTTFTUs <= 0 {
		return fmt.Errorf("workloadTargets[%q].tauTtftUs must be > 0, got %d", c.ActiveWorkload, targets.TauTTFTUs)
	}
	if targets.TauE2EUs <= 0 {
		return fmt.Errorf("workloadTargets[%q].tauE2eUs must be > 0, got %d: because a resident's "+
			"realized TTFT is placement-invariant it factors out of the charge, so a zero E2E target "+
			"makes the ENTIRE decode-side externality identically zero and degenerates this arm to "+
			"its own-good term alone", c.ActiveWorkload, targets.TauE2EUs)
	}

	// --- Engine agreement.
	if c.Engine.ChunkTokens <= 0 {
		return fmt.Errorf("engine.chunkTokens must be > 0 and equal the engine's max_num_batched_tokens, got %d", c.Engine.ChunkTokens)
	}
	if c.Engine.BlockSize <= 0 {
		return fmt.Errorf("engine.blockSize must be > 0 and equal the engine's block_size, got %d", c.Engine.BlockSize)
	}
	if c.Engine.MaxBatchSize <= 0 {
		return fmt.Errorf("engine.maxBatchSize must be > 0 and equal the engine's max_num_seqs, got %d", c.Engine.MaxBatchSize)
	}

	// --- Signals. An EMPTY producer name is worse than a wrong one, so both are
	// required rather than defaulted.
	//
	// DataKey.WithNonEmptyProducerName("") returns the key UNCHANGED (datakey.go:36-43),
	// leaving producerName at the plugin TYPE string. A WRONG explicit name errors at
	// startup (data_graph.go:94-97); an EMPTY one silently resolves to the type string,
	// and if the operator declared their producer under any other instance name the
	// required key looks unproduced -- so CreateMissingDataProducers auto-instantiates a
	// SECOND, nil-parameter producer (data_graph.go:72-118) which this arm then reads.
	// That yields two indexers with two block sizes, bypasses the operator's tuning, and
	// can bind the two arms to different producers -- the exact hazard config.md signal 11
	// exists to prevent.
	//
	// It is required even though PluginSpec.Name defaults to Type, which makes an empty
	// value accidentally work in the shipped overlay: that is what would let this pass a
	// smoke test and break on a renamed producer.
	if c.Signals.PrefixMatchInfoProducerName == "" {
		return errors.New("signals.prefixMatchInfoProducerName is required: an empty value leaves " +
			"the data key at its type-string default, so a producer declared under any other " +
			"instance name causes a second, nil-parameter producer to be auto-created and read " +
			"instead -- two indexers, two block sizes, and the two arms potentially bound to " +
			"different prefix signals")
	}
	if c.Signals.TokenizedPromptProducerName == "" {
		return errors.New("signals.tokenizedPromptProducerName is required: same reason as " +
			"signals.prefixMatchInfoProducerName -- an empty binding can silently resolve to a " +
			"second, auto-created token producer")
	}
	if c.Signals.GPUTypeLabelKey == "" {
		return errors.New("signals.gpuTypeLabelKey is required: theta is keyed by pod label, and a " +
			"defaulted GPU type is wrong on EVERY decision because heterogeneity rides the " +
			"per-iteration intercept (the two fleet types differ by a factor of 1.539)")
	}

	// --- The estimator. D1's substitution is acceptable; a silent switch is not.
	if c.AdmissionEstimator != estimatorRollforward {
		return fmt.Errorf("admissionEstimator must be %q, got %q: `waiting` reads a per-endpoint "+
			"work account that does not exist at this pin, `little` reads an arrival rate with no "+
			"route, and the published scheduler-rollout needs the engine's ordered wait queue, its "+
			"current grants, and its step start instant -- against which the target exposes "+
			"Metrics.WaitingQueueSize, one integer",
			estimatorRollforward, c.AdmissionEstimator)
	}

	// --- Transfer model, degradation D5.
	if c.Transfer.SizeAware {
		if c.Transfer.XferBandwidthGBps <= 0 {
			return fmt.Errorf("transfer.xferBandwidthGbps must be > 0 when transfer.sizeAware is true, got %g",
				c.Transfer.XferBandwidthGBps)
		}
		if c.Transfer.KVBytesPerTokenPerGPU <= 0 {
			return fmt.Errorf("transfer.kvBytesPerTokenPerGpu must be > 0 when transfer.sizeAware is true, got %g: "+
				"leaving it 0 makes transferBytes 0, so every request is charged a flat xferBaseUs -- a "+
				"4k-token prompt should be charged ~13500 us, roughly 270x under-priced, with the error "+
				"growing linearly in prompt length", c.Transfer.KVBytesPerTokenPerGPU)
		}
	} else if c.Transfer.FlatCXferUs < 0 {
		return fmt.Errorf("transfer.flatCXferUs must be >= 0, got %d", c.Transfer.FlatCXferUs)
	}

	// --- theta. Reject an empty table and validate every entry.
	if len(c.CoeffsByGPUType) == 0 {
		return errors.New("coeffsByGpuType is required and must carry one entry per GPU-type label " +
			"value in the fleet (config.md section 4); there is deliberately no fallback entry, " +
			"because a silent default would price an endpoint under the wrong physics on every decision")
	}
	for _, gpuType := range sortedKeys(c.CoeffsByGPUType) {
		if err := c.CoeffsByGPUType[gpuType].validate(); err != nil {
			return fmt.Errorf("coeffsByGpuType[%q]: %w", gpuType, err)
		}
	}

	// --- Shadow table. A non-positive TTL disables the sweep, which is the failure
	// the sweep exists to prevent.
	if c.ShadowTable.EntryTTLSeconds <= 0 {
		return fmt.Errorf("shadowTable.entryTtlSeconds must be > 0, got %d: without the sweep, "+
			"requests that terminate without a final chunk are charged as residents forever and "+
			"permanently inflate every externality on their endpoint", c.ShadowTable.EntryTTLSeconds)
	}
	if c.ShadowTable.SweepIntervalSeconds <= 0 {
		return fmt.Errorf("shadowTable.sweepIntervalSeconds must be > 0, got %d", c.ShadowTable.SweepIntervalSeconds)
	}
	if c.ShadowTable.ResidentPrefillTokensCap <= 0 {
		return fmt.Errorf("shadowTable.residentPrefillTokensCap must be > 0, got %d", c.ShadowTable.ResidentPrefillTokensCap)
	}
	// config.md signal 14 caps S_pf at max_num_batched_tokens, so the two settings are one
	// fact expressed twice and must agree. A larger cap would let the shadow sum claim more
	// prefill work than the engine can schedule in a step.
	if c.ShadowTable.ResidentPrefillTokensCap != int64(c.Engine.ChunkTokens) {
		return fmt.Errorf("shadowTable.residentPrefillTokensCap (%d) must equal engine.chunkTokens (%d): "+
			"S_pf is what is being prefilled in ONE engine step, so its cap is the engine's "+
			"max_num_batched_tokens", c.ShadowTable.ResidentPrefillTokensCap, c.Engine.ChunkTokens)
	}

	// --- Capacity account. Only checked when it is actually live.
	if !c.Ablation.NoCapacity {
		if c.Capacity.TauRefUs <= 0 {
			return fmt.Errorf("capacity.tauRefUs must be > 0 when the capacity term is enabled "+
				"(ablation.noCapacity is false), got %d", c.Capacity.TauRefUs)
		}
		if c.Capacity.NomPrefillTokens <= 0 {
			return fmt.Errorf("capacity.nomPrefillTokens must be > 0 when the capacity term is enabled, got %d",
				c.Capacity.NomPrefillTokens)
		}
		if c.Ablation.OccupancyCapacity && c.Capacity.ReferenceBatch <= 0 {
			return fmt.Errorf("capacity.referenceBatch must be > 0 when ablation.occupancyCapacity is true, got %d",
				c.Capacity.ReferenceBatch)
		}
		if targets.TauITLUs <= int64(maxCoeffAlphaD(c.CoeffsByGPUType)) {
			return fmt.Errorf("workloadTargets[%q].tauItlUs (%d) must exceed the largest alphaD (%g) "+
				"when the capacity term is enabled: muDNom is only meaningful above the per-iteration intercept",
				c.ActiveWorkload, targets.TauITLUs, maxCoeffAlphaD(c.CoeffsByGPUType))
		}
	}

	if c.OutputTokenProcessingUs < 0 {
		return fmt.Errorf("outputTokenProcessingUs must be >= 0, got %g", c.OutputTokenProcessingUs)
	}

	return nil
}

// maxCoeffAlphaD returns the largest decode intercept across configured GPU types.
func maxCoeffAlphaD(byType map[string]Coeffs) float64 {
	var out float64
	for _, c := range byType {
		if c.AlphaD > out {
			out = c.AlphaD
		}
	}
	return out
}

// sortedKeys returns map keys in deterministic order, so error messages and
// validation order do not vary between runs.
func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

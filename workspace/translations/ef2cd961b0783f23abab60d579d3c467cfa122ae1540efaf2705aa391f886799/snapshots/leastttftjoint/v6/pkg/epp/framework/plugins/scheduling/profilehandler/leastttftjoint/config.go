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
//
// D1 IS INHERITED IN FULL FROM THE FOCAL ARM, BUT ITS DIRECTION IS NOT. For the focal
// arm the substitution biases toward remote prefill. For THIS objective the sign is
// load- and pool-dependent, because the two placements join their clocks differently:
// local ADDS tAdmD with slope 1 while disaggregated takes max(remoteLead, tAdmD). When
// tAdmD dominates that max, understating it is charged identically to both placements
// and CANCELS; when remoteLead dominates, the disaggregated projection is insensitive
// to tAdmD altogether, so understating it lowers ONLY the local candidate -- biasing
// toward LOCAL. The toward-remote direction survives only through the tAdmP channel,
// and only while remoteLead dominates.
//
// That matters for the ATTRIBUTION ARGUMENT and not merely for this arm's accuracy: it
// would be convenient to say both arms inherit D1 identically so the comparison stays
// fair, but a load-dependent direction cannot support that claim. Under decode-side
// congestion -- the regime the high-load and burst cohorts target -- D1 shifts the two
// arms' local/remote splits by different amounts. Treat the comparison as fair only
// where the substitution counter shows the same estimator regime on both arms.
//
// AND IT MOVES THIS ARM BETWEEN TWO REGIMES, NOT MERELY ACROSS A THRESHOLD. Because the
// disaggregated projection joins its clocks with max(remoteLead, tAdmD), tAdmD decides WHICH
// regime the arm is in:
//
//	tAdmD dominates   -> the prefill choice cannot move J at all. The arm is
//	                     PREFILL-INDIFFERENT and the ascending-ID tie-break decides placement.
//	remoteLead dominates -> the arm is PREFILL-SENSITIVE and prices prefill hardware.
//
// tAdmD is exactly the term the rollforward fallback understates, so D1 can carry the arm from
// one regime into the other rather than shifting a boundary within one. Under decode-side
// congestion -- the high-load and burst cohorts -- that is a difference in KIND between what the
// simulation measured and what the target does.
//
// NO DIRECTION IS CLAIMED FOR THIS, and none should be inferred: which regime the published
// estimator would have selected on a given cell is not recoverable from the target, and the two
// regimes do not order by goodput. It is recorded because it bears on the ATTRIBUTION argument
// -- the two arms' local/remote splits can diverge for a reason unrelated to the objective --
// and because a reader comparing placement counters between arms needs to know that a flat
// prefill distribution may mean indifference rather than a preference.
const estimatorRollforward = "rollforward"

// EngineAgreement carries the three engine settings no metric exposes usefully.
//
// Each MUST equal the engine flag of the same meaning, and BlockSize is additionally
// validated against the scraped Metrics.CacheBlockSize at decision time with a loud
// failure on disagreement: upstream has ONE block size, and a port carrying two that
// silently convert leaves a latent unit bug in the admission test, which compares
// blocks accumulated in one unit against a need expressed in another.
//
// Identical values to the focal arm (config.md section 9.1) -- if the two arms carried
// different engine agreement the comparison would not be a comparison.
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
	//
	// This arm is hardware-aware BY CONSTRUCTION and deliberately so: each side of
	// every candidate prices with its own candidate's theta. A hardware-BLIND
	// least-TTFT baseline would lose to the focal arm partly because it cannot see the
	// fleet, which would confound the mechanism with hardware awareness.
	GPUTypeLabelKey string `json:"gpuTypeLabelKey"`
	// PrefixMatchInfoProducerName binds the a_p read BY PRODUCER NAME -- signal 11.
	// A data key's identity is "<DataType>/<ProducerName>", so an unbound read can
	// silently become a second, differently-configured signal.
	//
	// BOTH ARMS MUST BE GIVEN THE SAME VALUE, and that requirement is stronger here
	// than it looks: they are separate plugin instances with separate bindings, so a
	// divergence does not fail -- it silently prices different prompts in the two arms
	// and the measured difference stops attributing to the objective.
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName"`
	// TokenizedPromptProducerName binds the a_r read BY PRODUCER NAME -- signal 10,
	// same reasoning, and see degradation D8 in promptTokens for why a divergence in
	// the FAILURE RATE of this read is just as confounding as a divergence in its value.
	TokenizedPromptProducerName string `json:"tokenizedPromptProducerName"`
}

// ShadowTable configures the EPP-side resident index -- signals 12-18.
//
// STILL REQUIRED IN THIS ARM, even though it reads no resident VALUE. It reads the
// resident POPULATIONS in three places: ResidentPrefillTokens enters tIterDecode and
// tIterPrefill as S_pf on every candidate (D4), RunningDecode feeds decodeRemStepsEst
// and the rollforward KV walk (D2, via StepsDone and KVBlocks), and RunningPrefill
// feeds prefillRemStepsEst (D6). What it does NOT inherit is D2c, the late-first-token
// bias, because no kernel here reads ArrivalUs, FirstTokenUs, or TTFTSet.
type ShadowTable struct {
	// EntryTTLSeconds bounds how long an entry with no further chunks is charged as
	// a resident. THE SWEEP IS REQUIRED, NOT HYGIENE: requests terminate without a
	// final chunk (client disconnect, upstream error) and the ResponseBodyProcessor
	// signature carries no error or termination state, so a disconnect is
	// indistinguishable from a completion. Without the sweep those entries are
	// charged forever and permanently inflate S_pf and the remaining-steps estimates
	// on that endpoint.
	EntryTTLSeconds int `json:"entryTtlSeconds"`
	// SweepIntervalSeconds is how often the reaper runs.
	SweepIntervalSeconds int `json:"sweepIntervalSeconds"`
	// ResidentPrefillTokensCap caps S_pf both per occupant and in total --
	// degradation D4. Still an over-estimate, biasing toward remote.
	ResidentPrefillTokensCap int64 `json:"residentPrefillTokensCap"`
}

// Transfer is the KV-transfer price model -- degradation D5, config.md section 7.
//
// UNMEASURED ON THIS TARGET, AND IT MATTERS MORE HERE THAN IN THE FOCAL ARM: there is
// no externality term to partially offset a mis-priced transfer, so an unmeasured
// c_xfer moves this arm's decisions systematically. It enters the objective at exactly
// one place, remoteLeadUs in jointCandidateTTFT.
//
// It is NOT the only size-dependent price of going remote, however. The disaggregated
// projection also adds tIterFirstDecode, the B+1 re-timed first decode iteration, which
// the LOCAL projection does not -- local samples its first token when prefill completes,
// so no decode iteration precedes it. That term's KV component scales with the
// request's own input length, so prompt size prices the remote path through two
// channels, not one.
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

// Config is this arm's full configuration, and it is a STRICT SUBSET of the focal arm's
// (see doc.go). Every field here has the same meaning and MUST be given the same value
// (config.md section 9.1); the focal arm's objective-specific fields are absent because
// nothing here reads them.
//
// WHAT IS ABSENT, AND WHY EACH ABSENCE IS A DECISION RATHER THAN AN OMISSION. This arm
// drops fields rather than populating unread ones: a populated-but-never-read field is a
// reader trap, because it implies a consumer exists somewhere.
//
//   - NO tau TRIPLE (no SLOTargets, ActiveWorkload, or WorkloadTargets). No SLO
//     threshold reaches this arm's score anywhere. The consequence is worth stating
//     because it is asymmetric: the tau selector of config.md section 6 is IRRELEVANT
//     to this arm, and -- unlike the focal arm -- this arm CANNOT be flattened by a
//     misconfigured zero triple. There is no failure mode here to validate against.
//   - NO V. The objective is a latency in microseconds, not a weighted sum, so there is
//     no penalty/stability multiplier to carry.
//   - NO Ablation. Its three externality/own-good switches name terms this arm does not
//     have. Its Decomposed switch is genuinely unreachable here for a structural
//     reason, not merely unused: restricting the decode enumeration presupposes the
//     scorer-first ordering that Decide deliberately does not apply (see Decide's doc
//     on PLAIN ASCENDING-ID ORDER). The config.md section 10 decomposition ablation is
//     therefore run on the FOCAL arm, which is where it belongs -- it prices the joint
//     SHAPE, and both arms share the shape.
//   - NO Capacity, and no NomPrefillTokens. This arm has no capacity term. Both were
//     considered and both are genuinely unreachable: muDNom and muPNom take their
//     arguments BY PARAMETER, so the verbatim-copied Coeffs methods compile without any
//     Config field, and this arm calls neither -- its admission contexts use muDecode
//     and muPrefill, which read only batch state.
//
// Decoded with plugin.StrictDecoder, so DisallowUnknownFields is set and a misspelled
// key is a STARTUP ERROR rather than a silently ignored one. That strictness is what
// turns the absences above into loud failures: pasting the focal arm's overlay into this
// arm's parameters block does not silently ignore `v` and `workloadTargets`, it refuses
// to start.
//
// There are deliberately NO defaults for the physics: every field that changes a
// decision must be stated in the scenario overlay, so a missing value fails loudly
// instead of quietly pricing candidates under invented physics.
type Config struct {
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
	//
	// IDENTICAL VALUES TO THE FOCAL ARM. If the two arms carried different
	// coefficients the comparison would be meaningless -- they would differ in the
	// objective AND in the physics, and no measurement could separate them.
	CoeffsByGPUType map[string]Coeffs `json:"coeffsByGpuType"`

	ShadowTable ShadowTable `json:"shadowTable"`

	// OutputTokenProcessingUs is the client-visible per-token post-processing
	// latency (streaming detokenization), added explicitly to every TTFT projection
	// because it sits outside the calibrated theta -- config.md section 5 signal 25.
	//
	// config.md gives no value for it. UNLIKE IN THE FOCAL ARM IT IS DECISION-NEUTRAL
	// HERE, and the difference is instructive: there it enters ownGood through a
	// sigmoid, so a common additive constant still moves two candidates' values by
	// different amounts. Here the objective IS the latency, so adding the same constant
	// to every candidate cannot change the argmin. It is carried anyway because the
	// projected TTFT is logged and compared against the focal arm's, and a term present
	// in one arm's projection and absent from the other's would make those two numbers
	// incomparable.
	OutputTokenProcessingUs float64 `json:"outputTokenProcessingUs"`
}

// validate enforces every invariant whose violation would otherwise be SILENT.
//
// Each check below corresponds to a failure mode named in config.md or in the
// specification layer, where the consequence is a policy that keeps running and keeps
// reporting a latency while computing something other than the published objective.
func (c *Config) validate() error {
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
	// startup (data_graph.go:96-98); an EMPTY one silently resolves to the type string,
	// and if the operator declared their producer under any other instance name the
	// required key looks unproduced -- so CreateMissingDataProducers auto-instantiates a
	// SECOND, nil-parameter producer (data_graph.go:72-118) which this arm then reads.
	//
	// THAT IS THE EXACT FAILURE THE TWO-ARM COMPARISON CANNOT SURVIVE. It yields two
	// indexers with two block sizes, bypasses the operator's tuning, and -- because each
	// arm resolves its own binding independently -- can bind THIS arm to a different
	// prefix producer than the focal arm. The two arms would then price different
	// prompts while both looked healthy, and the measured difference would no longer
	// attribute to the objective.
	//
	// It is required even though PluginSpec.Name defaults to Type, which makes an empty
	// value accidentally work in the shipped overlay: that is what would let this pass a
	// smoke test and break on a renamed producer.
	if c.Signals.PrefixMatchInfoProducerName == "" {
		return errors.New("signals.prefixMatchInfoProducerName is required: an empty value leaves " +
			"the data key at its type-string default, so a producer declared under any other " +
			"instance name causes a second, nil-parameter producer to be auto-created and read " +
			"instead -- two indexers, two block sizes, and the two arms potentially bound to " +
			"different prefix signals, which destroys the attribution the arms exist to make")
	}
	if c.Signals.TokenizedPromptProducerName == "" {
		return errors.New("signals.tokenizedPromptProducerName is required: same reason as " +
			"signals.prefixMatchInfoProducerName -- an empty binding can silently resolve to a " +
			"second, auto-created token producer")
	}
	if c.Signals.GPUTypeLabelKey == "" {
		return errors.New("signals.gpuTypeLabelKey is required: theta is keyed by pod label, and a " +
			"defaulted GPU type is wrong on EVERY decision because heterogeneity rides the " +
			"per-iteration intercept (the two fleet types differ by a factor of 1.539). This arm " +
			"is hardware-aware by construction and that is deliberate: a hardware-blind " +
			"least-TTFT baseline would confound the mechanism with hardware awareness")
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
				"growing linearly in prompt length. This arm has no externality term to partially offset "+
				"a mis-priced transfer, so the error moves its decisions systematically",
				c.Transfer.KVBytesPerTokenPerGPU)
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
			"permanently inflate S_pf and the remaining-steps estimates on their endpoint",
			c.ShadowTable.EntryTTLSeconds)
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

	if c.OutputTokenProcessingUs < 0 {
		return fmt.Errorf("outputTokenProcessingUs must be >= 0, got %g", c.OutputTokenProcessingUs)
	}

	return nil
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

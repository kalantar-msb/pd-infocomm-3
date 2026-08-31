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
	"encoding/json"
	"fmt"
	"net"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

// HandlerPluginType is the config `type:` for the state-owning half of this arm.
const HandlerPluginType = "causal-slo-externality-handler"

// Profile result keys in the assembled SchedulingResult.
//
// These name entries of SchedulingResult.ProfileResults, which this handler constructs
// itself -- they are NOT required to be configured scheduling-profile names. The decode
// key is the PRIMARY profile, and it carries the decode pick ALONE.
const (
	resultProfileDecode  = "decode"
	resultProfilePrefill = "prefill"
)

// jointProfileName is the single role-blind profile this handler runs.
//
// ONE profile, not two, and that is the required structural shape rather than a
// simplification: all profiles receive the same candidateEndpoints slice
// (scheduler.go:81), so one unfiltered profile sees BOTH pools, which is what lets a
// single picker run enumerate D local plus D*P disaggregated candidates and compare them
// in ONE argmin.
const jointProfileName = "default"

var (
	_ fwksched.ProfileHandler     = &Handler{}
	_ fwkrc.PreRequest            = &Handler{}
	_ fwkrc.ResponseBodyProcessor = &Handler{}
	_ fwkplugin.ConsumerPlugin    = &Handler{}
)

// Handler owns this arm's shared state and drives the joint decision.
//
// TWO REGISTRATIONS, NOT ONE PLUGIN WITH TWO INTERFACES: ProfileHandler.Pick and
// Picker.Pick share a method name with different signatures
// (scheduling/plugins.go:47, :77), so no single Go type can satisfy both. This half
// carries ProfileHandler + PreRequest + ResponseBodyProcessor; the Picker half carries
// Filter + Picker and holds a pointer to this one.
//
// State ownership follows lifecycle ownership: the response hooks that WRITE the shadow
// table live here, so the table and the Policy live here too, and the picker reads them
// through the pointer it resolved at factory time.
//
// EXACTLY ONE ProfileHandler IS PERMITTED across all instantiated plugins
// (configloader.go:380-394 errors with "multiple profile handlers found"), so this arm's
// plugin config must NOT also declare disagg-profile-handler or single-profile-handler.
// That is why this handler sets the prefill routing header itself rather than delegating.
type Handler struct {
	typedName fwkplugin.TypedName

	cfg    Config
	policy *Policy
	table  *shadowTable

	prefixDataKey fwkplugin.DataKey
	tokenDataKey  fwkplugin.DataKey

	metrics *pluginMetrics
}

// HandlerFactory builds the handler, validates the whole configuration, and starts the
// shadow table's TTL sweeper.
//
// Validation is deliberately strict and total: every check in Config.validate
// corresponds to a failure mode whose consequence is a policy that keeps running and
// keeps reporting goodput while computing something other than the published objective.
func HandlerFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	cfg := Config{}
	if rawParameters == nil {
		return nil, fmt.Errorf("%s requires parameters: there are deliberately no defaults for the "+
			"physics, so that a missing value fails loudly instead of pricing candidates under "+
			"invented coefficients", HandlerPluginType)
	}
	if err := rawParameters.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("failed to parse parameters of the %s plugin: %w", HandlerPluginType, err)
	}
	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("%s configuration is invalid: %w", HandlerPluginType, err)
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}

	metrics := newPluginMetrics(name, HandlerPluginType)
	table := newShadowTable(cfg.ShadowTable, int64(cfg.Engine.BlockSize), metrics)
	table.startSweeper(time.Duration(cfg.ShadowTable.SweepIntervalSeconds) * time.Second)

	h := &Handler{
		typedName: fwkplugin.TypedName{Type: HandlerPluginType, Name: name},
		cfg:       cfg,
		table:     table,
		metrics:   metrics,
		// Bound BY PRODUCER NAME. A data key's identity is
		// "<DataType>/<ProducerName>" (plugin/datakey.go:45-49), so an unbound read can
		// silently become a second, differently-configured signal. Both arms must be
		// given the SAME producer names in config so they read one signal -- that is
		// what attributes the effect to the mechanism rather than to the machinery.
		prefixDataKey: attrprefix.PrefixCacheMatchInfoDataKey.
			WithNonEmptyProducerName(cfg.Signals.PrefixMatchInfoProducerName),
		tokenDataKey: tokenproducer.TokenizedPromptDataKey.
			WithNonEmptyProducerName(cfg.Signals.TokenizedPromptProducerName),
	}
	h.policy = newPolicy(cfg, table, metrics)
	return h, nil
}

// TypedName implements fwkplugin.Plugin.
func (h *Handler) TypedName() fwkplugin.TypedName { return h.typedName }

// Consumes declares the produced data this arm reads, mirroring the picker's
// declaration so the DAG orders the producers before either half.
func (h *Handler) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			h.tokenDataKey: fwksched.TokenizedPrompt{},
		},
		Optional: map[fwkplugin.DataKey]any{
			h.prefixDataKey: attrprefix.PrefixCacheMatchInfo{},
		},
	}
}

// snapshotFor assembles one endpoint's routing view from its scraped metrics plus the
// shadow table.
//
// TWO SCRAPE TRAPS ARE HANDLED HERE, both silent if got wrong, both confirmed at this
// pin:
//
//   - Metrics.KVCacheUsagePercent IS A FRACTION IN [0,1], not a percent, despite the
//     name. Dividing by 100 would under-count KV by 100x and collapse the C1*KV term
//     that carries the hardware heterogeneity this experiment exists to measure.
//
//   - Metrics.KvCacheMaxTokenCapacity IS UNUSABLE: it is assigned nowhere outside tests
//     at this pin, so it always reads 0 and any branch guarded on it being positive is
//     dead code. KV capacity is derived from CacheNumBlocks * CacheBlockSize instead.
func (h *Handler) snapshotFor(id, gpuType string, metrics *fwkdl.Metrics, blockSize int64) Snapshot {
	numBlocks := int64(metrics.CacheNumBlocks)
	usage := metrics.KVCacheUsagePercent // a FRACTION in [0,1]

	kvTokens := int64(usage * float64(blockSize) * float64(numBlocks))
	// Degradation D7: int64 truncation on top of exporter quantisation makes this a
	// FLOOR on free blocks, which over-states admission delay.
	freeBlocks := int64((1.0 - usage) * float64(numBlocks))
	if freeBlocks < 0 {
		freeBlocks = 0
	}

	// ONE lock hold for all three, so S_pf cannot describe a resident population this
	// snapshot does not contain.
	decode, prefill, prefillTokens := h.table.viewFor(id)

	return Snapshot{
		ID:                    id,
		GPUType:               gpuType,
		BatchSize:             metrics.RunningRequestsSize,
		QueueDepth:            metrics.WaitingQueueSize,
		KvTokensInUse:         kvTokens,
		FreeKVBlocks:          freeBlocks,
		ResidentPrefillTokens: prefillTokens,
		MaxBatchSize:          int64(h.cfg.Engine.MaxBatchSize),
		BlockSizeTokens:       blockSize,
		RunningDecode:         decode,
		RunningPrefill:        prefill,
		// Degradation D1: permanently false at this pin. No route exists to the
		// engine's wait-queue contents, its current grants, or its step start instant.
		SchedulerStateObserved: false,
	}
}

// Pick selects which profiles to run. scheduler.go:68-92 calls it in a loop until it
// returns an empty map.
//
// ONE role-blind profile runs once. The single picker run inside it produces BOTH picks,
// so there is no second stage to gate: the decode/prefill split is an OUTPUT of the
// joint argmin, not an input to a sequence of profile runs. That is the structural
// difference from the stock decode-then-decide handler.
func (h *Handler) Pick(ctx context.Context, _ *fwksched.InferenceRequest, profiles map[string]fwksched.SchedulerProfile,
	profileResults map[string]*fwksched.ProfileRunResult) map[string]fwksched.SchedulerProfile {
	if len(profileResults) > 0 {
		return nil // already ran: done
	}
	profile, ok := profiles[jointProfileName]
	if !ok {
		// Fall back to the only configured profile when it is named something else, so
		// a renamed profile does not silently produce zero results.
		if len(profiles) == 1 {
			for name, p := range profiles {
				return map[string]fwksched.SchedulerProfile{name: p}
			}
		}
		log.FromContext(ctx).Error(nil, "no joint profile configured",
			"plugin", h.typedName.String(), "expected", jointProfileName, "have", len(profiles))
		return nil
	}
	return map[string]fwksched.SchedulerProfile{jointProfileName: profile}
}

// ProcessResults SPLITS the picker's two endpoints into two profile results.
//
// THIS SPLIT IS MANDATORY, NOT COSMETIC. The director turns EVERY endpoint in the
// primary profile's TargetEndpoints into a routing destination and comma-joins them into
// reqCtx.TargetEndpoint (director.go:486-496). Leaving both picks in the primary profile
// would dispatch each request to BOTH pods -- a silent misroute that still returns 200s.
//
// So the primary profile carries the DECODE pick alone, and the prefill pick moves to
// its own key where PreRequest reads it for the header. The joint argmin is untouched;
// only the reporting shape splits.
func (h *Handler) ProcessResults(ctx context.Context, request *fwksched.InferenceRequest,
	profileResults map[string]*fwksched.ProfileRunResult) (*fwksched.SchedulingResult, error) {
	logger := log.FromContext(ctx)

	var joint *fwksched.ProfileRunResult
	for _, result := range profileResults {
		if result != nil && len(result.TargetEndpoints) > 0 {
			joint = result
			break
		}
	}
	if joint == nil {
		return nil, fmt.Errorf("%s: no endpoint selected by the joint argmin", h.typedName.String())
	}

	// The picker's contract: TargetEndpoints[0] is the decode pick, [1] the prefill
	// pick when the argmin chose to disaggregate.
	decodeEP := joint.TargetEndpoints[0]
	out := map[string]*fwksched.ProfileRunResult{
		resultProfileDecode: {TargetEndpoints: []fwksched.Endpoint{decodeEP}},
	}
	var prefillEP fwksched.Endpoint
	if len(joint.TargetEndpoints) > 1 {
		prefillEP = joint.TargetEndpoints[1]
		out[resultProfilePrefill] = &fwksched.ProfileRunResult{
			TargetEndpoints: []fwksched.Endpoint{prefillEP},
		}
	}

	// Record the placement in the shadow table. This is the arrival stamp for signals
	// 12 and 18: the timestamp is the ROUTING INSTANT, not the client's arrival, which
	// is a stated bias rather than an oversight.
	h.recordPlacement(ctx, request, decodeEP, prefillEP)

	logger.V(logutil.DEBUG).Info("assembled joint scheduling result",
		"plugin", h.typedName.String(), "requestID", requestID(request),
		"decode", endpointID(decodeEP), "prefill", endpointID(prefillEP))

	// TWO INVARIANTS HERE ARE LOAD-BEARING FOR CRASH-SAFETY, not just correctness, because
	// the director dereferences both without a guard:
	//
	//   1. PrimaryProfileName must name a key present in ProfileResults. director.go:481 looks
	//      it up and :485 dereferences the result with no nil check.
	//   2. That entry must carry at least one endpoint. director.go:495 indexes
	//      targetMetadatas[0] unguarded.
	//
	// Both hold unconditionally above: the primary name is always resultProfileDecode, and
	// out[resultProfileDecode] is always populated with exactly the one decode endpoint. A
	// refactor that made either conditional would turn a routing bug into an EPP panic.
	return &fwksched.SchedulingResult{
		ProfileResults:     out,
		PrimaryProfileName: resultProfileDecode,
	}, nil
}

// recordPlacement inserts the shadow-table entry for a request just placed.
func (h *Handler) recordPlacement(ctx context.Context, request *fwksched.InferenceRequest,
	decodeEP, prefillEP fwksched.Endpoint) {
	id := requestID(request)
	if id == "" {
		// Without a request ID the response hooks cannot find the entry again, so
		// inserting it would create an entry that is never advanced and is only ever
		// reaped by the TTL sweep -- charging a phantom resident until then.
		log.FromContext(ctx).V(logutil.DEBUG).Info("no request ID; not tracking resident",
			"plugin", h.typedName.String())
		return
	}
	promptTokens, ok := promptTokens(request)
	if !ok {
		return
	}
	h.table.place(residentEntry{
		requestID:    id,
		decodeID:     endpointID(decodeEP),
		prefillID:    endpointID(prefillEP),
		sloClass:     sloClassOf(request),
		promptTokens: int64(promptTokens),
		arrivalUs:    time.Now().UnixMicro(),
		kvBlocks:     h.policy.reqKVNeed(promptTokens),
	})
}

// PreRequest writes the prefill routing header so the sidecar contacts the pod the
// joint argmin chose.
//
// It runs AFTER the picker and after ProcessResults, and receives the finished
// SchedulingResult (director.go:507).
//
// THE UNCONDITIONAL DELETE COMES FIRST, and it is a silent-misroute guard rather than
// hygiene: this arm chooses local placement for some requests, and a stale
// x-prefiller-host-port left by an earlier hop would send a locally-placed request's
// prefill to a remote pod, quietly converting a local decision into a disaggregated one.
// The stock disagg handler does the same (disagg_profile_handler.go:557).
func (h *Handler) PreRequest(ctx context.Context, request *fwksched.InferenceRequest,
	schedulingResult *fwksched.SchedulingResult) error {
	if request == nil || request.Headers == nil {
		return nil
	}
	delete(request.Headers, routing.PrefillEndpointHeader)
	if schedulingResult == nil {
		return nil
	}
	prefill := schedulingResult.ProfileResults[resultProfilePrefill]
	if prefill == nil || len(prefill.TargetEndpoints) == 0 {
		return nil // local placement: no prefill header, by design
	}
	metadata := prefill.TargetEndpoints[0].GetMetadata()
	if metadata == nil {
		// THE ONE PATH THAT RETURNS AN ERROR, and it is the silent-fallback guard the
		// rest of this arm exists to prevent. The joint argmin chose to disaggregate,
		// but the prefill pick carries no address, so the header cannot be written and
		// the request would run LOCALLY while every counter and log records a
		// disaggregated placement. Failing the request makes the disagreement visible
		// instead of quietly transferring a different algorithm's decision.
		return fmt.Errorf("%s: the joint argmin chose to disaggregate but the prefill "+
			"endpoint carries no metadata, so the prefill header cannot be set",
			h.typedName.String())
	}
	request.Headers[routing.PrefillEndpointHeader] = net.JoinHostPort(metadata.Address, metadata.Port)
	log.FromContext(ctx).V(logutil.DEBUG).Info("set prefill header",
		"plugin", h.typedName.String(), "requestID", requestID(request),
		"prefill", net.JoinHostPort(metadata.Address, metadata.Port))
	return nil
}

// ResponseBody advances the shadow table -- signals 15, 16, and 17.
//
// Usage.CompletionTokens is monotonic per request and REQUIRES the engine flag
// --enable-force-include-usage (config.md section 11): without it, usage arrives only in
// the FINAL chunk, so StepsDone stays 0 for every request's whole lifetime and every
// remaining-steps estimate is wrong while the table looks present and correct. The EPP
// cannot detect that in one sample, which is why it is an operator requirement rather
// than a runtime check.
//
// reqCtx.Usage is only refreshed when the parsed chunk actually carried a usage block
// (handlers/response.go:80-81); otherwise it retains its previous value, so a repeated
// count is not evidence of a stalled request.
//
// CONCURRENCY: non-final chunks are delivered from an ASYNC QUEUE on a different
// goroutine (director.go:601-632) while the final chunk runs on the request goroutine
// (:539). That is the concrete reason the shadow table is mutex-guarded rather than
// relying on per-request framework state.
func (h *Handler) ResponseBody(ctx context.Context, request *fwksched.InferenceRequest,
	response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil {
		return
	}
	id := requestID(request)
	if id == "" {
		return
	}

	h.table.observeChunk(id, response.Usage.CompletionTokens, time.Now().UnixMicro())

	if !response.EndOfStream {
		return
	}
	// TERMINAL CHUNK -- AND DEGRADATION D9 LIVES HERE.
	//
	// Only a genuinely COMPLETED request yields a usable realized output length: one that
	// terminated early carries a truncated count that drags the per-class N_out mean down.
	// This hook cannot tell the two apart. It carries no error or termination state
	// (requestcontrol/plugins.go:76-86 records an upstream TODO to add one), and the server
	// FORCES a terminal call with EndOfStream true for every request that picked a pod but
	// never completed -- error, client disconnect, or panic (pkg/epp/handlers/server.go:373-379).
	// The flag that separates them, reqCtx.ResponseComplete, is director-internal and is
	// already set before the hook runs on the normal paths (server.go:537-540, :608-610).
	//
	// So the fold happens on every terminal observation, and the counter is what keeps that
	// from being silent. Declining to fold is not the conservative alternative -- it is
	// STRICTLY WORSE, and it fails the same way as a trap this arm rejects at startup. No
	// request can be confirmed complete, so the mean would never update and nHatFor would
	// return its seed of 1 forever. varDecodeInputs then floors every resident's rem at 1, so
	// every resident looks one step from completion, cBase and cLocalAfter land within one
	// iteration of each other, and varDecodeContribution goes to ~0 across the whole
	// population. The decode-side externality collapses and the arm degenerates to its own-good
	// term -- the same degeneration as the tau_e2e = 0 trap, which Config.validate refuses to
	// start under. Folding truncated counts biases the mechanism; declining to fold disables it.
	//
	// BIAS: a truncated count understates N_out, so residents look closer to finishing, less
	// value is at risk, and the externality is UNDER-counted -- toward LOCAL, the same
	// direction as D2b and D2c.
	//
	// The TTL sweep is still required, for the narrower population this path cannot reach at
	// all: a request whose entry exists but whose abort cleanup never ran, e.g. TargetPod nil
	// or a failure before the deferred cleanup.
	class, outputTokens, ok := h.table.complete(id)
	if !ok {
		return
	}
	h.metrics.recordOutputLengthFolded()
	h.policy.observeCompletedOutput(class, outputTokens)
	log.FromContext(ctx).V(logutil.TRACE).Info("resident completed",
		"plugin", h.typedName.String(), "requestID", id, "class", class, "outputTokens", outputTokens)
}

// endpointID returns an endpoint's stable identity, or "" for a nil endpoint.
//
// EndpointMetadata.ID.String() is the identity the in-tree per-endpoint index also uses
// (inflightload/producer.go:343), so the shadow table keys agree with the rest of the
// EPP. Address/Port are the ROUTING pair, not the identity.
func endpointID(ep fwksched.Endpoint) string {
	if ep == nil || ep.GetMetadata() == nil {
		return ""
	}
	return ep.GetMetadata().ID.String()
}

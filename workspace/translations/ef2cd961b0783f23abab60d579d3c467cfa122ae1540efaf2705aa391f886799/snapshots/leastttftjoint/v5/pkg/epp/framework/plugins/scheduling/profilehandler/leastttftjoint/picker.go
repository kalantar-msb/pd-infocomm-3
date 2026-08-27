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
	"fmt"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// PickerPluginType is the config `type:` for the argmin half of this arm.
const PickerPluginType = "least-ttft-joint-picker"

// decisionAttributeKey is the per-endpoint attribute under which the Filter hands the
// request-derived operands to the Picker.
//
// THIS IS THE ONLY RACE-FREE CHANNEL AT THIS PIN. Picker.Pick receives only
// []*ScoredEndpoint and no request (plugins.go:77), and `CycleState` does not exist in
// this tree. The director builds a FRESH Endpoint per request via NewEndpoint
// (director.go:479-486, types.go:114-124), and object identity is preserved from Filter
// output through the scorer map into ScoredEndpoint.Endpoint
// (scheduler_profile.go:201), so an attribute written in the Filter is readable in the
// Picker and is private to one request.
//
// Stashing the request in a plugin struct field would NOT work: one plugin instance
// serves concurrent requests.
const decisionAttributeKey = "LeastTTFTJointDecisionInput"

var (
	_ fwksched.Filter          = &Picker{}
	_ fwksched.Picker          = &Picker{}
	_ fwkplugin.ConsumerPlugin = &Picker{}
)

// pickerParameters is the picker's whole config surface: it owns no policy of its own
// and reads the handler's.
type pickerParameters struct {
	// HandlerPluginName names the handler that owns the shared Policy and shadow
	// table. Resolved at factory time; a forward reference is a startup error.
	HandlerPluginName string `json:"handlerPluginName"`
}

// decisionInput carries the per-request operands the argmin needs, written by Filter
// and read by Pick.
//
// It must implement fwkdl.Cloneable, and note that Attributes.Get returns
// value.Clone() -- a COPY, not the stored pointer (attributemap.go:83-85). So this type
// carries only immutable per-request facts. The mutable shared state (the shadow table)
// travels the other way: by factory-time lookup of the owning handler.
type decisionInput struct {
	requestID string
	sloClass  string
	inputLen  int
	nowUs     float64

	// apByEndpoint is a_p per endpoint ID. Computed here because the per-endpoint
	// prefix attribute AND the request are both in scope only in the Filter.
	apByEndpoint map[string]int

	// decodeIDs and prefillIDs are the role split, by pod label.
	decodeIDs  []string
	prefillIDs []string

	// snapshots is the routing view per endpoint ID.
	snapshots map[string]Snapshot
}

// Clone implements fwkdl.Cloneable.
func (d *decisionInput) Clone() fwkdl.Cloneable {
	if d == nil {
		return nil
	}
	out := &decisionInput{
		requestID: d.requestID,
		sloClass:  d.sloClass,
		inputLen:  d.inputLen,
		nowUs:     d.nowUs,
	}
	out.apByEndpoint = make(map[string]int, len(d.apByEndpoint))
	for k, v := range d.apByEndpoint {
		out.apByEndpoint[k] = v
	}
	out.decodeIDs = append([]string(nil), d.decodeIDs...)
	out.prefillIDs = append([]string(nil), d.prefillIDs...)
	out.snapshots = make(map[string]Snapshot, len(d.snapshots))
	for k, v := range d.snapshots {
		out.snapshots[k] = v
	}
	return out
}

// Picker implements Filter and Picker for the joint argmin.
//
// WHY A PICKER AND NOT A SCORER, verified at this pin:
//   - Scorer.Score returns map[Endpoint]float64 (plugins.go:71) -- per endpoint, with
//     NO WAY TO NAME A PAIR, while the objective is over (decode, prefill placement);
//   - every scorer contribution passes through enforceScoreRange, which clamps to
//     [0,1] (scheduler_profile.go:178, :215-223), while J is a LATENCY IN MICROSECONDS:
//     non-negative, but unbounded above and routinely in the tens of thousands. Clamping
//     does not degrade the ranking, it DESTROYS it -- essentially every candidate
//     saturates at 1 and the argmin degenerates to enumeration order.
//
// The focal arm faces the same hazard for the opposite reason: its J is signed, so
// clamping collapses every candidate with J <= 0 instead. Both arms are barred from the
// Scorer pin, by different arithmetic.
//
// A Picker's own computed objective does NOT pass through that clamp -- runPickerPlugin
// never calls enforceScoreRange -- so J keeps its full range here. The corollary is that
// ScoredEndpoint.Score carries inherited, CLAMPED scorer contributions and this picker
// ignores that field entirely.
type Picker struct {
	typedName fwkplugin.TypedName

	handler *Handler

	prefixDataKey fwkplugin.DataKey
	tokenDataKey  fwkplugin.DataKey

	metrics *pluginMetrics
}

// PickerFactory builds the picker and binds it to the handler that owns the state.
func PickerFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	parameters := pickerParameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse parameters of the %s plugin: %w", PickerPluginType, err)
		}
	}
	if parameters.HandlerPluginName == "" {
		return nil, fmt.Errorf("%s requires handlerPluginName: the argmin and the response hooks must "+
			"share ONE Policy and ONE shadow table, and the handler owns them", PickerPluginType)
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}

	// Resolve the state owner. instantiatePlugins builds plugins in CONFIG ORDER and
	// registers each only after construction (configloader.go:215-239), so this
	// resolves BACKWARD only: the handler must be declared BEFORE this picker.
	//
	// Erroring on nil is what prevents the real hazard -- two divergent copies of the
	// shadow table, each seeing a fraction of the residents. In this arm that would
	// under-count S_pf on every candidate and collapse both remaining-steps estimates
	// toward their floor of 1, so every admission delay would be under-priced, while both
	// plugins looked healthy.
	raw := handle.Plugin(parameters.HandlerPluginName)
	if raw == nil {
		return nil, fmt.Errorf("%s: handlerPluginName %q not found; it must name a %s declared EARLIER "+
			"in the plugins list (by-name references resolve backward only)",
			PickerPluginType, parameters.HandlerPluginName, HandlerPluginType)
	}
	handler, ok := raw.(*Handler)
	if !ok {
		return nil, fmt.Errorf("%s: plugin %q is not a %s", PickerPluginType, parameters.HandlerPluginName, HandlerPluginType)
	}

	return &Picker{
		typedName:     fwkplugin.TypedName{Type: PickerPluginType, Name: name},
		handler:       handler,
		prefixDataKey: handler.prefixDataKey,
		tokenDataKey:  handler.tokenDataKey,
		metrics:       newPluginMetrics(name, PickerPluginType),
	}, nil
}

// TypedName implements fwkplugin.Plugin.
func (p *Picker) TypedName() fwkplugin.TypedName { return p.typedName }

// Consumes declares the produced data this arm reads, so the producer DAG orders them
// before it and a MISSING producer is an init-time error rather than a per-request miss.
func (p *Picker) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			p.tokenDataKey: fwksched.TokenizedPrompt{},
		},
		// The prefix read is OPTIONAL rather than Required: a miss is a legitimate
		// runtime state that the arm handles by treating the prompt as fully uncached
		// and counting the miss. Declaring it Required would refuse to start on a
		// fleet whose prefix producer has not warmed up.
		Optional: map[fwkplugin.DataKey]any{
			p.prefixDataKey: attrprefix.PrefixCacheMatchInfo{},
		},
	}
}

// Filter builds the whole routing view and attaches it to the endpoints.
//
// It is NOT a role filter in the usual sense: it returns every endpoint it accepted,
// both pools, because the joint argmin needs the full cross product in ONE picker run.
// The role SPLIT is recorded in the attached decisionInput rather than applied by
// narrowing the slice.
//
// It rejects, and COUNTS, an endpoint that cannot be priced: absent metadata, an absent
// or unmapped GPU-type label, or an absent role label. A rejection must be visible,
// because a mislabelled endpoint leaving the candidate set is otherwise
// indistinguishable from a routing preference.
func (p *Picker) Filter(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)
	cfg := &p.handler.cfg

	inputLen, tokenized := promptTokens(request)
	if !tokenized {
		// DEGRADATION D8. a_r is absent, so nothing is scorable and this arm computes no
		// objective. It is counted per arm, because if the RATE differs between the two
		// arms it confounds the comparison the arms exist to make.
		//
		// THERE IS NO "INHERITED PICK" TO FALL BACK ON in this arm's shape: the profile is
		// role-blind and carries no scorers, and nothing downstream of the picker checks
		// role -- the director reads Address/Port off whatever the primary profile returns
		// and dispatches (director.go:461-472). So returning the unfiltered candidate set
		// here would let a decode request be sent to the PREFILL pod.
		//
		// The candidate set is therefore still validated, and narrowed to decode-eligible
		// endpoints, before being handed back. The arm declines to RANK them; it does not
		// decline to enforce which pods may serve a decode request.
		p.metrics.recordTokenizationUnavailable()
		logger.V(logutil.DEBUG).Info("tokenized prompt absent, declining to score; "+
			"returning decode-eligible candidates only",
			"plugin", p.typedName.String(), "requestID", requestID(request))
		return p.decodeEligible(ctx, endpoints)
	}

	in := &decisionInput{
		requestID:    requestID(request),
		sloClass:     sloClassOf(request),
		inputLen:     inputLen,
		nowUs:        float64(time.Now().UnixMicro()),
		apByEndpoint: map[string]int{},
		snapshots:    map[string]Snapshot{},
	}

	accepted := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		if ep == nil || ep.GetMetadata() == nil {
			p.metrics.recordCandidateRejected(rejectReasonEndpointMetadataNil)
			continue
		}
		metadata := ep.GetMetadata()
		id := metadata.NamespacedName.String()

		// --- Role, signal 9. Read the SAME label the stock filters read, via their
		// own constants, so the two cannot drift apart.
		//
		// DELIBERATE DIVERGENCE FROM decode-filter: it passes allowsNoLabel = true
		// (bylabel/roles.go:47), so an UNLABELLED pod is decode-eligible there. This
		// arm REJECTS an absent role label instead, because the specification's
		// endpointRole contract requires it: an unlabelled pod would otherwise be
		// priced as a decode candidate under a role it may not serve, and the mistake
		// would look like a routing preference.
		role, roleOK := metadata.Labels[bylabel.RoleLabel]
		if !roleOK || role == "" {
			p.metrics.recordCandidateRejected(rejectReasonRoleLabelAbsent)
			continue
		}
		isDecode, isPrefill := isDecodeRole(role), isPrefillRole(role)
		if !isDecode && !isPrefill {
			p.metrics.recordCandidateRejected(rejectReasonRoleUnknown)
			continue
		}

		// --- theta, signal 8. REJECT on absent or unmapped; NEVER default.
		//
		// Heterogeneity rides the per-iteration intercept alpha, present on every
		// iteration regardless of KV state (25563.82 us on A100 against 16613.54 us on
		// H100, a factor of 1.539), so a defaulted label is wrong on EVERY decision
		// rather than only under load.
		gpuType, gpuOK := metadata.Labels[cfg.Signals.GPUTypeLabelKey]
		if !gpuOK || gpuType == "" {
			p.metrics.recordCandidateRejected(rejectReasonGPUTypeLabelAbsent)
			logger.V(logutil.DEBUG).Info("rejecting endpoint: GPU-type label absent",
				"plugin", p.typedName.String(), "endpoint", id, "labelKey", cfg.Signals.GPUTypeLabelKey)
			continue
		}
		if _, mapped := cfg.CoeffsByGPUType[gpuType]; !mapped {
			p.metrics.recordCandidateRejected(rejectReasonGPUTypeUnmapped)
			logger.V(logutil.DEBUG).Info("rejecting endpoint: GPU-type label not in coeffsByGpuType",
				"plugin", p.typedName.String(), "endpoint", id, "gpuType", gpuType)
			continue
		}

		metrics := ep.GetMetrics()
		if metrics == nil {
			p.metrics.recordCandidateRejected(rejectReasonMetricsUnavailable)
			continue
		}

		// --- a_p, signal 11. Read the block count AND the block size off the SAME
		// PrefixCacheMatchInfo.
		//
		// TWO BLOCK SIZES ARE REQUIRED HERE. The engine reports 16 while the prefix
		// producer clamps its own block size up to a 64-token floor
		// (approximateprefix/plugin.go:42-50, minBlockSizeTokens = 64), so multiplying
		// the producer's block COUNT by the ENGINE's block SIZE would understate the
		// cached span by 4x -- inflating a_p, over-pricing prefill work, and biasing
		// toward remote prefill. Taking both numbers from the same attribute makes a
		// producer reconfiguration unable to rescale a_p silently.
		//
		// Use CachedBlockCount(), NOT MatchBlocks(): the latter is a device-tier
		// WEIGHTED score suitable for ranking, and is <= the literal count, so it
		// would overstate the uncached suffix.
		ap := inputLen
		if raw, found := ep.Get(p.prefixDataKey.String()); found && raw != nil {
			if info, isPrefixInfo := raw.(*attrprefix.PrefixCacheMatchInfo); isPrefixInfo {
				ap = inputLen - info.CachedBlockCount()*info.BlockSizeTokens()
			} else {
				p.metrics.recordPrefixUnobserved()
			}
		} else {
			// A miss means "no information", which is NOT "nothing cached". Charging
			// the full prompt over-prices the candidate rather than asserting a cold
			// cache as fact, and leaves it in the argmin.
			p.metrics.recordPrefixUnobserved()
		}
		in.apByEndpoint[id] = ap

		// --- Block-size agreement, signal 23. Config and engine must agree, and a
		// disagreement is loud rather than silently converted: the admission test
		// compares blocks accumulated in one unit against a need expressed in another.
		blockSize := int64(metrics.CacheBlockSize)
		if blockSize > 0 && blockSize != int64(cfg.Engine.BlockSize) {
			p.metrics.recordBlockSizeMismatch()
			p.metrics.recordCandidateRejected(rejectReasonBlockSizeDisagrees)
			logger.Error(nil, "rejecting endpoint: scraped CacheBlockSize disagrees with configured engine.blockSize",
				"plugin", p.typedName.String(), "endpoint", id,
				"scraped", blockSize, "configured", cfg.Engine.BlockSize)
			continue
		}
		if blockSize <= 0 {
			blockSize = int64(cfg.Engine.BlockSize)
		}

		snap := p.handler.snapshotFor(id, gpuType, metrics, blockSize)
		in.snapshots[id] = snap
		if isDecode {
			in.decodeIDs = append(in.decodeIDs, id)
		}
		if isPrefill {
			in.prefillIDs = append(in.prefillIDs, id)
		}
		accepted = append(accepted, ep)
	}

	// Attach to every accepted endpoint. Put ignores a nil value
	// (attributemap.go:62-66), so the pointer must be non-nil, which it is.
	for _, ep := range accepted {
		ep.Put(decisionAttributeKey, in)
	}
	return accepted
}

// Pick runs the joint argmin and returns BOTH picks in one result.
//
// Returning two endpoints here is legitimate -- ProfileRunResult.TargetEndpoints is a
// slice (types.go:164-167) -- but it is NOT the final routing shape: the director turns
// every endpoint in the PRIMARY profile's TargetEndpoints into a destination and
// comma-joins them (director.go:461-472). Handler.ProcessResults therefore splits these
// two into separate profile results, with the decode pick alone as primary.
//
// The order is fixed and load-bearing: TargetEndpoints[0] is the DECODE pick,
// TargetEndpoints[1] the prefill pick when the argmin chose to disaggregate.
//
// This picker deliberately does NOT shuffle. max-score-picker shuffles for random tie
// breaking (picker/maxscore/picker.go:93), but this arm's determinism requirement is the
// opposite: candidates are enumerated over endpoints sorted by ID and the argmin uses a
// strict improvement threshold, so ties resolve to the first-enumerated candidate and
// two identical decisions cannot differ.
func (p *Picker) Pick(ctx context.Context, scoredEndpoints []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	logger := log.FromContext(ctx)
	if len(scoredEndpoints) == 0 {
		return &fwksched.ProfileRunResult{}
	}

	in, byID := p.readDecisionInput(scoredEndpoints)
	if in == nil {
		// No decision input: the Filter declined (tokenization unavailable, degradation
		// D8) or this profile was reached without it.
		//
		// The choice must be DETERMINISTIC and DECODE-ELIGIBLE, and neither is free here.
		// scoredEndpoints is map-ordered (scheduler_profile.go:199-205), so
		// scoredEndpoints[0] would be a uniformly random element; and nothing downstream
		// of the picker checks role, so a prefill endpoint reaching this return would be
		// dispatched the full decode request. The Filter has already narrowed its output
		// to decode-eligible endpoints on this path; taking the lowest ID makes the
		// remaining choice reproducible.
		if fallback := lowestIDDecodeEligible(scoredEndpoints); fallback != nil {
			logger.V(logutil.DEBUG).Info("no decision input; using the lowest-ID decode-eligible endpoint",
				"plugin", p.typedName.String(), "endpoint", endpointID(fallback))
			return &fwksched.ProfileRunResult{TargetEndpoints: []fwksched.Endpoint{fallback}}
		}
		logger.Error(nil, "no decision input and no decode-eligible endpoint; declining to pick",
			"plugin", p.typedName.String())
		return &fwksched.ProfileRunResult{}
	}

	decodeSnaps := make([]Snapshot, 0, len(in.decodeIDs))
	for _, id := range in.decodeIDs {
		decodeSnaps = append(decodeSnaps, in.snapshots[id])
	}
	prefillSnaps := make([]Snapshot, 0, len(in.prefillIDs))
	for _, id := range in.prefillIDs {
		prefillSnaps = append(prefillSnaps, in.snapshots[id])
	}
	if len(decodeSnaps) == 0 {
		logger.V(logutil.DEBUG).Info("no decode candidates after filtering", "plugin", p.typedName.String())
		return &fwksched.ProfileRunResult{}
	}

	ec := &evalCtx{
		class:        in.sloClass,
		inputLen:     in.inputLen,
		nowUs:        in.nowUs,
		requestID:    in.requestID,
		apByEndpoint: in.apByEndpoint,
	}
	ec.reqKVNeed = p.handler.policy.reqKVNeed(in.inputLen)

	// NO SCORER PREFERENCE IS COMPUTED, AND THAT IS A DELIBERATE DIVERGENCE FROM THE
	// FOCAL ARM rather than an omission. The focal arm reads the inherited scorer's pick
	// and moves it to the front of the enumeration as a tie-break order; this arm
	// enumerates in plain ascending ID order, because upstream does too -- the least-ttft
	// branch iterates decodeSnaps directly (sim/edpp.go:1456), and scorerFirstSnapshots
	// is applied only in the SLO-externality and causal-VaR branches (sim/edpp.go:2176-2178).
	//
	// Adding the reorder here would change which candidate wins an EXACT tie, silently
	// and only sometimes. So Policy.decide takes no scorer arguments and there is nothing
	// to pass.
	//
	// ScoredEndpoint.Score is therefore ignored ENTIRELY by this arm, not merely
	// down-weighted: it carries inherited contributions already clamped to [0,1], which
	// cannot be combined with an objective measured in microseconds.
	start := time.Now()
	best, ok := p.handler.policy.decide(ec, decodeSnaps, prefillSnaps)
	p.metrics.observeArgminDuration(time.Since(start).Seconds())
	if !ok {
		return &fwksched.ProfileRunResult{}
	}

	targets := make([]fwksched.Endpoint, 0, 2)
	decodeEP := byID[best.dID]
	if decodeEP == nil {
		logger.Error(nil, "the argmin named a decode endpoint absent from the scored set",
			"plugin", p.typedName.String(), "requestID", in.requestID, "decode", best.dID)
		return &fwksched.ProfileRunResult{}
	}
	targets = append(targets, decodeEP)

	// THE PLACEMENT COUNTER RECORDS WHAT THE RESULT CARRIES, not what the argmin chose.
	// A disaggregated win whose prefill endpoint is missing from the scored set would
	// otherwise be counted as disaggregated while behaving as local -- a metric that
	// disagrees with behaviour in the one direction that matters for reading a result.
	placedLocal := best.local
	if !best.local {
		prefillEP := byID[best.pID]
		if prefillEP == nil {
			p.metrics.recordCandidateRejected(rejectReasonPrefillPickMissing)
			logger.Error(nil, "the argmin chose to disaggregate but its prefill endpoint is "+
				"absent from the scored set; the request will run LOCALLY and is counted as such",
				"plugin", p.typedName.String(), "requestID", in.requestID, "prefill", best.pID)
			placedLocal = true
		} else {
			targets = append(targets, prefillEP)
		}
	}
	p.metrics.recordPlacement(placedLocal)

	logger.V(logutil.DEBUG).Info("joint argmin chose placement",
		"plugin", p.typedName.String(), "requestID", in.requestID,
		"decode", best.dID, "prefill", best.pID, "local", best.local, "placedLocal", placedLocal,
		"J", best.J, "candidates", len(decodeSnaps)+len(decodeSnaps)*len(prefillSnaps))

	return &fwksched.ProfileRunResult{TargetEndpoints: targets}
}

// readDecisionInput recovers the Filter's attachment and indexes endpoints by ID.
//
// Attributes.Get returns a CLONE (attributemap.go:83-85), so the value read back is a
// copy -- fine, because decisionInput carries only immutable per-request facts.
func (p *Picker) readDecisionInput(scored []*fwksched.ScoredEndpoint) (*decisionInput, map[string]fwksched.Endpoint) {
	byID := make(map[string]fwksched.Endpoint, len(scored))
	var in *decisionInput
	for _, se := range scored {
		if se == nil || se.GetMetadata() == nil {
			continue
		}
		byID[se.GetMetadata().NamespacedName.String()] = se
		if in != nil {
			continue
		}
		if raw, ok := se.Get(decisionAttributeKey); ok && raw != nil {
			if parsed, isInput := raw.(*decisionInput); isInput {
				in = parsed
			}
		}
	}
	return in, byID
}

// isDecodeRole and isPrefillRole mirror the stock filters' value sets EXACTLY
// (bylabel.NewDecodeRole and NewPrefillRole), including the DEPRECATED RoleBoth: those
// filters still accept it, so omitting it here would silently exclude a pod the rest of
// the EPP treats as eligible, and the exclusion would look like a routing preference.
//
// The two sets are not disjoint: prefill-decode, both, and encode-prefill-decode satisfy
// either pool.
//
// NOTE the one deliberate divergence, which lives in the caller rather than here:
// bylabel.NewDecodeRole passes allowsNoLabel = true (bylabel/roles.go:47), so an
// UNLABELLED pod is decode-eligible for the stock filter. This arm rejects an absent role
// label instead, per the specification's endpointRole contract -- an unlabelled pod would
// otherwise be priced as a decode candidate under a role it may not serve.
func isDecodeRole(role string) bool {
	return role == bylabel.RoleDecode || role == bylabel.RolePrefillDecode ||
		role == bylabel.RoleBoth || role == bylabel.RoleEncodePrefillDecode //nolint:staticcheck // SA1019: matches the stock decode filter's value set
}

func isPrefillRole(role string) bool {
	return role == bylabel.RolePrefill || role == bylabel.RoleEncodePrefill ||
		role == bylabel.RolePrefillDecode || role == bylabel.RoleBoth || //nolint:staticcheck // SA1019: matches the stock prefill filter's value set
		role == bylabel.RoleEncodePrefillDecode
}

// decodeEligible validates the candidate set and narrows it to decode-eligible endpoints.
//
// It is the tokenization-decline path's candidate set. The arm cannot RANK without a_r, but
// role eligibility is not a ranking question: nothing downstream of the picker enforces it,
// so this is the last place a decode request can be prevented from reaching a prefill pod.
//
// The GPU-type label is checked too, and counted, even though theta is not consulted on this
// path: an endpoint that cannot be priced at all should not silently start serving traffic
// only when tokenization fails.
func (p *Picker) decodeEligible(ctx context.Context, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	logger := log.FromContext(ctx)
	cfg := &p.handler.cfg
	out := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		if ep == nil || ep.GetMetadata() == nil {
			p.metrics.recordCandidateRejected(rejectReasonEndpointMetadataNil)
			continue
		}
		metadata := ep.GetMetadata()
		role, roleOK := metadata.Labels[bylabel.RoleLabel]
		if !roleOK || role == "" {
			p.metrics.recordCandidateRejected(rejectReasonRoleLabelAbsent)
			continue
		}
		if !isDecodeRole(role) {
			// Not a rejection by reason: a prefill-only endpoint is simply not a
			// candidate for a decode placement, which is the normal case.
			continue
		}
		gpuType, gpuOK := metadata.Labels[cfg.Signals.GPUTypeLabelKey]
		if !gpuOK || gpuType == "" {
			p.metrics.recordCandidateRejected(rejectReasonGPUTypeLabelAbsent)
			continue
		}
		if _, mapped := cfg.CoeffsByGPUType[gpuType]; !mapped {
			p.metrics.recordCandidateRejected(rejectReasonGPUTypeUnmapped)
			continue
		}
		out = append(out, ep)
	}
	if len(out) == 0 {
		logger.Error(nil, "no decode-eligible endpoint survived validation on the "+
			"tokenization-decline path", "plugin", p.typedName.String())
	}
	return out
}

// lowestIDDecodeEligible returns the lowest-ID decode-eligible endpoint, or nil.
//
// Used on the tokenization-decline path, where there is no decision input to consult.
//
// IT MUST BE DETERMINISTIC, and the slice it walks is not: runPickerPlugin builds
// []*ScoredEndpoint by ranging a map[Endpoint]float64 (scheduler_profile.go:199-205),
// itself seeded by ranging the Filter's output at :154-157, so the Filter's ordering is
// discarded and Go's map iteration randomizes what arrives. Taking scored[0] would return
// a different endpoint for two byte-identical requests. Lowest ID agrees with the
// ID-sorted enumeration in Policy.decide.
func lowestIDDecodeEligible(scored []*fwksched.ScoredEndpoint) fwksched.Endpoint {
	var best fwksched.Endpoint
	var bestID string
	for _, se := range scored {
		if se == nil || se.GetMetadata() == nil {
			continue
		}
		if !isDecodeRole(se.GetMetadata().Labels[bylabel.RoleLabel]) {
			continue
		}
		if id := se.GetMetadata().NamespacedName.String(); best == nil || id < bestID {
			best, bestID = se, id
		}
	}
	return best
}

// promptTokens returns a_r -- signal 10, and degradation D8 lives here.
//
// Absence is a NIL TokenizedPrompt (requesthandling/types.go:98-100) and TokenCount()
// returns 0 on nil (:126-128), so the two states must be told apart by the pointer
// rather than by the count -- a legitimately empty prompt is also nil
// (tokenizer.go:264-266).
//
// WHAT THIS CANNOT DETECT, stated plainly because the plugin must not imply otherwise:
// the tokenizer's `estimate` backend is the ZERO-CONFIG DEFAULT (tokenizer.go:61-66,
// :172-197), and TokenizedPrompt carries NO provenance marker. So an unreachable
// tokenizer does not surface here as nil -- it surfaces as a plausible ESTIMATED count,
// which mis-prices every candidate instead of disabling the arm. There is no runtime
// discriminator. The operator preflight in config.md section 11 (curl the render
// sidecar) is the only check, and the D8 counter below covers the nil case only.
func promptTokens(request *fwksched.InferenceRequest) (int, bool) {
	if request == nil || request.Body == nil {
		return 0, false
	}
	tp := request.Body.TokenizedPrompt
	if tp == nil {
		return 0, false
	}
	return tp.TokenCount(), true
}

// sloClassOf returns the request's SLO class -- signal 19.
//
// Read from the objective header, which is where `blis observe` puts `slo_class`. Both
// the current key and its deprecated alias are accepted, because the load generator
// emits the older spelling (`x-gateway-inference-objective`) while the target has since
// introduced `x-llm-d-inference-objective`; honouring only one would silently resolve
// every request to the fallback class.
//
// scheduling.RequestObjectives carries only Priority at this pin, so it is not a route
// to the class name.
//
// Every cohort of all three workloads declares `slo_class: standard`, so in this campaign
// this is one constant string across the whole grid.
//
// IT SELECTS NOTHING SLO-SHAPED IN THIS ARM. No tau reaches this arm's objective anywhere,
// so unlike in the focal arm the class does not resolve a threshold triple and cannot
// flatten the policy if it resolves to the fallback. It is read for exactly one reason: it
// KEYS THE PER-CLASS OUTPUT-LENGTH MEAN (signal 17). nHatOut is a map from class to
// running mean, read by decodeRemStepsEst and by the rollout's censored output estimate,
// so a mixed-class workload would estimate N_out separately per class.
//
// The consequence of a misread is therefore bounded and different in kind: requests would
// share one N_out mean rather than being priced against a zero threshold.
func sloClassOf(request *fwksched.InferenceRequest) string {
	if request == nil {
		return sloClassStandard
	}
	for _, key := range []string{metadata.ObjectiveKey, metadata.OldObjectiveKey} {
		if v, ok := request.Headers[key]; ok && v != "" {
			return v
		}
	}
	return sloClassStandard
}

// requestID returns the request's ID, or "" when unavailable.
func requestID(request *fwksched.InferenceRequest) string {
	if request == nil {
		return ""
	}
	return request.RequestID
}

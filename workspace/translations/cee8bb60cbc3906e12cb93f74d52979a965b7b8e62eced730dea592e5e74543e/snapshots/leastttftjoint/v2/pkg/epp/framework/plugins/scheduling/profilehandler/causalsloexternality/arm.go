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
	"encoding/json"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// This file is the ARM SEAM: the one place where the two registered arms of the
// INFOCOM 2027 transfer are allowed to differ.
//
// WHY IT EXISTS. The comparator arm (algorithms/least_ttft_joint.go) shares the
// candidate set, the estimators, the physics, and the prefix reads with the focal arm
// and differs IN THE OBJECTIVE ONLY. That is what attributes a measured difference to
// the MECHANISM rather than to the MACHINERY: a weaker comparator would leave open
// that the focal arm won because it computes better physics, not because it prices SLO
// externality.
//
// The specification layer states that requirement as a VERBATIM-COPY CONTRACT --
// roughly forty symbols that the comparator must copy from the focal arm unchanged,
// package clause aside (algorithms/least_ttft_joint.go:47-96, config.md section 9.3).
// This port satisfies it by SHARING rather than by copying, which is strictly
// stronger: a copy can drift under an edit to one side, and the contract's own stated
// failure mode is exactly that -- "a re-derived-but-slightly-different estimator would
// silently destroy the attribution argument while every test still passed". There is
// one Coeffs, one AdmissionContext, one rollforward estimator, one shadow table, one
// candidate enumeration, and one argmin, and both arms run them.
//
// The seam mirrors the simulation's own shape. Upstream has ONE joint decider whose
// per-candidate cost function is swapped per rule: decideJoint at sim/edpp.go:1334
// sets costFn to jointSLOExternalityCandidateScore for the focal arm and to
// jointCandidateTTFT in the trailing else at :1448-1464. Objective below is that
// costFn, and Policy.decide is that decider.
//
// WHAT MUST NOT MOVE INTO AN ARM. Anything an arm can reach through Eval is shared by
// construction. Anything an arm reimplements is a silent divergence that no test
// catches, because both implementations compile and both produce plausible
// microsecond numbers. If a future arm needs a physics term Eval does not expose, add
// the accessor here -- do not re-derive the term in the arm's package.

// Objective is the per-candidate cost one arm minimises, plus the two enumeration
// facts that legitimately differ between the registered arms.
//
// It is deliberately narrow. Every method below corresponds to a divergence the
// specification layer NAMES; there is no general-purpose hook, because a hook nothing
// uses is a reader trap that implies a consumer exists somewhere.
type Objective interface {
	// Cost returns the candidate's objective value on the arm's own scale, where
	// LOWER IS BETTER. ps == nil means LOCAL -- prefill co-resident on the decode
	// endpoint ds; otherwise decode on ds and prefill on *ps.
	//
	// The scale is the arm's own and is neither normalised nor clamped: the focal
	// arm's J is signed and unbounded, the comparator's is a latency in microseconds,
	// non-negative and unbounded above. Both would be destroyed by the [0,1] clamp a
	// Scorer contribution passes through, which is why this arm attaches as a Picker
	// (see Picker's doc comment).
	Cost(e Eval, ds Snapshot, ps *Snapshot) float64

	// ScorerFirstEnumeration reports whether the inherited scorer's preferred
	// endpoint is moved to the FRONT of the enumeration as a tie-break order.
	//
	// The focal arm does this, and upstream's scorerFirstSnapshots doc calls the
	// result "the causal-VaR equality tie-break order" (sim/edpp.go:2176-2178). The
	// comparator does NOT: upstream's least-ttft branch iterates decodeSnaps directly
	// (sim/edpp.go:1456) and applies the reordering only in the SLO-externality and
	// causal-VaR branches. Reordering an arm that upstream does not reorder changes
	// which candidate wins an EXACT tie -- silently, and only sometimes.
	ScorerFirstEnumeration() bool

	// ReadsSLOValueConfig reports whether the objective reads the SLO-value half of
	// Config: v, ablation, activeWorkload, workloadTargets, and capacity.
	//
	// It gates VALIDATION, not just behaviour, and in both directions. For an
	// objective that reads them they are required, because an unresolvable tau triple
	// FLATTENS the policy rather than loosening it. For one that does not, they must
	// be ABSENT from the plugin config, because a populated-but-never-read field is a
	// reader trap: it implies a consumer exists, and an operator who tuned tau for
	// the comparator arm would be tuning nothing. The specification layer is explicit
	// that the comparator carries "NO tau FIELD, AND NO NomPrefillTokens", and that
	// "adding either field would imply a consumer that does not exist"
	// (algorithms/least_ttft_joint.go:237-242).
	ReadsSLOValueConfig() bool
}

// Arm binds one objective to the pair of plugin type strings that switch it on.
//
// TWO REGISTRATIONS PER ARM, not one, because ProfileHandler.Pick and Picker.Pick
// share a method name with different signatures
// (pkg/epp/framework/interface/scheduling/plugins.go:47, :77), so no single Go type
// can satisfy both. The types are carried here rather than read from a constant so
// the shared Handler and Picker can serve either arm while still reporting their own
// identity in TypedName, in logs, and in the plugin_type metric label.
//
// EXACTLY ONE ProfileHandler IS PERMITTED across all instantiated plugins
// (pkg/epp/config/loader/configloader.go:382-391 errors with "multiple profile
// handlers found"), so although both arms are registered in the binary, only ONE can
// be instantiated per EPP. The arms are selected by the treatment overlay naming their
// types, one overlay per arm, and they are never co-resident. That is why the shared
// decisionAttributeKey and the shared metric collectors are unambiguous at runtime.
type Arm struct {
	// HandlerType is the config `type:` of the state-owning half.
	HandlerType string
	// PickerType is the config `type:` of the argmin half.
	PickerType string
	// Objective is the per-candidate cost this arm minimises.
	Objective Objective
}

// FocalArm is the focal arm: causal_externality_no_capacity_v8, config.md section 9.2.
var FocalArm = Arm{
	HandlerType: HandlerPluginType,
	PickerType:  PickerPluginType,
	Objective:   focalObjective{},
}

// focalObjective is J = V*(externality - ownGood) + capacity, evaluated by
// scoreCandidate. sim/edpp.go:1688-1779.
type focalObjective struct{}

func (focalObjective) Cost(e Eval, ds Snapshot, ps *Snapshot) float64 {
	return e.p.scoreCandidate(e.ec, ds, ps).total
}

// ScorerFirstEnumeration is true for the focal arm, and it is not cosmetic: it is what
// makes restricting the enumeration to the first decode endpoint reproduce the
// decomposed control exactly (see Ablation.Decomposed).
func (focalObjective) ScorerFirstEnumeration() bool { return true }

func (focalObjective) ReadsSLOValueConfig() bool { return true }

// Eval is the shared physics bound to ONE decision, and it is the whole of the
// interface an arm's objective is given.
//
// EVERY METHOD IS A ONE-LINE DELEGATION to the corresponding Policy method. That is
// the point: an arm cannot hold a coefficient, an admission estimate, a chunk count, a
// transfer price, or a rollout guard of its own, so the two arms cannot drift apart on
// any of them. The verbatim-copy contract is satisfied by there being one copy.
//
// LOCKING. An Eval is only ever constructed inside Policy.decide, which holds
// Policy.mu for the whole decision, and none of the methods below take the lock again.
// nHatOut is resolved once per decision before any candidate is evaluated, so every
// candidate in one argmin sees identical operands (sim/edpp.go:1643-1655).
//
// It is passed BY VALUE and holds only two pointers, so an objective cannot retain
// per-decision state across decisions by accident.
type Eval struct {
	p  *Policy
	ec *evalCtx
}

// InputLen is a_r, the arriving request's prompt length in tokens -- signal 10.
func (e Eval) InputLen() int { return e.ec.inputLen }

// CoeffsFor is theta for a GPU-type label value -- the SINGLE selection point for
// per-endpoint heterogeneous coefficients, so both arms are hardware-aware by the same
// route. An unmapped label was already rejected by the filter.
func (e Eval) CoeffsFor(gpuType string) Coeffs { return e.p.coeffsFor(gpuType) }

// TIterDecode and TIterPrefill expose the two iteration-time laws.
//
// They are routed through Eval rather than exported on Coeffs so the arm-visible
// surface stays in this one file and can be audited without reading the whole package.
// Coeffs.Wp is already exported and is called directly.
func (e Eval) TIterDecode(theta Coeffs, bDec int, kv, sPf int64) float64 {
	return theta.tIterDecode(bDec, kv, sPf)
}

func (e Eval) TIterPrefill(theta Coeffs, sPf int64) float64 { return theta.tIterPrefill(sPf) }

// APForEndpoint is a_p for one endpoint -- the uncached suffix, signal 11. A prefix
// miss returns the full prompt, which over-prices the candidate rather than asserting
// a cold cache as fact.
func (e Eval) APForEndpoint(id string) int { return e.p.apForEndpoint(e.ec, id) }

// ChunkTerms is (nChunks, deltaPfChunk) for ap uncached tokens under the engine's
// batched-token budget.
func (e Eval) ChunkTerms(theta Coeffs, ap int) (nChunks, deltaPfChunk float64) {
	return e.p.chunkTerms(theta, ap)
}

// ProjectedLocalTTFT and ProjectedDisaggTTFT are the two client-visible first-token
// clocks, INCLUDING the post-processing term that sits outside the calibrated theta.
//
// The asymmetry between them is real rather than an oversight: local execution samples
// its first token when prefill completes, so no decode iteration precedes it, while
// the disaggregated form adds the re-timed B+1 first decode iteration
// (sim/edpp.go:1245-1250 against :1252-1255).
func (e Eval) ProjectedLocalTTFT(tAdmD, nChunks, tIterD, wpLoc float64) float64 {
	return e.p.projectedLocalTTFT(tAdmD, nChunks, tIterD, wpLoc)
}

func (e Eval) ProjectedDisaggTTFT(decodeJoinUs, tIterFirstDecode float64) float64 {
	return e.p.projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode)
}

// DecodeAdmissionUs and PrefillAdmissionUs are t_adm on each pool.
//
// EVERY CALL INCREMENTS THE D1 SUBSTITUTION COUNTER, on both arms, because every call
// is one the published policy would have served with the scheduler rollout. Both arms
// must be compared at the same estimator regime -- the comparator's own header is
// explicit that D1's direction is NOT a single sign for a least-TTFT objective, so
// "both arms inherit D1 identically" is not a claim this port may make
// (algorithms/least_ttft_joint.go:100-125).
func (e Eval) DecodeAdmissionUs(ds Snapshot) float64 {
	return e.p.estimateTAdm(e.p.decodeAdmissionCtx(e.ec, ds))
}

func (e Eval) PrefillAdmissionUs(ps Snapshot) float64 {
	return e.p.estimateTAdm(e.p.prefillAdmissionCtx(e.ec, ps))
}

// CXferUs is the KV-transfer price of going remote -- degradation D5, UNMEASURED. It
// takes no argument because reqKVNeed is candidate-invariant and already resolved for
// this decision.
func (e Eval) CXferUs() float64 { return e.p.cXferUsFor(e.ec.reqKVNeed) }

// RolloutLocalTTFT, RolloutDecodeAdmission, and RolloutPrefillCompletion are the
// DEGRADATION D1 branches: the published scheduler rollout, which is unreachable at
// this pin because Snapshot.SchedulerStateObserved is permanently false.
//
// Each returns ok == false today, so the closed-form estimate above it stands. They
// are exposed rather than elided so that a future engine patch supplying the wait
// queue activates BOTH arms at once -- an arm that had quietly dropped the branch
// would keep running the substitute while the other switched, which is the worst
// possible state for a comparison.
func (e Eval) RolloutLocalTTFT(ds Snapshot, theta Coeffs) (tAdm, ttft float64, ok bool) {
	return e.p.rolloutLocalTTFT(e.ec, ds, theta)
}

func (e Eval) RolloutDecodeAdmission(ds Snapshot, theta Coeffs) (float64, bool) {
	return e.p.rolloutDecodeAdmission(e.ec, ds, theta)
}

func (e Eval) RolloutPrefillCompletion(ps Snapshot, theta Coeffs) (tAdm, completion float64, ok bool) {
	return e.p.rolloutPrefillCompletion(e.ec, ps, theta)
}

// NewHandlerFactory returns the ProfileHandler-half factory for one arm.
//
// The returned closure is the whole of what differs between arms at construction time:
// the type string it reports, the objective it installs in the shared Policy, and the
// half of Config its validation demands.
func NewHandlerFactory(arm Arm) fwkplugin.FactoryFunc {
	return func(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return newHandler(arm, name, rawParameters, handle)
	}
}

// NewPickerFactory returns the Filter+Picker-half factory for one arm.
func NewPickerFactory(arm Arm) fwkplugin.FactoryFunc {
	return func(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
		return newPicker(arm, name, rawParameters, handle)
	}
}

// NewPickerConfigParser returns the ConfigParserFunc that exposes an arm's picker
// dependency on its handler, for RegisterWithPluginDependencies.
//
// IT MUST BE REGISTERED ALONGSIDE THE FACTORY. Registering the factory with plain
// Register leaves the type out of PluginsWithPluginDependencies, the `pluginRef` tag on
// pickerParameters.HandlerPluginName is then never read, and construction order falls
// back to ranging a Go map -- a nondeterministic crashloop. See dependency_test.go.
func NewPickerConfigParser(arm Arm) fwkplugin.ConfigParserFunc {
	return func(rawParameters *json.Decoder, _ fwkplugin.Handle) (any, error) {
		return parsePickerParameters(arm, rawParameters)
	}
}

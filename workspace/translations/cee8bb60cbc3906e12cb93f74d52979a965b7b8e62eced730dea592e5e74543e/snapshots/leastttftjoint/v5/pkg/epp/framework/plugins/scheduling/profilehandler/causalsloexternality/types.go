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

// RunningReqState mirrors sim/admission_estimator.go:19-40.
//
// TrueRemaining is the ORACLE remaining step count and is ALWAYS -1 on this
// target: the oracle reads hidden output length, so it is a diagnostic upper
// bound and never a deployable policy. The port therefore always takes the
// censored branch in varDecodeInputs.
//
// One exception to that sign convention, inherited from upstream: on the PREFILL
// slice TrueRemaining carries remaining PROMPT tokens, which is known input and
// needs no censoring.
type RunningReqState struct {
	// RequestID is the shadow table's key, carried so populations can be returned in a
	// deterministic order -- the table's own index is a map.
	RequestID     string
	StepsDone     int64
	KVBlocks      int64
	TrueRemaining int64

	SLOClass     string
	ArrivalUs    int64
	FirstTokenUs int64
	TTFTSet      bool
}

// SchedulerReqState mirrors sim/admission_estimator.go:44-53.
//
// It carries no output length: the rollout supplies the censored per-class
// running mean instead. Populated only if a future engine patch exports the wait
// queue -- see Snapshot.SchedulerStateObserved.
type SchedulerReqState struct {
	ID              string
	SLOClass        string
	PromptTokens    int64
	ComputedTokens  int64
	ScheduledTokens int64
	KVBlocks        int64
	ArrivalUs       int64
}

// Snapshot is the per-endpoint routing view the candidate scorer reads. It
// mirrors the simulator's RoutingSnapshot for the fields this arm uses.
//
// The comment on each scraped field names the signal number in config.md
// section 5 and the accessor it comes from, so the mapping can be audited
// without leaving this file.
type Snapshot struct {
	// ID is the endpoint's stable identity, GetMetadata().ID.String().
	ID string
	// GPUType is the signal-8 pod-label value. An endpoint reaching here always
	// has a non-empty value that keys Config.CoeffsByGPUType: the filter rejects
	// every other endpoint rather than defaulting one.
	GPUType string

	BatchSize             int   // signal 1:  Metrics.RunningRequestsSize
	QueueDepth            int   // signal 2:  Metrics.WaitingQueueSize
	KvTokensInUse         int64 // signal 6:  derived, usage * blockSize * numBlocks
	FreeKVBlocks          int64 // signal 7:  derived, D7 -- a floor
	ResidentPrefillTokens int64 // signal 14: D4 -- capped shadow sum
	MaxBatchSize          int64 // config, = max_num_seqs (no metric exposes it)
	BlockSizeTokens       int64 // signal 3:  Metrics.CacheBlockSize

	RunningDecode  []RunningReqState // signal 12: built shadow table, D2
	RunningPrefill []RunningReqState // signal 13: same table, prefill index, D6

	// SchedulerStateObserved is the DEGRADATION D1 GUARD, and it is permanently
	// false at this pin. No route exists to the engine's ordered wait queue, its
	// current grants, or its step start instant -- the target exposes
	// Metrics.WaitingQueueSize, one integer. It is carried explicitly so the
	// substitution is visible at the point it happens, and so a future engine
	// patch supplying the queue has a reachable branch rather than needing the
	// rollout re-derived (sim/edpp_scheduler_rollout.go:299).
	SchedulerStateObserved    bool
	SchedulerRunning          []SchedulerReqState
	SchedulerWaiting          []SchedulerReqState
	CurrentScheduled          []SchedulerReqState
	CurrentStepStartUs        int64
	MaxScheduledTokens        int64
	LongPrefillTokenThreshold int64

	// AdmissionRate feeds only the `little` estimator, which no registered arm
	// selects, so it is unset on this target. Declared rather than removed so the
	// omission reads as a decision: a port switching TAdmEstimator to `little`
	// must first build the arrival-rate account.
	AdmissionRate float64
}

// evalCtx holds the per-decision, candidate-invariant terms, so every candidate
// evaluation in one argmin uses identical operands. sim/edpp.go:1643-1655.
type evalCtx struct {
	class     string
	inputLen  int
	reqKVNeed int64
	nHatOut   float64
	nowUs     float64
	requestID string

	// apByEndpoint is a_p per endpoint ID -- the uncached suffix, computed in the
	// filter where the request is in scope and the per-endpoint prefix attribute
	// is readable. Picker.Pick receives no request, so this map is how a_p reaches
	// the argmin.
	apByEndpoint map[string]int
}

// pathBreakdown separates the three resident populations a placement can harm.
// sim/edpp_var.go:694-705. Kept explicit so the aggregate is auditable rather
// than a single opaque number.
type pathBreakdown struct {
	decode        float64
	collocPrefill float64
	prefillPool   float64
}

func (v pathBreakdown) total() float64 {
	return v.decode + v.collocPrefill + v.prefillPool
}

// candidateScore is one candidate's score breakdown. sim/edpp.go:1676-1682.
//
// The capacity fields are dead while Ablation.NoCapacity is true, which is the
// focal arm's setting. They are retained because the ablation cohort's validity
// gate reads the breakdown, not just the total.
type candidateScore struct {
	externalityBreakdown pathBreakdown
	externality          float64
	ownGood              float64
	netGoodCost          float64

	capacityQueueDecode   float64
	capacityQueuePrefill  float64
	capacityDemandDecode  float64
	capacityDemandPrefill float64
	capacityDecode        float64
	capacityPrefill       float64
	capacityTotal         float64

	total float64
}

// candidate is one enumerated (decode, placement) pair with its objective.
// sim/edpp.go:1630-1636.
type candidate struct {
	dID   string
	pID   string
	local bool
	J     float64
}

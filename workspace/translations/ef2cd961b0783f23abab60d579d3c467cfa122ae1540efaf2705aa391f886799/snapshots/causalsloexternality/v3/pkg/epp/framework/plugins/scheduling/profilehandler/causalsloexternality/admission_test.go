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
	"math"
	"testing"
)

// TestFlooredTAdmLowerBoundsByOneIteration pins the floor's meaning: even with a free
// slot, a request waits for the current decode step to finish before the next batch
// formation admits it.
func TestFlooredTAdmLowerBoundsByOneIteration(t *testing.T) {
	ctx := AdmissionContext{TIter: 17000}
	if got := flooredTAdm(0, ctx); got != 17000 {
		t.Errorf("floor not applied: got %g, want 17000", got)
	}
	if got := flooredTAdm(50000, ctx); got != 50000 {
		t.Errorf("floor must not lower an estimate already above it: got %g, want 50000", got)
	}
	// No-op when TIter is unavailable.
	if got := flooredTAdm(0, AdmissionContext{}); got != 0 {
		t.Errorf("with no TIter the floor is a no-op: got %g, want 0", got)
	}
}

// TestRollforwardAdmitsImmediatelyWhenSlotAndKVFit is the fast path: a slot free AND
// enough KV already fits, so the only wait is the current iteration.
func TestRollforwardAdmitsImmediatelyWhenSlotAndKVFit(t *testing.T) {
	ctx := AdmissionContext{
		BatchSize:    4,
		MaxBatchSize: 256,
		FreeKVBlocks: 1000,
		ReqKVNeed:    100,
		TIter:        17000,
	}
	if got := rollforwardEstimateTAdm(ctx); got != 17000 {
		t.Errorf("expected the one-iteration floor, got %g", got)
	}
}

// TestRollforwardWaitsForKVEvenWithFreeSlots is the case a slot-only test would miss:
// the batch is not full, but the arriving request's KV does not fit, so admission must
// wait for a departure to free blocks.
func TestRollforwardWaitsForKVEvenWithFreeSlots(t *testing.T) {
	ctx := AdmissionContext{
		BatchSize:         2,
		MaxBatchSize:      256,
		FreeKVBlocks:      10,
		ReqKVNeed:         100,
		TIter:             1000,
		QueueDepth:        0,
		RemainingStepsEst: 5,
		Running: []RunningReqState{
			{TrueRemaining: -1, KVBlocks: 20}, // frees 20 -> 30, still short
			{TrueRemaining: -1, KVBlocks: 80}, // frees 80 -> 110, now enough
		},
	}
	// Both have TrueRemaining -1, so each departs at RemainingStepsEst = 5 steps.
	// The second departure satisfies the KV need: 5 * 1000 = 5000.
	if got := rollforwardEstimateTAdm(ctx); got != 5000 {
		t.Errorf("expected 5000 (departure step 5 x TIter 1000), got %g", got)
	}
}

// TestRollforwardUsesCensoredEstimateNotOracle is the deployability property: on this
// target TrueRemaining is always -1, so the estimator must fall back to the class
// estimate. A port that read the oracle would be using hidden output length.
func TestRollforwardUsesCensoredEstimateNotOracle(t *testing.T) {
	base := AdmissionContext{
		BatchSize:         256,
		MaxBatchSize:      256,
		FreeKVBlocks:      0,
		ReqKVNeed:         1,
		TIter:             1000,
		QueueDepth:        0,
		RemainingStepsEst: 7,
		Running:           []RunningReqState{{TrueRemaining: -1, KVBlocks: 10}},
	}
	if got := rollforwardEstimateTAdm(base); got != 7000 {
		t.Errorf("censored branch: expected 7 x 1000, got %g", got)
	}

	// If an oracle value were ever present it would be honoured -- the branch exists --
	// but nothing on this target produces one.
	oracle := base
	oracle.Running = []RunningReqState{{TrueRemaining: 3, KVBlocks: 10}}
	if got := rollforwardEstimateTAdm(oracle); got != 3000 {
		t.Errorf("oracle branch: expected 3 x 1000, got %g", got)
	}
}

// TestRollforwardFloorsRemainingAtOne guards the degenerate case where the class
// estimate rounds below one step.
func TestRollforwardFloorsRemainingAtOne(t *testing.T) {
	ctx := AdmissionContext{
		BatchSize:         256,
		MaxBatchSize:      256,
		FreeKVBlocks:      100,
		ReqKVNeed:         1,
		TIter:             1000,
		RemainingStepsEst: 0.4, // int64() truncates to 0, must floor to 1
		Running:           []RunningReqState{{TrueRemaining: -1, KVBlocks: 1}},
	}
	if got := rollforwardEstimateTAdm(ctx); got != 1000 {
		t.Errorf("expected the floored 1-step departure, got %g", got)
	}
}

// TestRollforwardFallsBackToWaveFormWhenQueueDeeperThanOneDrain exercises DEGRADATION
// D1's biasing branch by name. Past one batch drain the estimator leaves the exact
// departure walk and uses the fluid wave form, which UNDERSTATES admission delay.
func TestRollforwardFallsBackToWaveFormWhenQueueDeeperThanOneDrain(t *testing.T) {
	ctx := AdmissionContext{
		BatchSize:         2,
		MaxBatchSize:      2,
		FreeKVBlocks:      0,
		ReqKVNeed:         1,
		TIter:             1000,
		QueueDepth:        10, // needs 11 slots; only 2 departures available
		RemainingStepsEst: 4,
		Running: []RunningReqState{
			{TrueRemaining: -1, KVBlocks: 1},
			{TrueRemaining: -1, KVBlocks: 1},
		},
	}
	// waves = ceil((10+1)/2) = 6; 6 * 4 * 1000 = 24000.
	want := math.Ceil(11.0/2.0) * 4 * 1000
	if got := rollforwardEstimateTAdm(ctx); got != want {
		t.Errorf("wave fallback: got %g, want %g", got, want)
	}

	// And confirm the DIRECTION D1 claims: the wave form here is smaller than an honest
	// serialization of 11 admissions at 4 steps each would be.
	serialized := 11.0 * 4 * 1000
	if got := rollforwardEstimateTAdm(ctx); got >= serialized {
		t.Errorf("wave form should understate a serialized estimate (%g), got %g", serialized, got)
	}
}

// TestRollforwardStableSortDeterminism is the property the specification calls out as
// not cosmetic: two running requests with EQUAL remaining steps but different KV
// holdings are accumulated in slice order, so an unstable sort could change which
// departure first satisfies the KV condition and therefore change the returned delay.
func TestRollforwardStableSortDeterminism(t *testing.T) {
	ctx := AdmissionContext{
		BatchSize:         256,
		MaxBatchSize:      256,
		FreeKVBlocks:      0,
		ReqKVNeed:         50,
		TIter:             1000,
		RemainingStepsEst: 5,
		Running: []RunningReqState{
			{TrueRemaining: 2, KVBlocks: 10},
			{TrueRemaining: 2, KVBlocks: 60},
			{TrueRemaining: 9, KVBlocks: 5},
		},
	}
	first := rollforwardEstimateTAdm(ctx)
	for i := 0; i < 50; i++ {
		if got := rollforwardEstimateTAdm(ctx); got != first {
			t.Fatalf("non-deterministic result on iteration %d: %g then %g", i, first, got)
		}
	}
}

// TestFluidEstimatorWaveForm pins the wave arithmetic, which is live on this target
// because rollforward falls back to it.
func TestFluidEstimatorWaveForm(t *testing.T) {
	ctx := AdmissionContext{
		BatchSize:         4,
		MaxBatchSize:      4,
		FreeKVBlocks:      0,
		ReqKVNeed:         1,
		TIter:             1000,
		QueueDepth:        7,
		RemainingStepsEst: 3,
	}
	// waves = ceil(8/4) = 2; 2 * 3 * 1000 = 6000. Deliberately NOT the naive fluid
	// drain, which would divide the queue by the batch without the wave rounding.
	if got := fluidEstimateTAdm(ctx); got != 6000 {
		t.Errorf("fluid wave form: got %g, want 6000", got)
	}
}

// TestUnselectedEstimatorsReadOnlyTheirOwnInputs documents why QWork and AdmissionRate
// are declared-but-unread on this target: each has exactly one consumer, and neither
// consumer is selectable here.
func TestUnselectedEstimatorsReadOnlyTheirOwnInputs(t *testing.T) {
	// `waiting` is the only consumer of QWork.
	if got := waitingEstimateTAdm(AdmissionContext{QWork: 5000, Mu: 0.5}); got != 10000 {
		t.Errorf("waiting: got %g, want 10000", got)
	}
	if got := waitingEstimateTAdm(AdmissionContext{QWork: 5000, Mu: 0}); got != 0 {
		t.Errorf("waiting with zero mu must not divide: got %g", got)
	}
	// `little` is the only consumer of AdmissionRate.
	if got := littleEstimateTAdm(AdmissionContext{QueueDepth: 10, AdmissionRate: 0.001, TIter: 1}); got != 10000 {
		t.Errorf("little: got %g, want 10000", got)
	}
	// With the target's unset AdmissionRate it degenerates to the floor -- which is
	// exactly why it is not selectable.
	if got := littleEstimateTAdm(AdmissionContext{QueueDepth: 10, TIter: 500}); got != 500 {
		t.Errorf("little with unset rate: got %g, want the 500 floor", got)
	}
}

// TestCensorOracleRemainingForcesMinusOne pins the invariant that no estimator can read
// a hidden output length.
func TestCensorOracleRemainingForcesMinusOne(t *testing.T) {
	in := []RunningReqState{
		{TrueRemaining: 42, StepsDone: 5, KVBlocks: 7},
		{TrueRemaining: -1, StepsDone: 1},
	}
	out := censorOracleRemaining(in)
	for i, r := range out {
		if r.TrueRemaining != -1 {
			t.Errorf("entry %d: TrueRemaining = %d, want -1", i, r.TrueRemaining)
		}
	}
	// It must COPY: the caller's slice is not reordered or mutated.
	if in[0].TrueRemaining != 42 {
		t.Error("censorOracleRemaining mutated its input")
	}
	// Other fields survive.
	if out[0].StepsDone != 5 || out[0].KVBlocks != 7 {
		t.Error("censoring must not disturb other fields")
	}
	if censorOracleRemaining(nil) != nil {
		t.Error("nil input must yield nil")
	}
}

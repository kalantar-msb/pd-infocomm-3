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

import "testing"

// The scheduler rollout is UNREACHABLE IN PRODUCTION at this pin -- degradation D1 --
// because Snapshot.SchedulerStateObserved is permanently false: no route exists to the
// engine's ordered wait queue, its current grants, or its step start instant.
//
// It is tested anyway, with a synthesised observed snapshot, because "unreachable" is a
// property of the target's metrics surface rather than of this algebra. A future engine
// patch exporting the wait queue makes the branch live without re-deriving anything, and
// these tests are what would catch a transcription error at that point.

// TestRolloutGuardIsClosedOnThisTarget is the D1 fact, asserted rather than assumed: with a
// realistic snapshot the rollout declines and every caller falls through to the closed-form
// substitute.
func TestRolloutGuardIsClosedOnThisTarget(t *testing.T) {
	p := testPolicy(t, armConfig())
	// A snapshot as the handler actually builds it: SchedulerStateObserved is false.
	snap := decodeSnap("d1", "H100_SXM_80GB", 4, 0, 20000)
	ec := testEvalCtx(p, 4000, map[string]int{"d1": 4000})

	if _, ok := p.rolloutTimes(ec, snap, h100, 0, false, false); ok {
		t.Error("the rollout guard must be closed at this pin")
	}
	if _, _, ok := p.rolloutLocalTTFT(ec, snap, h100); ok {
		t.Error("rolloutLocalTTFT must decline")
	}
	if _, ok := p.rolloutDecodeAdmission(ec, snap, h100); ok {
		t.Error("rolloutDecodeAdmission must decline")
	}
	if _, _, ok := p.rolloutPrefillCompletion(ec, snap, h100); ok {
		t.Error("rolloutPrefillCompletion must decline")
	}
}

// observedSnap synthesises the scheduler state the engine does not export, so the rollout
// branch can be exercised.
func observedSnap(id string, waiting []SchedulerReqState) Snapshot {
	return Snapshot{
		ID: id, GPUType: "H100_SXM_80GB",
		BatchSize: 0, MaxBatchSize: 256,
		FreeKVBlocks: 10000, BlockSizeTokens: 16,
		SchedulerStateObserved: true,
		SchedulerRunning:       nil,
		SchedulerWaiting:       waiting,
		MaxScheduledTokens:     2048,
	}
}

func TestRolloutAdmitsTargetFromAnEmptyEngine(t *testing.T) {
	p := testPolicy(t, armConfig())
	snap := observedSnap("d1", nil)
	ec := testEvalCtx(p, 100, map[string]int{"d1": 100})

	result, ok := p.rolloutTimes(ec, snap, h100, 0, false, false)
	if !ok {
		t.Fatal("the target must be admitted on an empty engine")
	}
	if !result.admitted {
		t.Error("admitted must be set")
	}
	if !result.firstToken {
		t.Error("the target must reach its first token")
	}
	// A 100-token prompt fits in one 2048-token chunk, so prefill completes in one step and
	// that step IS the first token.
	if result.firstTokenUs <= 0 {
		t.Errorf("firstTokenUs = %g, want positive", result.firstTokenUs)
	}
	if result.admissionUs > result.firstTokenUs {
		t.Error("admission cannot come after the first token")
	}
}

// TestRolloutChunksLongPromptsAgainstTheTokenBudget pins the chunked-prefill model: a prompt
// longer than the budget takes multiple steps to reach its first token.
func TestRolloutChunksLongPromptsAgainstTheTokenBudget(t *testing.T) {
	p := testPolicy(t, armConfig())
	ec := testEvalCtx(p, 100, map[string]int{"d1": 100})
	shortResult, _ := p.rolloutTimes(ec, observedSnap("d1", nil), h100, 0, false, false)

	ecLong := testEvalCtx(p, 6000, map[string]int{"d1": 6000}) // 3 chunks at 2048
	longResult, ok := p.rolloutTimes(ecLong, observedSnap("d1", nil), h100, 0, false, false)
	if !ok {
		t.Fatal("the long prompt must still be admitted")
	}
	if longResult.firstTokenUs <= shortResult.firstTokenUs {
		t.Errorf("a longer prompt must take longer to first token: short=%g long=%g",
			shortResult.firstTokenUs, longResult.firstTokenUs)
	}
}

// TestRolloutDecodeOnlyTreatsPromptAsComputed models a TRANSFERRED request: its prompt is
// already computed elsewhere, so it reaches its first token on a decode grant rather than
// after prefill.
func TestRolloutDecodeOnlyTreatsPromptAsComputed(t *testing.T) {
	p := testPolicy(t, armConfig())
	ec := testEvalCtx(p, 6000, map[string]int{"d1": 6000})

	local, _ := p.rolloutTimes(ec, observedSnap("d1", nil), h100, 0, false, false)
	transferred, ok := p.rolloutTimes(ec, observedSnap("d1", nil), h100, 6000, true, false)
	if !ok {
		t.Fatal("a transferred request must be admitted")
	}
	if transferred.firstTokenUs >= local.firstTokenUs {
		t.Errorf("a transferred request skips prefill, so it must reach its first token sooner: "+
			"transferred=%g local=%g", transferred.firstTokenUs, local.firstTokenUs)
	}
}

// TestRolloutPrefillPoolUsesPrefillIntercept confirms the alpha selection: a prefill-pool
// endpoint is charged alpha_p, not the decode intercept.
func TestRolloutPrefillPoolUsesPrefillIntercept(t *testing.T) {
	p := testPolicy(t, armConfig())
	ec := testEvalCtx(p, 100, map[string]int{"p1": 100})

	// Use coefficients whose two intercepts differ sharply, so the selection is visible.
	theta := h100
	theta.AlphaP = theta.AlphaD * 1.05 // within validate()'s 10% bound

	decodeAlpha, _ := p.rolloutTimes(ec, observedSnap("p1", nil), theta, 0, false, false)
	prefillAlpha, ok := p.rolloutTimes(ec, observedSnap("p1", nil), theta, 0, false, true)
	if !ok {
		t.Fatal("the prefill-pool rollout must admit")
	}
	if prefillAlpha.firstTokenUs <= decodeAlpha.firstTokenUs {
		t.Errorf("the prefill pool must be charged the larger alpha_p: prefill=%g decode=%g",
			prefillAlpha.firstTokenUs, decodeAlpha.firstTokenUs)
	}
}

// TestRolloutQueuedWorkDelaysAdmission pins that the target is appended to the WAIT QUEUE,
// so its admission is predicted rather than assumed.
func TestRolloutQueuedWorkDelaysAdmission(t *testing.T) {
	p := testPolicy(t, armConfig())
	p.observeCompletedOutput(sloClassStandard, 50)
	ec := testEvalCtx(p, 100, map[string]int{"d1": 100})

	empty, _ := p.rolloutTimes(ec, observedSnap("d1", nil), h100, 0, false, false)

	// Queue several requests ahead of the target, each with a full prompt to prefill.
	waiting := []SchedulerReqState{}
	for i := 0; i < 5; i++ {
		waiting = append(waiting, SchedulerReqState{
			ID: string(rune('a' + i)), SLOClass: sloClassStandard,
			PromptTokens: 2000, ComputedTokens: 0,
		})
	}
	busy, ok := p.rolloutTimes(ec, observedSnap("d1", waiting), h100, 0, false, false)
	if !ok {
		t.Fatal("the target must still be admitted eventually")
	}
	if busy.firstTokenUs <= empty.firstTokenUs {
		t.Errorf("queued work ahead of the target must delay it: empty=%g busy=%g",
			empty.firstTokenUs, busy.firstTokenUs)
	}
}

// TestRolloutReqForCensorsOutputEstimate pins the censoring: a request that has already
// decoded decodeDone tokens has total output >= decodeDone, so the class mean is floored by
// that before subtracting, and the remainder floored at 1.
func TestRolloutReqForCensorsOutputEstimate(t *testing.T) {
	p := testPolicy(t, armConfig())
	p.observeCompletedOutput(sloClassStandard, 100) // class mean 100

	// Still prefilling: no decode done, so the full class mean remains.
	prefilling := p.rolloutReqFor(SchedulerReqState{
		SLOClass: sloClassStandard, PromptTokens: 2000, ComputedTokens: 500,
	})
	if prefilling.outputRemaining != 100 {
		t.Errorf("outputRemaining = %d, want the full class mean 100", prefilling.outputRemaining)
	}

	// Mid-decode: 40 tokens produced, so 60 remain.
	midDecode := p.rolloutReqFor(SchedulerReqState{
		SLOClass: sloClassStandard, PromptTokens: 2000, ComputedTokens: 2040,
	})
	if midDecode.outputRemaining != 60 {
		t.Errorf("outputRemaining = %d, want 60", midDecode.outputRemaining)
	}

	// Past the class mean: the estimate is floored BY decodeDone, and the remainder at 1.
	past := p.rolloutReqFor(SchedulerReqState{
		SLOClass: sloClassStandard, PromptTokens: 2000, ComputedTokens: 2500,
	})
	if past.outputRemaining != 1 {
		t.Errorf("outputRemaining = %d, want the floor of 1", past.outputRemaining)
	}

	// Negative inputs are clamped rather than propagated.
	negative := p.rolloutReqFor(SchedulerReqState{
		SLOClass: sloClassStandard, PromptTokens: -5, ComputedTokens: -10, KVBlocks: -3,
	})
	if negative.prompt != 0 || negative.computed != 0 || negative.kvBlocks != 0 {
		t.Errorf("negative fields must clamp to 0, got %+v", negative)
	}
}

func TestCeilBlocks(t *testing.T) {
	for _, tc := range []struct {
		tokens, blockSize, want int64
	}{
		{0, 16, 0},
		{-5, 16, 0},
		{1, 16, 1},
		{16, 16, 1},
		{17, 16, 2},
		{4000, 16, 250},
		{100, 0, 100}, // a non-positive block size degrades to 1 token per block
	} {
		if got := ceilBlocks(tc.tokens, tc.blockSize); got != tc.want {
			t.Errorf("ceilBlocks(%d, %d) = %d, want %d", tc.tokens, tc.blockSize, got, tc.want)
		}
	}
}

// TestRolloutGrantTimeChargesPrefillCausally pins the per-grant charges: a prefill grant is
// charged causally against the prefix already computed, a decode grant C0 + C1*context.
func TestRolloutGrantTimeChargesPrefillCausally(t *testing.T) {
	req := &rolloutReq{}
	// One prefill grant of 500 tokens starting from a prefix of 1000.
	prefill := []rolloutGrant{{req: req, before: 1000, grant: 500, prefill: true}}
	want := h100.AlphaD + h100.CPf*500 + h100.CAttn*500*(1000+250)
	closeTo(t, rolloutGrantTime(prefill, h100, h100.AlphaD), want, "prefill grant time")

	// One decode grant at context 2000.
	decode := []rolloutGrant{{req: req, before: 2000, grant: 1, prefill: false}}
	wantDecode := h100.AlphaD + h100.C0 + h100.C1*2000
	closeTo(t, rolloutGrantTime(decode, h100, h100.AlphaD), wantDecode, "decode grant time")

	// A later prefix costs more than an earlier one for the same grant, which is the
	// causal part.
	early := rolloutGrantTime([]rolloutGrant{{req: req, before: 0, grant: 500, prefill: true}}, h100, h100.AlphaD)
	late := rolloutGrantTime([]rolloutGrant{{req: req, before: 10000, grant: 500, prefill: true}}, h100, h100.AlphaD)
	if late <= early {
		t.Error("a prefill grant deeper into the prompt must cost more")
	}
}

// TestCurrentScheduledTimeIgnoresUngrantedRequests pins that only requests with a positive
// scheduled-token count contribute to the in-flight step's duration.
func TestCurrentScheduledTimeIgnoresUngrantedRequests(t *testing.T) {
	states := []SchedulerReqState{
		{PromptTokens: 2000, ComputedTokens: 0, ScheduledTokens: 0}, // no grant
		{PromptTokens: 2000, ComputedTokens: 0, ScheduledTokens: 0}, // no grant
	}
	if got := currentScheduledTime(states, h100, h100.AlphaD); got != 0 {
		t.Errorf("with no grants the in-flight step has no duration, got %g", got)
	}

	withGrant := []SchedulerReqState{{PromptTokens: 2000, ComputedTokens: 0, ScheduledTokens: 512}}
	if got := currentScheduledTime(withGrant, h100, h100.AlphaD); got <= 0 {
		t.Errorf("a granted step must have a positive duration, got %g", got)
	}
}

// TestRolloutRespectsLongPrefillThreshold pins that the chunk cap is the MIN of the token
// budget and the long-prefill threshold when the latter is set.
func TestRolloutRespectsLongPrefillThreshold(t *testing.T) {
	p := testPolicy(t, armConfig())
	ec := testEvalCtx(p, 6000, map[string]int{"d1": 6000})

	snap := observedSnap("d1", nil)
	unthrottled, _ := p.rolloutTimes(ec, snap, h100, 0, false, false)

	throttled := snap
	throttled.LongPrefillTokenThreshold = 512 // far below the 2048 budget
	got, ok := p.rolloutTimes(ec, throttled, h100, 0, false, false)
	if !ok {
		t.Fatal("the throttled rollout must still admit")
	}
	if got.firstTokenUs <= unthrottled.firstTokenUs {
		t.Errorf("a lower prefill chunk cap needs more steps to first token: "+
			"unthrottled=%g throttled=%g", unthrottled.firstTokenUs, got.firstTokenUs)
	}
}

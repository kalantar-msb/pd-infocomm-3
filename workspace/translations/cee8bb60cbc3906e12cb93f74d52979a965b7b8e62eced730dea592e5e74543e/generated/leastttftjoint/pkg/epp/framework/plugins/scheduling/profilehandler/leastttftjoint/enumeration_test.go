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
	"reflect"
	"testing"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality"
)

// The tests in this file run the REAL Filter and Picker over the real shared decider, and
// they cover the two properties that no per-candidate arithmetic test can reach:
//
//   - THE CANDIDATE SET IS SHARED. The arms exist to differ in the objective only, so they
//     must enumerate the same D local plus D*P disaggregated candidates, on one scale, in
//     one argmin. An arm that enumerated its own candidates would no longer be comparable,
//     and nothing in a single-arm test would notice.
//   - THE ENUMERATION ORDER DIVERGES, deliberately and in one direction only. This is a
//     tie-break order, so it changes behaviour only on an EXACT tie -- silently, and only
//     sometimes. It is asserted directly for that reason.
//
// The mechanism is a recording objective wrapped around each arm's real one, which is
// possible precisely because the decider takes the objective as a parameter.

// recordingObjective records every candidate the shared decider evaluates, in order, and
// delegates the cost to the arm's real objective.
type recordingObjective struct {
	inner causalsloexternality.Objective
	seen  *[]string
}

func (r recordingObjective) Cost(e causalsloexternality.Eval, ds causalsloexternality.Snapshot,
	ps *causalsloexternality.Snapshot) float64 {
	key := ds.ID + " local"
	if ps != nil {
		key = ds.ID + " via " + ps.ID
	}
	*r.seen = append(*r.seen, key)
	return r.inner.Cost(e, ds, ps)
}

func (r recordingObjective) ScorerFirstEnumeration() bool { return r.inner.ScorerFirstEnumeration() }
func (r recordingObjective) ReadsSLOValueConfig() bool    { return r.inner.ReadsSLOValueConfig() }

// recordingArm wraps one arm's objective. The type strings are local to the test: these
// arms are constructed directly, never registered, so they cannot collide with the real
// ones. They must still differ from each other, because the picker's cross-arm guard
// compares them.
func recordingArm(name string, inner causalsloexternality.Objective, seen *[]string) causalsloexternality.Arm {
	return causalsloexternality.Arm{
		HandlerType: name + "-handler",
		PickerType:  name + "-picker",
		Objective:   recordingObjective{inner: inner, seen: seen},
	}
}

// runDecision drives one whole decision through the production path: Filter assembles the
// routing view and attaches it to the endpoints, then Pick runs the argmin.
func runDecision(t *testing.T, cfgArm causalsloexternality.Arm, cfg causalsloexternality.Config,
	request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint,
	scores map[string]float64) *fwksched.ProfileRunResult {
	t.Helper()
	_, plugin := buildArmWith(t, cfgArm, cfg)

	filter, ok := plugin.(fwksched.Filter)
	if !ok {
		t.Fatal("the picker half must implement Filter")
	}
	picker, ok := plugin.(fwksched.Picker)
	if !ok {
		t.Fatal("the picker half must implement Picker")
	}

	ctx := context.Background()
	accepted := filter.Filter(ctx, request, endpoints)
	return picker.Pick(ctx, scoredFrom(accepted, scores))
}

// The endpoint IDs of the test fleet. EndpointMetadata.ID.String() is the identity the arm
// keys on, so these are "<namespace>/<name>".
const (
	decodeH100ID = "default/d1"
	decodeA100ID = "default/d2"
	prefillID    = "default/p1"
)

// fleet is the 2D+1P fleet the campaign's heterogeneity condition needs: one H100 decode
// instance, one A100 decode instance, and one H100 prefill instance. config.md section 2
// notes the scenario schema cannot express the per-instance decode hardware, which is why
// theta is keyed by pod label instead.
func fleet(decodeMetrics *fwkdl.Metrics) []fwksched.Endpoint {
	return []fwksched.Endpoint{
		testEndpoint("d1", bylabel.RoleDecode, gpuH100, decodeMetrics),
		testEndpoint("d2", bylabel.RoleDecode, gpuA100, decodeMetrics),
		testEndpoint("p1", bylabel.RolePrefill, gpuH100, nil),
	}
}

// congestedMetrics puts the decode endpoints at full batch with their KV exhausted, which
// makes decode admission dominate the disaggregated clock join.
func congestedMetrics() *fwkdl.Metrics {
	return &fwkdl.Metrics{
		CacheBlockSize:      16,
		CacheNumBlocks:      10000,
		KVCacheUsagePercent: 1.0,
		RunningRequestsSize: 256,
		WaitingQueueSize:    1000,
	}
}

// ---------------------------------------------------------------------------
// The shared candidate set
// ---------------------------------------------------------------------------

// TestArgminEnumeratesTheFullCrossProduct pins the REQUIRED STRUCTURAL SHAPE: D local
// candidates PLUS D*P disaggregated candidates, all scored on one scale in one argmin.
//
// The alternative the target invites is decode-first decomposition -- pick a decode
// endpoint with the stock scorers, then decide placement -- which is the simulation's own
// ablation arm and is priced at +0.0485 equal-cell mean goodput in favour of the joint
// shape, worst cell falling 0.918 to 0.700. A decomposed port still runs and still reports
// goodput, so the shape is asserted by counting the candidates rather than inferred from
// the result.
func TestArgminEnumeratesTheFullCrossProduct(t *testing.T) {
	var seen []string
	cfgArm := recordingArm("comparator", Objective{}, &seen)

	runDecision(t, cfgArm, comparatorConfig(), testRequest(), fleet(nil), nil)

	want := []string{
		decodeH100ID + " local",
		decodeH100ID + " via " + prefillID,
		decodeA100ID + " local",
		decodeA100ID + " via " + prefillID,
	}
	if !reflect.DeepEqual(seen, want) {
		t.Errorf("candidate enumeration =\n  %v\nwant\n  %v\n(2 decode endpoints and 1 prefill "+
			"endpoint give 2 local plus 2 disaggregated candidates, in ascending decode-ID order, "+
			"each decode endpoint's local candidate first)", seen, want)
	}
}

// TestBothArmsEnumerateTheSameCandidateSet is the ATTRIBUTION test at the level of the
// candidate set: the two arms must consider exactly the same candidates in exactly the
// same order, so that a measured difference is attributable to the objective and to
// nothing else.
//
// With no scorers attached -- the shape both arms ship -- every inherited score is 0, so
// the focal arm's scorer-first reordering resolves to the lowest-ID endpoint, which is
// already first, and the two orders coincide. That coincidence is the point: under the
// shipped configuration the arms are exactly comparable. TestScorerPreferenceReorders
// covers what happens when they are not.
func TestBothArmsEnumerateTheSameCandidateSet(t *testing.T) {
	var comparatorSeen, focalSeen []string

	runDecision(t, recordingArm("comparator", Objective{}, &comparatorSeen),
		comparatorConfig(), testRequest(), fleet(nil), nil)
	runDecision(t, recordingArm("focal", causalsloexternality.FocalArm.Objective, &focalSeen),
		focalConfig(), testRequest(), fleet(nil), nil)

	if !reflect.DeepEqual(comparatorSeen, focalSeen) {
		t.Errorf("the two arms must enumerate one candidate set:\n  comparator %v\n  focal      %v",
			comparatorSeen, focalSeen)
	}
	if len(comparatorSeen) == 0 {
		t.Fatal("no candidates were evaluated; the fixture is not exercising the argmin")
	}
}

// TestScorerPreferenceReordersTheFocalArmButNotThisOne pins the one enumeration divergence
// this arm declares, in both directions.
//
// Upstream's least-ttft branch iterates decodeSnaps directly (sim/edpp.go:1456), while
// scorerFirstSnapshots is applied only in the SLO-externality and causal-VaR branches, its
// own doc calling it "the causal-VaR equality tie-break order" (sim/edpp.go:2176-2178).
// Copying the focal arm's reordering into this arm would change which candidate wins an
// exact tie -- and exact ties are the NORM during warmup, while the shadow table is empty
// and every candidate's externality is identically zero.
func TestScorerPreferenceReordersTheFocalArmButNotThisOne(t *testing.T) {
	// Prefer the SECOND decode endpoint by inherited score, so a scorer-first arm must
	// enumerate it before the lower-ID one.
	scores := map[string]float64{decodeA100ID: 0.9, decodeH100ID: 0.1}

	var comparatorSeen, focalSeen []string
	runDecision(t, recordingArm("comparator", Objective{}, &comparatorSeen),
		comparatorConfig(), testRequest(), fleet(nil), scores)
	runDecision(t, recordingArm("focal", causalsloexternality.FocalArm.Objective, &focalSeen),
		focalConfig(), testRequest(), fleet(nil), scores)

	if got := comparatorSeen[0]; got != decodeH100ID+" local" {
		t.Errorf("this arm must enumerate in plain ascending-ID order regardless of inherited "+
			"scores; first candidate was %q, want %q", got, decodeH100ID+" local")
	}
	if got := focalSeen[0]; got != decodeA100ID+" local" {
		t.Errorf("the focal arm must move the inherited scorer's pick to the front; first "+
			"candidate was %q, want %q", got, decodeA100ID+" local")
	}
	if reflect.DeepEqual(comparatorSeen, focalSeen) {
		t.Error("with an inherited preference injected the two orders must differ; identical " +
			"orders mean one arm's declared enumeration property is not being honoured")
	}
	// Both must still enumerate the SAME SET -- only the order differs.
	if len(comparatorSeen) != len(focalSeen) {
		t.Errorf("the candidate SET must be identical even when the order is not: %d against %d",
			len(comparatorSeen), len(focalSeen))
	}
}

// ---------------------------------------------------------------------------
// The argmin's output
// ---------------------------------------------------------------------------

// TestIdleFleetIsPlacedLocally covers the argmin's local outcome, and with it the fact that
// the decode choice is part of the output on BOTH outcomes.
//
// On an idle H100 fleet the local projection is roughly 40 ms against roughly 60 ms
// disaggregated, because the remote path pays prefill admission, the transfer, and the
// re-timed first decode iteration while the local path samples its first token when
// prefill completes.
func TestIdleFleetIsPlacedLocally(t *testing.T) {
	result := runDecision(t, arm, comparatorConfig(), testRequest(), fleet(nil), nil)

	if len(result.TargetEndpoints) != 1 {
		t.Fatalf("a local placement returns the decode pick alone, got %d endpoints",
			len(result.TargetEndpoints))
	}
	if got := endpointID(result.TargetEndpoints[0]); got != decodeH100ID {
		t.Errorf("the H100 decode endpoint should win on an idle fleet, got %q", got)
	}
}

// TestDecodeCongestionIsPlacedRemotely covers the disaggregated outcome and the order
// contract on the result.
//
// TargetEndpoints[0] is the DECODE pick and [1] the prefill pick, and that order is
// load-bearing: Handler.ProcessResults splits them into separate profile results with the
// decode pick alone as primary, because the director turns every endpoint in the primary
// profile into a routing destination and comma-joins them. Returning them the other way
// round would dispatch decode traffic to the prefill pod and still return 200s.
func TestDecodeCongestionIsPlacedRemotely(t *testing.T) {
	result := runDecision(t, arm, comparatorConfig(), testRequest(),
		fleet(congestedMetrics()), nil)

	if len(result.TargetEndpoints) != 2 {
		t.Fatalf("a disaggregated placement returns the decode pick and the prefill pick, got %d",
			len(result.TargetEndpoints))
	}
	if got := endpointID(result.TargetEndpoints[0]); got != decodeH100ID && got != decodeA100ID {
		t.Errorf("TargetEndpoints[0] must be the decode pick, got %q", got)
	}
	if got := endpointID(result.TargetEndpoints[1]); got != prefillID {
		t.Errorf("TargetEndpoints[1] must be the prefill pick, got %q", got)
	}
}

// TestTheA100DecodeEndpointLosesOnAnIdleFleet pins hardware awareness end to end, at the
// argmin rather than in one candidate's arithmetic.
//
// The per-iteration intercept is 25563.82 us on A100 against 16613.54 us on H100, a factor
// of 1.539 present on EVERY iteration regardless of KV state. A port that defaulted or
// ignored the GPU-type label would price the two decode endpoints identically and resolve
// this by enumeration order, which looks like a working argmin.
func TestTheA100DecodeEndpointLosesOnAnIdleFleet(t *testing.T) {
	// Present the A100 endpoint FIRST by inherited score, so that a hardware-blind port
	// choosing on anything but physics would have a reason to prefer it. This arm ignores
	// inherited scores entirely.
	scores := map[string]float64{decodeA100ID: 1.0}
	result := runDecision(t, arm, comparatorConfig(), testRequest(), fleet(nil), scores)

	if len(result.TargetEndpoints) == 0 {
		t.Fatal("no endpoint selected")
	}
	if got := endpointID(result.TargetEndpoints[0]); got != decodeH100ID {
		t.Errorf("the A100 decode endpoint must lose to the H100 one on an idle fleet: its "+
			"per-iteration intercept is 1.539x larger on every iteration. Got %q", got)
	}
}

// TestTokenizationUnavailableDeclinesToRankButStillEnforcesRole covers degradation D8.
//
// a_r is BUILT on this target and its carrier is a nullable pointer, so absence is a real
// runtime state. On failure this arm computes no objective at all and the choice falls back
// to a THIRD policy -- neither arm -- which is why the D8 counter must be compared between
// the two arms before any result is read.
//
// What it must NOT do is return the unfiltered candidate set. The profile is role-blind and
// carries no scorers, and nothing downstream of the picker checks role: the director reads
// Address and Port off whatever the primary profile returns and dispatches. So a prefill
// endpoint surviving this path would be sent a full decode request. Declining to RANK is
// not declining to enforce which pods may serve a decode request.
func TestTokenizationUnavailableDeclinesToRankButStillEnforcesRole(t *testing.T) {
	request := testRequest()
	request.Body.TokenizedPrompt = nil

	result := runDecision(t, arm, comparatorConfig(), request, fleet(nil), nil)

	if len(result.TargetEndpoints) != 1 {
		t.Fatalf("the decline path must still return exactly one decode-eligible endpoint, got %d",
			len(result.TargetEndpoints))
	}
	got := endpointID(result.TargetEndpoints[0])
	if got == prefillID {
		t.Fatal("a prefill endpoint reached the routing destination on the tokenization-decline " +
			"path; nothing downstream of the picker checks role")
	}
	// Deterministic, because the scored slice this path walks is map-ordered.
	if got != decodeH100ID {
		t.Errorf("the decline path must take the lowest-ID decode-eligible endpoint, got %q", got)
	}
}

// TestUnlabelledGPUTypeIsRejectedRatherThanDefaulted pins config.md signal 8 for this arm.
//
// Heterogeneity rides the per-iteration intercept, so a defaulted GPU type is wrong on
// EVERY decision rather than only under load. The endpoint leaves the candidate set and the
// rejection is counted, because a mislabelled endpoint quietly leaving the candidate set is
// otherwise indistinguishable from a routing preference.
func TestUnlabelledGPUTypeIsRejectedRatherThanDefaulted(t *testing.T) {
	endpoints := []fwksched.Endpoint{
		testEndpoint("d1", bylabel.RoleDecode, "", nil), // no GPU-type label
		testEndpoint("d2", bylabel.RoleDecode, gpuA100, nil),
		testEndpoint("p1", bylabel.RolePrefill, gpuH100, nil),
	}

	var seen []string
	runDecision(t, recordingArm("comparator", Objective{}, &seen), comparatorConfig(),
		testRequest(), endpoints, nil)

	for _, candidate := range seen {
		if candidate == decodeH100ID+" local" || candidate == decodeH100ID+" via "+prefillID {
			t.Fatalf("an endpoint with no GPU-type label was priced: %q. It must be rejected, "+
				"never defaulted", candidate)
		}
	}
	if len(seen) != 2 {
		t.Errorf("one decode endpoint and one prefill endpoint remain, so 1 local plus 1 "+
			"disaggregated candidate; got %d: %v", len(seen), seen)
	}
}

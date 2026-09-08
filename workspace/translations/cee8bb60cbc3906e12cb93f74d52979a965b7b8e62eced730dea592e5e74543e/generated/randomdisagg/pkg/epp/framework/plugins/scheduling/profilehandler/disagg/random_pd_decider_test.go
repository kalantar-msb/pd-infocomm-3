package disagg

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func randomDeciderOrFail(t *testing.T, cfg RandomPDDeciderConfig) *RandomPDDecider {
	t.Helper()
	d, err := NewRandomPDDecider(cfg)
	if err != nil {
		t.Fatalf("NewRandomPDDecider(%+v) returned error: %v", cfg, err)
	}
	return d
}

func reqID(id string) *scheduling.InferenceRequest {
	return &scheduling.InferenceRequest{RequestID: id}
}

func TestRandomPDDeciderConfigValidation(t *testing.T) {
	for _, tc := range []struct {
		name    string
		p       float64
		wantErr bool
	}{
		{"zero", 0, false},
		{"one", 1, false},
		{"mid", 0.075, false},
		{"negative", -0.001, true},
		{"above one", 1.001, true},
		{"NaN", math.NaN(), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewRandomPDDecider(RandomPDDeciderConfig{DisaggregateProbability: tc.p})
			if tc.wantErr && err == nil {
				t.Fatalf("p=%v: expected an error, got nil", tc.p)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("p=%v: unexpected error: %v", tc.p, err)
			}
		})
	}
}

// p=0 and p=1 must reproduce the simulator's `never` and `always` rules exactly,
// with no dependence on the hash.
func TestRandomPDDeciderEndpointsAreExact(t *testing.T) {
	ctx := context.Background()
	never := randomDeciderOrFail(t, RandomPDDeciderConfig{DisaggregateProbability: 0})
	always := randomDeciderOrFail(t, RandomPDDeciderConfig{DisaggregateProbability: 1})

	for i := 0; i < 2000; i++ {
		r := reqID(fmt.Sprintf("req-%d", i))
		if never.disaggregate(ctx, r, nil) {
			t.Fatalf("p=0 disaggregated %s", r.RequestID)
		}
		if !always.disaggregate(ctx, r, nil) {
			t.Fatalf("p=1 did not disaggregate %s", r.RequestID)
		}
	}
}

// The verdict must be a property of the request, so repeated calls agree. This is
// what makes the arm safe against the handler invoking the hook more than once.
func TestRandomPDDeciderIsDeterministicPerRequest(t *testing.T) {
	ctx := context.Background()
	d := randomDeciderOrFail(t, RandomPDDeciderConfig{DisaggregateProbability: 0.5, Seed: 42})

	for i := 0; i < 1000; i++ {
		r := reqID(fmt.Sprintf("req-%d", i))
		first := d.disaggregate(ctx, r, nil)
		for call := 0; call < 5; call++ {
			if got := d.disaggregate(ctx, r, nil); got != first {
				t.Fatalf("%s: call %d returned %v, first call returned %v", r.RequestID, call, got, first)
			}
		}
	}
}

// The realized rate must track p. Tolerance is 4 binomial standard errors, which
// is generous enough not to flake and tight enough to catch an off-by-a-factor.
func TestRandomPDDeciderRealizedRateTracksProbability(t *testing.T) {
	ctx := context.Background()
	const n = 200000

	for _, p := range []float64{0.075, 0.25, 0.5, 0.9} {
		t.Run(fmt.Sprintf("p=%v", p), func(t *testing.T) {
			d := randomDeciderOrFail(t, RandomPDDeciderConfig{DisaggregateProbability: p, Seed: 7})
			hits := 0
			for i := 0; i < n; i++ {
				if d.disaggregate(ctx, reqID(fmt.Sprintf("req-%d", i)), nil) {
					hits++
				}
			}
			got := float64(hits) / n
			tol := 4 * math.Sqrt(p*(1-p)/n)
			if math.Abs(got-p) > tol {
				t.Fatalf("realized rate %.5f differs from p=%.5f by more than %.5f", got, p, tol)
			}
		})
	}
}

// Changing the seed at fixed p must select a different subset while holding the
// rate. Identical subsets would mean the seed is not reaching the hash.
func TestRandomPDDeciderSeedChangesSubsetNotRate(t *testing.T) {
	ctx := context.Background()
	const n = 20000
	a := randomDeciderOrFail(t, RandomPDDeciderConfig{DisaggregateProbability: 0.3, Seed: 1})
	b := randomDeciderOrFail(t, RandomPDDeciderConfig{DisaggregateProbability: 0.3, Seed: 2})

	var hitsA, hitsB, differ int
	for i := 0; i < n; i++ {
		r := reqID(fmt.Sprintf("req-%d", i))
		da, db := a.disaggregate(ctx, r, nil), b.disaggregate(ctx, r, nil)
		if da {
			hitsA++
		}
		if db {
			hitsB++
		}
		if da != db {
			differ++
		}
	}
	if differ == 0 {
		t.Fatal("two seeds selected identical subsets; seed is not reaching the hash")
	}
	// Both rates near 0.3, so the disagreement fraction should be near
	// 2*0.3*0.7 = 0.42. A very low value would mean the seeds are near-aliased.
	if frac := float64(differ) / n; frac < 0.3 {
		t.Fatalf("subsets differ on only %.3f of requests; expected near 0.42", frac)
	}
	t.Logf("hitsA=%d hitsB=%d differ=%d", hitsA, hitsB, differ)
}

// IdentityHeader, when configured and present, must key the decision instead of
// RequestID -- that is what buys replay across runs (R2).
func TestRandomPDDeciderPrefersIdentityHeader(t *testing.T) {
	ctx := context.Background()
	d := randomDeciderOrFail(t, RandomPDDeciderConfig{
		DisaggregateProbability: 0.5,
		Seed:                    11,
		IdentityHeader:          "x-blis-request-id",
	})

	// Same stable header, different Envoy IDs: the verdict must not move.
	same := 0
	for i := 0; i < 500; i++ {
		r := &scheduling.InferenceRequest{
			RequestID: fmt.Sprintf("envoy-%d", i),
			Headers:   map[string]string{"x-blis-request-id": "stable-7"},
		}
		if d.disaggregate(ctx, r, nil) {
			same++
		}
	}
	if same != 0 && same != 500 {
		t.Fatalf("stable header did not pin the verdict: %d/500 disaggregated", same)
	}

	// Header absent: falls back to RequestID, so verdicts must vary across ids.
	varied := map[bool]int{}
	for i := 0; i < 500; i++ {
		varied[d.disaggregate(ctx, reqID(fmt.Sprintf("envoy-%d", i)), nil)]++
	}
	if varied[true] == 0 || varied[false] == 0 {
		t.Fatalf("fallback to RequestID did not vary: %+v", varied)
	}
}

// An empty header value must not be treated as an identity; it should fall back.
func TestRandomPDDeciderEmptyHeaderFallsBack(t *testing.T) {
	ctx := context.Background()
	d := randomDeciderOrFail(t, RandomPDDeciderConfig{
		DisaggregateProbability: 0.5, Seed: 3, IdentityHeader: "x-id",
	})
	withEmpty := &scheduling.InferenceRequest{
		RequestID: "envoy-1",
		Headers:   map[string]string{"x-id": ""},
	}
	if got, want := d.disaggregate(ctx, withEmpty, nil), d.disaggregate(ctx, reqID("envoy-1"), nil); got != want {
		t.Fatalf("empty header value changed the verdict: got %v, want %v", got, want)
	}
}

// A nil request must not panic on the decision path.
func TestRandomPDDeciderNilRequest(t *testing.T) {
	d := randomDeciderOrFail(t, RandomPDDeciderConfig{DisaggregateProbability: 0.5, Seed: 5})
	_ = d.disaggregate(context.Background(), nil, nil)
}

func TestRandomPDDeciderTypedName(t *testing.T) {
	d := randomDeciderOrFail(t, RandomPDDeciderConfig{DisaggregateProbability: 0.5}).WithName("my-decider")
	if got := d.TypedName().Type; got != RandomPDDeciderPluginType {
		t.Errorf("Type = %q, want %q", got, RandomPDDeciderPluginType)
	}
	if got := d.TypedName().Name; got != "my-decider" {
		t.Errorf("Name = %q, want %q", got, "my-decider")
	}
}

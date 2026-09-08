package disagg

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"math"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// Specification of record: algorithms/random_disagg.go in the pd-infocomm-3
// bundle. That file carries the rationale and the declared degradations (R1-R5).
// The link is by convention only -- `sim2real translation append` records
// source_path = null for appended arms (R5), so nothing machine-checks it.

const (
	// RandomPDDeciderPluginType is the type-name of the randomPDDecider plugin.
	RandomPDDeciderPluginType = "random-pd-decider"
)

// RandomPDDeciderConfig holds the configuration for the randomPDDecider plugin.
type RandomPDDeciderConfig struct {
	// DisaggregateProbability is the probability p in [0,1] that a request is
	// disaggregated. 0 never disaggregates, 1 always does.
	DisaggregateProbability float64 `json:"disaggregateProbability"`

	// Seed selects which subset of requests is disaggregated at a given
	// probability. Changing it at fixed probability draws an independent subset
	// at the same rate.
	Seed int64 `json:"seed"`

	// IdentityHeader, when non-empty and present on the request, names the header
	// whose value keys the decision. Empty, or absent on the request, falls back
	// to the Envoy-generated RequestID -- correct but not replayable across runs.
	IdentityHeader string `json:"identityHeader"`
}

func (c RandomPDDeciderConfig) validate() error {
	if math.IsNaN(c.DisaggregateProbability) {
		return errors.New("disaggregateProbability parameter of random disaggregation decider cannot be NaN")
	}

	// Rejected rather than clamped on purpose: a clamped rate would run the arm
	// at an operating point other than the one reported.
	if c.DisaggregateProbability < 0 || c.DisaggregateProbability > 1 {
		return fmt.Errorf("disaggregateProbability parameter of random disaggregation decider must be within [0,1], got %v",
			c.DisaggregateProbability)
	}

	return nil
}

// compile-time type assertion
var _ deciderPlugin = &RandomPDDecider{}

// RandomPDDecider is a PD decider plugin which disaggregates a fixed expected
// fraction of requests, chosen independently of any load or content signal.
//
// It reads no metrics and no endpoint state; the endpoint argument to
// disaggregate is ignored. It exists as a matched-rate control against which
// signal-driven deciders can be measured.
type RandomPDDecider struct {
	typedName plugin.TypedName
	config    RandomPDDeciderConfig
}

// RandomPDDeciderPluginFactory defines the factory function for creating
// a new instance of the randomPDDecider.
func RandomPDDeciderPluginFactory(name string, rawParameters *json.Decoder,
	_ plugin.Handle) (plugin.Plugin, error) {
	config := RandomPDDeciderConfig{
		DisaggregateProbability: 0,
		Seed:                    0,
		IdentityHeader:          "",
	}

	if rawParameters != nil {
		if err := rawParameters.Decode(&config); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", RandomPDDeciderPluginType, err)
		}
	}

	decider, err := NewRandomPDDecider(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s plugin: %w", RandomPDDeciderPluginType, err)
	}

	return decider.WithName(name), nil
}

// NewRandomPDDecider initializes a random PD decider plugin and returns its
// pointer. If the configuration is invalid an error is returned.
func NewRandomPDDecider(config RandomPDDeciderConfig) (*RandomPDDecider, error) {
	if err := config.validate(); err != nil {
		return nil, err
	}

	return &RandomPDDecider{
		typedName: plugin.TypedName{Type: RandomPDDeciderPluginType},
		config:    config,
	}, nil
}

// TypedName returns the typed name of the plugin.
func (d *RandomPDDecider) TypedName() plugin.TypedName {
	return d.typedName
}

// WithName sets the name of the plugin.
func (d *RandomPDDecider) WithName(name string) *RandomPDDecider {
	d.typedName.Name = name
	return d
}

// identity returns the string the decision is keyed on.
func (d *RandomPDDecider) identity(request *scheduling.InferenceRequest) string {
	if request == nil {
		return ""
	}

	if d.config.IdentityHeader != "" {
		if value, ok := request.Headers[d.config.IdentityHeader]; ok && value != "" {
			return value
		}
	}

	return request.RequestID
}

// mix64 is murmur3's fmix64 finalizer.
//
// REQUIRED, NOT DECORATIVE. FNV-1a alone has weak avalanche on inputs that differ
// only in their trailing bytes, which is exactly the shape of request identifiers
// ("req-1", "req-2", ... / sequential UUID suffixes). Its high bits then cluster,
// and since unitInterval reads the HIGH bits the realized rate drifts from p by
// percentage points rather than by sampling error -- measured at 0.541 against
// p = 0.5 over 200k identifiers before this finalizer was added, roughly 37
// binomial standard errors. Do not remove it, and do not replace the hash with
// hash/maphash, whose seed is randomized per process and would destroy replay.
func mix64(x uint64) uint64 {
	x ^= x >> 33
	x *= 0xff51afd7ed558ccd
	x ^= x >> 33
	x *= 0xc4ceb9fe1a85ec53
	x ^= x >> 33
	return x
}

// unitInterval maps (seed, identity) deterministically into [0,1).
//
// Derived by hash rather than drawn from an RNG stream so that the verdict is a
// property of the request alone: the EPP serves requests concurrently, so a
// stream's position -- and therefore every decision after it -- would depend on
// arrival interleaving, and repeated calls for one request could disagree.
func (d *RandomPDDecider) unitInterval(identity string) float64 {
	hash := fnv.New64a()

	var seed [8]byte
	binary.LittleEndian.PutUint64(seed[:], uint64(d.config.Seed))
	_, _ = hash.Write(seed[:]) // hash.Hash writes never fail
	_, _ = hash.Write([]byte(identity))

	// Top 53 bits: the float64 mantissa width, so the quotient is exact and the
	// comparison against p is not skewed by rounding.
	return float64(mix64(hash.Sum64())>>11) / float64(uint64(1)<<53)
}

// disaggregate reports whether this request should run the disaggregated stage.
// The endpoint is deliberately ignored -- see the type comment.
func (d *RandomPDDecider) disaggregate(_ context.Context,
	request *scheduling.InferenceRequest, _ scheduling.Endpoint) bool {
	// Exact at the endpoints: p == 0 and p == 1 must never depend on the hash.
	if d.config.DisaggregateProbability <= 0 {
		return false
	}
	if d.config.DisaggregateProbability >= 1 {
		return true
	}

	return d.unitInterval(d.identity(request)) < d.config.DisaggregateProbability
}

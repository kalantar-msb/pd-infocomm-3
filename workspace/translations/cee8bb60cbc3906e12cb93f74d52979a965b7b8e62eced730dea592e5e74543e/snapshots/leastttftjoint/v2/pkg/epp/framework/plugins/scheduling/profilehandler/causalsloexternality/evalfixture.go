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
	"github.com/prometheus/client_golang/prometheus"
)

// EvalFixture assembles an Eval over a fresh Policy so that ONE arm's Objective can be
// exercised directly, in the arm's own package, against the real shared physics.
//
// WHY THIS IS EXPORTED, stated plainly because an exported test seam deserves a reason.
// Each registered arm lives in its own Go package and contributes exactly one thing: an
// Objective (see arm.go). Objective.Cost takes an Eval, and Eval is constructed only
// inside Policy.decide. Without this fixture an arm's package could reach its own
// objective only end to end through the Picker, which returns the WINNING ENDPOINT and
// never the objective's VALUE. A test that can see only the winner cannot catch a wrong
// coefficient, a dropped post-processing term, or a serialized clock join: it catches
// them only when the error is large enough to flip a ranking on the particular fixture
// the test happened to build. For an objective whose whole content is an arithmetic
// expression in microseconds, that is not adequate, and getting that arithmetic exactly
// right is the entire point of the verbatim-copy contract (config.md section 9.3).
//
// The alternative -- each arm re-deriving the physics in its own test oracle -- is the
// specific failure the contract forbids, one layer down.
//
// It is NOT a shortcut around the plugin surface. It builds no plugin, registers no
// type, and reads no configuration file; arms must still test their factories, their
// Filter/Pick path, and their overlay through the real entry points.
type EvalFixture struct {
	// Config is the arm's plugin configuration. It is NOT validated here: a fixture
	// that validated would be unable to exercise an objective against a deliberately
	// degenerate configuration, and validation has its own tests.
	Config Config

	// Objective is the arm under test. It is stored on the Policy, so anything the
	// objective reaches through Eval sees the same arm the argmin would.
	Objective Objective

	// Snapshots are every endpoint in the decision, both pools. They populate the
	// ID-to-GPU-type map exactly as Policy.decide does, so a candidate's physics
	// cannot be selected differently here than in production.
	Snapshots []Snapshot

	// SLOClass, InputLen, and APByEndpoint are the per-request operands the Filter
	// would have assembled. An endpoint absent from APByEndpoint is treated as fully
	// UNCACHED, which is the production behaviour on a prefix miss.
	SLOClass     string
	InputLen     int
	APByEndpoint map[string]int

	// NowUs is the routing instant. Zero is a valid choice and is what a test wants
	// unless it is exercising a time-dependent term.
	NowUs float64

	// RequestID is carried into the eval context for log parity. Optional.
	RequestID string

	// Registerer receives the arm's metric collectors. A nil value gets a fresh
	// registry, so a fixture never mutates the process-default one.
	Registerer prometheus.Registerer

	// PluginType labels the counters this fixture's evaluations increment -- most
	// usefully the arm's own handler type, so a test that reads a counter sees it
	// under the same label production would.
	PluginType string
}

// Build returns the Eval and a stop function that must be called to halt the shadow
// table's TTL sweeper.
//
// The eval context is assembled in the SAME ORDER Policy.decide assembles it -- the
// GPU-type map from the snapshots, then reqKVNeed, then the per-class N_out mean -- so
// that nHatOut is candidate-invariant and every candidate evaluated against one fixture
// sees identical operands (sim/edpp.go:1643-1655).
func (f EvalFixture) Build() (Eval, func(), error) {
	registerer := f.Registerer
	if registerer == nil {
		registerer = prometheus.NewRegistry()
	}
	if err := registerMetrics(registerer); err != nil {
		return Eval{}, func() {}, err
	}

	pluginType := f.PluginType
	if pluginType == "" {
		pluginType = "eval-fixture"
	}
	metrics := newPluginMetrics("fixture", pluginType)
	table := newShadowTable(f.Config.ShadowTable, int64(f.Config.Engine.BlockSize), metrics)
	policy := newPolicy(f.Config, table, metrics, f.Objective)

	policy.gpuTypeByID = make(map[string]string, len(f.Snapshots))
	for _, s := range f.Snapshots {
		policy.gpuTypeByID[s.ID] = s.GPUType
	}

	class := f.SLOClass
	if class == "" {
		class = sloClassStandard
	}
	ec := &evalCtx{
		class:        class,
		inputLen:     f.InputLen,
		nowUs:        f.NowUs,
		requestID:    f.RequestID,
		apByEndpoint: f.APByEndpoint,
		reqKVNeed:    policy.reqKVNeed(f.InputLen),
	}
	ec.nHatOut = policy.nHatFor(class)

	return Eval{p: policy, ec: ec}, table.stop, nil
}

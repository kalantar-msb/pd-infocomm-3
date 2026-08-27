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
	"math"
	"sync"
)

// Policy holds this arm's long-lived state.
//
// It is shared by BOTH plugin registrations -- the handler owns it and the picker holds
// a pointer, resolved at factory time by name. nHatOut is guarded by mu, because the
// response-body hooks that write it run on a different goroutine from the scheduling
// path that reads it (non-final chunks drain from an async queue, director.go:541-552,
// while the final chunk runs on the request goroutine at :539).
//
// IT IS DELIBERATELY SMALLER THAN THE FOCAL ARM'S: no shadow-table value inputs, no
// capacity account, and no per-decision GPU-type map (that map existed only so the
// capacity COMMIT could not select different physics than the candidate score, and
// there is no commit here).
//
// It also drops the focal arm's `table *shadowTable` field. The shadow table is still
// REQUIRED -- see ShadowTable's doc for the three places this arm reads the resident
// populations -- but the reads all happen through Snapshot, which the Handler assembles
// from the table it owns. Nothing on Policy ever dereferenced that field in the focal
// arm either; carrying it here would be a populated-but-never-read field, which is a
// reader trap because it implies a consumer.
//
// nHatOut IS still required, and is not optional state: decodeRemStepsEst and the
// rollout's censored output estimate both read it. The per-class running mean is
// maintained with the same rules as the focal arm's -- seeded at 1 (not 0), and updated
// ONLY on requests that actually completed.
type Policy struct {
	cfg Config

	// metrics carries the counters every declared degradation needs to be visible.
	metrics *pluginMetrics

	mu sync.Mutex

	// nHatOut is the per-class running mean of realized output lengths.
	// sim/edpp.go:412-425.
	//
	// SEEDED AT 1, NOT 0 (mean() returns 1 when n == 0): a zero seed makes the decode
	// horizon vanish and collapses every remaining-steps floor.
	//
	// Updated ONLY on requests that actually COMPLETED. A request reaching a terminal
	// state without completing carries no realized output length, and folding its
	// truncated count in would drag the estimate down.
	nHatOut map[string]*runningMean
}

// newPolicy builds a Policy over a validated Config.
func newPolicy(cfg Config, metrics *pluginMetrics) *Policy {
	return &Policy{
		cfg:     cfg,
		metrics: metrics,
		nHatOut: map[string]*runningMean{},
	}
}

// jointCandidateTTFT is THE OBJECTIVE: the deciding request's own forward
// time-to-first-token for one candidate, in microseconds. sim/edpp.go:1891-1940, the
// least-ttft branch's costFn (sim/edpp.go:1453).
//
// ps == nil means LOCAL (prefill and decode co-resident on ds); otherwise decode on ds
// and prefill on *ps.
//
// WHAT IT CARRIES, AND MORE IMPORTANTLY WHAT IT DOES NOT:
//   - no backlog or congestion drift,
//   - no SLO virtual queues,
//   - NO EXTERNALITY OVER RESIDENTS -- that is the whole difference from the focal arm,
//   - NO tau AT ALL: no SLO threshold reaches this number anywhere,
//   - no transfer penalty beyond the transfer LATENCY already inside the disaggregated
//     TTFT.
//
// THE ARITHMETIC REPRODUCES EXACTLY THE FOCAL ARM'S tHatLocal AND tHatDisagg, and that
// exactness IS the attribution argument: the two arms compute the same TTFT from the
// same estimators, the same physics, and the same prefix reads, and differ only in what
// they do with it. Every operand below comes from a symbol that is a verbatim copy of
// the focal arm's (see doc.go and shared.go). A re-derived-but-slightly-different
// estimator here would silently destroy that argument while every test still passed:
// the arms would then differ in the objective AND in the physics, and no measurement
// could separate the mechanism from the machinery.
//
// Each side uses its OWN candidate's theta, so the arm is hardware-aware by
// construction. That is deliberate -- a hardware-blind least-TTFT baseline would lose to
// the focal arm partly because it cannot see the fleet.
//
// THE REDUCED-RULE EQUIVALENCE HOLDS ONLY UNDER HOMOGENEOUS theta, which is worth
// planning around because it is the cheapest single-endpoint check available for
// validating this port. Upstream asserts that restricting this to a single decode
// endpoint reproduces the reduced least-ttft decision (sim/edpp.go:1889), and that is
// true when one coefficient set is in play. It is NOT true with coeffsByGpuType loaded:
// the reduced path prices remote prefill with the GLOBAL coefficients
// (sim/edpp.go:906, and again in its prefill admission context at :882), while this
// objective uses the prefill endpoint's own thetaP (sim/edpp.go:1915). config.md
// section 4 puts AlphaP at 16617.85 us on H100 against 25568.35 us on A100, so the two
// disagree by roughly 8950 us per prefill iteration -- larger than the entire transfer
// term. So run that check on a HOMOGENEOUS fleet: it WILL diverge on the
// h100_a100_realistic cohorts, and a divergence there leaves no way to tell whether the
// port or the equivalence claim is at fault.
//
// Caller must hold p.mu -- decide does.
func (p *Policy) jointCandidateTTFT(ec *evalCtx, ds Snapshot, ps *Snapshot) float64 {
	thetaD := p.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	tIterD := thetaD.tIterDecode(bDec, kv, sPfD)
	// The B+1 re-timed FIRST decode iteration: batch grows by one, KV grows by the
	// arriving request's full input length.
	tIterFirstDecode := thetaD.tIterDecode(bDec+1, kv+int64(ec.inputLen), sPfD)
	tAdmD := p.estimateTAdm(p.decodeAdmissionCtx(ec, ds))

	if ps == nil {
		// ---- LOCAL: prefill and decode co-resident on ds, so prefill uses the DECODE
		// theta.
		apLoc := p.apForEndpoint(ec, ds.ID)
		nChunksLoc, _ := p.chunkTerms(thetaD, apLoc)
		// Wp (not nChunks*deltaPfChunk) so the estimate keeps the QUADRATIC attention
		// cost, matching the executor.
		wpLoc := thetaD.Wp(maxInt(apLoc, 0), ec.inputLen)
		tHat := p.projectedLocalTTFT(tAdmD, nChunksLoc, tIterD, wpLoc)
		// D1: the rollout is unreachable at this pin (SchedulerStateObserved is
		// permanently false), so the closed-form tHat stands and the substitution
		// counter has already been incremented inside estimateTAdm.
		if _, rolloutTTFT, ok := p.rolloutLocalTTFT(ec, ds, thetaD); ok {
			tHat = rolloutTTFT
		}
		return tHat
	}

	// ---- DISAGGREGATED: decode on ds, prefill on *ps, so prefill uses the PREFILL
	// endpoint's theta.
	thetaP := p.coeffsFor(ps.GPUType)
	apP := p.apForEndpoint(ec, ps.ID)
	nChunksP, _ := p.chunkTerms(thetaP, apP)
	wpP := thetaP.Wp(maxInt(apP, 0), ec.inputLen)
	tIterP := thetaP.tIterPrefill(ps.ResidentPrefillTokens)
	if rolloutAdm, ok := p.rolloutDecodeAdmission(ec, ds, thetaD); ok {
		tAdmD = rolloutAdm // D1
	}
	tAdmP := p.estimateTAdm(p.prefillAdmissionCtx(ec, *ps))
	// THE ONLY PLACE c_xfer ENTERS THE OBJECTIVE -- degradation D5, and it matters more
	// here than in the focal arm because there is no externality term to partially
	// offset a mis-priced transfer.
	cXferUs := p.cXferUsFor(ec.reqKVNeed)
	prefillCompletionUs := tAdmP + nChunksP*tIterP + wpP
	if _, rolloutCompletion, ok := p.rolloutPrefillCompletion(ec, *ps, thetaP); ok {
		prefillCompletionUs = rolloutCompletion // D1
	}
	remoteLeadUs := prefillCompletionUs + cXferUs

	// MAX, NOT SUM: remote preparation and decode-queue drainage are CONCURRENT clocks
	// from the routing instant. Adding them serially over-prices disaggregation by up to
	// a full decode admission delay.
	//
	// THIS max() IS UNCONDITIONAL, AND THE FLAG THAT LOOKS LIKE ITS SWITCH IS NOT. The
	// campaign passes --edpp-ttft-overlap-aware (run_decisive_campaign.py:325-331), but
	// cfg.TTFTOverlapAware gates only the REDUCED path (sim/edpp.go:909) and is never
	// consulted on the joint path -- its only two occurrences upstream are the config
	// field (sim/edpp.go:110) and that one gate. So the flag is a NO-OP for this arm,
	// which is why it is not represented in Config.
	//
	// A port MUST NOT make this max() conditional on that flag: doing so reintroduces
	// the serialized form whenever the flag is off, which is a different algorithm and
	// one that systematically under-selects remote prefill.
	decodeJoinUs := math.Max(remoteLeadUs, tAdmD)
	return p.projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode)
}

// decide enumerates every candidate and returns the argmin of projected TTFT.
// decideJoint begins at sim/edpp.go:1334; THIS arm's branch is the trailing else at
// sim/edpp.go:1448-1464, where costFn is set to jointCandidateTTFT at :1453.
//
// THE SHAPE IS THE SAME AS THE FOCAL ARM'S: D local candidates plus D*P disaggregated
// candidates, scored on ONE scale and compared in ONE argmin. It attaches at the same
// extension point for the same two reasons -- a per-endpoint Scorer cannot name a PAIR,
// and score-range clamping to [0,1] would destroy the ranking. The clamping hazard
// applies here for a DIFFERENT reason than in the focal arm: this objective is a latency
// in microseconds, non-negative and unbounded above, so clamping would collapse
// essentially every candidate to a single score rather than collapsing the J <= 0 half.
//
// THE DECODE CHOICE IS PART OF THE OUTPUT ON BOTH OUTCOMES. Upstream encodes a local win
// as {Disaggregate:false, DecodePodOverride:dID} and a disaggregated win as
// {Disaggregate:true, DecodePodOverride:dID, PrefillPodHint:pID}
// (sim/edpp.go:1470-1475): the decode pod is overridden EVEN WHEN THE RULE DECLINES TO
// DISAGGREGATE. A port that applies the decode selection only on the disaggregated
// branch, and otherwise lets an inherited scorer place decode, has silently discarded
// half the joint argmin.
//
// DETERMINISM, AND THE ONE PLACE THIS ARM DELIBERATELY DIVERGES FROM THE FOCAL ARM:
// endpoints are sorted by ID and the argmin uses a strict improvement threshold, so ties
// resolve to the first-enumerated candidate rather than to map iteration order (the input
// slice is map-ordered -- runPickerPlugin ranges a map at scheduler_profile.go:199-205).
//
// The enumeration order is PLAIN ASCENDING ID, NOT scorer-first. The focal arm moves the
// stock scorer's pick to the front so that restricting enumeration to it reproduces the
// decomposed rule; THIS arm does not, and upstream does not either -- the least-ttft
// branch iterates decodeSnaps directly (sim/edpp.go:1456), and scorerFirstSnapshots is
// applied only in the SLO-externality and causal-VaR branches, its own doc calling it
// "the causal-VaR equality tie-break order" (sim/edpp.go:2176-2178). Reordering here
// would change which candidate wins an exact tie, silently and only sometimes -- which
// is why this arm's picker does not compute a scorer preference at all and this method
// takes no scorer arguments.
func (p *Policy) decide(ec *evalCtx, decodeSnaps, prefillSnaps []Snapshot) (candidate, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()

	decodeSnaps = sortedByID(decodeSnaps)
	prefillSnaps = sortedByID(prefillSnaps)

	// nHatOut is candidate-invariant, so it is resolved ONCE per decision -- every
	// candidate evaluation then uses identical operands. It is read here rather than in
	// the picker because it is mutable shared state, and this is the lock hold.
	//
	// The class is resolved because decodeRemStepsEst and rolloutReqFor take it, NOT
	// because any tau is read -- there is no tau in this arm.
	ec.nHatOut = p.nHatFor(ec.class)

	var best *candidate
	consider := func(c candidate) {
		if best == nil || c.J < best.J-1e-12 {
			cc := c
			best = &cc
		}
	}

	for _, ds := range decodeSnaps {
		consider(candidate{dID: ds.ID, local: true, J: p.jointCandidateTTFT(ec, ds, nil)})
		for i := range prefillSnaps {
			ps := prefillSnaps[i]
			consider(candidate{dID: ds.ID, pID: ps.ID, local: false, J: p.jointCandidateTTFT(ec, ds, &ps)})
		}
	}
	if best == nil {
		return candidate{}, false
	}
	return *best, true
}

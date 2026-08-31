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

import "math"

// THE CAPACITY ACCOUNT.
//
// DISABLED BY THIS ARM (Ablation.NoCapacity, config.md section 9.2), so sloCapacity
// is dead state in the focal configuration. Ported in full because the algebra is
// part of the published objective and flipping the switch back must not require
// re-deriving it -- and because a HALF-specified subsystem is worse than an absent
// one: the read side would look complete while the commit side silently drifted.
//
// Ported from sim/edpp.go:1148-1206 and :2381-2426.

// capacityState is one endpoint's virtual workload queue. In occupancy mode q is
// microseconds of physical occupancy, mu is one, and scale converts both factors to
// seconds. sim/edpp.go:405-410.
type capacityState struct {
	q, mu, scale float64
	gpuType      string
}

// refreshCapacity drains every queue by the elapsed wall time and refreshes each
// endpoint's nominal drain rate and scale. sim/edpp.go:1148-1189.
func (p *Policy) refreshCapacity(nowUs int64, decodeSnaps, prefillSnaps []Snapshot) {
	if !p.sloCapacityClockSet {
		p.sloCapacityClock = nowUs
		p.sloCapacityClockSet = true
	} else if nowUs > p.sloCapacityClock {
		elapsed := float64(nowUs - p.sloCapacityClock)
		for _, state := range p.sloCapacity {
			drain := state.mu * elapsed
			if p.cfg.Ablation.OccupancyCapacity {
				drain = elapsed
			}
			state.q = math.Max(state.q-drain, 0)
		}
		p.sloCapacityClock = nowUs
	}
	set := func(snap Snapshot, mu float64) {
		if snap.ID == "" {
			return
		}
		state := p.sloCapacity[snap.ID]
		if state == nil {
			state = &capacityState{}
			p.sloCapacity[snap.ID] = state
		}
		if p.cfg.Ablation.OccupancyCapacity {
			state.mu = 1
			state.scale = 1_000_000 // express Q*DeltaT in physical seconds squared
		} else {
			state.mu = clampMu(mu)
			state.scale = state.mu * float64(p.cfg.Capacity.TauRefUs)
		}
		state.gpuType = snap.GPUType
	}
	// Decode endpoints use the NOMINAL decode drain rate at the SLO-critical batch;
	// prefill endpoints the nominal prefill rate at the nominal chunk.
	tauITL := float64(p.cfg.activeTargets().TauITLUs)
	for _, snap := range decodeSnaps {
		theta := p.coeffsFor(snap.GPUType)
		set(snap, theta.muDNom(tauITL))
	}
	for _, snap := range prefillSnaps {
		theta := p.coeffsFor(snap.GPUType)
		set(snap, theta.muPNom(p.cfg.Capacity.NomPrefillTokens))
	}
}

// capacityTerm is the quadratic-drift cross term (q/scale)*(work/scale).
// sim/edpp.go:1191-1197.
func (p *Policy) capacityTerm(id string, work float64) float64 {
	state := p.sloCapacity[id]
	if state == nil || state.scale <= 0 || work <= 0 {
		return 0
	}
	return (state.q / state.scale) * (work / state.scale)
}

// capacityQueue reports an endpoint's queue occupancy. sim/edpp.go:1199-1206.
func (p *Policy) capacityQueue(id string) float64 {
	if state := p.sloCapacity[id]; state != nil {
		return state.q
	}
	return 0
}

// The occupancy-time capacity account, at a fixed decode width B.
// sim/edpp.go:2381-2392.
//
// Dedicated prefill pays one baseline per chunk; decode pays a 1/B share of its
// iteration baseline per output token; collocation SHARES the prefill baselines with
// the existing decode batch and therefore adds only Wp.
func (p *Policy) decodeOccupancy(theta Coeffs, ar int, nOut float64) float64 {
	return nOut*theta.AlphaD/float64(p.cfg.Capacity.ReferenceBatch) + theta.Wd(ar, nOut)
}

func (p *Policy) prefillOccupancy(theta Coeffs, ap, ar int) float64 {
	nChunks, _ := p.chunkTerms(theta, ap)
	return nChunks*theta.AlphaP + theta.Wp(maxInt(ap, 0), ar)
}

func (p *Policy) localOccupancy(theta Coeffs, ap, ar int, nOut float64) float64 {
	return p.decodeOccupancy(theta, ar, nOut) + theta.Wp(maxInt(ap, 0), ar)
}

// bookCapacityWork is the COMMIT half of the capacity account, applied after the
// decision at the winning endpoints. sim/edpp.go:2394-2426, reached from the route
// callback at sim/edpp.go:2466-2470.
//
// DEAD ON THIS ARM alongside the rest of the capacity account.
//
// THE INVARIANT THAT MATTERS: the queue is fed by the SAME demand expression the
// candidate score used, evaluated at the COMMITTED endpoints with their own theta and
// their own a_p. Candidate/commit agreement is what makes the drift argument work --
// a port that SCORES with one demand and BOOKS another drifts without erroring, and
// the drift is invisible in any single decision.
//
// NOT MAINTAINED, deliberately: upstream's route callback also calls
// bookSLOAdmissionWork (sim/edpp.go:2432), which feeds the SEPARATE per-instance
// backlog behind the admission estimator's QWork -- and then returns early, skipping
// the historical bookkeeping entirely. This port declares QWork as 0 and
// decision-neutral because rollforward never reads it. A port switching the estimator
// to `waiting` must build BOTH that account and this commit path.
func (p *Policy) bookCapacityWork(ec *evalCtx, toPrefill bool, decodeID, prefillID string) {
	if p.cfg.Ablation.NoCapacity || decodeID == "" {
		return
	}
	thetaD := p.coeffsFor(p.gpuTypeOf(decodeID))
	nOut := ec.nHatOut
	wd := thetaD.Wd(ec.inputLen, nOut)

	if toPrefill {
		thetaP := p.coeffsFor(p.gpuTypeOf(prefillID))
		apP := p.apForEndpoint(ec, prefillID)
		wpP := thetaP.Wp(maxInt(apP, 0), ec.inputLen)
		decodeDemand, prefillDemand := wd, wpP
		if p.cfg.Ablation.OccupancyCapacity {
			decodeDemand = p.decodeOccupancy(thetaD, ec.inputLen, nOut)
			prefillDemand = p.prefillOccupancy(thetaP, apP, ec.inputLen)
		}
		if state := p.sloCapacity[prefillID]; state != nil {
			state.q += prefillDemand
		}
		if state := p.sloCapacity[decodeID]; state != nil {
			state.q += decodeDemand
		}
		return
	}

	// Local: prefill and decode both land on the decode endpoint, so the demand is
	// that endpoint's own a_p under its own theta.
	apD := p.apForEndpoint(ec, decodeID)
	demand := thetaD.Wp(maxInt(apD, 0), ec.inputLen) + wd
	if p.cfg.Ablation.OccupancyCapacity {
		demand = p.localOccupancy(thetaD, apD, ec.inputLen, nOut)
	}
	if state := p.sloCapacity[decodeID]; state != nil {
		state.q += demand
	}
}

// gpuTypeOf resolves a committed endpoint's GPU type for theta selection.
//
// Upstream's counterpart is sloCoeffsForInstance (sim/edpp.go:2370-2375), which MUST
// agree with what the candidate score used, or the commit books work under different
// physics than it scored. Here both read the same snapshot map built once per
// decision, so they cannot diverge.
func (p *Policy) gpuTypeOf(id string) string {
	if gpuType, ok := p.gpuTypeByID[id]; ok {
		return gpuType
	}
	return ""
}

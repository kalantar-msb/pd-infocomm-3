// Package leastttftjoint specifies the least-TTFT joint comparator arm for the
// INFOCOM 2027 transfer.
//
// THIS FILE IS A SPECIFICATION LAYER, NOT COMPILING CODE. Bodiless functions are
// TARGET-API ADAPTERS ONLY; every quantity the simulation computes is stated in
// full, here or by the verbatim-copy contract below.
//
// # PINS
//
//	simulation  vishakha-ramani/inference-sim  871b169bb13934ca8dd1e002638e1f6bf490b3b5  (infocom-implementation)
//	target      llm-d/llm-d-router             71f4f0999f95b96c49a9d0c4afbd18dfdb943c26  (v0.10.0)
//	engine      vllm-project/vllm              v0.26.0
//
// # WHY THIS ARM EXISTS
//
// It shares the candidate set, the estimators, the physics, and the prefix reads
// with the focal arm, and differs IN THE OBJECTIVE ONLY. That is what attributes
// the measured effect to the MECHANISM rather than to the MACHINERY: a weaker
// baseline would leave open that the focal arm won because it computes better
// physics, not because it prices SLO externality.
//
// Cited evidence for the comparison (INFOCOM_REPRODUCIBILITY.md expected
// checkpoints, reproduced in sim_results/main/):
//
//	causal externality  worst static-plan gap  0.0100
//	least-TTFT          worst static-plan gap  0.0542
//
// and in the stress cohort (sim_results/stress/): 0.0031 against 0.1110.
//
// # THE OBJECTIVE
//
// The argmin of the ARRIVING REQUEST'S OWN projected time-to-first-token, over the
// same candidate set. It carries:
//
//   - no backlog or congestion drift,
//   - no SLO virtual queues,
//   - no externality over residents,
//   - NO tau AT ALL -- see "What this arm does not read",
//   - no transfer penalty beyond the transfer LATENCY already inside the
//     disaggregated TTFT.
//
// Each side uses its own candidate's theta, so the arm is hardware-aware by
// construction. That is deliberate: a hardware-BLIND least-TTFT baseline would
// lose to the focal arm partly because it cannot see the fleet, which would
// confound the mechanism with hardware awareness.
//
// # THE VERBATIM-COPY CONTRACT
//
// Every symbol listed below MUST be a VERBATIM COPY of the focal arm's
// (algorithms/causal_slo_externality.go), package clause aside. This is not style.
// A re-derived-but-slightly-different estimator would silently destroy the
// attribution argument while every test still passed: the two arms would then
// differ in the objective AND in the physics, and no measurement could separate
// them. Any change to one of these must be made in BOTH arms, or the comparison
// stops being a comparison.
//
// Shared, copy verbatim:
//
//	Coeffs and every method:      Wp, Wd, tIterDecode, tIterPrefill,
//	                              muDecode, muPrefill, muDNom, muPNom,
//	                              deltaBarDecode, validate
//	clampMu, maxInt, minI64, maxI64
//	AdmissionContext, flooredTAdm,
//	  waitingEstimateTAdm, littleEstimateTAdm,
//	  fluidEstimateTAdm, rollforwardEstimateTAdm, stableSortByStep
//	the whole scheduler rollout:  rolloutReq, rolloutGrant, rolloutResult,
//	                              rolloutContext, ceilBlocks, rolloutGrantTime,
//	                              currentScheduledTime, schedulerRollout,
//	                              rolloutReqFor, rolloutTimes,
//	                              rolloutLocalTTFT, rolloutDecodeAdmission,
//	                              rolloutPrefillCompletion
//	helpers:                      apForInstance, chunkTerms,
//	                              projectedLocalTTFT, projectedDisaggTTFT,
//	                              reqKVNeed, cXferUsFor,
//	                              decodeRemStepsEst, prefillRemStepsEst,
//	                              decodeAdmissionCtx, prefillAdmissionCtx,
//	                              censorOracleRemaining, estimateTAdm,
//	                              coeffsFor, nHatFor, runningMean
//	types:                        Snapshot, RunningReqState, SchedulerReqState,
//	                              Endpoint, Request, Metrics, candidate, evalCtx
//	adapters:                     endpointGPUType, endpointRole, endpointMetrics,
//	                              promptTokens, cachedBlockCount,
//	                              residentDecodeState, residentPrefillState,
//	                              residentPrefillTokens, requestSLOClass, nowUs,
//	                              sortedByID, inputLen, requestID, endpointByID
//
// residentDecodeState and residentPrefillState are the load-bearing two: they
// populate Snapshot.RunningDecode and Snapshot.RunningPrefill, which this arm reads
// through decodeAdmissionCtx and prefillRemStepsEst. They are the shadow table, so
// D2 and D6 arrive with them.
//
// Both arms MUST also point at the SAME prefix-cache producer instance, bound by
// producer name, so a_r and a_p cannot differ between arms. A data key's identity
// includes its producer name, so an unbound read can silently become a second,
// differently-configured signal -- and then the two arms are pricing different
// prompts.
//
// # DECLARED DEGRADATIONS
//
// D1 (scheduler rollout unobtainable) is INHERITED IN FULL, and the same substitution
// counter is required here.
//
// BUT ITS DIRECTION IS NOT A SINGLE SIGN FOR THIS OBJECTIVE, and the focal arm's
// one-line summary must not be copied over. The two placements join their clocks
// differently: local ADDS tAdmD with slope 1 (sim/edpp.go:1246), while disagg takes
// max(remoteLead, tAdmD) (sim/edpp.go:1928). Writing
//
//	local = tAdmD + P_loc
//	disagg = max(tAdmP + W_P + c_xfer, tAdmD) + tIterFirstDecode
//
// then when tAdmD dominates the max, understating it is charged identically to both
// placements and CANCELS; when remoteLead dominates, disagg is insensitive to tAdmD
// altogether, so understating it lowers ONLY the local candidate -- biasing toward
// LOCAL, the opposite of the focal arm's direction. The toward-remote direction
// survives only through the tAdmP channel, and only while remoteLead dominates.
//
// So: understating DECODE admission biases local or cancels; understating PREFILL
// admission biases remote. The net is load- and pool-dependent.
//
// THIS MATTERS FOR THE ATTRIBUTION ARGUMENT. It would be convenient to say both arms
// inherit D1 identically so the comparison stays fair, but a load-dependent
// direction cannot support that claim: under decode-side congestion -- the regime the
// high-load and burst cohorts target -- D1 shifts the two arms' local/remote splits
// by different amounts. Treat the comparison as fair only where the substitution
// counter shows the same estimator regime on both arms.
//
// AND THE ROLLOUT WAS THE LIVE PATH UPSTREAM FOR THIS ARM, not a fallback.
// Selecting this rule makes the simulation record resident and scheduler detail,
// which is what sets SchedulerStateObserved true and activates the rollout; the
// closed-form projection below is only reached when that guard fails. So on the
// target -- where the guard is permanently false -- BOTH arms run an estimator that
// produced none of the numbers in sim_results/. That is the whole of D1, and it is
// the single most important thing to instrument on this arm as well as the focal one.
//
// D5 (c_xfer unmeasured) is inherited and matters MORE here than in the focal arm:
// there is no externality term to partially offset a mis-priced transfer, so an
// unmeasured c_xfer moves this arm's decisions systematically. It enters at exactly
// one place, remoteLeadUs below.
//
// It is NOT, however, the only size-dependent price of going remote. The
// disaggregated projection also adds tIterFirstDecode, the B+1 re-timed first decode
// iteration, which the LOCAL projection does not (sim/edpp.go:1252 against :1245 --
// local samples its first token when prefill completes, so no decode iteration
// precedes it). That term's KV component scales with the request's own input length,
// so prompt size prices the remote path through two channels, not one.
//
// D7 (FreeKVBlocks is a floor) is inherited via the admission context.
//
// D8  TOKENIZATION CAN BE UNAVAILABLE AT RUNTIME, AND THE FALLBACK IS A THIRD POLICY.
//     a_r is BUILT on the target (config.md §5 signal 10) via a tokenizer sidecar
//     plus a token-producer plugin, and its carrier is a NULLABLE pointer --
//     TokenizedPrompt is documented as "parser-derived tokenization results when
//     available" (pkg/epp/framework/interface/requesthandling/types.go:115-117), with
//     TokenCount() at :191. Absence is a real runtime state. Upstream has no
//     counterpart guard, because the simulation's input length is exact DES state and
//     cannot fail.
//     WHY THE DIRECTION IS NOT A ROUTING BIAS: on failure this arm returns no
//     decision at all, so the stock scorer's pick stands. That is a silent fallback
//     to a THIRD policy -- neither arm -- and if the fallback RATE differs between the
//     two arms (they are separate plugin instances with separate token-producer
//     bindings) it confounds the very comparison the arms exist to make. It is
//     likeliest on long prompts, exactly where local-vs-remote is most contested.
//     A per-arm counter is required alongside D1's substitution counter, and the two
//     arms' rates must be compared before any result is read.
//
// D2, D4, and D6 ARE ALSO INHERITED. This arm reads no resident VALUE, but it does
// read the resident POPULATIONS, in three places:
//   - D4: ResidentPrefillTokens enters tIterDecode and tIterPrefill as S_pf, on
//     every candidate (see jointCandidateTTFT below).
//   - D2: RunningDecode feeds decodeRemStepsEst and the rollforward KV walk through
//     decodeAdmissionCtx (sim/edpp.go:1610-1611), so StepsDone and KVBlocks are read.
//   - D6: RunningPrefill feeds prefillRemStepsEst.
//
// What it does NOT inherit is D2c, the late-first-token bias: no value kernel here
// reads ArrivalUs, FirstTokenUs, or TTFTSet.
//
// # WHAT THIS ARM DOES NOT READ, AND WHY THAT MATTERS
//
// It drops fields rather than populating unread ones. A populated-but-never-read
// field is a reader trap, because it implies a consumer exists somewhere.
//
//	tau_ttft / tau_itl / tau_e2e   DROPPED ENTIRELY. No SLO threshold reaches this
//	                               arm's score anywhere. Consequence: the tau
//	                               selector (config.md §6) is IRRELEVANT to this
//	                               arm, and it cannot be flattened by a zero
//	                               triple the way the focal arm can.
//	RunningDecode resident detail  DROPPED for value purposes. StepsDone and
//	                               KVBlocks are KEPT, because decodeRemStepsEst
//	                               and the rollforward KV walk read them.
//	ArrivalUs / FirstTokenUs /
//	  TTFTSet                      DROPPED. There is no value kernel here, so a
//	                               resident's deadline slack is never evaluated.
//	RunningPrefill                 DROPPED except for prefillRemStepsEst, which
//	                               reads remaining prompt tokens only.
//	the capacity account           NOT PRESENT. This arm has no capacity term.
//	the value kernels              NOT PRESENT. No sloCompositeValue, no
//	                               gDecodeComposite, no goodSelf.
package leastttftjoint

import "math"

// Config holds the knobs this arm reads. It is a STRICT SUBSET of the focal arm's:
// every field here has the same meaning and the same value (config.md §9.1), and
// the focal arm's objective-specific and tau fields are absent because nothing
// here reads them.
type Config struct {
	// ChunkTokens MUST equal the engine's max_num_batched_tokens = 2048
	// (config.md §1, §5 signal 22).
	ChunkTokens int

	// BlockSize MUST equal the engine's block_size = 16, and MUST be validated
	// against the scraped Metrics.CacheBlockSize with a loud failure on
	// disagreement (config.md §5 signal 23).
	BlockSize int

	// MaxBatchSize is the engine's max_num_seqs = 256 (config.md §5 signal 24).
	MaxBatchSize int

	// Coeffs / CoeffsByGPU: config.md §4, keyed by the POD LABEL value. Identical
	// values to the focal arm -- if the two arms carried different coefficients
	// the comparison would be meaningless.
	Coeffs      Coeffs
	CoeffsByGPU map[string]Coeffs

	// CXferSizeAware: true (config.md §9.1).
	CXferSizeAware bool

	// Transfer model. UNMEASURED -- degradation D5, config.md §7.
	XferBaseUs            float64
	XferBandwidthGBps     float64
	KVBytesPerTokenPerGPU float64

	// OutputTokenProcessingUs is outside the calibrated theta and must be added to
	// every TTFT projection (config.md §5 signal 25).
	OutputTokenProcessingUs float64

	// NO tau FIELD, AND NO NomPrefillTokens. Both were considered and both are
	// genuinely unreachable from this arm: muDNom and muPNom take their arguments BY
	// PARAMETER, so the verbatim-copied Coeffs methods compile without any Config
	// field, and this arm calls neither -- its admission contexts use muDecode and
	// muPrefill, which read only batch state. Adding either field would imply a
	// consumer that does not exist.
}

// jointCandidateTTFT is THE OBJECTIVE: the deciding request's own forward
// time-to-first-token for one candidate. sim/edpp.go:1891-1940.
//
// ps == nil means LOCAL (prefill and decode co-resident on ds); otherwise decode
// on ds and prefill on *ps.
//
// The arithmetic reproduces EXACTLY the focal arm's tHatLocal and tHatDisagg. That
// exactness is the attribution argument: the two arms compute the same TTFT and
// differ only in what they do with it.
//
// THE REDUCED-RULE EQUIVALENCE HOLDS ONLY UNDER HOMOGENEOUS theta. Upstream asserts
// that restricting this to a single decode endpoint reproduces the reduced
// least-ttft decision (sim/edpp.go:1889), and that is true when one coefficient set
// is in play. It is NOT true with coeffs_by_gpu loaded: the reduced path prices
// remote prefill with the GLOBAL coefficients (d.coeffs.tIterPrefill at
// sim/edpp.go:906, and again in its prefill admission context at :882), while this
// objective uses the prefill endpoint's own thetaP (sim/edpp.go:1915). config.md §4
// puts AlphaP at 16617.85 us on H100 against 25568.35 us on A100, so the two
// disagree by roughly 8950 us per prefill iteration -- larger than the entire
// transfer term.
//
// Consequence worth planning around: that equivalence is the cheapest
// single-endpoint check available for validating this port, and it WILL diverge on
// the h100_a100_realistic cohorts -- the fleets this experiment is about. Run it on
// a homogeneous fleet, or a divergence leaves no way to tell whether the port or the
// equivalence claim is at fault.
func (p *Policy) jointCandidateTTFT(ec *evalCtx, ds Snapshot, ps *Snapshot) float64 {
	thetaD := p.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	tIterD := thetaD.tIterDecode(bDec, kv, sPfD)
	// The B+1 re-timed FIRST decode iteration: batch grows by one, KV grows by the
	// arriving request's full input length.
	tIterFirstDecode := thetaD.tIterDecode(bDec+1, kv+int64(ec.inputLen), sPfD)
	tAdmD := p.estimateTAdm(p.decodeAdmissionCtx(ec, ds))

	if ps == nil {
		// --- LOCAL: prefill and decode co-resident on ds, so prefill uses the
		// DECODE theta.
		apLoc := p.apForInstance(ec.req, ds.ID)
		nChunksLoc, _ := p.chunkTerms(thetaD, apLoc)
		wpLoc := thetaD.Wp(maxInt(apLoc, 0), ec.inputLen)
		// Wp (not nChunks*deltaPf) so the estimate keeps the QUADRATIC attention
		// cost, matching the executor.
		tHat := p.projectedLocalTTFT(tAdmD, nChunksLoc, tIterD, wpLoc)
		// D1: unreachable on the target, so the closed-form tHat stands.
		// Increment the substitution counter here.
		if _, rolloutTTFT, ok := p.rolloutLocalTTFT(ec, ds, thetaD); ok {
			tHat = rolloutTTFT
		}
		return tHat
	}

	// --- DISAGGREGATED: decode on ds, prefill on *ps, so prefill uses the PREFILL
	// endpoint's theta.
	thetaP := p.coeffsFor(ps.GPUType)
	apP := p.apForInstance(ec.req, ps.ID)
	nChunksP, _ := p.chunkTerms(thetaP, apP)
	wpP := thetaP.Wp(maxInt(apP, 0), ec.inputLen)
	tIterP := thetaP.tIterPrefill(ps.ResidentPrefillTokens)
	if rolloutAdm, ok := p.rolloutDecodeAdmission(ec, ds, thetaD); ok {
		tAdmD = rolloutAdm // D1
	}
	tAdmP := p.estimateTAdm(p.prefillAdmissionCtx(ec, *ps))
	cXferUs := p.cXferUsFor(ec.req) // D5 -- the ONLY size-dependent remote price here
	prefillCompletionUs := tAdmP + nChunksP*tIterP + wpP
	if _, rolloutCompletion, ok := p.rolloutPrefillCompletion(ec, *ps, thetaP); ok {
		prefillCompletionUs = rolloutCompletion // D1
	}
	remoteLeadUs := prefillCompletionUs + cXferUs
	// MAX, NOT SUM: remote preparation and decode-queue drainage are CONCURRENT
	// clocks from the routing instant. Adding them serially over-prices
	// disaggregation by up to a full decode admission delay.
	//
	// THIS max() IS UNCONDITIONAL, AND THE FLAG THAT LOOKS LIKE ITS SWITCH IS NOT.
	// The campaign passes --edpp-ttft-overlap-aware
	// (run_decisive_campaign.py:325-331), but cfg.TTFTOverlapAware gates only the
	// REDUCED path (sim/edpp.go:909) and is never consulted on the joint path --
	// its only two occurrences upstream are the config field (sim/edpp.go:110) and
	// that one gate. So the flag is a NO-OP for this arm.
	//
	// A port MUST NOT make this max() conditional on that flag: doing so
	// reintroduces the serialized form whenever the flag is off, which is a
	// different algorithm and one that systematically under-selects remote prefill.
	decodeJoinUs := math.Max(remoteLeadUs, tAdmD)
	return p.projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode)
}

// Decide enumerates every candidate and returns the argmin of projected TTFT.
// decideJoint begins at sim/edpp.go:1334, but THIS arm's branch is the trailing else
// at sim/edpp.go:1448-1464, where costFn is set to jointCandidateTTFT at :1453.
//
// THE SHAPE IS THE SAME AS THE FOCAL ARM'S: D local candidates plus D*P
// disaggregated candidates, one scale, one argmin. It must attach at the same
// extension point for the same two reasons -- a per-endpoint scorer cannot name a
// PAIR, and score-range clamping to [0,1] would destroy the ranking. Confirm both
// interfaces against the pinned target checkout.
//
// Note this objective is a LATENCY in microseconds and is therefore non-negative
// and unbounded above, so clamping to [0,1] would collapse essentially every
// candidate to a single score. The clamping hazard applies to this arm as much as
// to the focal one, for a different reason.
//
// DETERMINISM: endpoints sorted by ID, strict improvement threshold, so ties
// resolve to the first-enumerated candidate rather than to map iteration order.
func (p *Policy) Decide(req Request, decodeSnaps, prefillSnaps []Snapshot) (candidate, bool) {
	inputLen, ok := promptTokens(req)
	if !ok {
		// Tokenization failed -- distinct from a legitimately empty prompt, which
		// must still be scored.
		return candidate{}, false
	}

	decodeSnaps = sortedByID(decodeSnaps)
	prefillSnaps = sortedByID(prefillSnaps)

	// class is resolved because decodeRemStepsEst takes it, NOT because any tau is
	// read. See "What this arm does not read".
	ec := &evalCtx{
		req:       req,
		class:     requestSLOClass(req),
		inputLen:  inputLen,
		reqKVNeed: p.reqKVNeed(req),
		nHatOut:   p.nHatFor(requestSLOClass(req)),
		nowUs:     nowUs(),
	}

	// PLAIN ASCENDING-ID ORDER, NOT scorer-first. The focal arm moves the stock
	// scorer's pick to the front so that restricting enumeration to it reproduces the
	// decomposed rule; THIS arm does not, and upstream does not either -- the
	// least-ttft branch iterates decodeSnaps directly (sim/edpp.go:1456), and
	// scorerFirstSnapshots is applied only in the SLO-externality and causal-VaR
	// branches, its own doc calling it "the causal-VaR equality tie-break order"
	// (sim/edpp.go:2176-2178). Reordering here would change which candidate wins an
	// exact tie, silently and only sometimes.
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

// Policy holds this arm's long-lived state. It is DELIBERATELY SMALLER than the
// focal arm's: no shadow-table value inputs, no capacity account.
//
// nHatOut IS still required -- decodeRemStepsEst and the rollout's censored output
// estimate both read it -- so the per-class running mean must be maintained here
// too, with the same rules: seeded at 1 (not 0), and updated ONLY on requests that
// actually completed.
//
// The shadow table is still needed for StepsDone and KVBlocks (degradation D2
// applies to those reads), but this arm never reads ArrivalUs, FirstTokenUs, or
// TTFTSet, so D2c does not affect it.
type Policy struct {
	cfg     Config
	nHatOut map[string]*runningMean
}

// ---------------------------------------------------------------------------
// Shared symbols -- see THE VERBATIM-COPY CONTRACT in the package comment.
// Declared here so this file reads as a complete specification of the arm; their
// bodies are the focal arm's, copied unchanged.
// ---------------------------------------------------------------------------

type (
	Coeffs           struct{ AlphaD, AlphaP, C0, C1, CPf, CAttn float64 }
	Endpoint         any
	Request          any
	Metrics          any
	RunningReqState  struct{}
	SchedulerReqState struct{}
	Snapshot         struct{}
	AdmissionContext struct{}
	evalCtx          struct {
		req       Request
		class     string
		inputLen  int
		reqKVNeed int64
		nHatOut   float64
		nowUs     float64
	}
	candidate struct {
		dID, pID string
		local    bool
		J        float64
	}
	runningMean struct{}
	rolloutResult struct{}
)

func maxInt(a, b int) int

func (c Coeffs) Wp(ap, ar int) float64
func (c Coeffs) tIterDecode(bDec int, kv, sPf int64) float64
func (c Coeffs) tIterPrefill(sPf int64) float64

func (p *Policy) coeffsFor(gpuType string) Coeffs
func (p *Policy) nHatFor(class string) float64
func (p *Policy) apForInstance(req Request, instID string) int
func (p *Policy) chunkTerms(theta Coeffs, ap int) (nChunks, deltaPfChunk float64)
func (p *Policy) projectedLocalTTFT(tAdmD, nChunks, tIterD, wpLoc float64) float64
func (p *Policy) projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode float64) float64
func (p *Policy) reqKVNeed(req Request) int64
func (p *Policy) cXferUsFor(req Request) float64
func (p *Policy) estimateTAdm(ctx AdmissionContext) float64
func (p *Policy) decodeAdmissionCtx(ec *evalCtx, ds Snapshot) AdmissionContext
func (p *Policy) prefillAdmissionCtx(ec *evalCtx, ps Snapshot) AdmissionContext
func (p *Policy) rolloutLocalTTFT(ec *evalCtx, ds Snapshot, theta Coeffs) (tAdm, ttft float64, ok bool)
func (p *Policy) rolloutDecodeAdmission(ec *evalCtx, ds Snapshot, theta Coeffs) (float64, bool)
func (p *Policy) rolloutPrefillCompletion(ec *evalCtx, ps Snapshot, theta Coeffs) (tAdm, completion float64, ok bool)

func promptTokens(req Request) (int, bool)
func requestSLOClass(req Request) string
func nowUs() float64
func sortedByID(snaps []Snapshot) []Snapshot

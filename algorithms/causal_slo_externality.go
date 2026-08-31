// Package causalsloexternality specifies the INFOCOM 2027 joint causal-SLO-externality
// routing policy for transfer from the BLIS discrete-event simulator to
// llm-d-router.
//
// THIS FILE IS A SPECIFICATION LAYER, NOT COMPILING CODE. It states the complete
// computation the simulation performs, cited to the simulation source, against the
// target's real interface names. Functions whose bodies are absent are TARGET-API
// ADAPTERS ONLY: their exact accessor must be confirmed against the pinned target
// checkout, and a wrong guess must fail loudly rather than mis-score silently.
// Every quantity the simulation computes is stated here in full.
//
// # PINS
//
//	simulation  vishakha-ramani/inference-sim  871b169bb13934ca8dd1e002638e1f6bf490b3b5  (infocom-implementation)
//	target      llm-d/llm-d-router             71f4f0999f95b96c49a9d0c4afbd18dfdb943c26  (v0.10.0)
//	engine      vllm-project/vllm              v0.26.0
//
// Simulator citations are relative to the simulation pin; target citations to the
// target pin. Deployment knobs are cited to config.md sections.
//
// # THE MECHANISM
//
// The policy is a joint argmin over (decode endpoint, prefill placement) of
//
//	J(d, p) = V * ( externality(d, p) - ownGood(d, p) ) + capacity(d, p)
//
// with V = 8 (config.md §9.2). The focal arm disables the capacity term
// (SLOExternalityNoCapacity), so the operative objective is
//
//	J(d, p) = 8 * ( externality(d, p) - ownGood(d, p) )
//
// WHAT DIFFERS BETWEEN CONDITIONS is the SLO value already banked in a candidate
// endpoint's residents. Two endpoints with identical queue depth and identical KV
// occupancy are not interchangeable: one may hold residents sitting just inside
// their end-to-end deadline, where one extra co-scheduled prefill flips them to a
// miss; the other may hold residents with comfortable slack, or residents already
// doomed, where the same prefill costs nothing recoverable. Load-shaped signals --
// queue depth, KV utilization, projected TTFT -- cannot see that difference,
// because it is a property of the residents' deadlines, not of the endpoint's load.
//
// THE MECHANISM THAT EXPLOITS IT prices each candidate by the smooth SLO value it
// destroys among residents, evaluated CAUSALLY: every completion model is gated on
// an admission window (admissionSteps / arrivalSteps below), so a resident that
// finishes before the arrival is admitted contributes exactly zero. That gating is
// what makes the term an externality rather than another load proxy, and it is why
// the effect survives on a fleet whose load signals are already balanced.
// Heterogeneous fleets give it the most room, because per-GPU coefficients make the
// same resident set cost different amounts on different hardware (config.md §2, §4).
//
// The routing value is a smooth TTFT x E2E surrogate. Reported goodput remains the
// HARD conjunction of TTFT, mean ITL, and E2E; mean ITL is an evaluation gate, not
// a routing term -- the composite kernel never reads tau_itl.
//
// # DECLARED DEGRADATIONS
//
// Each states what is lost AND the direction of the resulting bias. Net direction
// across all of them is NOT known and is NOT claimed: D1 biases toward remote,
// D2/D4/D6/D7 do not all point the same way. This list is what to measure before
// trusting a magnitude -- not a claim that the errors cancel.
//
// D1  SCHEDULER ROLLOUT UNOBTAINABLE. The published TTFT and admission clock replay
//     the engine scheduler over its ordered wait queue (schedulerRollout below,
//     ported from sim/edpp_scheduler_rollout.go:79-243). The target exposes
//     Metrics.WaitingQueueSize -- one integer
//     (pkg/epp/framework/interface/datalayer/metrics.go:33). There is no route to
//     the queue contents, the current grants, or the step start instant, so the
//     rollout's own guard (snap.SchedulerStateObserved,
//     sim/edpp_scheduler_rollout.go:299) is permanently false on the target and the
//     closed-form rollforwardEstimateTAdm substitutes -- which is what the
//     simulation itself does when that guard fails. BOTH estimators are stated in
//     full below so the substitution is legible rather than implied by an absent
//     branch. A substitution COUNTER is required: the fallback is acceptable, the
//     silence is the defect.
//     BIAS: rollforward walks only the current running set's departures then falls
//     back to a wave form (sim/admission_estimator.go:167-172). Past one batch
//     drain it UNDERSTATES admission delay => smaller admissionSteps => residents
//     charged for more of the arrival's interference => local candidate
//     OVER-priced => biased toward REMOTE prefill.
//
// D2  PER-RESIDENT DECODE STATE MUST BE BUILT. varDecodeInputs reads per-resident
//     StepsDone, ArrivalUs, FirstTokenUs, TTFTSet (sim/edpp_var.go:615-641). vLLM
//     exports resident state only in aggregate, so the port maintains an EPP-side
//     index of the requests it placed. epp.replicas is pinned to 1 in the BASELINE
//     so both arms match (config.md §11), which removes the partition case. Two
//     degradations remain:
//     D2b traffic bypassing this EPP is invisible -- aggregate RunningRequestsSize
//         still sees it, so batch size stays right while resident detail is short.
//         BIAS: under-counts the externality => biased toward LOCAL.
//     D2c the recorded first token is a DEQUEUE instant, so realized TTFT is late.
//         BIAS: under the composite kernel a resident's charge is
//         gDecodeComposite(cb) - gDecodeComposite(cp), and because realized TTFT is
//         fixed by the decision it factors out as sigmoid((tau_ttft-ttft)/tau_ttft)
//         multiplying BOTH terms. A late first token shrinks that common positive
//         factor and scales the whole difference down => biased toward LOCAL.
//
// D4  ResidentPrefillTokens (S_pf) ESTIMATED. S_pf is what is being prefilled THIS
//     STEP, not the outstanding backlog. The EPP cannot know what the engine
//     scheduled, so the port sums its shadow-table prefill occupants with a
//     per-occupant cap and a total cap at ChunkTokens (config.md §5 signal 14).
//     BIAS: still an over-estimate => over-states local prefill inflation =>
//     biased toward REMOTE.
//
// D6  PRE-FIRST-TOKEN OCCUPANTS INCOMPLETE. RunningPrefill (sim/edpp_var.go:643-673)
//     is the population whose first token is still live. The shadow table sees only
//     occupants this EPP placed. BIAS: under-counts collocPrefill => biased toward
//     LOCAL.
//
// D7  FreeKVBlocks IS A FLOOR. Derived as (1 - KVCacheUsagePercent) * CacheNumBlocks
//     with int64 truncation, on top of exporter quantisation. BIAS: understates free
//     KV => rollforward admits later => over-states admission delay.
//
// D5  c_xfer UNMEASURED. XferBaseUs and XferBandwidthGBps carry the simulator's
//     transfer-executor settings, not target measurements (config.md §7). c_xfer is
//     the only price of going remote that c_xfer itself controls, entering at exactly
//     one place (remoteLeadUs below), so an unmeasured value mis-prices
//     SYSTEMATICALLY rather than noisily. BIAS: under-pricing transfer => biased
//     toward REMOTE.
//     It is NOT the only SIZE-DEPENDENT remote price: projectedDisaggTTFT also adds
//     tIterFirstDecode, which projectedLocalTTFT does not (local samples its first
//     token when prefill completes, so no decode iteration precedes it), and that
//     term's KV component scales with the request's own input length. Prompt size
//     therefore prices the remote path through two channels.
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
// # WHAT THIS FILE DOES NOT ASSERT
//
// It requests that downstream stages confirm the target-API adapters against the
// pinned checkout. It does not assert what any downstream stage does.
package causalsloexternality

import "math"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Config holds the knobs this arm reads. Values and their provenance are in
// config.md §9.1 (shared) and §9.2 (focal). Durations are microseconds.
type Config struct {
	// V is Neely's penalty/stability tradeoff knob. config.md §9.2: 8.
	//
	// KEPT, NOT FOLDED AWAY, even though with capacity disabled it is a common
	// positive multiplier that cannot change the argmin: the ablation cohort's
	// validity gate asserts score = 8 * (externality - ownGood) exactly, and a
	// port that folds V away cannot reproduce that check.
	V float64

	// TauTTFTUs / TauITLUs / TauE2EUs are the DEFAULT SLO targets, and the
	// ByClass maps override them per SLO class (sim/edpp.go:682-708).
	//
	// Every cohort in all three workloads declares slo_class: standard
	// (workloads/interactive-chat-single-turn.yaml:21 and the other two), so the
	// class is one constant string across the whole grid and a per-request lookup
	// CANNOT select the triple -- the simulation varies tau per invocation.
	// config.md §6: carry all three workload triples and select one, then run one
	// assemble per workload.
	//
	// An unresolvable selection MUST fail at startup. A zero triple does not
	// loosen the policy, it FLATTENS it: sloCompositeValue returns 1.0 for every
	// disabled target, so every resident charge becomes 1.0 - 1.0 = 0, the
	// externality is 0 on every candidate, and the argmin is a tie broken by
	// enumeration order -- while the policy keeps running and keeps reporting
	// goodput.
	TauTTFTUs, TauITLUs, TauE2EUs             int64
	TauTTFTByClassUs, TauITLByClassUs         map[string]int64
	TauE2EByClassUs                           map[string]int64

	// TauRefUs is the fixed reference tau for the capacity account's scale:
	// state.scale = state.mu * TauRefUs (sim/edpp.go:1177). Independent of the
	// operating tau_ttft. Dead on this arm while the capacity term is disabled.
	TauRefUs int64

	// ChunkTokens is the per-step prefill token budget. MUST equal the engine's
	// max_num_batched_tokens = 2048 (config.md §1, §5 signal 22). 0 = no cap.
	ChunkTokens int

	// BlockSize is the KV block size in tokens. MUST equal the engine's
	// block_size = 16 (config.md §1, §5 signal 23), and MUST be validated against
	// the scraped Metrics.CacheBlockSize with a loud failure on disagreement:
	// upstream has ONE block size, and a port carrying two that silently convert
	// leaves a latent unit bug in the admission test, which compares blocks
	// accumulated in one unit against a need expressed in another.
	BlockSize int

	// MaxBatchSize is the engine's max_num_seqs = 256 (config.md §1, §5 signal
	// 24). No metric exposes it, so it is config.
	MaxBatchSize int

	// NomPrefillTokens is S_nom, the nominal prefill chunk for the capacity
	// account's fixed prefill normalizer. Simulator default 512
	// (cmd/root.go:1506).
	NomPrefillTokens int

	// SLOCapacityReferenceBatch is the fixed decode width B used by the
	// occupancy-time capacity account (sim/edpp.go:2381).
	SLOCapacityReferenceBatch int

	// Coeffs is the fallback theta; CoeffsByGPU holds the per-GPU-type overrides.
	// config.md §4 carries both files' values. CoeffsByGPU is keyed by the POD
	// LABEL value, not by the scenario's acceleratorType -- see config.md §2.
	Coeffs      Coeffs
	CoeffsByGPU map[string]Coeffs

	// Ablation switches (config.md §10). The focal arm sets NoCapacity true and
	// the other three false.
	SLOExternalityNoCapacity        bool
	SLOExternalityNoExternality     bool
	SLOExternalityNoOwnGood         bool
	SLOExternalityOccupancyCapacity bool

	// DecomposedSLOExternality fixes decode placement to the stock scorer's choice
	// instead of enumerating decode candidates (sim/edpp.go:1385-1387). This is
	// the matched decomposition control, NOT the focal arm: config.md §10 prices
	// it at +0.0485 equal-cell mean goodput against the full joint shape.
	DecomposedSLOExternality bool

	// CXferSizeAware selects the size-aware transfer price. config.md §9.1: true.
	CXferSizeAware bool

	// CXferUs is the FLAT KV-transfer cost, used when CXferSizeAware is false. It is a
	// DISTINCT field from XferBaseUs (upstream sim/edpp.go:77, validated >= 0 at
	// :269-270, simulator default 5 ms at cmd/root.go:1505). XferBaseUs is only the
	// additive base of the SIZE-AWARE form. Conflating them makes the flat path
	// unrepresentable and silently reprices every disaggregated candidate if
	// CXferSizeAware is ever turned off.
	CXferUs int64

	// Transfer model, size-aware form. UNMEASURED on the target -- degradation D5,
	// config.md §7.
	XferBaseUs            float64
	XferBandwidthGBps     float64
	KVBytesPerTokenPerGPU float64

	// OutputTokenProcessingUs is the client-visible per-token post-processing
	// latency (streaming detokenization). It is OUTSIDE the calibrated theta and
	// must be added explicitly to every TTFT projection (sim/edpp.go:1235-1242).
	// config.md §5 signal 25.
	OutputTokenProcessingUs float64
}

// minMu floors every drain rate so predictor denominators never collapse.
// Upstream: `const edppMinMu = 1e-3` (sim/edpp.go:341), consumed only by clampMu
// (sim/edpp.go:670-678), which is reached from muDecode, muPrefill, muDNom, muPNom.
//
// THE VALUE IS 1e-3, NOT 1e-6. Under this arm's configuration the difference is
// latent rather than active -- Mu reaches AdmissionContext but rollforward never
// reads it -- yet it becomes live the moment the capacity account is enabled (muDNom
// and muPNom set the scale) or the estimator is switched to `waiting`.
const minMu = 1e-3

func clampMu(mu float64) float64 {
	if mu < minMu {
		return minMu
	}
	if mu > 1.0 {
		return 1.0
	}
	return mu
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ---------------------------------------------------------------------------
// TARGET-API ADAPTERS -- bodies deliberately absent
//
// Each of these is a lookup against the target's live interfaces. The exact
// accessor must be CONFIRMED against the pinned target checkout; a wrong guess
// must fail loudly rather than return a plausible wrong number. Nothing below
// computes a quantity the simulation computes -- every one of those is stated in
// full further down.
// ---------------------------------------------------------------------------

// endpointGPUType returns the endpoint's GPU-type label value, which selects
// theta. config.md §5 signal 8: read the POD label, never the node's
// nvidia.com/gpu.product -- plugins see endpoints, not nodes, and
// EndpointMetadata carries only the pod's labels.
//
// On an absent or unknown value the endpoint MUST be REJECTED from the candidate
// set and the rejection COUNTED -- never defaulted. Heterogeneity rides the
// per-iteration INTERCEPT alpha, which is present on every iteration regardless
// of KV state (config.md §4: 25563.82 A100 vs 16613.54 H100, a factor of 1.539),
// so a defaulted label is wrong on EVERY decision rather than only under load.
// Without the counter, the removal would look like a routing preference.
func endpointGPUType(ep Endpoint) (gpuType string, ok bool)

// endpointRole returns the endpoint's prefill/decode role from its pod label.
// config.md §5 signal 9.
func endpointRole(ep Endpoint) (role string, ok bool)

// endpointMetrics returns the scraped metrics snapshot. The target's ENTIRE
// native surface is the seven fields on
// pkg/epp/framework/interface/datalayer/metrics.go:26-42.
//
// Two traps, both silent, both confirmed against the pin:
//   - KVCacheUsagePercent is a FRACTION in [0,1], not a percent, despite the
//     name. Dividing by 100 under-counts KV by 100x and collapses the C1*KV term
//     that carries the hardware heterogeneity this experiment exists to measure.
//   - KvCacheMaxTokenCapacity is UNUSABLE: declared at pkg/epp/framework/interface/datalayer/metrics.go:35 and copied
//     in Clone at :77, it is assigned nowhere else outside tests, so it always
//     reads 0 and any branch guarded on it being positive is dead code. Derive KV
//     capacity from CacheNumBlocks * CacheBlockSize instead.
func endpointMetrics(ep Endpoint) Metrics

// promptTokens returns a_r, the request's full prompt length in tokens.
// config.md §5 signal 10: BUILT -- it needs a tokenizer sidecar plus a
// token-producer plugin. The absent-sidecar failure mode returns 0 WITH NO ERROR,
// which reads as a shorter-than-any-threshold prompt and silently disables
// disaggregation for every request. Distinguish a render failure from a
// legitimately empty prompt using the raw request body, and count the failure.
func promptTokens(req Request) (int, bool)

// cachedBlockCount returns the endpoint's cached prefix for this request, from
// which a_p is derived in apForInstance. config.md §5 signal 11.
//
// Three silent-wrong routes to avoid, per the realized mapping:
//   - use the literal cached-block count, NOT a device-tier-weighted ranking
//     score: the latter is <= the former and would overstate cached tokens and
//     under-price prefill work.
//   - use the PREFIX PRODUCER's block size, not the engine's. The producer clamps its
//     own block size up to a 64-token floor
//     (approximateprefix/plugin.go:50, minBlockSizeTokens = 64) while the engine
//     reports 16. So it reports N blocks of >= 64 tokens and the true cached span is
//     64N; multiplying by the engine's 16 yields 16N, which UNDERSTATES cached tokens
//     by 4x. The consequence is that a_p is INFLATED and prefill work is
//     OVER-priced -- biasing toward remote prefill, not toward local.
//   - bind the producer BY NAME so both arms read the SAME prefix signal: a data
//     key's identity includes its producer name, so an unbound read can silently
//     become a second, differently-configured signal.
//
// On a producer miss, treat the prompt as fully UNCACHED and count the miss. A
// miss means "no information", which is not "nothing cached"; charging the full
// prompt over-prices the candidate rather than asserting a cold cache as fact,
// and leaves it in the argmin.
func cachedBlockCount(ep Endpoint, req Request) (blocks int, ok bool)

// residentDecodeState returns the per-resident decode population for an endpoint.
// This is the BUILT shadow table -- degradation D2. It must return the requests
// this EPP placed on ep that have produced a first token, each carrying
// StepsDone, KVBlocks, ArrivalUs, FirstTokenUs, TTFTSet and SLOClass.
//
// Design points that are load-bearing, from the realized port:
//   - a mutex-guarded LONG-LIVED index, not framework per-request state: entries
//     are written from the async response-body goroutine and read from the
//     scheduling path, and the index outlives any one request.
//   - a TTL SWEEP is REQUIRED, not hygiene. Requests terminate without a final
//     chunk (client disconnect, upstream error) and the response hooks carry no
//     termination state, so without the sweep those entries are charged as
//     residents forever and permanently inflate every externality on ep.
//   - StepsDone requires the engine flag --enable-force-include-usage. Without
//     it the usage count arrives only in the FINAL chunk, so StepsDone stays 0
//     for every request's whole lifetime and every remaining-steps estimate is
//     wrong while the table looks present and correct (config.md §11).
//   - a repeated request ID must REPLACE, so a retried request is not counted
//     twice.
func residentDecodeState(ep Endpoint) []RunningReqState

// residentPrefillState returns the pre-first-token occupant population on ep --
// requests a prior decision placed there that have NOT yet produced a first
// token. Same shadow table, prefill index. Degradation D6.
//
// A resident whose prefill runs REMOTELY occupies no prefill capacity on its
// decode endpoint and must be skipped there rather than double-counted.
//
// This population exists as a separate return because the decode-side value
// kernel returns zero for a resident with no first token (gDecodeComposite
// returns 0 when TTFTSet is false), so such a request would be missed entirely
// if it were not moved here, where its first token is still live.
func residentPrefillState(ep Endpoint) []RunningReqState

// residentPrefillTokens returns S_pf, the resident prefill tokens being processed
// this step on ep. Degradation D4: capped per occupant and capped in total at
// Config.ChunkTokens.
func residentPrefillTokens(ep Endpoint) int64

// requestSLOClass returns the request's SLO class, which selects the tau triple.
// config.md §5 signal 19.
func requestSLOClass(req Request) string

// nowUs returns the decision instant in microseconds on the same clock the
// shadow table stamps arrivals and first tokens with.
func nowUs() float64

// Endpoint, Request, Metrics, and RunningReqState are the target's own types plus
// the shadow table's record. Their exact definitions must be confirmed against
// the pinned checkout.
type (
	Endpoint any
	Request  any
	Metrics  any
)

// RunningReqState mirrors sim/admission_estimator.go:19-40. TrueRemaining is the
// ORACLE remaining step count and is -1 on the target -- the oracle is a
// diagnostic upper bound that reads hidden output length, never a deployable
// policy, so the port always takes the censored branch in varDecodeInputs.
type RunningReqState struct {
	StepsDone     int64
	KVBlocks      int64
	TrueRemaining int64 // -1 on the target (deployable); prefill slice carries remaining PROMPT tokens

	SLOClass     string
	ArrivalUs    int64
	FirstTokenUs int64
	TTFTSet      bool

	OracleOutputLen int64 // -1 on the target
}

// Snapshot is the per-endpoint routing view the candidate scorer reads. It
// mirrors the simulator's RoutingSnapshot for the fields this arm uses.
//
// SchedulerStateObserved is the D1 guard: it is PERMANENTLY FALSE on the target
// at this pin, because no route exists to SchedulerRunning, SchedulerWaiting,
// CurrentScheduled, or CurrentStepStartUs. It is carried explicitly so the
// substitution is visible at the point it happens, and so a future engine patch
// supplying the queue has a reachable branch (sim/edpp_scheduler_rollout.go:299).
type Snapshot struct {
	ID      string
	GPUType string

	BatchSize             int   // signal 1: Metrics.RunningRequestsSize
	QueueDepth            int   // signal 2: Metrics.WaitingQueueSize
	KvTokensInUse         int64 // signal 6: derived, usage * blockSize * numBlocks
	FreeKVBlocks          int64 // signal 7: derived, D7
	ResidentPrefillTokens int64 // signal 14: D4
	MaxBatchSize          int64 // config, = max_num_seqs
	BlockSizeTokens       int64

	RunningDecode  []RunningReqState // D2
	RunningPrefill []RunningReqState // D6

	// D1 -- none of these has a route at this pin.
	SchedulerStateObserved    bool
	SchedulerRunning          []SchedulerReqState
	SchedulerWaiting          []SchedulerReqState
	CurrentScheduled          []SchedulerReqState
	CurrentStepStartUs        int64
	MaxScheduledTokens        int64
	LongPrefillTokenThreshold int64

	// AdmissionRate feeds only the `little` estimator, which no registered arm
	// selects. Stated for completeness of the estimator set below.
	AdmissionRate float64
}

// SchedulerReqState mirrors sim/admission_estimator.go:44-53. It carries no
// output length: the rollout supplies the censored per-class running mean.
type SchedulerReqState struct {
	ID              string
	SLOClass        string
	PromptTokens    int64
	ComputedTokens  int64
	ScheduledTokens int64
	KVBlocks        int64
	Priority        float64
	ArrivalUs       int64
}

// ---------------------------------------------------------------------------
// The calibrated latency law (theta)
//
// Ported from sim/edpp_coeffs.go. Units: AlphaD/AlphaP in us, C0 in us/req,
// C1/CPf in us/token, CAttn in us/token^2. Values in config.md §4.
// ---------------------------------------------------------------------------

type Coeffs struct {
	AlphaD float64 // alpha:   decode per-iteration fixed cost
	AlphaP float64 // alpha_p: prefill per-iteration fixed cost (~= AlphaD)
	C0     float64 // decode per-request overhead
	C1     float64 // decode KV-read per resident token
	CPf    float64 // exposed prefill compute per token
	CAttn  float64 // prefill attention term
}

// Wp is the prefill demand of ap uncached tokens for a prompt of full length ar,
// in us. sim/edpp_coeffs.go:66-71.
//
// It is the trajectory sum of the causal per-step prefill charge
// CPf*s + CAttn*s*(prefix + s/2), integrated over the prefill from prefix ar-ap
// to ar -- hence the (ar - ap/2) form. At ap == ar (no cache) this is
// CPf*ar + 0.5*CAttn*ar^2.
func (c Coeffs) Wp(ap, ar int) float64 {
	a := float64(ap)
	r := float64(ar)
	return c.CPf*a + c.CAttn*a*(r-a/2.0)
}

// Wd is the decode demand for a prompt of length ar generating o output tokens,
// in us. sim/edpp_coeffs.go:76-83.
//
// The exact discrete per-step sum Sum_{k=0}^{o-1}(C0 + C1*(ar+k))
// = C0*o + C1*o*(ar + (o-1)/2), matching the per-decode-step charge where
// context = ar + k. o is the N_out estimate at routing time.
func (c Coeffs) Wd(ar int, o float64) float64 {
	if o <= 0 {
		return 0
	}
	r := float64(ar)
	return c.C0*o + c.C1*o*(r+(o-1)/2.0)
}

// tIterDecode is the decode iteration time (us) at the given batch state.
// sim/edpp_coeffs.go:85-87.
//
// NOTE ON bDec: upstream's BatchSize() INCLUDES prefilling requests, and so does
// the target's Metrics.RunningRequestsSize. The mapping is EXACT, so the port
// preserves it and deliberately does not rename it or "fix" it to a decode-only
// count. Upstream is internally inconsistent here -- its calibration fits C0
// against a decode-only column while the policy consumes the running total -- and
// the port INHERITS that inconsistency rather than introducing a different one.
// Cost: C0 is ~5.3-5.9 us/req (config.md §4), so miscounting four prefilling
// requests is ~21 us against a 17000-27000 us iteration, under 0.1%.
func (c Coeffs) tIterDecode(bDec int, kv, sPf int64) float64 {
	return c.AlphaD + c.C0*float64(bDec) + c.C1*float64(kv) + c.CPf*float64(sPf)
}

// tIterPrefill is the prefill iteration time (us). A dedicated prefill server
// runs no decode work, so B_dec = 0 and KV = 0. sim/edpp_coeffs.go:91-93.
func (c Coeffs) tIterPrefill(sPf int64) float64 {
	return c.AlphaP + c.CPf*float64(sPf)
}

// muDecode / muPrefill are the live drain rates mu = 1 - alpha/T_iter, clamped.
// sim/edpp_coeffs.go:97-105.
func (c Coeffs) muDecode(bDec int, kv, sPf int64) float64 {
	return clampMu(1.0 - c.AlphaD/c.tIterDecode(bDec, kv, sPf))
}

func (c Coeffs) muPrefill(sPf int64) float64 {
	return clampMu(1.0 - c.AlphaP/c.tIterPrefill(sPf))
}

// muDNom is the fixed nominal decode drain rate at the SLO-critical batch, where
// T_iter == tau_itl. sim/edpp_coeffs.go:107-110. Caller guarantees
// tau_itl > AlphaD.
func (c Coeffs) muDNom(tauITLUs float64) float64 {
	return clampMu(1.0 - c.AlphaD/tauITLUs)
}

// muPNom is the fixed nominal prefill drain rate at the nominal operating chunk.
// sim/edpp_coeffs.go:113-116.
func (c Coeffs) muPNom(sPfNom int) float64 {
	return clampMu(1.0 - c.AlphaP/(c.AlphaP+c.CPf*float64(sPfNom)))
}

// deltaBarDecode is the marginal decode work per step at context length ctxLen.
// sim/edpp_coeffs.go:119-121. Read by the comparator's ITL term, not by this
// arm's composite objective; stated so both arms share one law.
func (c Coeffs) deltaBarDecode(ctxLen float64) float64 {
	return c.C0 + c.C1*ctxLen
}

// validate mirrors sim/edpp_coeffs.go:127-146. The alpha ~= alpha_p bound is not
// cosmetic: a divergence over 10% means the JSON was fit on mismatched hardware
// or regimes. Both files in config.md §4 are within 0.03%.
func (c Coeffs) validate() error {
	switch {
	case c.AlphaD <= 0:
		return errCoeff("AlphaD must be > 0")
	case c.AlphaP <= 0:
		return errCoeff("AlphaP must be > 0")
	case c.C0 < 0:
		return errCoeff("C0 must be >= 0")
	case c.C1 < 0:
		return errCoeff("C1 must be >= 0")
	case c.CPf <= 0:
		return errCoeff("CPf must be > 0")
	case c.CAttn < 0:
		return errCoeff("CAttn must be >= 0")
	}
	if rel := math.Abs(c.AlphaD-c.AlphaP) / c.AlphaD; rel > 0.10 {
		return errCoeff("AlphaD and AlphaP diverge by more than 10%")
	}
	return nil
}

// errCoeff is the specification's stand-in for the target's error construction.
func errCoeff(msg string) error

// ---------------------------------------------------------------------------
// Admission-delay estimators
//
// Ported from sim/admission_estimator.go. ALL FOUR are stated because D1 makes
// the choice among them a port decision rather than an inherited one: the focal
// arm selects `rollforward` (config.md §9.1), and a reader must be able to see
// what the published policy would have done instead.
//
// AdmissionContext is a pure input bundle: an estimator is a pure function of it.
// ---------------------------------------------------------------------------

// AdmissionContext mirrors sim/admission_estimator.go:57-70. Times and work in us.
type AdmissionContext struct {
	QWork             float64 // waiting-backlog work (us) -- read only by `waiting`
	Mu                float64 // occupancy-aware drain rate
	BatchSize         int
	MaxBatchSize      int
	FreeKVBlocks      int64
	ReqKVNeed         int64
	TIter             float64
	QueueDepth        int
	AdmissionRate     float64 // req/us -- read only by `little`
	RemainingStepsEst float64
	Running           []RunningReqState
}

// flooredTAdm lower-bounds an estimate by one iteration: even with a free slot, a
// request waits for the current decode step to finish before the next batch
// formation admits it. No-op when TIter is unavailable or already exceeded.
// sim/admission_estimator.go:79-86.
func flooredTAdm(est float64, ctx AdmissionContext) float64 {
	if ctx.TIter > est {
		return ctx.TIter
	}
	return est
}

// waitingEstimateTAdm is the `waiting` estimator. sim/admission_estimator.go:89-95.
//
// NOT SELECTED by any registered arm. It is the only consumer of ctx.QWork, which
// is why QWork is a declared-but-unread input on the target: reconstructing the
// per-instance work account would be dead state that implies a consumer.
func waitingEstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.Mu <= 0 {
		return 0
	}
	return ctx.QWork / ctx.Mu
}

// littleEstimateTAdm is the `little` estimator. sim/admission_estimator.go:99-105.
//
// NOT SELECTED by any registered arm. It is the only consumer of
// ctx.AdmissionRate, for the same reason as QWork above.
func littleEstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.AdmissionRate <= 0 {
		return flooredTAdm(0, ctx)
	}
	return flooredTAdm(float64(ctx.QueueDepth)/ctx.AdmissionRate, ctx)
}

// fluidEstimateTAdm is the `fluid` estimator. sim/admission_estimator.go:109-124.
//
// NOT SELECTED, but its wave form IS reached: rollforward falls back to it for the
// deep tail, so the arithmetic below is live on the target.
func fluidEstimateTAdm(ctx AdmissionContext) float64 {
	// Admit next iteration if a slot AND enough KV already fit.
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return flooredTAdm(0, ctx)
	}
	if ctx.BatchSize <= 0 || ctx.RemainingStepsEst <= 0 || ctx.TIter <= 0 {
		return flooredTAdm(0, ctx)
	}
	// Synchronized batch: occupants finish ~R steps together, so slots free in
	// WAVES of BatchSize every ~R iterations. A request at queue position
	// QueueDepth waits ceil((QueueDepth+1)/BatchSize) waves. This is deliberately
	// NOT the naive fluid drain /BatchSize.
	waves := math.Ceil(float64(ctx.QueueDepth+1) / float64(ctx.BatchSize))
	return flooredTAdm(waves*ctx.RemainingStepsEst*ctx.TIter, ctx)
}

// rollforwardEstimateTAdm is the `rollforward` estimator, which THIS ARM SELECTS
// (config.md §9.1). sim/admission_estimator.go:127-176.
//
// Deterministic look-ahead: each running request departs after its remaining steps
// (oracle TrueRemaining if >= 0, else the N_out estimate), freeing its KV.
// Accumulate departureStep * T_iter until a slot AND enough free KV exist.
//
// DEGRADATION D1 LIVES HERE. This is the SUBSTITUTE for the scheduler rollout, not
// the published estimator. Its bias is stated in the file header: past one batch
// drain it understates admission delay, biasing toward remote prefill. Every call
// that reaches this function on the target instead of schedulerRollout MUST
// increment a substitution counter.
func rollforwardEstimateTAdm(ctx AdmissionContext) float64 {
	if ctx.BatchSize < ctx.MaxBatchSize && ctx.FreeKVBlocks >= ctx.ReqKVNeed {
		return flooredTAdm(0, ctx)
	}
	type dep struct{ step, kv int64 }
	deps := make([]dep, 0, len(ctx.Running))
	for _, r := range ctx.Running {
		rem := r.TrueRemaining
		if rem < 0 {
			// The deployable branch, and the ONLY one reachable on the target:
			// TrueRemaining is -1 because the oracle reads hidden output length.
			rem = int64(ctx.RemainingStepsEst)
			if rem < 1 {
				rem = 1
			}
		}
		deps = append(deps, dep{step: rem, kv: r.KVBlocks})
	}
	// Sort by departure step ascending, STABLY: the tie-break must be
	// deterministic or two identical decisions can differ.
	stableSortByStep(deps)
	// The request sits at queue position QueueDepth: the QueueDepth requests ahead
	// fill the first QueueDepth freed slots and ours takes the next, so
	// QueueDepth+1 slots must free (plus our KV) before we are admitted.
	needSlots := int64(ctx.QueueDepth + 1)
	freeSlots := int64(ctx.MaxBatchSize - ctx.BatchSize)
	freeKV := ctx.FreeKVBlocks
	for _, d := range deps {
		freeSlots++
		freeKV += d.kv
		if freeSlots >= needSlots && freeKV >= ctx.ReqKVNeed {
			return flooredTAdm(float64(d.step)*ctx.TIter, ctx)
		}
	}
	// The running set's departures were exhausted before freeing QueueDepth+1
	// slots (queue deeper than one batch drain). Fall back to the fluid wave form.
	// THIS is the branch that understates delay -- see D1.
	if ctx.BatchSize > 0 {
		waves := math.Ceil(float64(ctx.QueueDepth+1) / float64(ctx.BatchSize))
		return flooredTAdm(waves*ctx.RemainingStepsEst*ctx.TIter, ctx)
	}
	// BatchSize <= 0: no wave cadence available -- cap at the last known departure.
	if len(deps) > 0 {
		return flooredTAdm(float64(deps[len(deps)-1].step)*ctx.TIter, ctx)
	}
	return flooredTAdm(0, ctx)
}

// stableSortByStep sorts ascending by .step, STABLY. sim/admission_estimator.go:147.
//
// Stability is not cosmetic: two running requests with equal remaining steps but
// different KV holdings are accumulated in slice order, so an unstable sort can
// change which departure first satisfies the KV condition, and therefore change the
// returned delay. Adapter for the target's sort import; the CONTRACT (ascending by
// step, stable) is the specified part.
func stableSortByStep(deps []struct{ step, kv int64 })

// ---------------------------------------------------------------------------
// The scheduler rollout -- THE PUBLISHED ESTIMATOR
//
// Ported in full from sim/edpp_scheduler_rollout.go. Every number in
// sim_results/ was produced by this, not by rollforwardEstimateTAdm.
//
// ITS INPUTS ARE UNOBTAINABLE ON THE TARGET (degradation D1): Snapshot's
// SchedulerStateObserved is permanently false at this pin, so on the target this
// code is unreachable and rollforwardEstimateTAdm substitutes. It is stated in
// full REGARDLESS, because an unobtainable INPUT does not license omitting the
// algebra that consumes it -- the two are independent facts and both are recorded.
// A future engine patch exporting the wait queue makes this branch reachable
// without re-deriving anything.
// ---------------------------------------------------------------------------

// rolloutReq is a mutable copy used by the replay. outputRemaining counts only
// FUTURE decode grants. Prefill progress is represented exactly by
// prompt-computed, because token-budget contention can make an actual prefill
// grant smaller than the nominal chunk cap; combining the two into one grant count
// would let extra prefill steps incorrectly consume the predicted output lifetime.
// sim/edpp_scheduler_rollout.go:11-20.
type rolloutReq struct {
	id              string
	prompt          int64
	computed        int64
	kvBlocks        int64
	outputRemaining int64
	target          bool
}

type rolloutGrant struct {
	req     *rolloutReq
	before  int64
	grant   int64
	prefill bool
}

type rolloutResult struct {
	admissionUs  float64
	firstTokenUs float64
	admitted     bool
	firstToken   bool
}

type rolloutContext struct {
	running, waiting   []*rolloutReq
	target             *rolloutReq
	currentScheduled   []SchedulerReqState
	currentStepStartUs int64
	nowUs              float64
	freeKVBlocks       int64
	tokenBudget        int64
	prefillChunkCap    int64
	blockSize          int64
	maxBatch           int
	maxSteps           int
	theta              Coeffs
	alpha              float64
}

// ceilBlocks is ceil(tokens/blockSize), 0 for non-positive tokens.
// sim/edpp_scheduler_rollout.go:35-43.
func ceilBlocks(tokens, blockSize int64) int64 {
	if tokens <= 0 {
		return 0
	}
	if blockSize <= 0 {
		blockSize = 1
	}
	return (tokens + blockSize - 1) / blockSize
}

// rolloutGrantTime is one replayed step's duration: the per-iteration intercept
// plus each grant's marginal charge. A prefill grant is charged causally against
// the prefix already computed; a decode grant is charged C0 + C1*context.
// sim/edpp_scheduler_rollout.go:45-57.
func rolloutGrantTime(grants []rolloutGrant, theta Coeffs, alpha float64) float64 {
	t := alpha
	for _, item := range grants {
		if item.prefill {
			grant := float64(item.grant)
			before := float64(item.before)
			t += theta.CPf*grant + theta.CAttn*grant*(before+grant/2)
		} else {
			t += theta.C0 + theta.C1*float64(item.before)
		}
	}
	return math.Max(t, 0)
}

// currentScheduledTime is the duration of the step ALREADY IN FLIGHT at the
// decision instant, reconstructed from its grants.
// sim/edpp_scheduler_rollout.go:59-77.
func currentScheduledTime(states []SchedulerReqState, theta Coeffs, alpha float64) float64 {
	grants := make([]rolloutGrant, 0, len(states))
	for i := range states {
		state := states[i]
		if state.ScheduledTokens <= 0 {
			continue
		}
		grants = append(grants, rolloutGrant{
			req:     &rolloutReq{},
			before:  state.ComputedTokens,
			grant:   state.ScheduledTokens,
			prefill: state.ComputedTokens < state.PromptTokens,
		})
	}
	if len(grants) == 0 {
		return 0
	}
	return rolloutGrantTime(grants, theta, alpha)
}

// schedulerRollout replays future scheduler steps until the target request is
// admitted and reaches its first token. sim/edpp_scheduler_rollout.go:79-243.
//
// The replay models, in order per step: continuing the running set under the token
// budget, KV-pressure preemption from the tail, then admitting from the wait queue
// while a slot and budget remain. The target request is appended to the WAIT QUEUE,
// so its admission is predicted rather than assumed.
func schedulerRollout(ctx rolloutContext) rolloutResult {
	result := rolloutResult{}
	// The in-flight step's REMAINING time: its full duration less however much of
	// it has already elapsed.
	elapsed := math.Max(
		currentScheduledTime(ctx.currentScheduled, ctx.theta, ctx.alpha)-
			math.Max(ctx.nowUs-float64(ctx.currentStepStartUs), 0),
		0,
	)
	running := append([]*rolloutReq(nil), ctx.running...)
	waiting := append([]*rolloutReq(nil), ctx.waiting...)
	waiting = append(waiting, ctx.target)
	freeKV := ctx.freeKVBlocks
	if freeKV < 0 {
		freeKV = 0
	}

	for step := 0; step < ctx.maxSteps; step++ {
		budget := ctx.tokenBudget
		grants := make([]rolloutGrant, 0, len(running)+len(waiting))
		preempted := make([]*rolloutReq, 0)

		// --- Continue the running set under the token budget.
		for index := 0; index < len(running) && budget > 0; index++ {
			req := running[index]
			before := req.computed
			isPrefill := before < req.prompt
			demand := int64(1) // a decode step demands exactly one token
			if isPrefill {
				demand = req.prompt - before
				if ctx.prefillChunkCap > 0 {
					demand = minI64(demand, ctx.prefillChunkCap)
				}
			}
			grant := minI64(demand, budget)
			if grant <= 0 {
				continue
			}
			newBlocks := ceilBlocks(before+grant, ctx.blockSize)
			deltaBlocks := newBlocks - req.kvBlocks
			if deltaBlocks < 0 {
				deltaBlocks = 0
			}
			canSchedule := true
			// KV pressure: preempt from the TAIL until the delta fits.
			for deltaBlocks > freeKV {
				if len(running) == 0 {
					canSchedule = false
					break
				}
				victim := running[len(running)-1]
				running = running[:len(running)-1]
				freeKV += victim.kvBlocks
				victim.kvBlocks = 0
				// A preempted request must RECOMPUTE. One already in decode
				// recomputes its prompt plus the output it had produced, so its
				// effective prompt grows by one; one still prefilling recomputes
				// its prompt.
				resumeTokens := victim.prompt
				if victim.computed >= victim.prompt {
					resumeTokens = victim.computed + 1
				}
				victim.prompt = resumeTokens
				victim.computed = 0
				preempted = append([]*rolloutReq{victim}, preempted...)
				if victim == req {
					canSchedule = false
					break
				}
			}
			if !canSchedule {
				break
			}
			freeKV -= deltaBlocks
			req.kvBlocks = newBlocks
			grants = append(grants, rolloutGrant{req: req, before: before, grant: grant, prefill: isPrefill})
			budget -= grant
		}

		// Preempted requests go to the FRONT of the wait queue, and no new
		// admission happens on a step that preempted.
		if len(preempted) > 0 {
			waiting = append(preempted, waiting...)
		}
		for len(preempted) == 0 && len(waiting) > 0 && budget > 0 && len(running) < ctx.maxBatch {
			req := waiting[0]
			before := req.computed
			isPrefill := before < req.prompt
			demand := int64(1)
			if isPrefill {
				demand = req.prompt - before
				if ctx.prefillChunkCap > 0 {
					demand = minI64(demand, ctx.prefillChunkCap)
				}
			}
			grant := minI64(demand, budget)
			if grant <= 0 {
				break
			}
			newBlocks := ceilBlocks(before+grant, ctx.blockSize)
			deltaBlocks := newBlocks - req.kvBlocks
			if deltaBlocks < 0 {
				deltaBlocks = 0
			}
			if deltaBlocks > freeKV {
				break
			}
			// THE ADMISSION INSTANT: recorded when the target first receives a
			// grant from the wait queue.
			if req.target {
				result.admissionUs = elapsed
				result.admitted = true
			}
			waiting = waiting[1:]
			freeKV -= deltaBlocks
			req.kvBlocks = newBlocks
			running = append(running, req)
			grants = append(grants, rolloutGrant{req: req, before: before, grant: grant, prefill: isPrefill})
			budget -= grant
		}

		// A step that only preempted makes no progress but is not a dead end.
		if len(grants) == 0 && len(preempted) > 0 {
			continue
		}
		if len(grants) == 0 {
			return result
		}

		// THE FIRST-TOKEN CONDITION: the target either took a decode grant, or
		// took the prefill grant that COMPLETED its prompt.
		targetFirstToken := false
		for _, item := range grants {
			if item.req.target && ((!item.prefill) || item.before+item.grant >= item.req.prompt) {
				targetFirstToken = true
			}
		}
		elapsed += rolloutGrantTime(grants, ctx.theta, ctx.alpha)

		// Advance every granted request, retire the finished, reclaim their KV.
		kept := running[:0]
		for _, req := range running {
			var grant *rolloutGrant
			for i := range grants {
				if grants[i].req == req {
					grant = &grants[i]
					break
				}
			}
			if grant == nil {
				kept = append(kept, req)
				continue
			}
			req.computed = grant.before + grant.grant
			if !grant.prefill {
				req.outputRemaining--
			}
			if grant.prefill || req.outputRemaining > 0 {
				kept = append(kept, req)
			} else {
				freeKV += req.kvBlocks
			}
		}
		running = kept

		if targetFirstToken {
			result.firstTokenUs = elapsed
			result.firstToken = true
			return result
		}
	}
	return result
}

func minI64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

// rolloutReqFor converts one observed scheduler request into a replay copy,
// supplying the censored per-class output estimate. sim/edpp_scheduler_rollout.go:244-260.
//
// The output estimate is CENSORED: a request that has already decoded decodeDone
// tokens has total output >= decodeDone, so the class mean is floored by that
// before subtracting, and the remainder floored at 1.
func (p *Policy) rolloutReqFor(state SchedulerReqState, chunkCap int64) *rolloutReq {
	computed := state.ComputedTokens
	if computed < 0 {
		computed = 0
	}
	prompt := state.PromptTokens
	if prompt < 0 {
		prompt = 0
	}
	outputEstimate := p.nHatFor(state.SLOClass)
	decodeDone := computed - prompt
	if decodeDone < 0 {
		decodeDone = 0
	}
	totalOutput := math.Max(outputEstimate, float64(decodeDone))
	remainingOutput := int64(math.Ceil(totalOutput)) - decodeDone
	if remainingOutput < 1 {
		remainingOutput = 1
	}
	return &rolloutReq{
		id: state.ID, prompt: prompt, computed: computed,
		kvBlocks:        maxI64(state.KVBlocks, 0),
		outputRemaining: remainingOutput,
	}
}

func maxI64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// rolloutTimes applies the replay to one candidate endpoint.
// sim/edpp_scheduler_rollout.go:262-307.
//
// cachedTokens is the target's known prefix on that endpoint. decodeOnly models a
// TRANSFERRED request: its prompt is already computed elsewhere, but its KV blocks
// still have to fit at the candidate decoder. prefillPool selects the prefill
// intercept alpha_p instead of the decode intercept alpha.
//
// THE GUARD ON LINE ONE IS DEGRADATION D1: SchedulerStateObserved is permanently
// false on the target, so this returns (zero, false) on every call and every
// caller falls through to rollforwardEstimateTAdm.
func (p *Policy) rolloutTimes(req Request, snap Snapshot, theta Coeffs, cachedTokens int, decodeOnly, prefillPool bool, nHatOut, now float64) (rolloutResult, bool) {
	if !snap.SchedulerStateObserved || snap.MaxScheduledTokens <= 0 || snap.MaxBatchSize <= 0 {
		return rolloutResult{}, false
	}
	blockSize := snap.BlockSizeTokens
	if blockSize <= 0 {
		blockSize = int64(maxInt(p.cfg.BlockSize, 1))
	}
	chunkCap := snap.MaxScheduledTokens
	if snap.LongPrefillTokenThreshold > 0 {
		chunkCap = minI64(chunkCap, snap.LongPrefillTokenThreshold)
	}
	running := make([]*rolloutReq, 0, len(snap.SchedulerRunning))
	for _, state := range snap.SchedulerRunning {
		running = append(running, p.rolloutReqFor(state, chunkCap))
	}
	waiting := make([]*rolloutReq, 0, len(snap.SchedulerWaiting))
	for _, state := range snap.SchedulerWaiting {
		waiting = append(waiting, p.rolloutReqFor(state, chunkCap))
	}
	prompt := int64(p.inputLen(req))
	computed := int64(maxInt(cachedTokens, 0))
	targetKV := ceilBlocks(computed, blockSize)
	if decodeOnly {
		computed = prompt
		targetKV = 0
	}
	target := &rolloutReq{
		id: p.requestID(req), prompt: prompt, computed: computed, kvBlocks: targetKV,
		outputRemaining: maxI64(int64(math.Ceil(math.Max(nHatOut, 1))), 1),
		target:          true,
	}
	alpha := theta.AlphaD
	if prefillPool {
		alpha = theta.AlphaP
	}
	result := schedulerRollout(rolloutContext{
		running: running, waiting: waiting, target: target,
		currentScheduled: snap.CurrentScheduled, currentStepStartUs: snap.CurrentStepStartUs,
		nowUs: now, freeKVBlocks: snap.FreeKVBlocks,
		tokenBudget: snap.MaxScheduledTokens, prefillChunkCap: chunkCap,
		blockSize: blockSize, maxBatch: int(snap.MaxBatchSize), maxSteps: 100000,
		theta: theta, alpha: alpha,
	})
	return result, result.admitted
}

// rolloutLocalTTFT predicts (admission, TTFT) for a LOCAL placement.
// sim/edpp_scheduler_rollout.go:309-316. Note the client-visible TTFT adds the
// output-token post-processing latency, which is outside theta.
func (p *Policy) rolloutLocalTTFT(ec *evalCtx, ds Snapshot, theta Coeffs) (tAdm, ttft float64, ok bool) {
	cached := ec.inputLen - maxInt(p.apForInstance(ec.req, ds.ID), 0)
	result, ok := p.rolloutTimes(ec.req, ds, theta, cached, false, false, ec.nHatOut, ec.nowUs)
	if !ok || !result.firstToken {
		return 0, 0, false
	}
	return result.admissionUs, result.firstTokenUs + p.cfg.OutputTokenProcessingUs, true
}

// rolloutDecodeAdmission predicts decode admission for a TRANSFERRED request: its
// prompt counts as already computed. sim/edpp_scheduler_rollout.go:318-324.
func (p *Policy) rolloutDecodeAdmission(ec *evalCtx, ds Snapshot, theta Coeffs) (float64, bool) {
	result, ok := p.rolloutTimes(ec.req, ds, theta, ec.inputLen, true, false, ec.nHatOut, ec.nowUs)
	if !ok {
		return 0, false
	}
	return result.admissionUs, true
}

// rolloutPrefillCompletion predicts (admission, completion) on a prefill pool
// endpoint. sim/edpp_scheduler_rollout.go:326-333. No post-processing term: this
// clock ends at prefill completion, not at a client-visible token.
func (p *Policy) rolloutPrefillCompletion(ec *evalCtx, ps Snapshot, theta Coeffs) (tAdm, completion float64, ok bool) {
	cached := ec.inputLen - maxInt(p.apForInstance(ec.req, ps.ID), 0)
	result, ok := p.rolloutTimes(ec.req, ps, theta, cached, false, true, ec.nHatOut, ec.nowUs)
	if !ok || !result.firstToken {
		return 0, 0, false
	}
	return result.admissionUs, result.firstTokenUs, true
}

// ---------------------------------------------------------------------------
// The causal re-timing model
//
// Ported from sim/edpp_var.go:100-188. These are BATCH-LEVEL per-iteration times,
// not per-resident: adding one request changes the whole batch's iteration time.
//
// THE ADMISSION GATING IN cLocalAfter AND cDisagg IS THE CAUSAL MECHANISM. A
// resident that completes before the arrival is admitted is delayed by exactly
// nothing, so its contribution is exactly zero. That is what makes this term an
// externality rather than another load proxy.
// ---------------------------------------------------------------------------

// reTiming holds the three per-iteration decode times a placement can produce.
// sim/edpp_var.go:109-127.
//
//	tIter0       current batch B per-iter time -- the baseline.
//	tIterOverlap LOCAL placement, while the arriving request R prefills
//	             co-scheduled on the decode batch: chunked prefill adds `chunk`
//	             resident prefill tokens, so this is tIter0 + CPf*chunk.
//	             DEAD ON THIS ARM: computed by reTimingFor on every call but never
//	             read, because exactPrefillOverlap is forced true and the exact
//	             branch of cLocalAfter returns before touching it. Stated so a port
//	             does not mistake it for a live term -- or delete it and then be
//	             unable to run the legacy branch at all.
//	tIterAfter   after R JOINS the decode batch: tIterDecode(B+1, kv+dkv_R, sPf),
//	             where dkv_R is R's full input length. This is FULL re-timing --
//	             recompute tIterDecode with B+1 and kv+dkv_R, not a marginal add.
type reTiming struct {
	tIter0       float64
	tIterOverlap float64
	tIterAfter   float64

	// cAttn and chunk parameterize the causal prefill attention added to the
	// overlap window: each of R's co-scheduled chunks attends to its causal prefix.
	cAttn float64
	chunk float64

	// exactPrefillOverlap and its operands select the EXACT marginal-overlap form.
	// THIS ARM FORCES IT TRUE (sim/edpp.go:1709, :1745), overriding config: the
	// legacy form assumes every overlapping chunk is full and starts at prefix
	// zero, while the exact form uses the known uncached span [ar-ap, ar), handles
	// the partial last chunk, and charges only marginal work above the baseline
	// decode iteration.
	exactPrefillOverlap bool
	cPf                 float64
	ap                  float64
	ar                  float64
}

// reTimingFor builds the batch-level per-iteration times under decode physics
// thetaD at batch state (bDec, kv, sPf). sim/edpp_var.go:675-692.
//
// dkv_R is R's FULL input length -- its resident context once it joins the decode
// batch. Input-only, so it reads no hidden output length.
func (p *Policy) reTimingFor(inputLen int, thetaD Coeffs, bDec int, kv, sPf int64, chunk int) reTiming {
	dkv := int64(inputLen)
	return reTiming{
		tIter0:       thetaD.tIterDecode(bDec, kv, sPf),
		tIterOverlap: thetaD.tIterDecode(bDec, kv, sPf+int64(chunk)),
		tIterAfter:   thetaD.tIterDecode(bDec+1, kv+dkv, sPf),
		cAttn:        thetaD.CAttn,
		chunk:        float64(chunk),
	}
}

// cBase is a decode resident's projected completion with rem steps left at the
// current (pre-R) batch per-iteration time. sim/edpp_var.go:130-132.
func (rt reTiming) cBase(now float64, rem int64) float64 {
	return now + float64(rem)*rt.tIter0
}

// cLocal is the zero-admission-delay compatibility form of cLocalAfter.
// sim/edpp_var.go:135-137. Not reached by this arm, which always supplies an
// admission window; stated because it is the published function's other entry.
func (rt reTiming) cLocal(now float64, rem int64, nChunks float64) float64 {
	return rt.cLocalAfter(now, rem, 0, nChunks)
}

// cLocalAfter is the resident's completion under LOCAL placement when R waits
// admissionSteps baseline iterations before joining. sim/edpp_var.go:144-166.
//
// THE CAUSAL STRUCTURE, in three phases:
//   - pre:       min(admissionSteps, rem) iterations run UNDISTURBED. A resident
//                that finishes inside the admission window is untouched.
//   - overlap:   of the surviving tail, the first min(nChunks, remaining)
//                iterations overlap R's prefill.
//   - remainder: the rest run at the B+1 re-timed rate.
func (rt reTiming) cLocalAfter(now float64, rem int64, admissionSteps, nChunks float64) float64 {
	pre := math.Min(math.Max(admissionSteps, 0), float64(rem))
	remaining := float64(rem) - pre
	overlap := math.Min(nChunks, remaining)
	if rt.exactPrefillOverlap {
		// THE BRANCH THIS ARM TAKES. Baseline iteration time for the overlap
		// window plus only R's MARGINAL prefill work.
		return now + pre*rt.tIter0 + overlap*rt.tIter0 +
			prefillMarginalWork(rt.cPf, rt.cAttn, rt.ap, rt.ar, rt.chunk, overlap) +
			(remaining-overlap)*rt.tIterAfter
	}
	// LEGACY form, retained because it is what the published function computes when
	// the exact flag is off. Causal prefill attention over R's co-scheduled chunks
	// j = 0..overlap-1, each charged against causal prefix j*chunk + chunk/2
	// (start prefix 0), summing to cAttn*chunk^2*overlap^2/2.
	attn := rt.cAttn * rt.chunk * rt.chunk * overlap * overlap / 2.0
	return now + pre*rt.tIter0 + overlap*rt.tIterOverlap + attn + (remaining-overlap)*rt.tIterAfter
}

// prefillMarginalWork is the exact work added by the first `iterations` chunks of
// an arriving request's uncached prefill. sim/edpp_var.go:168-179.
//
//	processed    = min(ap, iterations*chunk)
//	cachedPrefix = max(ar-ap, 0)
//	work         = CPf*processed + CAttn*processed*(cachedPrefix + processed/2)
//
// It EXCLUDES baseline iteration time: residents would pay that even if the
// arriving request were absent. That exclusion is why cLocalAfter's exact branch
// charges overlap*tIter0 separately rather than overlap*tIterOverlap.
func prefillMarginalWork(cPf, cAttn, ap, ar, chunk, iterations float64) float64 {
	if ap <= 0 || chunk <= 0 || iterations <= 0 {
		return 0
	}
	processed := math.Min(ap, iterations*chunk)
	cachedPrefix := math.Max(ar-ap, 0)
	return cPf*processed + cAttn*processed*(cachedPrefix+processed/2.0)
}

// cDisagg is the resident's completion under DISAGGREGATED placement.
// sim/edpp_var.go:181-188.
//
// R prefills remotely, so the decode endpoint is undisturbed for arrivalSteps
// iterations (R's remote prefill plus KV transfer, in units of tIter0), and only
// the tail max(rem - arrivalSteps, 0) steps run at the B+1 re-timed time. There is
// NO prefill-overlap inflation on this endpoint -- that is the asymmetry the
// policy trades against transfer cost.
func (rt reTiming) cDisagg(now float64, rem int64, arrivalSteps float64) float64 {
	pre := math.Min(arrivalSteps, float64(rem))
	return now + pre*rt.tIter0 + (float64(rem)-pre)*rt.tIterAfter
}

// ---------------------------------------------------------------------------
// SLO value kernels
//
// THIS ARM USES THE `composite` KERNEL EXCLUSIVELY -- forced at sim/edpp.go:1712
// and :1748, not read from config. The other three kernels (flip, util, hazard)
// are NOT stated here: they are selected by a different rule (Rule=="var") that
// this arm does not use, and no switch in this arm's configuration reaches them.
// ---------------------------------------------------------------------------

// slo is one resident's resolved SLO thresholds in microseconds.
// sim/edpp_var.go:597-608.
type slo struct {
	tauTTFTUs float64
	tauITLUs  float64
	tauE2EUs  float64
}

// sigmoid is the standard logistic. sim/edpp_var.go:992.
func sigmoid(x float64) float64 { return 1.0 / (1.0 + math.Exp(-x)) }

// sloCompositeValue is the bounded routing value: a smooth TTFT x E2E surrogate.
// sim/edpp_var.go:242-251.
//
// Each ENABLED dimension uses ITS OWN threshold as the transition bandwidth, so
// the value is scale-free in tau. A DISABLED target (tau <= 0) contributes exactly
// one -- which is why a zero triple flattens the policy to a constant rather than
// loosening it (see Config.TauTTFTUs).
//
// NOTE WHAT IS ABSENT: tau_itl. The composite kernel never reads it. Mean ITL is
// an evaluation gate on reported goodput, not a routing term.
func sloCompositeValue(s slo, ttftUs, e2eUs float64) float64 {
	u := 1.0
	if s.tauTTFTUs > 0 {
		u *= sigmoid((s.tauTTFTUs - ttftUs) / s.tauTTFTUs)
	}
	if s.tauE2EUs > 0 {
		u *= sigmoid((s.tauE2EUs - e2eUs) / s.tauE2EUs)
	}
	return u
}

// decodeResident is one already-decoding resident's value inputs.
// sim/edpp_var.go co-resident struct, assembled at :615-641.
type decodeResident struct {
	rem          int64 // remaining decode steps; < 0 means SKIP
	arrivalUs    int64
	firstTokenUs int64
	ttftSet      bool
	slo          slo
}

// gDecodeComposite is a decoding resident's composite good at completion cUs.
// sim/edpp_var.go:253-260.
//
// Its realized TTFT is FIXED by history, so the placement cannot move it: it
// enters as a constant factor. THAT IS WHY DEGRADATION D2c HAS THE DIRECTION IT
// DOES -- a late-recorded first token shrinks sigmoid((tau_ttft - ttft)/tau_ttft),
// a COMMON POSITIVE MULTIPLIER on both terms of the charge below, scaling the
// whole difference down and biasing toward local.
//
// A resident with no first token yet returns 0 both before and after, so it
// contributes nothing here. Such requests are NOT lost: they are carried in the
// prefill-occupant population instead, where their first token is still live.
// THE tau_e2e ZERO TRAP, sharper than the general flattening note on
// Config.TauE2EUs and stated where it bites. Because the TTFT factor is REALIZED
// and therefore placement-invariant, it is identical on both sides of the charge
// and factors out. So if tau_e2e resolves to 0 for a resident, this function
// reduces to that invariant factor alone and the resident's contribution is
// EXACTLY ZERO -- and if tau_e2e is 0 for every resident, THE ENTIRE DECODE-SIDE
// EXTERNALITY IS IDENTICALLY ZERO and the arm degenerates to its own-good term
// while still running and still reporting goodput. A positive E2E target is a
// correctness requirement, not a tuning choice (the simulation's own tests set
// TauE2EUs = 500000, sim/edpp_slo_externality_test.go:17).
func gDecodeComposite(cr decodeResident, cUs float64) float64 {
	if !cr.ttftSet {
		return 0
	}
	ttft := float64(cr.firstTokenUs - cr.arrivalUs)
	return sloCompositeValue(cr.slo, ttft, cUs-float64(cr.arrivalUs))
}

// varDecodeContribution is one decoding resident's charge: the good it HAD at its
// baseline completion cb minus the good it HAS at its placed completion cp.
// sim/edpp_var.go:319-338 (composite branch).
//
// Positive means the placement destroyed value. When cp == cb -- the resident
// finished inside the admission window -- this is exactly zero.
func varDecodeContribution(cr decodeResident, cb, cp float64) float64 {
	return gDecodeComposite(cr, cb) - gDecodeComposite(cr, cp)
}

// varDecodeLocalAfter sums the decode-side externality of a LOCAL placement.
// sim/edpp_var.go:288-301.
//
// Residents with rem < 0 are SKIPPED, not treated as zero-remaining: a censored
// resident carries no information and must not be charged.
func varDecodeLocalAfter(now float64, crs []decodeResident, rt reTiming, nChunks, admissionSteps float64) float64 {
	var sum float64
	for _, cr := range crs {
		if cr.rem < 0 {
			continue
		}
		cb := rt.cBase(now, cr.rem)
		cp := rt.cLocalAfter(now, cr.rem, admissionSteps, nChunks)
		sum += varDecodeContribution(cr, cb, cp)
	}
	return sum
}

// varDecodeDisagg is the disaggregated mirror. sim/edpp_var.go:303-317.
func varDecodeDisagg(now float64, crs []decodeResident, rt reTiming, arrivalSteps float64) float64 {
	var sum float64
	for _, cr := range crs {
		if cr.rem < 0 {
			continue
		}
		cb := rt.cBase(now, cr.rem)
		cp := rt.cDisagg(now, cr.rem, arrivalSteps)
		sum += varDecodeContribution(cr, cb, cp)
	}
	return sum
}

// prefillResident is one pre-first-token occupant's value inputs.
// sim/edpp_var.go assembled at :643-673.
type prefillResident struct {
	remPrefillTokens int64 // remaining PROMPT tokens; < 0 means SKIP
	remDecodeSteps   int64 // decode horizon once it reaches its first token
	arrivalUs        int64
	slo              slo
}

// varPrefillTTFTContribution scores one occupant's FIRST-TOKEN value at risk.
// sim/edpp_var.go:411-441 (composite branch, which equals the util branch).
//
// The deadline is arrival + tau_ttft and the bandwidth is tau_ttft. Shared by the
// remote prefill-pool term and the collocated decode-endpoint term so both score
// first-token risk with identical arithmetic.
func varPrefillTTFTContribution(k prefillResident, cb, cp float64) float64 {
	deadline := float64(k.arrivalUs) + k.slo.tauTTFTUs
	scale := k.slo.tauTTFTUs
	if scale <= 0 {
		scale = 1
	}
	return sigmoid((deadline-cb)/scale) - sigmoid((deadline-cp)/scale)
}

// gCollocComposite is a collocated occupant's composite good when it reaches its
// first token at ftUs. sim/edpp_var.go:560-571.
//
// IT DELIBERATELY IGNORES THE END-TO-END COMPLETION, and the second parameter is
// unused for that reason. A resident still prefilling has no assigned decoder
// state in the routing view, so its declared phase value is TTFT-only; adding a
// synthetic decode horizon here would make the one-step potential depend on state
// the router does not observe. Do not "improve" this to read eUs -- it would make
// the collocated term inconsistent with the decode term's information set.
func gCollocComposite(k prefillResident, ftUs, _ float64) float64 {
	if k.slo.tauTTFTUs <= 0 {
		return 1
	}
	return sigmoid((k.slo.tauTTFTUs - (ftUs - float64(k.arrivalUs))) / k.slo.tauTTFTUs)
}

// varCollocContribution is one collocated occupant's charge under the composite
// kernel. sim/edpp_var.go:505-521 (composite branch).
func varCollocContribution(k prefillResident, ftB, ftP, eB, eP float64) float64 {
	return gCollocComposite(k, ftB, eB) - gCollocComposite(k, ftP, eP)
}

// varCollocPrefillLocalAfter sums the value at risk imposed on the DECODE
// endpoint's collocated prefill occupants by a LOCAL placement.
// sim/edpp_var.go:450-475.
//
// Such an occupant was placed here by a prior decision and has not produced a
// first token, so the decode-side terms miss it entirely. A local placement harms
// it TWICE: it needs ceil(remPrefillTokens/chunk) more batch iterations to reach
// its first token and R's co-scheduled prefill slows those, and then R joins the
// decode batch (B -> B+1) and slows its decode steps too. Both projections use the
// same cBase -> cLocalAfter model.
func varCollocPrefillLocalAfter(now float64, ks []prefillResident, rt reTiming, chunk, nChunks, admissionSteps float64) float64 {
	if chunk < 1 {
		chunk = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remPf := int64(math.Ceil(float64(k.remPrefillTokens) / chunk))
		ftB := rt.cBase(now, remPf)
		ftP := rt.cLocalAfter(now, remPf, admissionSteps, nChunks)
		eB, eP := ftB, ftP
		if k.remDecodeSteps > 0 {
			total := remPf + k.remDecodeSteps
			eB = rt.cBase(now, total)
			eP = rt.cLocalAfter(now, total, admissionSteps, nChunks)
		}
		sum += varCollocContribution(k, ftB, ftP, eB, eP)
	}
	return sum
}

// varCollocPrefillDisagg is the disaggregated mirror. sim/edpp_var.go:477-503.
//
// An occupant that reaches its first token inside the arrival window
// (remPf <= arrivalSteps) sees cDisagg == cBase and contributes exactly zero:
// remote placement does not delay it.
func varCollocPrefillDisagg(now float64, ks []prefillResident, rt reTiming, chunk, arrivalSteps float64) float64 {
	if chunk < 1 {
		chunk = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remPf := int64(math.Ceil(float64(k.remPrefillTokens) / chunk))
		ftB := rt.cBase(now, remPf)
		ftP := rt.cDisagg(now, remPf, arrivalSteps)
		eB, eP := ftB, ftP
		if k.remDecodeSteps > 0 {
			total := remPf + k.remDecodeSteps
			eB = rt.cBase(now, total)
			eP = rt.cDisagg(now, total, arrivalSteps)
		}
		sum += varCollocContribution(k, ftB, ftP, eB, eP)
	}
	return sum
}

// varPrefillDisaggAfter is the LEGACY remote prefill-pool externality.
// sim/edpp_var.go:347-366.
//
// NOT the branch this arm takes -- it forces the exact form below. Stated because
// it is what the published function computes when the exact flag is off, and
// because the difference between the two is the point of the exact correction:
// this form charges an occupant R's ENTIRE prefill duration rPrefillUs, even an
// occupant with one iteration left.
func varPrefillDisaggAfter(now float64, ks []prefillResident, tIterP, chunkP, admissionSteps, rPrefillUs float64) float64 {
	if chunkP < 1 {
		chunkP = 1
	}
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remIters := math.Ceil(float64(k.remPrefillTokens) / chunkP)
		cb := now + remIters*tIterP
		cp := cb
		if remIters > admissionSteps {
			cp += rPrefillUs
		}
		sum += varPrefillTTFTContribution(k, cb, cp)
	}
	return sum
}

// varPrefillDisaggExactAfter is the marginal-overlap correction, and THE BRANCH
// THIS ARM TAKES. sim/edpp_var.go:372-404.
//
// After R's admission, an occupant is delayed only by R's prefill chunks that
// execute BEFORE that occupant reaches its first token. Baseline prefill
// iterations are not charged again, and an occupant with one remaining iteration
// is not charged R's entire multi-chunk prompt.
func varPrefillDisaggExactAfter(
	now float64,
	ks []prefillResident,
	tIterP, chunkP, admissionSteps float64,
	rAp, rAr int,
	coeffs Coeffs,
) float64 {
	if chunkP < 1 {
		chunkP = 1
	}
	rChunks := math.Ceil(float64(maxInt(rAp, 0)) / chunkP)
	var sum float64
	for _, k := range ks {
		if k.remPrefillTokens < 0 {
			continue
		}
		remIters := math.Ceil(float64(k.remPrefillTokens) / chunkP)
		remainingAfterAdmission := math.Max(remIters-admissionSteps, 0)
		overlap := math.Min(rChunks, remainingAfterAdmission)
		cb := now + remIters*tIterP
		cp := cb + prefillMarginalWork(
			coeffs.CPf, coeffs.CAttn,
			float64(rAp), float64(rAr),
			chunkP, overlap,
		)
		sum += varPrefillTTFTContribution(k, cb, cp)
	}
	return sum
}

// pathBreakdown separates the three resident populations a placement can harm.
// sim/edpp_var.go:694-705. Kept explicit so the aggregate is auditable.
type pathBreakdown struct {
	decode        float64
	collocPrefill float64
	prefillPool   float64
}

func (v pathBreakdown) total() float64 {
	return v.decode + v.collocPrefill + v.prefillPool
}

// goodSelf scores the ARRIVING request's OWN smoothed goodput under a candidate.
// sim/edpp_var.go:958-990 (composite branch).
//
// R arrives at the decision instant, so its TTFT measured from arrival IS
// tHatFromNow. It then decodes nOut steps at tIterAfter -- the B+1 re-timed
// per-iteration time it experiences once it joins the batch -- so its end-to-end
// latency from arrival is tHatFromNow + nOut*tIterAfter.
//
// The rule charges externality - ownGood: the value this placement DESTROYS among
// residents minus the value it EARNS for R.
func goodSelf(s slo, tHatFromNow, tIterAfter, nOut float64) float64 {
	ttft := tHatFromNow
	e2e := tHatFromNow + nOut*tIterAfter
	return sloCompositeValue(s, ttft, e2e)
}

// ---------------------------------------------------------------------------
// Policy state and input assembly
// ---------------------------------------------------------------------------

// Policy holds the arm's long-lived state.
type Policy struct {
	cfg Config

	// nHatOut is the per-class running mean of realized output lengths.
	// sim/edpp.go:412-425.
	//
	// SEEDED AT 1, NOT 0 (mean() returns 1 when n == 0): a zero seed makes the
	// decode demand vanish and collapses every remaining-steps floor.
	//
	// Updated ONLY on requests that actually COMPLETED. A request reaching a
	// terminal state without completing carries no realized output length, and
	// folding its truncated count in would drag the estimate down.
	nHatOut map[string]*runningMean

	// sloCapacity holds the per-endpoint virtual workload queues. DEAD STATE in
	// this arm, which disables the capacity term; maintained only if
	// SLOExternalityNoCapacity is flipped off. See capacity account below.
	sloCapacity         map[string]*capacityState
	sloCapacityClock    int64
	sloCapacityClockSet bool
}

// runningMean is a per-class running mean. sim/edpp.go:412-425.
type runningMean struct {
	n   int64
	sum float64
}

func (r *runningMean) update(v float64) { r.n++; r.sum += v }

func (r *runningMean) mean() float64 {
	if r.n == 0 {
		return 1 // no completions yet: conservative 1-token seed, NOT 0
	}
	return r.sum / float64(r.n)
}

func (p *Policy) nHatFor(class string) float64 {
	m := p.nHatOut[class]
	if m == nil {
		return 1
	}
	return m.mean()
}

// sloFor resolves a class's SLO thresholds. sim/edpp.go:682-708 via
// sim/edpp_var.go:597-608. A per-class override wins over the default, and the
// empty class uses the default.
func (p *Policy) sloFor(class string) slo {
	tauTTFT, tauITL, tauE2E := p.cfg.TauTTFTUs, p.cfg.TauITLUs, p.cfg.TauE2EUs
	if v, ok := p.cfg.TauTTFTByClassUs[class]; ok {
		tauTTFT = v
	}
	if v, ok := p.cfg.TauITLByClassUs[class]; ok {
		tauITL = v
	}
	if v, ok := p.cfg.TauE2EByClassUs[class]; ok {
		tauE2E = v
	}
	return slo{tauTTFTUs: float64(tauTTFT), tauITLUs: float64(tauITL), tauE2EUs: float64(tauE2E)}
}

// coeffsFor is the SINGLE selection point for per-endpoint heterogeneous
// coefficients. sim/edpp.go:709-719.
//
// On the target the key is the POD LABEL value (config.md §2, §5 signal 8), and an
// unmapped label must have already caused the endpoint to be rejected by
// endpointGPUType -- reaching a silent fallback here would price the endpoint
// under the wrong physics on every decision.
func (p *Policy) coeffsFor(gpuType string) Coeffs {
	if gpuType != "" {
		if c, ok := p.cfg.CoeffsByGPU[gpuType]; ok {
			return c
		}
	}
	return p.cfg.Coeffs
}

// varDecodeInputs converts an endpoint's decode resident slice into value inputs.
// sim/edpp_var.go:615-641.
//
// THIS ARM ALWAYS TAKES THE DEPLOYABLE (CENSORED) BRANCH -- varDeployable is
// forced true at sim/edpp.go:1710 and :1746. The oracle branch reads
// TrueRemaining, which is hidden output length, and is never a deployable policy.
//
// The censoring: a resident that has produced StepsDone tokens has total output
// >= StepsDone, so the class mean is FLOORED BY StepsDone before subtracting, and
// the remainder floored at 1 because it is still decoding.
func (p *Policy) varDecodeInputs(running []RunningReqState) []decodeResident {
	if len(running) == 0 {
		return nil
	}
	out := make([]decodeResident, 0, len(running))
	for _, r := range running {
		nHat := p.nHatFor(r.SLOClass)
		rem := int64(math.Max(math.Max(nHat, float64(r.StepsDone))-float64(r.StepsDone), 1))
		out = append(out, decodeResident{
			rem:          rem,
			arrivalUs:    r.ArrivalUs,
			firstTokenUs: r.FirstTokenUs,
			ttftSet:      r.TTFTSet,
			slo:          p.sloFor(r.SLOClass),
		})
	}
	return out
}

// varPrefillInputs converts an endpoint's prefill occupant slice into value
// inputs. sim/edpp_var.go:643-673.
//
// remPrefillTokens is remaining PROMPT tokens -- known input, never oracle-gated.
//
// remDecodeSteps is the occupant's decode horizon once it reaches its first token.
// It has produced no output yet, so the deployable estimate is the FULL censored
// class mean with no StepsDone to subtract, floored at 1.
func (p *Policy) varPrefillInputs(running []RunningReqState) []prefillResident {
	if len(running) == 0 {
		return nil
	}
	out := make([]prefillResident, 0, len(running))
	for _, r := range running {
		remDec := int64(math.Max(p.nHatFor(r.SLOClass), 1))
		out = append(out, prefillResident{
			remPrefillTokens: r.TrueRemaining,
			remDecodeSteps:   remDec,
			arrivalUs:        r.ArrivalUs,
			slo:              p.sloFor(r.SLOClass),
		})
	}
	return out
}

// ---------------------------------------------------------------------------
// Per-candidate helpers
// ---------------------------------------------------------------------------

// apForInstance is a_p: the UNCACHED SUFFIX of the prompt on this endpoint.
// sim/edpp.go:1055-1064.
//
//	a_p = a_r - cachedBlocks*BlockSize
//
// With no cache query available the full prompt is uncached. Note a_p may be
// NEGATIVE if the cache reports more blocks than the prompt covers; callers clamp
// with maxInt(ap, 0) at every use, and the raw value is passed to chunkTerms which
// returns (0,0) for ap <= 0.
// TWO BLOCK SIZES ARE REQUIRED HERE, AND THIS SITE NEEDS THE PREFIX ONE. Upstream
// can use a single d.cfg.BlockSize (sim/edpp.go:1059) because the simulator's
// prefix-cache index and its KV blocks genuinely share one block size. On the target
// they do not: the engine reports 16 while the prefix producer clamps to a >= 64-token
// floor (approximateprefix/plugin.go:50). Multiplying the producer's block count by
// the ENGINE's block size understates the cached span by 4x -- see cachedBlockCount.
//
// Config.BlockSize stays the ENGINE block size, because reqKVNeed and the rollout need
// that one. This call site must use the PRODUCER's, read from the producer itself
// rather than configured, so the two cannot drift.
func (p *Policy) apForInstance(req Request, instID string) int {
	ap := p.inputLen(req)
	if blocks, ok := cachedBlockCount(p.endpointByID(instID), req); ok {
		ap = p.inputLen(req) - blocks*p.prefixBlockSizeTokens(p.endpointByID(instID))
	}
	return ap
}

// prefixBlockSizeTokens returns the PREFIX PRODUCER's block size in tokens -- not the
// engine's Config.BlockSize. Target-API adapter: read it from the producer that
// supplied the block count, so a producer reconfiguration cannot silently rescale a_p.
func (p *Policy) prefixBlockSizeTokens(ep Endpoint) int

// chunkTerms returns (nChunks, deltaPfChunk) for ap uncached prefill tokens under
// the batched-token budget. sim/edpp.go:1220-1228.
//
// deltaPfChunk = theta.CPf*chunk is charged on the pool the prefill runs on:
// decode theta when local, prefill theta when disaggregated. ap <= 0 (fully cached
// or empty) yields (0, 0) -- no prefill work and no per-chunk ITL inflation.
func (p *Policy) chunkTerms(theta Coeffs, ap int) (nChunks, deltaPfChunk float64) {
	if ap <= 0 {
		return 0, 0
	}
	chunk := ap
	if p.cfg.ChunkTokens > 0 && p.cfg.ChunkTokens < chunk {
		chunk = p.cfg.ChunkTokens
	}
	return math.Ceil(float64(ap) / float64(chunk)), theta.CPf * float64(chunk)
}

// projectedLocalTTFT ends at the LOCAL request's client-visible first token.
// sim/edpp.go:1245-1250.
//
// Local execution samples the first token when prefill completes, so there is NO
// separate decode iteration before post-processing -- unlike the disaggregated
// form below. That asymmetry is real, not an oversight.
func (p *Policy) projectedLocalTTFT(tAdmD, nChunks, tIterD, wpLoc float64) float64 {
	return tAdmD + nChunks*tIterD + wpLoc + p.cfg.OutputTokenProcessingUs
}

// projectedDisaggTTFT ends at the decode endpoint's first client-visible token.
// sim/edpp.go:1252-1255. decodeJoinUs covers prefill admission and work, transfer,
// and decode admission; the first B+1 decode iteration and post-processing follow.
func (p *Policy) projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode float64) float64 {
	return decodeJoinUs + tIterFirstDecode + p.cfg.OutputTokenProcessingUs
}

// reqKVNeed is ceil(a_r / BlockSize). sim/edpp.go:1257-1262.
func (p *Policy) reqKVNeed(req Request) int64 {
	if p.cfg.BlockSize <= 0 {
		return 0
	}
	return int64((p.inputLen(req) + p.cfg.BlockSize - 1) / p.cfg.BlockSize)
}

// cXferUsFor is the KV-transfer cost for routing this request remotely.
// sim/edpp.go:1126-1140.
//
// DEGRADATION D5: XferBaseUs and XferBandwidthGBps are UNMEASURED on the target
// (config.md §7). This is the ONLY size-dependent price of going remote, entering
// the objective at exactly one place (remoteLeadUs), so a wrong value mis-prices
// SYSTEMATICALLY rather than noisily.
//
// KVBytesPerTokenPerGPU = 81920 for this model at TP=4 (config.md §7). Leaving it
// 0 makes transferBytes 0, so every request is charged a flat XferBaseUs -- a
// 4k-token prompt should be charged ~13500 us, roughly 270x under-priced, with the
// error growing linearly toward the deep-research workload's ~45k-token prompts.
func (p *Policy) cXferUsFor(req Request) float64 {
	if !p.cfg.CXferSizeAware {
		return float64(p.cfg.CXferUs) // the FLAT field, not XferBaseUs
	}
	bwBytesPerUs := p.cfg.XferBandwidthGBps * 1000.0 // GB/s -> bytes/us
	if bwBytesPerUs <= 0 || p.cfg.BlockSize <= 0 {
		return p.cfg.XferBaseUs
	}
	transferBytes := float64(p.reqKVNeed(req)) * float64(p.cfg.BlockSize) * p.cfg.KVBytesPerTokenPerGPU
	return p.cfg.XferBaseUs + transferBytes/bwBytesPerUs
}

// decodeRemStepsEst is the mean censored remaining-steps estimate over an
// endpoint's decode residents. sim/edpp.go:1070-1092.
//
// Deliberately NOT a mean that can go negative: the class estimate is floored by
// the LONGEST in-flight elapsed count before per-request subtraction, and each
// per-request remainder is floored at 1. An endpoint with no residents returns 1.
func (p *Policy) decodeRemStepsEst(snap Snapshot, class string) float64 {
	n := len(snap.RunningDecode)
	if n == 0 {
		return 1.0
	}
	nHatOut := p.nHatFor(class)
	var maxSteps int64
	for _, r := range snap.RunningDecode {
		if r.StepsDone > maxSteps {
			maxSteps = r.StepsDone
		}
	}
	nHatEff := math.Max(nHatOut, float64(maxSteps))
	var sum float64
	for _, r := range snap.RunningDecode {
		sum += math.Max(nHatEff-float64(r.StepsDone), 1)
	}
	return sum / float64(n)
}

// prefillRemStepsEst is the symmetric estimate for a prefill endpoint, over its
// occupants' remaining prefill-chunk counts floored at 1. sim/edpp.go:1094-1114.
func (p *Policy) prefillRemStepsEst(snap Snapshot) float64 {
	n := len(snap.RunningPrefill)
	if n == 0 {
		return 1.0
	}
	var sum float64
	for _, r := range snap.RunningPrefill {
		rem := r.TrueRemaining
		if rem < 1 {
			rem = 1
		}
		sum += float64(rem)
	}
	return sum / float64(n)
}

// decodeAdmissionCtx assembles the admission context for a decode candidate.
// sim/edpp.go:1601-1613.
//
// QWork is left ZERO on the target and that is DECISION-NEUTRAL for this arm: it
// is read only by waitingEstimateTAdm, and this arm selects rollforward. Stated
// rather than removed so the omission reads as a decision, not an oversight. A
// port switching TAdmEstimator to `waiting` must first build the per-endpoint work
// account.
func (p *Policy) decodeAdmissionCtx(ec *evalCtx, ds Snapshot) AdmissionContext {
	thetaD := p.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	return AdmissionContext{
		QWork:             0, // see note above
		Mu:                thetaD.muDecode(bDec, kv, sPfD),
		BatchSize:         ds.BatchSize,
		MaxBatchSize:      int(ds.MaxBatchSize),
		FreeKVBlocks:      ds.FreeKVBlocks,
		ReqKVNeed:         ec.reqKVNeed,
		TIter:             thetaD.tIterDecode(bDec, kv, sPfD),
		QueueDepth:        ds.QueueDepth,
		AdmissionRate:     ds.AdmissionRate, // read only by `little`; unset on the target
		RemainingStepsEst: p.decodeRemStepsEst(ds, ec.class),
		// The oracle remaining must be CENSORED to -1 before it reaches an
		// estimator. On the target it is already -1, so this is a no-op that
		// documents the invariant.
		Running: censorOracleRemaining(ds.RunningDecode),
	}
}

// prefillAdmissionCtx is the prefill-pool counterpart. sim/edpp.go:1616-1628.
//
// Note it uses muPrefill and tIterPrefill -- a dedicated prefill endpoint runs no
// decode work -- and passes RunningPrefill, whose TrueRemaining is remaining
// PROMPT tokens and therefore needs no censoring.
func (p *Policy) prefillAdmissionCtx(ec *evalCtx, ps Snapshot) AdmissionContext {
	thetaP := p.coeffsFor(ps.GPUType)
	sPfP := ps.ResidentPrefillTokens
	return AdmissionContext{
		QWork:             0, // see decodeAdmissionCtx
		Mu:                thetaP.muPrefill(sPfP),
		BatchSize:         ps.BatchSize,
		MaxBatchSize:      int(ps.MaxBatchSize),
		FreeKVBlocks:      ps.FreeKVBlocks,
		ReqKVNeed:         ec.reqKVNeed,
		TIter:             thetaP.tIterPrefill(sPfP),
		QueueDepth:        ps.QueueDepth,
		AdmissionRate:     ps.AdmissionRate,
		RemainingStepsEst: p.prefillRemStepsEst(ps),
		Running:           ps.RunningPrefill,
	}
}

// censorOracleRemaining forces TrueRemaining to -1 so no estimator can read a
// hidden output length. sim/edpp.go:2234-2248.
func censorOracleRemaining(running []RunningReqState) []RunningReqState {
	if len(running) == 0 {
		return nil
	}
	out := make([]RunningReqState, len(running))
	copy(out, running)
	for i := range out {
		out[i].TrueRemaining = -1
	}
	return out
}

// estimateTAdm dispatches to the configured estimator. This arm selects
// rollforward (config.md §9.1); the others are stated above for the D1 reason.
func (p *Policy) estimateTAdm(ctx AdmissionContext) float64 {
	return rollforwardEstimateTAdm(ctx)
}

// ---------------------------------------------------------------------------
// The capacity account
//
// DISABLED BY THIS ARM (SLOExternalityNoCapacity, config.md §9.2), so on the
// target sloCapacity is dead state. Stated in full because the algebra is part of
// the published objective and flipping the switch back must not require
// re-deriving it. A port that ENABLES the capacity term must additionally supply a
// monotonic clock and per-endpoint drain timestamps.
//
// Ported from sim/edpp.go:1148-1206 and :2381-2392.
// ---------------------------------------------------------------------------

// capacityState is one endpoint's virtual workload queue. In occupancy mode q is
// microseconds of physical occupancy, mu is one, and scale converts both factors
// to seconds. sim/edpp.go:405-410.
type capacityState struct {
	q, mu, scale float64
	gpuType      string
}

// refreshCapacity drains every queue by the elapsed wall time and refreshes each
// endpoint's nominal drain rate and scale. sim/edpp.go:1148-1189.
func (p *Policy) refreshCapacity(now int64, decodeSnaps, prefillSnaps []Snapshot) {
	if !p.sloCapacityClockSet {
		p.sloCapacityClock = now
		p.sloCapacityClockSet = true
	} else if now > p.sloCapacityClock {
		elapsed := float64(now - p.sloCapacityClock)
		for _, state := range p.sloCapacity {
			drain := state.mu * elapsed
			if p.cfg.SLOExternalityOccupancyCapacity {
				drain = elapsed
			}
			state.q = math.Max(state.q-drain, 0)
		}
		p.sloCapacityClock = now
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
		if p.cfg.SLOExternalityOccupancyCapacity {
			state.mu = 1
			state.scale = 1_000_000 // express Q*DeltaT in physical seconds squared
		} else {
			state.mu = clampMu(mu)
			state.scale = state.mu * float64(p.cfg.TauRefUs)
		}
		state.gpuType = snap.GPUType
	}
	// Decode endpoints use the NOMINAL decode drain rate at the SLO-critical
	// batch; prefill endpoints the nominal prefill rate at the nominal chunk.
	for _, snap := range decodeSnaps {
		theta := p.coeffsFor(snap.GPUType)
		set(snap, theta.muDNom(float64(p.cfg.TauITLUs)))
	}
	for _, snap := range prefillSnaps {
		theta := p.coeffsFor(snap.GPUType)
		set(snap, theta.muPNom(p.cfg.NomPrefillTokens))
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
// iteration baseline per output token; collocation SHARES the prefill baselines
// with the existing decode batch and therefore adds only Wp.
func (p *Policy) decodeOccupancy(theta Coeffs, ar int, nOut float64) float64 {
	return nOut*theta.AlphaD/float64(p.cfg.SLOCapacityReferenceBatch) + theta.Wd(ar, nOut)
}

func (p *Policy) prefillOccupancy(theta Coeffs, ap, ar int) float64 {
	nChunks, _ := p.chunkTerms(theta, ap)
	return nChunks*theta.AlphaP + theta.Wp(maxInt(ap, 0), ar)
}

func (p *Policy) localOccupancy(theta Coeffs, ap, ar int, nOut float64) float64 {
	return p.decodeOccupancy(theta, ar, nOut) + theta.Wp(maxInt(ap, 0), ar)
}

// bookCapacityWork is the COMMIT half of the capacity account, applied after the
// decision at the winning instances. sim/edpp.go:2394-2426, reached from the route
// callback at sim/edpp.go:2466-2470.
//
// DEAD ON THIS ARM alongside the rest of the capacity account, and stated for the
// same reason: a HALF-specified subsystem is worse than an absent one, because the
// read side above would look complete.
//
// THE INVARIANT THAT MATTERS: the queue is fed by the SAME demand expression the
// candidate score used, evaluated at the COMMITTED instances with their own theta
// and their own a_p. Candidate/commit agreement is what makes the drift argument
// work -- a port that SCORES with one demand and BOOKS another drifts without
// erroring, and the drift is invisible in any single decision.
func (p *Policy) bookCapacityWork(req Request, inputLen int, toPrefill bool, decodeInst, prefillInst string) {
	if decodeInst == "" {
		return
	}
	thetaD := p.coeffsFor(p.gpuTypeOfInstance(decodeInst))
	nOut := p.nHatFor(requestSLOClass(req))
	wd := thetaD.Wd(inputLen, nOut)

	if toPrefill {
		thetaP := p.coeffsFor(p.gpuTypeOfInstance(prefillInst))
		apP := p.apForInstance(req, prefillInst)
		wpP := thetaP.Wp(maxInt(apP, 0), inputLen)
		decodeDemand, prefillDemand := wd, wpP
		if p.cfg.SLOExternalityOccupancyCapacity {
			decodeDemand = p.decodeOccupancy(thetaD, inputLen, nOut)
			prefillDemand = p.prefillOccupancy(thetaP, apP, inputLen)
		}
		if state := p.sloCapacity[prefillInst]; state != nil {
			state.q += prefillDemand
		}
		if state := p.sloCapacity[decodeInst]; state != nil {
			state.q += decodeDemand
		}
		return
	}

	// Local: prefill and decode both land on the decode instance, so the demand is
	// that instance's own a_p under its own theta.
	apD := p.apForInstance(req, decodeInst)
	demand := thetaD.Wp(maxInt(apD, 0), inputLen) + wd
	if p.cfg.SLOExternalityOccupancyCapacity {
		demand = p.localOccupancy(thetaD, apD, inputLen, nOut)
	}
	if state := p.sloCapacity[decodeInst]; state != nil {
		state.q += demand
	}
}

// gpuTypeOfInstance resolves a committed instance's GPU type for theta selection.
// Target-API adapter; upstream's counterpart is sloCoeffsForInstance
// (sim/edpp.go:2370-2375), which MUST agree with what the candidate score used, or
// the commit books work under different physics than it scored.
func (p *Policy) gpuTypeOfInstance(id string) string

// THE OTHER BOOKKEEPING PATH, deliberately not maintained: upstream's route callback
// also calls bookSLOAdmissionWork (sim/edpp.go:2432), which feeds the SEPARATE
// per-instance backlog behind the admission estimator's QWork -- and then RETURNS
// EARLY, skipping the historical bookkeeping entirely. This port declares QWork as 0
// and decision-neutral (see decodeAdmissionCtx) because rollforward never reads it.
// A port switching the estimator to `waiting` must build BOTH that account and this
// commit path.

// ---------------------------------------------------------------------------
// The candidate score
// ---------------------------------------------------------------------------

// evalCtx holds the per-decision, candidate-invariant terms, so every candidate
// evaluation uses identical operands. sim/edpp.go:1643-1655.
type evalCtx struct {
	req       Request
	class     string
	inputLen  int
	reqKVNeed int64
	nHatOut   float64
	nowUs     float64
}

// candidateScore is one candidate's score breakdown. sim/edpp.go:1676-1682.
type candidateScore struct {
	externalityBreakdown                      pathBreakdown
	externality, ownGood, netGoodCost         float64
	capacityQueueDecode, capacityQueuePrefill float64
	capacityDemandDecode, capacityDemandPrefill float64
	capacityDecode, capacityPrefill           float64
	capacityTotal, total                      float64
}

// scoreCandidate implements the policy contract exactly:
//
//	total = V*(externality - ownGood) + capacity
//
// sim/edpp.go:1688-1779. ps == nil means LOCAL (prefill co-resident on the decode
// endpoint); otherwise decode on ds and prefill on *ps.
//
// It carries NO historical TTFT/ITL deficit term, no standalone transfer residue,
// and no per-decision normalization. Those belong to other rules in the published
// decider and are deliberately absent here.
func (p *Policy) scoreCandidate(ec *evalCtx, ds Snapshot, ps *Snapshot) candidateScore {
	thetaD := p.coeffsFor(ds.GPUType)
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	tIterD := thetaD.tIterDecode(bDec, kv, sPfD)
	// The B+1 re-timed FIRST decode iteration: batch grows by one and KV grows by
	// the arriving request's full input length.
	tIterFirstDecode := thetaD.tIterDecode(bDec+1, kv+int64(ec.inputLen), sPfD)
	wd := thetaD.Wd(ec.inputLen, ec.nHatOut)
	tAdmD := p.estimateTAdm(p.decodeAdmissionCtx(ec, ds))

	score := candidateScore{}

	if ps == nil {
		// ---- LOCAL candidate ----
		apLoc := p.apForInstance(ec.req, ds.ID)
		nChunksLoc, _ := p.chunkTerms(thetaD, apLoc)
		wpLoc := thetaD.Wp(maxInt(apLoc, 0), ec.inputLen)
		tHatLocal := p.projectedLocalTTFT(tAdmD, nChunksLoc, tIterD, wpLoc)
		// D1: on the target this never succeeds, so the closed-form tAdmD and
		// tHatLocal above stand. Increment the substitution counter here.
		if rolloutAdm, rolloutTTFT, ok := p.rolloutLocalTTFT(ec, ds, thetaD); ok {
			tAdmD, tHatLocal = rolloutAdm, rolloutTTFT
		}

		score.externalityBreakdown = p.externalityLocal(ec, ds, thetaD, apLoc, tAdmD)

		if !p.cfg.SLOExternalityNoOwnGood {
			score.ownGood = p.selfGood(ec, thetaD, ds, tHatLocal)
		}
		if !p.cfg.SLOExternalityNoCapacity {
			demand := wpLoc + wd
			if p.cfg.SLOExternalityOccupancyCapacity {
				demand = p.localOccupancy(thetaD, apLoc, ec.inputLen, ec.nHatOut)
			}
			score.capacityQueueDecode = p.capacityQueue(ds.ID)
			score.capacityDemandDecode = demand
			score.capacityDecode = p.capacityTerm(ds.ID, demand)
		}
	} else {
		// ---- DISAGGREGATED candidate ----
		if rolloutAdm, ok := p.rolloutDecodeAdmission(ec, ds, thetaD); ok {
			tAdmD = rolloutAdm // D1: unreachable on the target
		}
		thetaP := p.coeffsFor(ps.GPUType)
		apP := p.apForInstance(ec.req, ps.ID)
		nChunksP, _ := p.chunkTerms(thetaP, apP)
		wpP := thetaP.Wp(maxInt(apP, 0), ec.inputLen)
		tIterP := thetaP.tIterPrefill(ps.ResidentPrefillTokens)
		tAdmP := p.estimateTAdm(p.prefillAdmissionCtx(ec, *ps))
		prefillCompletionUs := tAdmP + nChunksP*tIterP + wpP
		if rolloutAdm, rolloutCompletion, ok := p.rolloutPrefillCompletion(ec, *ps, thetaP); ok {
			tAdmP, prefillCompletionUs = rolloutAdm, rolloutCompletion // D1
		}

		// THE ONLY PLACE c_xfer ENTERS THE OBJECTIVE -- degradation D5.
		remoteLeadUs := prefillCompletionUs + p.cXferUsFor(ec.req)

		// OVERLAP, NOT SERIALIZATION. The decode admission estimate is an absolute
		// wait measured at the routing instant. Remote prefill and transfer consume
		// part or all of that interval while the decode queue continues to drain,
		// so they must NOT be serialized with the full estimate a second time.
		//
		// UNCONDITIONAL. There is a --edpp-ttft-overlap-aware flag upstream, but it
		// gates only the REDUCED path (sim/edpp.go:909) and is never consulted on the
		// joint path -- its only occurrences are the config field (sim/edpp.go:110)
		// and that gate. Do NOT make this max() conditional on it: that reintroduces
		// the serialized form and over-prices every disaggregated candidate by up to
		// a full decode admission delay.
		decodeJoinUs := math.Max(remoteLeadUs, tAdmD)
		tHatDisagg := p.projectedDisaggTTFT(decodeJoinUs, tIterFirstDecode)

		score.externalityBreakdown = p.externalityDisagg(ec, ds, ps, thetaD, thetaP, apP, decodeJoinUs, tAdmP)

		if !p.cfg.SLOExternalityNoOwnGood {
			score.ownGood = p.selfGood(ec, thetaD, ds, tHatDisagg)
		}
		if !p.cfg.SLOExternalityNoCapacity {
			decodeDemand, prefillDemand := wd, wpP
			if p.cfg.SLOExternalityOccupancyCapacity {
				decodeDemand = p.decodeOccupancy(thetaD, ec.inputLen, ec.nHatOut)
				prefillDemand = p.prefillOccupancy(thetaP, apP, ec.inputLen)
			}
			score.capacityQueueDecode = p.capacityQueue(ds.ID)
			score.capacityQueuePrefill = p.capacityQueue(ps.ID)
			score.capacityDemandDecode = decodeDemand
			score.capacityDemandPrefill = prefillDemand
			score.capacityDecode = p.capacityTerm(ds.ID, decodeDemand)
			score.capacityPrefill = p.capacityTerm(ps.ID, prefillDemand)
		}
	}

	if !p.cfg.SLOExternalityNoExternality {
		score.externality = score.externalityBreakdown.total()
	}
	score.netGoodCost = score.externality - score.ownGood
	score.capacityTotal = score.capacityDecode + score.capacityPrefill
	score.total = p.cfg.V*score.netGoodCost + score.capacityTotal
	return score
}

// selfGood is the arriving request's own projected good under a candidate.
// sim/edpp.go:1666-1669.
//
// The prefill chunk does not affect tIterAfter -- only the overlap term -- so
// chunk is passed as 0. Decode always happens on ds, so ds's theta sets tIterAfter
// for BOTH local and disaggregated candidates.
func (p *Policy) selfGood(ec *evalCtx, thetaD Coeffs, ds Snapshot, tHat float64) float64 {
	rt := p.reTimingFor(ec.inputLen, thetaD, ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens, 0)
	return goodSelf(p.sloFor(ec.class), tHat, rt.tIterAfter, ec.nHatOut)
}

// externalityLocal is the LOCAL branch of sim/edpp_var.go:851-880, with this arm's
// forced settings (exact overlap, deployable, colloc-prefill on) already applied.
func (p *Policy) externalityLocal(ec *evalCtx, ds Snapshot, thetaD Coeffs, apLoc int, decodeAdmissionUs float64) pathBreakdown {
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	decode := p.varDecodeInputs(ds.RunningDecode)

	chunkLoc := apLoc
	if p.cfg.ChunkTokens > 0 && p.cfg.ChunkTokens < chunkLoc {
		chunkLoc = p.cfg.ChunkTokens
	}
	rt := p.reTimingFor(ec.inputLen, thetaD, bDec, kv, sPfD, chunkLoc)
	// Forced true by this arm (sim/edpp.go:1709).
	rt.exactPrefillOverlap = true
	rt.cPf = thetaD.CPf
	rt.ap = float64(maxInt(apLoc, 0))
	rt.ar = float64(ec.inputLen)

	nChunksLoc, _ := p.chunkTerms(thetaD, apLoc)
	// THE ADMISSION GATE, in baseline iterations. max(tIter0, 1) guards a
	// degenerate zero iteration time.
	admissionSteps := math.Ceil(decodeAdmissionUs / math.Max(rt.tIter0, 1))

	v := pathBreakdown{
		decode: varDecodeLocalAfter(ec.nowUs, decode, rt, nChunksLoc, admissionSteps),
	}
	if len(ds.RunningPrefill) > 0 {
		colloc := p.varPrefillInputs(ds.RunningPrefill)
		v.collocPrefill = varCollocPrefillLocalAfter(ec.nowUs, colloc, rt, float64(chunkLoc), nChunksLoc, admissionSteps)
	}
	// No prefillPool term: a local placement puts no work on any prefill endpoint.
	return v
}

// externalityDisagg is the DISAGGREGATED branch of sim/edpp_var.go:881-941.
//
// decodeJoinOverride is the join clock computed by the caller as
// max(remoteLeadUs, tAdmD) -- the OVERLAP form. The published function also
// contains a serialized fallback,
//
//	nChunksP*tIterP + Wp + prefillAdmissionUs + cXfer + decodeAdmissionUs
//
// used when no override is supplied (sim/edpp_var.go:899-905). This arm ALWAYS
// supplies the override (sim/edpp.go:1750-1752), so the serialized form is
// unreachable here; it is recorded so a reader does not mistake the override for
// the only model.
func (p *Policy) externalityDisagg(ec *evalCtx, ds Snapshot, ps *Snapshot, thetaD, thetaP Coeffs, apP int, decodeJoinOverride, prefillAdmissionUs float64) pathBreakdown {
	bDec, kv, sPfD := ds.BatchSize, ds.KvTokensInUse, ds.ResidentPrefillTokens
	decode := p.varDecodeInputs(ds.RunningDecode)

	chunkP := apP
	if p.cfg.ChunkTokens > 0 && p.cfg.ChunkTokens < chunkP {
		chunkP = p.cfg.ChunkTokens
	}
	// DECODE re-timing uses ds's theta even on a disaggregated candidate: decode
	// happens on ds in both placements. Only the prefill-pool term uses thetaP.
	rt := p.reTimingFor(ec.inputLen, thetaD, bDec, kv, sPfD, chunkP)
	rt.exactPrefillOverlap = true
	rt.cPf = thetaD.CPf
	rt.ap = float64(maxInt(apP, 0))
	rt.ar = float64(ec.inputLen)

	nChunksP, _ := p.chunkTerms(thetaP, apP)
	sPfP := ps.ResidentPrefillTokens
	tIterP := thetaP.tIterPrefill(sPfP)

	// Decode interference starts when R is ADMITTED, not when its first token
	// completes. This join-time clock is deliberately separate from the
	// client-visible TTFT projection.
	arrivalSteps := math.Ceil(decodeJoinOverride / math.Max(rt.tIter0, 1))

	v := pathBreakdown{
		decode: varDecodeDisagg(ec.nowUs, decode, rt, arrivalSteps),
	}
	// Collocated prefill occupants on ds are undisturbed until R arrives from the
	// pool; only those still prefilling past arrivalSteps have their first token
	// delayed.
	if len(ds.RunningPrefill) > 0 {
		colloc := p.varPrefillInputs(ds.RunningPrefill)
		v.collocPrefill = varCollocPrefillDisagg(ec.nowUs, colloc, rt, float64(chunkP), arrivalSteps)
	}

	prefill := p.varPrefillInputs(ps.RunningPrefill)
	prefillAdmissionSteps := math.Ceil(prefillAdmissionUs / math.Max(tIterP, 1))
	// Forced exact form (sim/edpp.go:1745). The legacy alternative
	// varPrefillDisaggAfter would charge each occupant R's ENTIRE prefill
	// duration, rPrefillUs = nChunksP*tIterP + Wp.
	v.prefillPool = varPrefillDisaggExactAfter(
		ec.nowUs, prefill, tIterP, float64(chunkP), prefillAdmissionSteps,
		maxInt(apP, 0), ec.inputLen, thetaP,
	)
	return v
}

// ---------------------------------------------------------------------------
// The argmin -- the REQUIRED STRUCTURAL SHAPE
// ---------------------------------------------------------------------------

// candidate is one enumerated (decode, placement) pair with its objective.
// sim/edpp.go:1630-1636.
type candidate struct {
	dID, pID string
	local    bool
	J        float64
}

// Decide enumerates every candidate and returns the argmin.
// sim/edpp.go:1334-1400, SLO-externality branch at :1380-1398.
//
// THE SHAPE: D local candidates PLUS D*P disaggregated candidates, scored on ONE
// scale and compared in ONE argmin. This cannot be expressed as a per-endpoint
// scorer on the target, for two independent reasons verified at the pin:
//   - a scorer returns a per-endpoint map and has no way to name a PAIR;
//   - every scorer contribution passes through the profile's score-range
//     enforcement, which CLAMPS to [0,1], while J is signed and unbounded.
//     Clamping does not degrade the ranking, it DESTROYS it: every candidate with
//     J <= 0 collapses to one score.
//
// The policy must therefore attach as a custom picker over a role-blind profile,
// and it must IGNORE inherited scorer contributions for the same clamping reason.
// Confirm both interfaces against the pinned target checkout.
//
// DETERMINISM: candidates are enumerated over endpoints sorted by ID, and the
// argmin uses a strict improvement threshold, so ties resolve to the
// first-enumerated candidate rather than to map iteration order.
//
// THE DECODE CHOICE IS PART OF THE OUTPUT ON BOTH OUTCOMES. Upstream encodes a
// local win as {Disaggregate:false, DecodePodOverride:dID} and a disaggregated win
// as {Disaggregate:true, DecodePodOverride:dID, PrefillPodHint:pID}
// (sim/edpp.go:1470-1475): the decode pod is overridden EVEN WHEN THE RULE DECLINES
// TO DISAGGREGATE. This is not merely a local/remote-prefill switch. A port that
// applies the decode selection only on the disaggregated branch, and otherwise lets
// the stock scorer place decode, has silently discarded half the joint argmin --
// what remains is closer to the `decomposed` control than to this arm.
func (p *Policy) Decide(req Request, decodeSnaps, prefillSnaps []Snapshot, scorerDecodeID string) (candidate, bool) {
	class := requestSLOClass(req)
	inputLen, ok := promptTokens(req)
	if !ok {
		// Tokenization failed -- distinct from a legitimately empty prompt, which
		// must still be scored. See promptTokens.
		return candidate{}, false
	}

	decodeSnaps = sortedByID(decodeSnaps)
	prefillSnaps = sortedByID(prefillSnaps)

	if !p.cfg.SLOExternalityNoCapacity {
		p.refreshCapacity(int64(nowUs()), decodeSnaps, prefillSnaps)
	}

	ec := &evalCtx{
		req:       req,
		class:     class,
		inputLen:  inputLen,
		reqKVNeed: p.reqKVNeed(req),
		nHatOut:   p.nHatFor(class),
		nowUs:     nowUs(),
	}

	// The stock scorer's decode pick is enumerated FIRST, so that restricting the
	// enumeration to it reproduces the decomposed rule exactly.
	orderedD := scorerFirstSnapshots(decodeSnaps, scorerDecodeID)

	// AND THE SAME APPLIES ON THE PREFILL SIDE. Upstream reorders the prefill
	// snapshots too, whenever a prefill scorer is injected (sim/edpp.go:1387-1390):
	//   orderedP := prefillSnaps
	//   if prefillScorer != nil && len(prefillSnaps) > 0 {
	//       orderedP = scorerFirstSnapshots(prefillSnaps, prefillScorer(req, prefillSnaps))
	//   }
	// Iterating plain prefillSnaps instead changes which candidate wins an exact tie.
	// With one prefill endpoint the two orders coincide, which is exactly why this is
	// easy to port wrong and hard to notice: config.md §2 deploys 1P, so a second
	// prefill endpoint would silently change tie resolution.
	orderedP := prefillSnaps
	if id, ok := p.prefillScorerPick(req, prefillSnaps); ok {
		orderedP = scorerFirstSnapshots(prefillSnaps, id)
	}
	if p.cfg.DecomposedSLOExternality && len(orderedD) > 1 {
		// The matched decomposition CONTROL, not the focal arm: decode is fixed to
		// the stock scorer's choice. config.md §10 prices this at +0.0485
		// equal-cell mean goodput against the full joint shape -- and flipping it
		// is the cheapest on-cluster check that the joint shape survived the port.
		orderedD = orderedD[:1]
	}

	var best *candidate
	consider := func(c candidate) {
		if best == nil || c.J < best.J-1e-12 {
			cc := c
			best = &cc
		}
	}

	for _, ds := range orderedD {
		s := p.scoreCandidate(ec, ds, nil)
		consider(candidate{dID: ds.ID, local: true, J: s.total})
		for i := range orderedP {
			ps := orderedP[i]
			s = p.scoreCandidate(ec, ds, &ps)
			consider(candidate{dID: ds.ID, pID: ps.ID, local: false, J: s.total})
		}
	}
	if best == nil {
		return candidate{}, false
	}
	return *best, true
}

// sortedByID returns the snapshots sorted ascending by endpoint ID.
// sim/edpp.go:2160-2168. Required for decision determinism: without it the
// enumeration order follows the caller's slice, and an exact tie resolves
// differently between two otherwise identical decisions.
//
// It COPIES before sorting -- the input slice must not be reordered in place, since
// the caller may hold it for other purposes.
func sortedByID(snaps []Snapshot) []Snapshot {
	if len(snaps) == 0 {
		return nil
	}
	out := make([]Snapshot, len(snaps))
	copy(out, snaps)
	sortSliceByID(out)
	return out
}

// scorerFirstSnapshots preserves ascending-ID order except that the stock scorer's
// preferred endpoint is moved to the FRONT. sim/edpp.go:2180-2196.
//
// It is a TIE-BREAK ORDER, not a filter: every endpoint is still enumerated, and
// because the argmin uses a strict improvement threshold, moving one to the front
// only changes which candidate wins an EXACT tie. That is also what makes
// restricting the enumeration to the first element reproduce the decomposed rule.
//
// Three early-return cases return the input UNCHANGED rather than a copy: no
// preference, fewer than two endpoints, or the preference already first.
func scorerFirstSnapshots(snaps []Snapshot, preferred string) []Snapshot {
	if preferred == "" || len(snaps) < 2 || snaps[0].ID == preferred {
		return snaps
	}
	out := make([]Snapshot, 0, len(snaps))
	for _, s := range snaps {
		if s.ID == preferred {
			out = append(out, s)
		}
	}
	for _, s := range snaps {
		if s.ID != preferred {
			out = append(out, s)
		}
	}
	return out
}

// sortSliceByID sorts in place, ascending by .ID. Adapter for the target's sort
// import; the ORDERING CONTRACT (ascending by ID) is the specified part.
func sortSliceByID(snaps []Snapshot)

// prefillScorerPick returns the injected prefill scorer's preferred endpoint, or
// ok == false when no prefill scorer is configured. Target-API adapter; upstream's
// counterpart is the d.prefillScorer hook (sim/edpp.go:1388).
func (p *Policy) prefillScorerPick(req Request, prefillSnaps []Snapshot) (string, bool)

// inputLen, requestID, and endpointByID are target-API adapters.
func (p *Policy) inputLen(req Request) int
func (p *Policy) requestID(req Request) string
func (p *Policy) endpointByID(id string) Endpoint

// ---------------------------------------------------------------------------
// Deliberate omissions
//
// Named so their absence reads as a decision rather than an oversight, and so a
// port re-enabling any of them knows what it must add.
//
//   - QWork (the per-endpoint waiting-work account) is read ONLY by
//     waitingEstimateTAdm. This arm selects rollforward, so it is
//     decision-neutral. A port switching to `waiting` must build it.
//   - AdmissionRate is read ONLY by littleEstimateTAdm. Same reasoning.
//   - The SLO deficit virtual queues (z_ttft, z_itl, sim/edpp.go:1338-1345) are
//     sampled for diagnostics and carried by OTHER rules in the published decider.
//     This arm's objective carries no historical deficit term
//     (sim/edpp.go:1683-1687 states that contract), so they do not enter J.
//   - The capacity account IS stated above but is dead state while
//     SLOExternalityNoCapacity is true. Re-enabling it requires a monotonic clock
//     and per-endpoint drain timestamps.
//   - The flip, util, and hazard value kernels are not stated: they are selected
//     only by rules this arm does not use, and no switch in this arm's
//     configuration reaches them.
//
// ---------------------------------------------------------------------------
// Required observability
//
// Every degradation above is INVISIBLE in the goodput number, and the target's own
// fallbacks are silent. These counters are a requirement of the port, not
// instrumentation:
//
//   - TTFT-estimator substitution (D1) -- otherwise the port runs a different
//     estimator than the one that produced every number in sim_results/.
//   - tokenization unavailable -- a_r reads 0 with no error, and nothing is
//     scorable.
//   - prefix unobserved -- a_p fell back to the full prompt, re-pricing the
//     local/remote boundary.
//   - candidate rejected, by reason -- a mislabelled endpoint leaving the
//     candidate set is otherwise indistinguishable from a routing PREFERENCE.
//   - block-size mismatch -- config/engine unit disagreement.
//   - shadow-table size (gauge) -- near zero under load means the resident
//     populations are empty and every externality is 0.
//   - shadow-table entries reaped -- entries dropped without a final chunk.
//   - placement chosen, local vs disaggregated -- what the argmin actually picked.
//   - argmin duration -- EPP-side work INSIDE the TTFT being measured.

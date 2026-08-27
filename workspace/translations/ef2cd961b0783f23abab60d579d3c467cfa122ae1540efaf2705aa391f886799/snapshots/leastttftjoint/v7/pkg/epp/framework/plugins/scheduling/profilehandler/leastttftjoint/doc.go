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

// Package leastttftjoint is the least-TTFT joint COMPARATOR arm for the INFOCOM 2027
// transfer. Its counterpart is the focal arm in
// pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality.
//
// # WHY THIS ARM EXISTS
//
// It shares the candidate set, the estimators, the physics, and the prefix reads with the
// focal arm, and differs IN THE OBJECTIVE ONLY. That is what attributes the measured
// effect to the MECHANISM rather than to the MACHINERY: a weaker baseline would leave open
// that the focal arm won because it computes better physics, not because it prices SLO
// externality.
//
// The objective is the argmin of the ARRIVING REQUEST'S OWN projected time-to-first-token
// over the same candidate set -- see Policy.jointCandidateTTFT. It carries no backlog
// drift, no SLO virtual queues, no externality over residents, no tau at all, and no
// transfer penalty beyond the transfer latency already inside the disaggregated TTFT.
//
// Each side prices with its OWN candidate's theta, so the arm is hardware-aware by
// construction. That is deliberate: a hardware-BLIND least-TTFT baseline would lose to the
// focal arm partly because it cannot see the fleet, which would confound the mechanism
// with hardware awareness.
//
// Cited evidence for the comparison (INFOCOM_REPRODUCIBILITY.md expected checkpoints,
// reproduced in sim_results/main/), worst static-plan gap:
//
//	causal externality  0.0100
//	least-TTFT          0.0542
//
// and in the stress cohort (sim_results/stress/): 0.0031 against 0.1110.
//
// # THE VERBATIM-COPY CONTRACT, AND HOW IT IS ENFORCED
//
// config.md section 9.3 requires this arm's routing view to be a VERBATIM COPY of the
// focal arm's, package clause aside. This is not style. A re-derived-but-slightly-different
// estimator would silently destroy the attribution argument WHILE EVERY TEST STILL PASSED:
// the two arms would then differ in the objective AND in the physics, and no measurement
// could separate them. Any change to a shared symbol must be made in BOTH arms, or the
// comparison stops being a comparison.
//
// A comment cannot enforce that, so contract_test.go does. It reads the focal arm's
// sources at test time and asserts:
//
//	coeffs.go, admission.go, rollout.go, shadow.go
//	    BYTE-IDENTICAL to the focal arm's, after rewriting the package clause.
//
//	types.go, shared.go
//	    DELETION-ONLY subsets of the focal arm's types.go and policy.go: every line
//	    present here appears in the focal file, in the same order. That is the property
//	    that matters -- it proves no RETAINED line was altered, while still allowing the
//	    objective-specific declarations to be absent.
//
// Editing a shared file in either arm without editing the other therefore fails
// `go test ./pkg/epp/framework/plugins/scheduling/profilehandler/leastttftjoint/...`. That
// test is the contract; this comment only explains it.
//
// A CONSEQUENCE OF BYTE-IDENTITY, STATED SO IT IS NOT MISTAKEN FOR AN ERROR: the four
// exactly-copied files retain the FOCAL ARM'S AUTHORIAL VOICE. In them, "this arm" means
// the focal arm, and their comments refer to the externality, the value kernels, and the
// capacity account -- none of which exist here. coeffs.go's deltaBarDecode even documents
// itself as being read by "the comparator arm's ITL term", which this arm does not have
// either. Those comments are left untouched on purpose: byte-identity is a property a test
// can check and a future editor cannot silently break, and it is worth far more than
// editorial polish in a copy. Read the copied files as the focal arm's, and read this
// package's own files (config.go, policy.go, metrics.go, handler.go, picker.go, doc.go) as
// this arm's.
//
// # WHAT IS DELIBERATELY ABSENT
//
// This arm drops fields and files rather than populating unread ones, because a
// populated-but-never-read field is a reader trap: it implies a consumer exists somewhere.
// Relative to the focal arm there is no kernels.go (no value kernels: no sloCompositeValue,
// no gDecodeComposite, no goodSelf), no capacity.go (no capacity account), and no
// retiming.go (nothing here needs reTiming -- the B+1 re-timed first decode iteration is
// computed directly from tIterDecode). Config drops the tau triple, V, Ablation, and
// Capacity; see Config's own doc for why each absence is unreachable rather than merely
// unused.
//
// # WHAT IT STILL READS, AND THE DEGRADATIONS THAT ARRIVE WITH IT
//
// It reads no resident VALUE, but it does read the resident POPULATIONS, so the shadow
// table and its degradations are inherited:
//
//	D4  ResidentPrefillTokens enters tIterDecode and tIterPrefill as S_pf, on every
//	    candidate.
//	D2  RunningDecode feeds decodeRemStepsEst and the rollforward KV walk, so StepsDone
//	    and KVBlocks are read.
//	D6  RunningPrefill feeds prefillRemStepsEst.
//	D7  FreeKVBlocks is a floor, inherited via the admission context.
//
// It does NOT inherit D2c, the late-first-token bias: no kernel here reads ArrivalUs,
// FirstTokenUs, or TTFTSet. Those three fields are still carried on RunningReqState and
// still written by the verbatim-copied shadow table, because that machinery is under the
// copy contract and forking it is the larger hazard. Nothing in this package reads them.
//
// D1 (scheduler rollout unobtainable) is inherited IN FULL, and the substitution counter
// is required here too -- but its DIRECTION is not a single sign for this objective, and
// the focal arm's one-line summary must not be copied over. It also moves this arm between a
// PREFILL-SENSITIVE and a PREFILL-INDIFFERENT regime rather than merely shifting a threshold,
// because tAdmD is both the term the fallback understates and the term that decides which side
// of max(remoteLead, tAdmD) binds. See estimatorRollforward's doc in config.go for the algebra,
// for why it bears on the attribution argument rather than only on this arm's accuracy, and for
// why no direction is claimed.
//
// D5 (c_xfer unmeasured) is inherited and matters MORE here than in the focal arm: there
// is no externality term to partially offset a mis-priced transfer. See Transfer's doc.
//
// D8 (tokenization can be unavailable) is inherited, and on that path this arm returns no
// ranking, so a third policy -- neither arm -- decides. See promptTokens and
// Picker.Filter. The per-arm counter is required, and the two arms' rates must be compared
// before any result is read.
//
// # PLACEMENT
//
// TWO plugin registrations sharing state, not one plugin with two interfaces:
// ProfileHandler.Pick and Picker.Pick share a method name with different signatures, so no
// single Go type can satisfy both. The handler carries ProfileHandler + PreRequest +
// ResponseBodyProcessor; the picker carries Filter + Picker and holds a pointer to the
// handler. The handler must be declared BEFORE the picker in the plugins list, because
// by-name references resolve backward only.
//
// Both arms register a ProfileHandler and only one is permitted per EPP, so the two arms
// are run as two scenarios in two processes and never share a shadow table, a Policy, or a
// decision. See Handler's doc.
package leastttftjoint

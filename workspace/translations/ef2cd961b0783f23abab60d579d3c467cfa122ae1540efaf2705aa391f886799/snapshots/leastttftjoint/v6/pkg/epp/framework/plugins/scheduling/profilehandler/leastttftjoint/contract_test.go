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
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/causalsloexternality"
)

// THIS FILE IS THE VERBATIM-COPY CONTRACT, ENFORCED.
//
// config.md section 9.3 requires this arm's routing view to be a verbatim copy of the focal
// arm's, package clause aside, differing in the objective ONLY. The reason is not style: a
// re-derived-but-slightly-different estimator would silently destroy the attribution
// argument WHILE EVERY OTHER TEST IN THIS PACKAGE STILL PASSED. The two arms would then
// differ in the objective AND in the physics, and no measurement could separate the
// mechanism from the machinery.
//
// A comment cannot enforce that, and a reviewer reading two 400-line files side by side
// cannot reliably enforce it either. These tests can. Editing a shared symbol in EITHER arm
// without editing the other fails here.

// focalDir locates the focal arm's package source, which is a sibling directory.
//
// The sources are read rather than reflected upon because the property under test is
// TEXTUAL: two functions can be behaviourally identical on every input a test tries and
// still differ in a way that matters at the third decimal place of a latency projection.
func focalDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(filepath.Join("..", "causalsloexternality"))
	if err != nil {
		t.Fatalf("resolve the focal arm's directory: %v", err)
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("the focal arm's package must be present for the copy contract to be checkable: %v", err)
	}
	return dir
}

func readLines(t *testing.T, path string) []string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return strings.Split(string(raw), "\n")
}

// rewritePackageClause is the ONLY difference the contract permits.
func rewritePackageClause(lines []string) []string {
	out := make([]string, len(lines))
	for i, line := range lines {
		if line == "package causalsloexternality" {
			out[i] = "package leastttftjoint"
			continue
		}
		out[i] = line
	}
	return out
}

// exactCopies are byte-identical to the focal arm's files after the package rewrite.
//
// coeffs.go   the calibrated latency law theta and its methods, plus the int helpers.
// admission.go all four admission estimators and the oracle censoring.
// rollout.go  the whole scheduler rollout -- degradation D1's unreachable branch.
// shadow.go   the EPP-side resident index -- degradations D2, D4, D6 arrive with it.
var exactCopies = []string{"coeffs.go", "admission.go", "rollout.go", "shadow.go"}

func TestSharedFilesAreByteIdenticalToTheFocalArm(t *testing.T) {
	dir := focalDir(t)
	for _, name := range exactCopies {
		t.Run(name, func(t *testing.T) {
			want := rewritePackageClause(readLines(t, filepath.Join(dir, name)))
			got := readLines(t, name)
			if len(got) != len(want) {
				t.Fatalf("%s has %d lines, the focal arm's has %d: this file is under the "+
					"verbatim-copy contract (config.md section 9.3) and may differ ONLY in its "+
					"package clause. If the focal arm changed, copy the change; do not re-derive it.",
					name, len(got), len(want))
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%s:%d diverges from the focal arm.\n  this arm:  %s\n  focal arm: %s\n\n"+
						"This file is under the verbatim-copy contract. A "+
						"re-derived-but-slightly-different estimator destroys the attribution "+
						"argument while every behavioural test still passes: the arms would differ "+
						"in the objective AND in the physics. Apply the change to BOTH arms.",
						name, i+1, got[i], want[i])
				}
			}
		})
	}
}

// deletionOnlyCopies are subsets: every line present here appears in the focal arm's file,
// in the same order, and nothing was rewritten.
//
// That is the property worth checking. It permits the objective-specific declarations to be
// ABSENT -- this arm has no pathBreakdown, no candidateScore, no value kernels, no capacity
// account -- while proving that no RETAINED line was altered. A subsequence check catches
// exactly the dangerous edit (a changed constant, a flipped comparison, a dropped term)
// while allowing the legitimate one (a whole declaration removed).
var deletionOnlyCopies = map[string]string{
	"types.go":  "types.go",
	"shared.go": "policy.go",
}

func TestSharedSubsetsAreDeletionOnly(t *testing.T) {
	dir := focalDir(t)
	for mine, focal := range deletionOnlyCopies {
		t.Run(mine, func(t *testing.T) {
			want := rewritePackageClause(readLines(t, filepath.Join(dir, focal)))
			got := readLines(t, mine)

			// Walk both in order; every line of `got` must be findable in `want` at or
			// after the current position.
			pos := 0
			for i, line := range got {
				found := -1
				for j := pos; j < len(want); j++ {
					if want[j] == line {
						found = j
						break
					}
				}
				if found < 0 {
					t.Fatalf("%s:%d is not a line of the focal arm's %s at or after line %d:\n  %q\n\n"+
						"This file must be a DELETION-ONLY subset of the focal arm's (config.md "+
						"section 9.3): declarations this arm does not have may be removed, but no "+
						"retained line may be rewritten. Either copy the focal arm's line verbatim, "+
						"or -- if this genuinely belongs to this arm's objective -- move it to "+
						"policy.go, config.go, or another file this arm owns.",
						mine, i+1, focal, pos+1, line)
				}
				pos = found + 1
			}
			if len(got) >= len(want) {
				t.Errorf("%s has %d lines and the focal arm's %s has %d: a subset must be strictly "+
					"smaller, so this file is probably not the subset it claims to be",
					mine, len(got), focal, len(want))
			}
		})
	}
}

// TestNoObjectiveSpecificSymbolsLeakedIn pins the other direction: the focal arm's
// objective must not reappear here under any name. Its presence would mean the comparator
// had grown the mechanism it exists to be compared against.
func TestNoObjectiveSpecificSymbolsLeakedIn(t *testing.T) {
	// Symbol names, matched against declaration sites rather than prose, so the doc
	// comments that legitimately DISCUSS the focal arm do not trip this.
	forbidden := []string{
		"func sloCompositeValue", "func gDecodeComposite", "func goodSelf",
		"func varDecodeContribution", "func varCollocContribution", "func varPrefillDisaggExactAfter",
		"func (p *Policy) externalityLocal", "func (p *Policy) externalityDisagg",
		"func (p *Policy) scoreCandidate", "func (p *Policy) selfGood", "func (p *Policy) sloFor",
		"func (p *Policy) capacityTerm", "func (p *Policy) refreshCapacity", "func (p *Policy) bookCapacityWork",
		"func (p *Policy) varDecodeInputs", "func (p *Policy) varPrefillInputs",
		"func scorerFirstSnapshots", "func preferredByScore", "func reTimingFor",
		"type candidateScore", "type pathBreakdown", "type Ablation", "type SLOTargets", "type Capacity",
	}
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("read package dir: %v", err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		raw, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		for _, symbol := range forbidden {
			if strings.Contains(string(raw), symbol) {
				t.Errorf("%s declares %q. This arm's objective is the arriving request's own "+
					"projected TTFT and nothing else: it has no value kernels, no externality, and "+
					"no capacity account. Growing one collapses the comparison.", name, symbol)
			}
		}
	}
}

// TestConfigIsAStrictSubsetOfTheFocalArms enforces the claim in Config's doc: every field
// here has the same meaning and the same value as the focal arm's field of the same name.
//
// Shape agreement is what a test can check, and it is the half that fails silently. If this
// arm renamed a json key, or narrowed a float64 to an int, both arms would still validate
// and still run -- and would be configured from overlays that no longer agree, so the two
// would price the same fleet differently for a reason unrelated to the objective.
//
// Value agreement cannot be checked from inside one package: it lives in the two generated
// overlays. overlay_test.go checks this arm's half against the shipped YAML.
func TestConfigIsAStrictSubsetOfTheFocalArms(t *testing.T) {
	mine := reflect.TypeOf(Config{})
	focal := reflect.TypeOf(causalsloexternality.Config{})

	if mine.NumField() >= focal.NumField() {
		t.Errorf("this arm's Config has %d fields and the focal arm's has %d: a STRICT subset must "+
			"be smaller", mine.NumField(), focal.NumField())
	}
	compareStructSubset(t, "Config", mine, focal)
}

// compareStructSubset asserts every field of `mine` exists in `focal` under the same json
// tag, with a structurally equal type. Named struct types differ between the two packages,
// so the comparison recurses on shape rather than comparing reflect.Type identity.
func compareStructSubset(t *testing.T, path string, mine, focal reflect.Type) {
	t.Helper()
	focalByTag := map[string]reflect.StructField{}
	for i := 0; i < focal.NumField(); i++ {
		f := focal.Field(i)
		focalByTag[jsonTagName(f)] = f
	}
	for i := 0; i < mine.NumField(); i++ {
		f := mine.Field(i)
		tag := jsonTagName(f)
		peer, ok := focalByTag[tag]
		if !ok {
			t.Errorf("%s: json key %q exists in this arm but not in the focal arm's Config. Every "+
				"field this arm carries must be the focal arm's field of the same name, or the two "+
				"overlays cannot be kept in agreement.", path, tag)
			continue
		}
		if err := sameShape(f.Type, peer.Type); err != nil {
			t.Errorf("%s.%s (json %q) differs in shape from the focal arm's: %v. The two arms must "+
				"be configurable from the same values for every shared key.", path, f.Name, tag, err)
		}
	}
}

func jsonTagName(f reflect.StructField) string {
	tag := f.Tag.Get("json")
	if tag == "" {
		return f.Name
	}
	if idx := strings.Index(tag, ","); idx >= 0 {
		tag = tag[:idx]
	}
	return tag
}

// sameShape compares two types structurally, ignoring the package a named type came from.
func sameShape(a, b reflect.Type) error {
	if a.Kind() != b.Kind() {
		return fmt.Errorf("kind %s against %s", a.Kind(), b.Kind())
	}
	switch a.Kind() {
	case reflect.Struct:
		if a.NumField() != b.NumField() {
			return fmt.Errorf("%d fields against %d", a.NumField(), b.NumField())
		}
		for i := 0; i < a.NumField(); i++ {
			fa, fb := a.Field(i), b.Field(i)
			if jsonTagName(fa) != jsonTagName(fb) {
				return fmt.Errorf("field %d json key %q against %q", i, jsonTagName(fa), jsonTagName(fb))
			}
			if err := sameShape(fa.Type, fb.Type); err != nil {
				return fmt.Errorf("field %q: %w", fa.Name, err)
			}
		}
		return nil
	case reflect.Map:
		if err := sameShape(a.Key(), b.Key()); err != nil {
			return fmt.Errorf("map key: %w", err)
		}
		return sameShape(a.Elem(), b.Elem())
	case reflect.Slice, reflect.Ptr:
		return sameShape(a.Elem(), b.Elem())
	default:
		return nil
	}
}

// TestCoeffsMatchTheFocalArmsFieldForField is the sharpest instance of the shape contract:
// theta is the physics, and both arms must read the same JSON keys into the same units. A
// renamed key here would leave this arm's coefficients at their zero values while
// validation still passed on the focal arm's overlay.
func TestCoeffsMatchTheFocalArmsFieldForField(t *testing.T) {
	if err := sameShape(reflect.TypeOf(Coeffs{}), reflect.TypeOf(causalsloexternality.Coeffs{})); err != nil {
		t.Fatalf("Coeffs diverges from the focal arm's: %v", err)
	}
}

// TestPluginTypesAreDistinctFromTheFocalArms pins that the two arms cannot collide in the
// plugin registry. Registration is global (fwkplugin.Register), so a shared type string
// would mean one arm silently overwrote the other's factory.
func TestPluginTypesAreDistinctFromTheFocalArms(t *testing.T) {
	pairs := [][2]string{
		{HandlerPluginType, causalsloexternality.HandlerPluginType},
		{PickerPluginType, causalsloexternality.PickerPluginType},
		{HandlerPluginType, causalsloexternality.PickerPluginType},
		{PickerPluginType, causalsloexternality.HandlerPluginType},
	}
	for _, pair := range pairs {
		if pair[0] == pair[1] {
			t.Errorf("plugin type %q is shared with the focal arm: registration is global, so one "+
				"arm's factory would overwrite the other's", pair[0])
		}
	}
	if HandlerPluginType == PickerPluginType {
		t.Error("this arm's two registrations must have distinct type strings")
	}
}

// TestDecisionAttributeKeyIsDistinctFromTheFocalArms pins the per-endpoint attribute
// namespace. The key is a plain string on a per-request Endpoint clone, so a shared key
// would let one arm read the other's decision input if both ever ran in one process.
//
// Only one arm can be instantiated per EPP (both declare a ProfileHandler), so this is
// defence in depth rather than a live hazard -- but it costs one constant and removes a
// whole class of question.
func TestDecisionAttributeKeyIsDistinctFromTheFocalArms(t *testing.T) {
	if strings.Contains(decisionAttributeKey, "CausalSLOExternality") {
		t.Errorf("decisionAttributeKey = %q, which is the focal arm's namespace", decisionAttributeKey)
	}
	if !strings.Contains(decisionAttributeKey, "LeastTTFTJoint") {
		t.Errorf("decisionAttributeKey = %q, want it namespaced to this arm", decisionAttributeKey)
	}
}

// TestMetricNamesAreDistinctFromTheFocalArms enforces the operator's decision on metric naming.
//
// The ten families here are `least_ttft_joint_*`, parallel to the focal arm's
// `causal_slo_externality_*` and term-for-term identical in every other respect. The reason is
// that the focal arm's names would be a MISNOMER on an arm with no externality term -- not that
// they would collide. See metrics.go's header, and TestTheTwoArmsCannotBeInstantiatedTogether
// below for why the collision argument would have been wrong.
//
// This test is the enforceable form of the decision in both directions: a family that keeps the
// focal arm's name is caught, and so is one that drifts out of the shared naming scheme.
func TestMetricNamesAreDistinctFromTheFocalArms(t *testing.T) {
	focalRaw, err := os.ReadFile(filepath.Join(focalDir(t), "metrics.go"))
	if err != nil {
		t.Fatalf("read the focal arm's metrics.go: %v", err)
	}
	focalSource := string(focalRaw)
	mineRaw, err := os.ReadFile("metrics.go")
	if err != nil {
		t.Fatalf("read metrics.go: %v", err)
	}

	mine := metricNames(t, string(mineRaw))
	focal := metricNames(t, focalSource)
	if len(mine) != len(focal) {
		t.Errorf("this arm declares %d metric families and the focal arm declares %d: the two sets "+
			"are meant to be term-for-term parallel, so a regex over __name__ can select both arms "+
			"in ONE cross-arm query", len(mine), len(focal))
	}
	for _, name := range mine {
		if strings.Contains(focalSource, `"`+name+`"`) {
			t.Errorf("metric %q is also declared by the focal arm. The families are deliberately "+
				"parallel-but-distinct: sharing a name would make this arm emit series named for an "+
				"externality it does not compute.", name)
		}
		if !strings.HasPrefix(name, "least_ttft_joint_") {
			t.Errorf("metric %q is not namespaced to this arm", name)
		}
	}

	// The parallelism is the property the single cross-arm regex query depends on, so check that
	// each of this arm's families has a focal counterpart differing only in the arm prefix.
	for _, name := range mine {
		peer := strings.Replace(name, "least_ttft_joint_", "causal_slo_externality_", 1)
		if !strings.Contains(focalSource, `"`+peer+`"`) {
			t.Errorf("metric %q has no focal-arm counterpart %q: without the term-for-term "+
				"parallelism, the cross-arm D1/D8 comparison stops being one regex query", name, peer)
		}
	}
}

// TestTheTwoArmsCannotBeInstantiatedTogether records why the metric-collision argument -- which
// looks like the obvious justification for distinct names -- is NOT the reason for them.
//
// Both arms' handlers implement fwksched.ProfileHandler, and buildSchedulerConfig permits
// exactly one across all instantiated plugins (configloader.go:275-284). So two arms in one
// EndpointPickerConfig is invalid before metrics matter, and registerMetrics runs at factory
// time rather than package init, so merely compiling both arms into one binary registers
// nothing. Distinct names are justified by the misnomer, not by a hazard.
//
// Distinct names do keep one failure mode legible, which this test also pins: because
// instantiatePlugins (configloader.go:147) runs BEFORE the uniqueness check (:160), a two-arm
// config fails during instantiation -- and with distinct families it can fail on the
// ProfileHandler rule rather than on a Prometheus duplicate registration that would mask it.
func TestTheTwoArmsCannotBeInstantiatedTogether(t *testing.T) {
	var _ fwksched.ProfileHandler = &Handler{}

	registry := prometheus.NewRegistry()
	mine, err := HandlerFactory("comparator", decoderFor(t, armConfig()), newFakeHandleWithRegistry(registry))
	if err != nil {
		t.Fatalf("building this arm failed: %v", err)
	}
	t.Cleanup(mine.(*Handler).table.stop)
	if _, ok := mine.(fwksched.ProfileHandler); !ok {
		t.Fatal("this arm's handler must be a ProfileHandler; that rule is what makes two arms in " +
			"one configuration invalid")
	}

	focal, err := causalsloexternality.HandlerFactory("focal", decoderFor(t, focalArmConfig()),
		newFakeHandleWithRegistry(registry))
	if err != nil {
		t.Fatalf("building the focal arm against the SAME registry failed: %v\n\n"+
			"With parallel-but-distinct family names this must succeed. If it does not, the two arms "+
			"share a family name and a two-arm config would fail on a Prometheus duplicate "+
			"registration, masking the clear \"multiple profile handlers found\" diagnosis.", err)
	}
	if _, ok := focal.(fwksched.ProfileHandler); !ok {
		t.Fatal("the focal arm's handler must also be a ProfileHandler")
	}
}

// metricNames extracts the Name: literals from a metrics source file.
func metricNames(t *testing.T, source string) []string {
	t.Helper()
	var out []string
	for _, line := range strings.Split(source, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "Name:") {
			continue
		}
		if first := strings.Index(line, `"`); first >= 0 {
			if last := strings.LastIndex(line, `"`); last > first {
				out = append(out, line[first+1:last])
			}
		}
	}
	if len(out) == 0 {
		t.Fatal("no metric names found; the extraction in this test has drifted from metrics.go")
	}
	return out
}

// focalArmConfig is a valid focal-arm configuration, used only to construct that arm in the
// coexistence test above. Its shared keys carry this arm's values, which is the agreement
// config.md section 9.1 requires; its objective-specific keys come from config.md section 9.2.
func focalArmConfig() causalsloexternality.Config {
	mine := armConfig()
	coeffs := map[string]causalsloexternality.Coeffs{}
	for gpuType, c := range mine.CoeffsByGPUType {
		coeffs[gpuType] = causalsloexternality.Coeffs{
			AlphaD: c.AlphaD, AlphaP: c.AlphaP, C0: c.C0, C1: c.C1, CPf: c.CPf, CAttn: c.CAttn,
		}
	}
	return causalsloexternality.Config{
		V:              8,
		Ablation:       causalsloexternality.Ablation{NoCapacity: true},
		ActiveWorkload: "interactive",
		WorkloadTargets: map[string]causalsloexternality.SLOTargets{
			"interactive": {TauTTFTUs: 1_000_000, TauITLUs: 50_000, TauE2EUs: 16_000_000},
		},
		Engine: causalsloexternality.EngineAgreement{
			ChunkTokens:  mine.Engine.ChunkTokens,
			BlockSize:    mine.Engine.BlockSize,
			MaxBatchSize: mine.Engine.MaxBatchSize,
		},
		Signals: causalsloexternality.Signals{
			GPUTypeLabelKey:             mine.Signals.GPUTypeLabelKey,
			PrefixMatchInfoProducerName: mine.Signals.PrefixMatchInfoProducerName,
			TokenizedPromptProducerName: mine.Signals.TokenizedPromptProducerName,
		},
		AdmissionEstimator: mine.AdmissionEstimator,
		Transfer: causalsloexternality.Transfer{
			SizeAware:             mine.Transfer.SizeAware,
			XferBaseUs:            mine.Transfer.XferBaseUs,
			XferBandwidthGBps:     mine.Transfer.XferBandwidthGBps,
			KVBytesPerTokenPerGPU: mine.Transfer.KVBytesPerTokenPerGPU,
			FlatCXferUs:           mine.Transfer.FlatCXferUs,
		},
		CoeffsByGPUType: coeffs,
		ShadowTable: causalsloexternality.ShadowTable{
			EntryTTLSeconds:          mine.ShadowTable.EntryTTLSeconds,
			SweepIntervalSeconds:     mine.ShadowTable.SweepIntervalSeconds,
			ResidentPrefillTokensCap: mine.ShadowTable.ResidentPrefillTokensCap,
		},
		Capacity:                causalsloexternality.Capacity{TauRefUs: 1_000_000, NomPrefillTokens: 512, ReferenceBatch: 256},
		OutputTokenProcessingUs: mine.OutputTokenProcessingUs,
	}
}

// TestTheTwoArmsSharedConfigValuesAgree is the value half of the section 9.1 agreement, which
// TestConfigIsAStrictSubsetOfTheFocalArms covers only in shape.
//
// If the two arms were configured with different engine agreement, different producer
// bindings, or different theta, they would differ in the PHYSICS as well as the objective and
// no measurement could separate the mechanism from the machinery. focalArmConfig above is
// derived from armConfig field by field, so this test is what keeps that derivation honest:
// it fails if a shared key is ever hardcoded on one side.
func TestTheTwoArmsSharedConfigValuesAgree(t *testing.T) {
	mine, focal := armConfig(), focalArmConfig()

	if mine.Engine.ChunkTokens != focal.Engine.ChunkTokens ||
		mine.Engine.BlockSize != focal.Engine.BlockSize ||
		mine.Engine.MaxBatchSize != focal.Engine.MaxBatchSize {
		t.Errorf("engine agreement differs: %+v against %+v", mine.Engine, focal.Engine)
	}
	if mine.Signals.PrefixMatchInfoProducerName != focal.Signals.PrefixMatchInfoProducerName {
		t.Errorf("the two arms bind DIFFERENT prefix producers (%q against %q): a data key's "+
			"identity includes its producer name, so the arms would price different prompts",
			mine.Signals.PrefixMatchInfoProducerName, focal.Signals.PrefixMatchInfoProducerName)
	}
	if mine.Signals.TokenizedPromptProducerName != focal.Signals.TokenizedPromptProducerName {
		t.Errorf("the two arms bind DIFFERENT token producers (%q against %q)",
			mine.Signals.TokenizedPromptProducerName, focal.Signals.TokenizedPromptProducerName)
	}
	if mine.Signals.GPUTypeLabelKey != focal.Signals.GPUTypeLabelKey {
		t.Errorf("the two arms read different GPU-type labels: %q against %q",
			mine.Signals.GPUTypeLabelKey, focal.Signals.GPUTypeLabelKey)
	}
	if mine.AdmissionEstimator != focal.AdmissionEstimator {
		t.Errorf("the two arms select different admission estimators: %q against %q",
			mine.AdmissionEstimator, focal.AdmissionEstimator)
	}
	if mine.Transfer.SizeAware != focal.Transfer.SizeAware ||
		mine.Transfer.XferBaseUs != focal.Transfer.XferBaseUs ||
		mine.Transfer.XferBandwidthGBps != focal.Transfer.XferBandwidthGBps ||
		mine.Transfer.KVBytesPerTokenPerGPU != focal.Transfer.KVBytesPerTokenPerGPU ||
		mine.Transfer.FlatCXferUs != focal.Transfer.FlatCXferUs {
		t.Errorf("the transfer models differ: %+v against %+v", mine.Transfer, focal.Transfer)
	}
	if len(mine.CoeffsByGPUType) != len(focal.CoeffsByGPUType) {
		t.Fatalf("theta tables differ in size: %d against %d",
			len(mine.CoeffsByGPUType), len(focal.CoeffsByGPUType))
	}
	for gpuType, c := range mine.CoeffsByGPUType {
		peer, ok := focal.CoeffsByGPUType[gpuType]
		if !ok {
			t.Errorf("theta for %q is missing from the focal arm's table", gpuType)
			continue
		}
		if c.AlphaD != peer.AlphaD || c.AlphaP != peer.AlphaP || c.C0 != peer.C0 ||
			c.C1 != peer.C1 || c.CPf != peer.CPf || c.CAttn != peer.CAttn {
			t.Errorf("theta for %q differs between the arms: %+v against %+v", gpuType, c, peer)
		}
	}
	if mine.ShadowTable != (ShadowTable{
		EntryTTLSeconds:          focal.ShadowTable.EntryTTLSeconds,
		SweepIntervalSeconds:     focal.ShadowTable.SweepIntervalSeconds,
		ResidentPrefillTokensCap: focal.ShadowTable.ResidentPrefillTokensCap,
	}) {
		t.Errorf("shadow-table settings differ: %+v against %+v", mine.ShadowTable, focal.ShadowTable)
	}
	if mine.OutputTokenProcessingUs != focal.OutputTokenProcessingUs {
		t.Errorf("outputTokenProcessingUs differs: %g against %g",
			mine.OutputTokenProcessingUs, focal.OutputTokenProcessingUs)
	}
}

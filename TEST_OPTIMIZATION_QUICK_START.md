# Test Suite Optimization - Quick Start Guide

**Fast reference for implementing test improvements to nnBenchmark**

---

## At a Glance

| Category | Count | Priority | Effort | Impact |
|----------|-------|----------|--------|--------|
| Tests to Add | 8-10 | Critical | 3-4 weeks | High |
| Tests to Consolidate | 70-80 | Medium | 1 week | Medium |
| Tests to Remove | 5-10 | Low | 3-4 days | Low |
| Tests to Improve | 3-5 | Medium | 1-2 weeks | Medium |
| **Total Impact** | - | - | 4-5 weeks | **Excellent** |

---

## 1. CRITICAL MISSING TESTS (Do First!)

### Implement These First - They Prevent Production Bugs

1. **End-to-End Planning Workflow** (Test 1)
   - File: `tests/test_planning.py`
   - Why: No integration test validates planning pipeline works together
   - How long: 2-3 hours
   - See: `TEST_RECOMMENDATIONS_CODE_EXAMPLES.md` Part 1, Test 1

2. **Corrupted Data Handling** (Test 2)
   - File: `tests/test_monai_integration.py`
   - Why: Real datasets have corrupted files; need graceful handling
   - How long: 2-3 hours
   - See: `TEST_RECOMMENDATIONS_CODE_EXAMPLES.md` Part 1, Test 2

3. **GPU Memory Constraints** (Test 4)
   - File: `tests/test_planning.py`
   - Why: Small GPUs (2-4GB) need different configs; should validate downscaling
   - How long: 1-2 hours
   - See: `TEST_RECOMMENDATIONS_CODE_EXAMPLES.md` Part 1, Test 4

4. **Splits Determinism** (Test 8)
   - File: `tests/test_splits.py`
   - Why: CLAUDE.md emphasizes reproducibility; need to verify determinism
   - How long: 1-2 hours
   - See: `TEST_RECOMMENDATIONS_CODE_EXAMPLES.md` Part 1, Test 5

**Total Time for Critical Tests**: ~1 week

---

## 2. QUICK WINS - Consolidate Over-Parameterized Tests

### These Give 15-30% Test Runtime Improvement

**Consolidate Sliding Window Tests**
- **Current**: 15 separate tests in `TestSlidingWindowVolumeVariations`
- **Target**: 1-2 parametrized tests
- **File**: `tests/test_inference_strategies.py` (lines 389-650)
- **Time**: 2-3 hours
- **Benefit**: 15 tests → 1 parametrized test = 14 fewer test runs
- **See**: `TEST_RECOMMENDATIONS_CODE_EXAMPLES.md` Part 2

**Consolidate File Type Detection Tests**
- **Current**: 16 parametrized cases
- **Target**: 5 representative cases
- **File**: `tests/test_monai_integration.py` (lines 20-62)
- **Time**: 30 minutes
- **Benefit**: Test clarity, fewer false positives
- **See**: `TEST_SUITE_ANALYSIS.md` Section 3, Test 4

---

## 3. REMOVE LOW-VALUE TESTS (2-3 hours)

Quick deletions that don't hurt coverage:

1. **Formatting tests** in `test_results.py`
   - Lines 54-83 (test_print_results_formatting, test_print_results_separator_lines)
   - Delete: ~30 lines
   - Keep: test_print_basic_results as smoke test

2. **Output value range test** in `test_inference_strategies.py`
   - Lines 565-579 (test_output_values_range)
   - Delete: ~15 lines
   - Reason: Tests library behavior, not our code

---

## 4. IMPROVE TEST QUALITY (2-3 hours)

Small fixes with big impact:

1. **Fix Weak Assertion**
   - File: `tests/test_planning.py`
   - Function: `TestModelTopology::test_anisotropic_pooling`
   - See: `TEST_RECOMMENDATIONS_CODE_EXAMPLES.md` Part 3, Improvement 1

2. **Verify Tensor→Float Conversion**
   - File: `tests/test_engines_handlers.py`
   - Function: `test_handler_converts_tensor_to_float`
   - See: `TEST_RECOMMENDATIONS_CODE_EXAMPLES.md` Part 3, Improvement 2

3. **Rename Tests for Clarity**
   - File: `tests/test_preprocessing_cropping.py`
   - Rename: `test_crop_with_precomputed_mask` → `test_crop_uses_provided_mask_avoids_recomputation`
   - Rename: `test_crop_empty_image` → `test_crop_all_zero_volume_returns_minimal_bbox`

---

## Implementation Roadmap

### Week 1: Critical Tests
- Monday-Tuesday: End-to-end planning test + corrupted data handling
- Wednesday: GPU memory constraints + splits determinism
- Thursday-Friday: Review, refine, ensure all pass

### Week 2: Consolidation & Cleanup
- Monday: Consolidate sliding window tests
- Tuesday: Consolidate file type detection, remove formatting tests
- Wednesday: Improve weak assertions and test names
- Thursday-Friday: Full test suite run, coverage check

### Week 3: Final Validation
- Monday-Wednesday: Integration testing with full suite
- Thursday: Documentation and handoff
- Friday: Buffer for fixes and edge cases

---

## Quick Reference: File-by-File Changes

### tests/test_planning.py
- **ADD**: `TestPlanningWorkflowIntegration` class with 2 tests
- **ADD**: `test_create_plan_detects_inconsistent_dimensionality` test
- **ADD**: `@pytest.mark.parametrize` GPU memory tests to `TestExperimentPlanner`
- **FIX**: `TestModelTopology::test_anisotropic_pooling` weak assertion
- **Total lines**: +150-200

### tests/test_monai_integration.py
- **ADD**: 2 tests to `TestFingerprintDatasetErrorHandling` class
- **CONSOLIDATE**: `TestDetectFileType` from 16 cases → 5 cases
- **Total lines**: +80, -30

### tests/test_inference_strategies.py
- **CONSOLIDATE**: `TestSlidingWindowVolumeVariations` from 15 tests → 3 parametrized tests
- **REMOVE**: `test_output_values_range` (15 lines)
- **Total lines**: -200, +80 (net -120)

### tests/test_engines_handlers.py
- **IMPROVE**: `test_handler_converts_tensor_to_float` to verify conversion
- **Total lines**: +10

### tests/test_results.py
- **REMOVE**: `test_print_results_formatting` (30 lines)
- **REMOVE**: `test_print_results_separator_lines` (14 lines)
- **Total lines**: -44

### tests/test_splits.py
- **ADD**: `TestSplitsDeterminism` class with 2 tests
- **Total lines**: +100

---

## Success Criteria

After implementation, verify:

- ✅ All 411+ existing tests still pass
- ✅ 8-10 new critical tests added
- ✅ Test runtime reduced by 15-30% (consolidation)
- ✅ No increase in test maintenance burden
- ✅ All weak assertions strengthened
- ✅ No cosmetic/formatting tests remain

---

## Integration Commands

```bash
# Run full test suite to baseline current state
uv run pytest --tb=short -v

# Run specific test category
uv run pytest tests/test_planning.py -v
uv run pytest tests/test_inference_strategies.py::TestSlidingWindowInferer -v

# Check coverage before/after
uv run pytest --cov=src --cov-report=term-missing

# Validate YAML configs
uv run pytest tests/test_config.py -v
```

---

## Dependencies & Compatibility

All recommended tests use existing test infrastructure:
- pytest (already used)
- unittest.mock (already used)
- fixtures in conftest.py (ready)
- No new external dependencies needed

---

## Documentation to Update

After implementing tests:

1. Update `CLAUDE.md` to mention test consolidation in "Testing Requirements"
2. Add note about deterministic splits verification
3. Document GPU memory constraint handling in training docs
4. Update "Key Terminology" if needed for new tests

---

## Common Pitfalls to Avoid

- ❌ Don't add tests that just verify library behavior (MONAI, PyTorch)
- ❌ Don't use `pytest.skip()` in exception handlers (use `xfail` or remove)
- ❌ Don't parametrize more than 3-4 variants per test (creates maintenance burden)
- ❌ Don't test internal state; test public behavior
- ✅ Do test error paths, not just happy paths
- ✅ Do name tests to describe what they test, not how
- ✅ Do verify both that tests pass AND what they assert

---

## Contact & Questions

For clarification on any recommendation, see:
- **Detailed Analysis**: `TEST_SUITE_ANALYSIS.md`
- **Code Examples**: `TEST_RECOMMENDATIONS_CODE_EXAMPLES.md`
- **This Guide**: `TEST_OPTIMIZATION_QUICK_START.md`

---

**Ready to start?** Begin with Critical Missing Tests (Section 1) - they take ~1 week and provide maximum impact.

**Timeline**: 4-5 weeks total for complete optimization
**Effort**: 1 developer, part-time (can be done in parallel with other work)
**Benefit**: 30% faster tests + higher confidence + better coverage

Good luck! 🚀

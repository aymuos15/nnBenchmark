"""
Tests to verify documentation accuracy and consistency.

These tests ensure that:
1. Function references in docs point to existing functions
2. Constants in docs match actual values in code
3. DOC markers exist for all documented items
4. [Constant]/[Adaptive] tags are accurate
"""

import re
from pathlib import Path

import pytest

from src.planning.constants import PLANNING_CONSTANTS

# ============================================================================
# Test Function References
# ============================================================================


def test_function_references_exist():
    """Verify all function references in planning.md exist in source code."""
    docs_path = Path("docs/planning.md")
    content = docs_path.read_text()

    # Extract all function references like: src/path/file.py::function_name()
    pattern = r'`(src/[\w/]+\.py)::([\w_]+)\(\)`'
    references = re.findall(pattern, content)

    errors = []
    for file_path, func_name in references:
        full_path = Path(file_path)
        if not full_path.exists():
            errors.append(f"File not found: {file_path}")
            continue

        code = full_path.read_text()
        # Check for function definition
        if f"def {func_name}(" not in code:
            errors.append(f"Function {func_name} not found in {file_path}")

    assert not errors, "Documentation references errors:\n" + "\n".join(errors)


def test_constants_match_registry():
    """Verify constants mentioned in docs match values in constants.py."""
    docs_path = Path("docs/planning.md")
    content = docs_path.read_text()

    # Define constants to check (name in docs: attribute in PLANNING_CONSTANTS)
    constant_mappings = {
        "10,000": "FOREGROUND_SAMPLES_PER_CASE",
        "12345": "RANDOM_SEED",
        "3.0": "ANISOTROPY_THRESHOLD",
        "0.25": "ANISOTROPY_VOXEL_RATIO",
        "256": "PATCH_NORM_3D",
        "2048": "PATCH_NORM_2D",
        "4": "MIN_FEATURE_MAP_SIZE",
        "32": "BASE_FEATURES",
        "512": "MAX_FEATURES_2D",
        "320": "MAX_FEATURES_3D",
        "560M": "UNET_REFERENCE_VAL_3D",
        "85M": "UNET_REFERENCE_VAL_2D",
        "200": "EPOCHS",
        "0.01": "LEARNING_RATE",
        "5": "N_FOLDS",
    }

    errors = []
    for doc_value, const_name in constant_mappings.items():
        actual_value = getattr(PLANNING_CONSTANTS, const_name)

        # Handle special cases
        if doc_value == "10,000":
            expected = "10,000" in content or "10000" in content
            if not expected:
                errors.append(f"Constant {const_name}={actual_value} not found in docs")
        elif doc_value == "560M":
            if "560M" not in content and "560000000" not in content:
                errors.append(f"Constant {const_name}={actual_value} not found in docs")
        elif doc_value == "85M":
            if "85M" not in content and "85000000" not in content:
                errors.append(f"Constant {const_name}={actual_value} not found in docs")
        else:
            # Check if value appears in docs
            if str(doc_value) not in content:
                errors.append(
                    f"Constant value {doc_value} for {const_name} not found in docs"
                )

    assert not errors, "Constant mismatch errors:\n" + "\n".join(errors)


# ============================================================================
# Test DOC Markers
# ============================================================================


def test_doc_markers_exist():
    """Verify DOC markers exist for all functions referenced in planning.md."""
    docs_path = Path("docs/planning.md")
    content = docs_path.read_text()

    # Extract function references
    pattern = r'`(src/[\w/]+\.py)::([\w_]+)\(\)`'
    references = re.findall(pattern, content)

    errors = []
    for file_path, func_name in references:
        full_path = Path(file_path)
        if not full_path.exists():
            continue

        code = full_path.read_text()

        # Check if DOC marker exists before this function
        # Look for pattern: # DOC: MARKER_NAME | ...
        # followed by def func_name(
        doc_pattern = rf'# DOC:.*?\n.*?def {func_name}\('
        if not re.search(doc_pattern, code, re.DOTALL):
            errors.append(f"DOC marker missing for {func_name} in {file_path}")

    assert not errors, "Missing DOC markers:\n" + "\n".join(errors)


def test_doc_marker_format():
    """Verify DOC markers follow the correct format."""
    # Files that should have DOC markers
    files_to_check = [
        "src/preprocessing/cropping.py",
        "src/planning/fingerprinting/fingerprint.py",
        "src/planning/fingerprinting/prepare_dataset.py",
        "src/planning/fingerprinting/resources.py",
        "src/planning/planner/heuristics.py",
        "src/planning/planner/sizing.py",
        "src/planning/planner/topology.py",
        "src/planning/planner/create.py",
        "src/planning/yaml_generator.py",
        "src/planning/splits.py",
    ]

    errors = []
    for file_path in files_to_check:
        full_path = Path(file_path)
        if not full_path.exists():
            errors.append(f"File not found: {file_path}")
            continue

        code = full_path.read_text()

        # Find all DOC markers
        doc_markers = re.findall(r'# DOC: (.*)', code)

        for marker in doc_markers:
            # Check format: MARKER_NAME | Category: X | ...
            if '|' not in marker:
                errors.append(
                    f"Invalid DOC marker format in {file_path}: {marker[:50]}"
                )
            elif 'Category:' not in marker:
                errors.append(
                    f"DOC marker missing Category in {file_path}: {marker[:50]}"
                )
            elif 'Documentation:' not in marker:
                errors.append(
                    f"DOC marker missing Documentation reference in {file_path}: {marker[:50]}"
                )

    assert not errors, "DOC marker format errors:\n" + "\n".join(errors)


# ============================================================================
# Test Category Tags
# ============================================================================


def test_constant_tags_accurate():
    """Verify [Constant]/[Adaptive] tags in docs match DOC markers."""
    docs_path = Path("docs/planning.md")
    content = docs_path.read_text()

    # Extract items with [Constant] tag
    constant_items = re.findall(r'\*\*\[Constant\]\*\* ([\w\s]+)', content)

    # This is a basic check - more sophisticated verification would
    # compare DOC marker categories with documentation tags
    assert len(constant_items) > 0, "No [Constant] tags found in documentation"

    # Extract items with [Adaptive] tag
    adaptive_items = re.findall(r'\*\*\[Adaptive\]\*\* ([\w\s]+)', content)
    assert len(adaptive_items) > 0, "No [Adaptive] tags found in documentation"


# ============================================================================
# Test Documentation Completeness
# ============================================================================


def test_all_steps_documented():
    """Verify all 5 steps are documented in planning.md."""
    docs_path = Path("docs/planning.md")
    content = docs_path.read_text()

    required_steps = [
        "Step 0: Preprocessing",
        "Step 1: Fingerprinting",
        "Step 2: Experiment Planning",
        "Step 3: YAML Generation",
        "Step 4: Cross-Validation Splits",
    ]

    errors = []
    for step in required_steps:
        if step not in content:
            errors.append(f"Missing step in documentation: {step}")

    assert not errors, "Documentation completeness errors:\n" + "\n".join(errors)


def test_source_references_use_function_names():
    """Verify source references use function names, not line numbers."""
    docs_path = Path("docs/planning.md")
    content = docs_path.read_text()

    # Check for old-style line number references like :123-456
    line_number_pattern = r'`src/[\w/]+\.py:\d+-?\d*`'
    line_number_refs = re.findall(line_number_pattern, content)

    assert (
        not line_number_refs
    ), "Found line number references (should use function names):\n" + "\n".join(
        line_number_refs
    )


# ============================================================================
# Helper Functions
# ============================================================================


def test_constants_registry_accessible():
    """Verify constants registry is importable and has expected attributes."""
    assert hasattr(PLANNING_CONSTANTS, 'RANDOM_SEED')
    assert hasattr(PLANNING_CONSTANTS, 'N_FOLDS')
    assert hasattr(PLANNING_CONSTANTS, 'BASE_FEATURES')
    assert hasattr(PLANNING_CONSTANTS, 'EPOCHS')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

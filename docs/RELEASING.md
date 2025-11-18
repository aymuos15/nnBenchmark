# Release Process

This document describes the complete process for releasing a new version of nnBenchmark.

## Overview

The release process consists of the following major steps:

1. Update version numbers across the project
2. Update the CHANGELOG
3. Create a release commit
4. Create a git tag
5. Build the distribution package
6. Publish to PyPI (Test and/or Production)
7. Push to remote repository
8. Create a GitHub Release

## Detailed Steps

### 1. Pre-Release Verification

Before starting the release process, run through this comprehensive checklist to ensure everything is ready:

#### 1.1 Version String Consistency Check

**CRITICAL**: Version strings must be consistent across ALL three files. Inconsistent versions will cause confusion and installation issues.

Check these files contain the SAME version number:

```bash
# Check current versions
grep "^version = " pyproject.toml
grep "^__version__ = " src/utils/__init__.py
grep "^__version__ = " src/__init__.py
```

All three should show the current version (e.g., "0.1.8"). If they differ, this is a critical bug that must be fixed immediately.

#### 1.2 CHANGELOG Structure Validation

**CRITICAL**: Verify the CHANGELOG.md follows the [Keep a Changelog](https://keepachangelog.com/) format correctly:

```bash
# View top of CHANGELOG
head -20 CHANGELOG.md
```

**Required structure**:
```markdown
# Changelog

## [Unreleased]
### Added
- ...

## [0.1.8] - 2025-01-17
### Changed
- ...
```

**Common mistake**: The `[Unreleased]` section must be at the TOP, before all versioned releases. If it's in the wrong position, the release process will fail.

#### 1.3 Documentation Sync Verification

**HIGH PRIORITY**: When code structure changes, documentation must be updated to match. Check for discrepancies:

```bash
# Search for old file path patterns that might be outdated
grep -r "checkpoint_epoch_" docs/
grep -r "training_history.json" docs/
grep -r "test_history.json" docs/

# Check README for outdated examples
grep -A5 "results/" README.md
```

**Files to check**:
- `docs/src/results.md` - Result directory structure examples
- `docs/src/checkpointing.md` - Checkpoint file path references
- `README.md` - Workflow output examples
- `AGENT.md` - Any structure references

If any outdated patterns are found, update documentation to reflect current code behavior.

#### 1.4 Lock File Update Check

**IMPORTANT**: If using `uv` for dependency management, the lock file must be updated:

```bash
# Check if uv.lock version matches pyproject.toml
grep "^version = " pyproject.toml
grep "^version = " uv.lock | head -1
```

If versions differ, run `uv lock` after updating `pyproject.toml` to sync the lock file.

#### 1.5 Migration Guide Consideration

**For BREAKING CHANGES only**: If this release contains breaking changes, consider whether users need migration guidance:

```bash
# Check if CHANGELOG contains breaking changes
grep -i "breaking" CHANGELOG.md
```

If breaking changes exist, consider:
- Adding a migration section to the CHANGELOG
- Creating a `MIGRATION.md` file
- Documenting upgrade path for users

#### 1.6 Pre-Release Summary

Run this command to get an overview of what will be released:

```bash
# See all changes since last release
git log $(git describe --tags --abbrev=0)..HEAD --oneline

# See all modified files
git status

# Check for uncommitted changes
git diff --stat
```

**Before proceeding**: Ensure all verification checks pass. Do not continue with the release if critical issues are found.

---

### 2. Plan the Release

After verification, plan the release details:

- The new version number (following [Semantic Versioning](https://semver.org/))
- What changes to include (already verified in step 1.6)
- The release date (typically today)

Example:
```bash
git log v0.1.8..HEAD --oneline
```

### 3. Update Version Numbers

Version numbers must be updated in **three** locations to maintain consistency:

#### Update `pyproject.toml`
Change the version in the `[project]` section:
```toml
[project]
name = "nnbenchmark"
version = "0.2.0"  # Update this
```

#### Update `src/utils/__init__.py`
Change the `__version__` variable:
```python
__version__ = "0.2.0"  # Update this
```

#### Update `src/__init__.py`
Change the `__version__` variable:
```python
__version__ = "0.2.0"  # Update this
```

**Verify consistency**:
```bash
grep "^version = " pyproject.toml
grep "^__version__ = " src/utils/__init__.py
grep "^__version__ = " src/__init__.py
```

All three should now show "0.2.0".

### 4. Update CHANGELOG.md

The CHANGELOG.md should already have an `[Unreleased]` section at the top with documented changes. Transform this into the release section:

#### Step 1: Rename the [Unreleased] section

Change:
```markdown
## [Unreleased]
```

To:
```markdown
## [0.2.0] - 2025-11-18
```

#### Step 2: Add a new empty [Unreleased] section

At the very top of the changelog (after the header), add:
```markdown
## [Unreleased]

```

**Final structure should be**:
```markdown
# Changelog

## [Unreleased]

## [0.2.0] - 2025-11-18

### BREAKING CHANGES
- Major change that breaks compatibility

### Added
- New feature 1
- New feature 2

### Fixed
- Bug fix 1

### Changed
- Change 1

## [0.1.8] - 2025-01-17
...
```

**Guidelines for changelog entries** (when adding to [Unreleased] during development):
- Use clear, descriptive language
- Group changes by category (BREAKING CHANGES, Added, Changed, Fixed, Removed)
- Focus on what changed and why, not implementation details
- Include new files or documentation as separate items
- For breaking changes, explain the impact and migration path

### 5. Update Lock File (if using uv)

If the project uses `uv` for dependency management, update the lock file:

```bash
uv lock
```

This syncs `uv.lock` with the new version in `pyproject.toml`.

**Verify**:
```bash
grep "^version = " uv.lock | head -1
```

Should show "0.2.0".

### 6. Create Release Commit

Stage and commit all version updates and any documentation changes:

```bash
# Stage all modified files related to the release
git add pyproject.toml src/utils/__init__.py src/__init__.py CHANGELOG.md uv.lock
git add docs/  # If documentation was updated
git add README.md AGENT.md  # If these were updated

git commit -m "$(cat <<'EOF'
chore: release v0.2.0

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Verify the commit was created:
```bash
git log --oneline -1
```

### 7. Create Git Tag

Create a tag pointing to the release commit:

```bash
git tag v0.2.0

# Verify the tag
git tag -l -n1 v0.2.0
```

### 8. Build Distribution Package

Build both source distribution (sdist) and wheel:

```bash
python -m build
```

This creates:
- `dist/nnbenchmark-0.2.0.tar.gz` (source distribution)
- `dist/nnbenchmark-0.2.0-py3-none-any.whl` (wheel)

Verify the build:
```bash
ls -lh dist/nnbenchmark-0.2.0*
```

### 9. Publish to PyPI

#### Prerequisites

Create a `~/.pypirc` file with your PyPI credentials:

```ini
[distutils]
index-servers =
    testpypi
    pypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-<your-test-token>

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-<your-production-token>
```

Ensure proper permissions:
```bash
chmod 600 ~/.pypirc
```

**⚠️ Security Warning**: Never expose these tokens in public repositories or commit them to git. Consider using [trusted publishing](https://docs.pypi.org/trusted-publishers/) with GitHub Actions instead.

#### Publish to Test PyPI (Recommended First)

**ALWAYS** publish to Test PyPI first to verify the package:

```bash
python -m twine upload --repository testpypi dist/nnbenchmark-0.2.0*
```

View the uploaded package:
- Test PyPI: https://test.pypi.org/project/nnbenchmark/0.2.0/

**Test the installation**:
```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate

# Install from Test PyPI
pip install -i https://test.pypi.org/simple/ nnbenchmark==0.2.0

# Verify it works
python -c "import nnbenchmark; print(nnbenchmark.__version__)"

# Clean up
deactivate
rm -rf test_env
```

#### Publish to Production PyPI (Optional)

**CAUTION**: Only publish to production PyPI for official releases. Test PyPI is sufficient for testing and development.

Once verified on Test PyPI and ready for official release:

```bash
python -m twine upload dist/nnbenchmark-0.2.0*
```

View the package:
- PyPI: https://pypi.org/project/nnbenchmark/0.2.0/

### 10. Push to Remote Repository

Push the commit and tag to GitHub:

```bash
git push origin master
git push origin v0.2.0
```

Verify:
- Commit appears on GitHub: https://github.com/aymuos15/nnBenchmark/commit/<commit-hash>
- Tag appears on GitHub: https://github.com/aymuos15/nnBenchmark/releases/tag/v0.2.0

### 11. Create GitHub Release

Create a formal GitHub Release with the changelog:

```bash
gh release create v0.2.0 \
  --title "v0.2.0" \
  --notes "$(cat CHANGELOG.md | sed -n '/## \[0.2.0\]/,/## \[/p' | head -n -1)"
```

Or manually on GitHub:
1. Go to https://github.com/aymuos15/nnBenchmark/releases/new
2. Select tag: `v0.2.0`
3. Title: `v0.2.0`
4. Description: Copy the relevant section from CHANGELOG.md
5. Click "Publish release"

## Complete Release Checklist

### Pre-Release Verification
- [ ] **Version consistency check**: All three files have matching versions
  - [ ] `pyproject.toml`
  - [ ] `src/utils/__init__.py`
  - [ ] `src/__init__.py`
- [ ] **CHANGELOG structure**: `[Unreleased]` section is at the top
- [ ] **Documentation sync**: All docs reflect current code structure
  - [ ] `docs/src/results.md`
  - [ ] `docs/src/checkpointing.md`
  - [ ] `README.md`
  - [ ] `AGENT.md`
- [ ] **Lock file**: `uv.lock` version matches `pyproject.toml` (if using uv)
- [ ] **Breaking changes**: Migration guide added if needed
- [ ] **Pre-release summary**: Reviewed all changes since last release

### Version Updates
- [ ] Updated `pyproject.toml` version to v0.2.0
- [ ] Updated `src/utils/__init__.py` version to v0.2.0
- [ ] Updated `src/__init__.py` version to v0.2.0
- [ ] Verified version consistency across all three files

### CHANGELOG Updates
- [ ] Moved `[Unreleased]` section to top (if needed)
- [ ] Renamed `[Unreleased]` to `[0.2.0] - 2025-11-18`
- [ ] Added new empty `[Unreleased]` section at top
- [ ] Verified CHANGELOG structure is correct

### Documentation Updates (if needed)
- [ ] Updated `docs/src/results.md` with new file paths
- [ ] Updated `docs/src/checkpointing.md` with new checkpoint paths
- [ ] Updated `README.md` workflow examples
- [ ] Updated `AGENT.md` structure references

### Lock File
- [ ] Ran `uv lock` to update lock file (if using uv)
- [ ] Verified lock file version matches v0.2.0

### Release Process
- [ ] Created release commit (`chore: release v0.2.0`)
- [ ] Created git tag (`v0.2.0`)
- [ ] Built distribution (`python -m build`)
- [ ] Published to Test PyPI (`twine upload --repository testpypi`)
- [ ] Tested installation from Test PyPI in clean environment
- [ ] Verified package version (`python -c "import nnbenchmark; print(nnbenchmark.__version__)"`)
- [ ] Published to Production PyPI (OPTIONAL - only for official releases)
- [ ] Pushed commit to GitHub (`git push origin master`)
- [ ] Pushed tag to GitHub (`git push origin v0.2.0`)
- [ ] Created GitHub Release with changelog (`gh release create v0.2.0`)

## Troubleshooting

### Build Fails
Ensure you have the required build tools:
```bash
pip install build twine
```

### PyPI Upload Fails (403 Forbidden)
The token may be invalid or expired:
1. Generate a new token at https://test.pypi.org/account/tokens/ or https://pypi.org/account/tokens/
2. Update ~/.pypirc with the new token
3. Retry the upload

### Tag Already Exists
If the tag was already pushed but you need to retag:
```bash
git tag -d v0.1.5  # Delete local tag
git push origin :refs/tags/v0.1.5  # Delete remote tag
git tag v0.1.5  # Create new tag
git push origin v0.1.5
```

## References

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [PyPI Help - Invalid Authentication](https://test.pypi.org/help/#invalid-auth)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub CLI - Release Creation](https://cli.github.com/manual/gh_release_create)

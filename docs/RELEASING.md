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

### 1. Plan the Release

Before starting, determine:
- The new version number (following [Semantic Versioning](https://semver.org/))
- What changes to include (run `git log <old-tag>..HEAD` to see commits since last release)
- The release date (typically today)

Example:
```bash
git log v0.1.2..HEAD --oneline
```

### 2. Update Version Numbers

Version numbers must be updated in two locations to maintain consistency:

#### Update `pyproject.toml`
Change the version in the `[project]` section:
```toml
[project]
name = "nnbenchmark"
version = "0.1.3"  # Update this
```

#### Update `src/utils/__init__.py`
Change the `__version__` variable:
```python
__version__ = "0.1.3"  # Update this
```

### 3. Update CHANGELOG.md

Add a new section at the top of `CHANGELOG.md` following the [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [0.1.3] - 2025-11-03

### Added
- Feature 1 description
- Feature 2 description

### Changed
- Change 1 description
- Change 2 description

### Fixed
- Bug fix 1 description
- Bug fix 2 description

### Removed
- Removed item 1 description
```

**Guidelines for changelog entries:**
- Use clear, descriptive language
- Group changes by category (Added, Changed, Fixed, Removed)
- Focus on what changed and why, not implementation details
- Include new files or documentation as separate items

### 4. Create Release Commit

Stage and commit all version updates:

```bash
git add pyproject.toml src/utils/__init__.py CHANGELOG.md

git commit -m "$(cat <<'EOF'
chore: release v0.1.3

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

Verify the commit was created:
```bash
git log --oneline -1
```

### 5. Create Git Tag

Create a signed tag pointing to the release commit:

```bash
git tag v0.1.3

# Verify the tag
git tag -l -n1 v0.1.3
```

### 6. Build Distribution Package

Build both source distribution (sdist) and wheel:

```bash
python -m build
```

This creates:
- `dist/nnbenchmark-0.1.3.tar.gz` (source distribution)
- `dist/nnbenchmark-0.1.3-py3-none-any.whl` (wheel)

Verify the build:
```bash
ls -lh dist/nnbenchmark-0.1.3*
```

### 7. Publish to PyPI

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

```bash
python -m twine upload --repository testpypi dist/nnbenchmark-0.1.3*
```

View the uploaded package:
- Test PyPI: https://test.pypi.org/project/nnbenchmark/0.1.3/

**Test the installation**:
```bash
pip install -i https://test.pypi.org/simple/ nnbenchmark==0.1.3
```

#### Publish to Production PyPI

Once verified on Test PyPI:

```bash
python -m twine upload dist/nnbenchmark-0.1.3*
```

View the package:
- PyPI: https://pypi.org/project/nnbenchmark/0.1.3/

### 8. Push to Remote Repository

Push the commit and tag to GitHub:

```bash
git push origin master
git push origin v0.1.3
```

Verify:
- Commit appears on GitHub: https://github.com/aymuos15/nnBenchmark/commit/<commit-hash>
- Tag appears on GitHub: https://github.com/aymuos15/nnBenchmark/releases/tag/v0.1.3

### 9. Create GitHub Release

Create a formal GitHub Release with the changelog:

```bash
gh release create v0.1.3 \
  --title "v0.1.3" \
  --notes "$(cat CHANGELOG.md | sed -n '/## \[0.1.3\]/,/## \[/p' | head -n -1)"
```

Or manually on GitHub:
1. Go to https://github.com/aymuos15/nnBenchmark/releases/new
2. Select tag: `v0.1.3`
3. Title: `v0.1.3`
4. Description: Copy the relevant section from CHANGELOG.md
5. Click "Publish release"

## Complete Release Checklist

- [ ] Planned release (identified changes)
- [ ] Updated `pyproject.toml` version
- [ ] Updated `src/utils/__init__.py` version
- [ ] Updated CHANGELOG.md with release notes
- [ ] Created release commit
- [ ] Created git tag (v0.1.3)
- [ ] Built distribution (`python -m build`)
- [ ] Published to Test PyPI and verified
- [ ] Tested installation from Test PyPI
- [ ] Published to Production PyPI
- [ ] Pushed commit to GitHub (`git push origin master`)
- [ ] Pushed tag to GitHub (`git push origin v0.1.3`)
- [ ] Created GitHub Release with changelog

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
git tag -d v0.1.3  # Delete local tag
git push origin :refs/tags/v0.1.3  # Delete remote tag
git tag v0.1.3  # Create new tag
git push origin v0.1.3
```

## References

- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [PyPI Help - Invalid Authentication](https://test.pypi.org/help/#invalid-auth)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub CLI - Release Creation](https://cli.github.com/manual/gh_release_create)

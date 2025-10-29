# Codecov Setup Instructions

To enable coverage reporting in your CI pipeline, follow these steps:

## 1. Sign up for Codecov

1. Go to [codecov.io](https://codecov.io)
2. Sign in with your GitHub account
3. Grant Codecov access to your repository

## 2. Get Your Codecov Token

1. Navigate to your repository on Codecov
2. Go to Settings > General
3. Copy the repository upload token

## 3. Add Token to GitHub Secrets

1. Go to your GitHub repository
2. Click on **Settings** > **Secrets and variables** > **Actions**
3. Click **New repository secret**
4. Name: `CODECOV_TOKEN`
5. Value: Paste your Codecov token
6. Click **Add secret**

## 4. Verify Setup

1. Create a pull request or push to trigger the CI workflow
2. Check that the workflow runs successfully
3. Visit your Codecov dashboard to see coverage reports

## Optional: Add Codecov Badge to README

Add this to your README.md to display coverage status:

```markdown
[![codecov](https://codecov.io/gh/YOUR_USERNAME/nnBenchmark/branch/master/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/nnBenchmark)
```

Replace `YOUR_USERNAME` with your GitHub username.

## Troubleshooting

- If coverage upload fails, the CI will continue (it's configured with `fail_ci_if_error: false`)
- Make sure your repository is public or you have a Codecov Pro account for private repos
- Check GitHub Actions logs for detailed error messages

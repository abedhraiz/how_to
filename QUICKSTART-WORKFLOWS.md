# 🚀 How to Run GitHub Actions in This Repository

This guide shows you how to run the automated workflows in this repository. No coding required!

---

## Option 1: Run Workflows Manually (Easiest!)

### Step-by-Step:

1. **Go to the Actions Tab**
   - Visit: https://github.com/abedhraiz/how_to/actions
   - Or click "Actions" at the top of this repository

2. **Select a Workflow**
   - Click on any workflow from the left sidebar:
     - ✅ Markdown Lint
     - 🔗 Link Checker
     - 📚 Documentation Validation
     - 📝 Spell Check
     - 📖 Generate Table of Contents

3. **Run the Workflow**
   - Click the "Run workflow" dropdown button (right side)
   - Select branch: `main`
   - Click the green "Run workflow" button
   - Watch it run in real-time! ⏱️

4. **View Results**
   - Click on the running workflow to see live logs
   - ✅ Green checkmark = Success
   - ❌ Red X = Failed (check logs for details)

### 🎥 Visual Guide:

```
GitHub Repo → Actions Tab → Workflow Name → Run workflow ▼ → Select branch → Run workflow
```

---

## Option 2: Trigger Workflows Automatically

### Create a Pull Request

Workflows automatically run when you:

1. **Fork this repository**
   - Click "Fork" at the top right

2. **Make a change**
   - Edit any `.md` file
   - Example: Fix a typo in README.md

3. **Create a Pull Request**
   - Go to your fork
   - Click "Pull request"
   - Create PR to `main` branch

4. **Watch the Magic! ✨**
   - ✅ Markdown Lint validates your changes
   - 🔗 Link Checker verifies links
   - 🏷️ Auto Label categorizes your PR
   - 📏 Size Labeler adds size label
   - 👋 Welcome message (if first-time contributor)

---

## Option 3: Run Locally with `act`

### For Advanced Users

1. **Install `act`**
   ```bash
   # macOS
   brew install act
   
   # Linux
   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
   
   # Windows
   choco install act-cli
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/abedhraiz/how_to.git
   cd how_to
   ```

3. **List available workflows**
   ```bash
   act -l
   ```

4. **Run a specific workflow**
   ```bash
   # Run documentation validation
   act push -W .github/workflows/docs-validation.yml
   
   # Run markdown lint
   act push -W .github/workflows/markdown-lint.yml
   
   # Dry run (don't actually execute)
   act -n
   ```

---

## 📚 Available Workflows

### 1. ✅ Markdown Lint
**File:** [`.github/workflows/markdown-lint.yml`](.github/workflows/markdown-lint.yml)

**What it does:** Checks markdown files for formatting issues

**Triggers:**
- Push to `main` with `.md` changes
- Pull requests with `.md` changes
- Manual run

**Try it:**
```bash
# Manual: Actions → Markdown Lint → Run workflow
# Auto: Edit any .md file and create PR
```

---

### 2. 🔗 Link Checker
**File:** [`.github/workflows/link-checker.yml`](.github/workflows/link-checker.yml)

**What it does:** Validates all links in markdown files aren't broken

**Triggers:**
- Push to `main`
- Pull requests
- Every Monday at midnight (scheduled)
- Manual run

**Bonus:** Creates a GitHub issue if broken links are found!

**Try it:**
```bash
# Manual: Actions → Link Checker → Run workflow
```

---

### 3. 📚 Documentation Validation
**File:** [`.github/workflows/docs-validation.yml`](.github/workflows/docs-validation.yml)

**What it does:**
- ✅ Verifies required files exist
- 🔍 Checks for broken internal links
- 📁 Validates directory structure
- 📊 Generates documentation statistics

**Triggers:**
- Push to `main` with docs changes
- Pull requests with docs changes
- Manual run

**Example output:**
```
✅ All required files exist
✅ No broken internal links found
✅ All required directories exist
📊 Documentation Statistics
========================
Total markdown files: 73
Total lines of documentation: 12,847
```

**Try it:**
```bash
# Manual: Actions → Documentation Validation → Run workflow
```

---

### 4. 📝 Spell Check
**File:** [`.github/workflows/spellcheck.yml`](.github/workflows/spellcheck.yml)

**What it does:** Checks spelling in all markdown files

**Triggers:**
- Push with `.md` changes
- Pull requests with `.md` changes
- Manual run

**Configuration:** [`.cspell.json`](.cspell.json)

**Add custom words:**
Edit `.cspell.json` and add to the `words` array:
```json
{
  "words": [
    "kubernetes",
    "your-custom-word"
  ]
}
```

---

### 5. 🏷️ Auto Label PRs
**File:** [`.github/workflows/auto-label-pr.yml`](.github/workflows/auto-label-pr.yml)

**What it does:** Automatically labels pull requests based on changed files

**Triggers:** Pull request opened/updated

**Labels applied:**
- `documentation` - Markdown changes
- `infrastructure` - Infrastructure docs
- `cloud` - Cloud platform docs
- `data-engineering` - Data engineering docs
- `ci-cd` - CI/CD docs
- `monitoring` - Monitoring docs
- `databases` - Database docs
- `version-control` - Version control docs
- `ai-ml` - AI/ML docs
- `github-actions` - Workflow changes

**Try it:** Create a PR that modifies `docs/cloud-platforms/` and watch it get labeled!

---

### 6. 📏 PR Size Labeler
**File:** [`.github/workflows/pr-size-labeler.yml`](.github/workflows/pr-size-labeler.yml)

**What it does:** Labels PRs by size for easier review

**Triggers:** Pull request opened/updated

**Size categories:**
| Label | Lines Changed |
|-------|---------------|
| `size/XS` | < 10 lines |
| `size/S` | < 50 lines |
| `size/M` | < 250 lines |
| `size/L` | < 1000 lines |
| `size/XL` | 1000+ lines |

---

### 7. 📖 Generate Table of Contents
**File:** [`.github/workflows/generate-toc.yml`](.github/workflows/generate-toc.yml)

**What it does:** Auto-generates table of contents in README files

**Triggers:**
- Push to `main` with docs changes
- Manual run

**How to use in your files:**
1. Add these markers to any markdown file:
   ```markdown
   <!-- toc -->
   <!-- tocstop -->
   ```

2. Push changes or run workflow manually

3. TOC is automatically generated and committed!

---

### 8. 👋 Welcome First-Time Contributors
**File:** [`.github/workflows/welcome-first-time.yml`](.github/workflows/welcome-first-time.yml)

**What it does:** Posts a friendly welcome message for first-time contributors

**Triggers:** First pull request or issue from new contributor

**Message includes:**
- Thank you note
- Contribution guidelines
- Next steps
- Where to get help

---

## 🎓 Quick Testing Guide

### Test 1: Run Markdown Lint (30 seconds)
1. Go to [Actions](https://github.com/abedhraiz/how_to/actions)
2. Click "Markdown Lint" (left sidebar)
3. Click "Run workflow" → "Run workflow"
4. Watch it complete (should take ~20-30 seconds)
5. Click on the workflow run to see detailed logs

✅ **Expected result:** Green checkmark

---

### Test 2: Create a Test PR (2 minutes)
1. Fork this repository
2. Edit `README.md` (add a space somewhere)
3. Commit: "test: triggering workflows"
4. Create pull request to main branch
5. Watch multiple workflows run:
   - Markdown Lint
   - Link Checker
   - Auto Label PR (adds `documentation` label)
   - PR Size Labeler (adds `size/XS` label)

✅ **Expected result:** All checks pass, labels automatically added

---

### Test 3: Check Documentation Stats (1 minute)
1. Go to [Actions](https://github.com/abedhraiz/how_to/actions)
2. Click "Documentation Validation"
3. Click "Run workflow" → "Run workflow"
4. Wait for completion (~30 seconds)
5. Click on workflow run
6. Click on "validate-structure" job
7. Scroll to "Generate documentation stats" step

✅ **Expected result:** See statistics like:
```
📊 Documentation Statistics
Total markdown files: 73
Total lines of documentation: 12,847
```

---

## 🐛 Troubleshooting

### Workflow doesn't start?
- ✅ Check if Actions are enabled: Settings → Actions → Allow all actions
- ✅ Verify you're on the `main` branch
- ✅ Wait a few seconds and refresh

### Workflow failed?
- 📋 Click on the failed workflow to see logs
- 🔍 Look for the specific step that failed
- 📖 Read the error message (usually very clear)
- 🔧 Fix the issue and try again

### Can't find "Run workflow" button?
- Some workflows don't have manual triggers
- They only run on push/PR automatically
- Check the workflow file for `workflow_dispatch` trigger

### Permission denied errors?
- Fork the repo if you're not a maintainer
- Workflows in forks may have limited permissions
- Some workflows require write access

---

## 📚 Learn More

- **[Workflows Documentation](.github/workflows/README.md)** - Detailed workflow guide
- **[GitHub Actions Guide](docs/cicd-automation/github-actions/github-actions-guide.md)** - Complete tutorial
- **[GitHub Actions Docs](https://docs.github.com/en/actions)** - Official documentation

---

## 🤝 Contributing

Want to add a new workflow?

1. Create a new `.yml` file in `.github/workflows/`
2. Test locally with `act` (optional)
3. Create a pull request
4. Update this guide with your new workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## 💡 Pro Tips

1. **Watch a workflow run in real-time**
   - Click on a running workflow
   - See live log output
   - Perfect for learning how workflows work!

2. **Use workflow badges in your README**
   ```markdown
   ![Workflow Name](https://github.com/abedhraiz/how_to/workflows/Workflow%20Name/badge.svg)
   ```

3. **Test workflows before pushing**
   - Use `act` to run workflows locally
   - Catch errors before creating commits
   - Faster iteration cycle

4. **Copy these workflows to your repos**
   - All workflows are reusable
   - Just copy the `.yml` file
   - Adjust paths and settings as needed

---

**Last Updated:** December 2024  
**Questions?** Open an [issue](https://github.com/abedhraiz/how_to/issues) or [discussion](https://github.com/abedhraiz/how_to/discussions)

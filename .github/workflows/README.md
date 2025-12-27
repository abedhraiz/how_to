# GitHub Actions Workflows

This directory contains automated workflows for maintaining and validating this documentation repository.

## 🚀 Active Workflows

### 1. **Markdown Lint** (`markdown-lint.yml`)
**Trigger:** Push/PR to main with `.md` changes, Manual  
**Purpose:** Ensures markdown files follow best practices and consistent formatting

**What it does:**
- Validates markdown syntax
- Checks for common formatting issues
- Uses `.markdownlint.json` configuration

**How to run manually:**
1. Go to Actions tab
2. Select "Markdown Lint"
3. Click "Run workflow"

---

### 2. **Link Checker** (`link-checker.yml`)
**Trigger:** Push/PR, Weekly schedule (Mondays), Manual  
**Purpose:** Validates all links in documentation aren't broken

**What it does:**
- Checks all markdown links
- Validates internal and external links
- Creates GitHub issue if broken links found
- Runs weekly to catch external link rot

**How to run manually:**
1. Go to Actions tab
2. Select "Link Checker"
3. Click "Run workflow"

---

### 3. **Documentation Validation** (`docs-validation.yml`)
**Trigger:** Push/PR to docs, Manual  
**Purpose:** Validates documentation structure and integrity

**What it does:**
- Checks required files exist
- Validates directory structure
- Checks for broken internal links
- Generates documentation statistics

**Expected output:**
```
✅ All required files exist
✅ No broken internal links found
✅ All required directories exist
📊 Documentation Statistics
```

---

### 4. **Spell Check** (`spellcheck.yml`)
**Trigger:** Push/PR with `.md` changes, Manual  
**Purpose:** Catches spelling errors in documentation

**What it does:**
- Runs cspell on all markdown files
- Uses `.cspell.json` for custom dictionary
- Reports spelling issues

**How to add custom words:**
Edit `.cspell.json` and add words to the `words` array.

---

### 5. **Auto Label PRs** (`auto-label-pr.yml`)
**Trigger:** PR opened/updated  
**Purpose:** Automatically categorizes pull requests

**What it does:**
- Analyzes changed files
- Applies relevant labels:
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

---

### 6. **PR Size Labeler** (`pr-size-labeler.yml`)
**Trigger:** PR opened/updated  
**Purpose:** Labels PRs by size for easier review

**Size categories:**
- `size/XS` - < 10 lines
- `size/S` - < 50 lines
- `size/M` - < 250 lines
- `size/L` - < 1000 lines
- `size/XL` - 1000+ lines

---

### 7. **Generate Table of Contents** (`generate-toc.yml`)
**Trigger:** Push to main with docs changes, Manual  
**Purpose:** Auto-generates TOC in README files

**What it does:**
- Finds README.md files with TOC markers
- Generates/updates table of contents
- Auto-commits changes

**To enable TOC in a file:**
Add these markers to your markdown:
```markdown
<!-- toc -->
<!-- tocstop -->
```

---

### 8. **Welcome First-Time Contributors** (`welcome-first-time.yml`)
**Trigger:** First PR or issue from new contributor  
**Purpose:** Creates welcoming community atmosphere

**What it does:**
- Detects first-time contributors
- Posts welcome message
- Provides contribution guidelines
- Thanks contributors

---

## 📊 Workflow Status

Check the current status of all workflows:
- [View Actions Tab](https://github.com/abedhraiz/how_to/actions)

### Status Badges

Add these to your README.md to show workflow status:

```markdown
![Markdown Lint](https://github.com/abedhraiz/how_to/workflows/Markdown%20Lint/badge.svg)
![Link Checker](https://github.com/abedhraiz/how_to/workflows/Link%20Checker/badge.svg)
![Documentation Validation](https://github.com/abedhraiz/how_to/workflows/Documentation%20Validation/badge.svg)
```

## 🔧 Configuration Files

### `.markdownlint.json`
Configures markdown linting rules:
- `MD013: false` - Allows long lines (for tables/links)
- `MD033: false` - Allows inline HTML
- `MD041: false` - Doesn't require H1 as first line
- `MD024` - Allows duplicate headers in different sections

### `.cspell.json`
Spell check configuration with custom dictionary of technical terms.

### `.github/markdown-link-check-config.json`
Link checker configuration:
- Ignores localhost links
- Configures retry logic
- Sets timeout and status codes

## 🎯 Testing Workflows Locally

### Using `act` (Local GitHub Actions)

1. **Install act:**
```bash
# macOS
brew install act

# Linux
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
```

2. **Run workflows locally:**
```bash
# List workflows
act -l

# Run specific workflow
act push -W .github/workflows/docs-validation.yml

# Run with specific event
act pull_request -W .github/workflows/auto-label-pr.yml
```

### Manual Validation Scripts

Run these locally before pushing:

**Check markdown files:**
```bash
# Install markdownlint-cli
npm install -g markdownlint-cli

# Run lint
markdownlint '**/*.md' --config .markdownlint.json
```

**Check links:**
```bash
# Install markdown-link-check
npm install -g markdown-link-check

# Check all markdown files
find . -name "*.md" -not -path "./node_modules/*" -exec markdown-link-check {} \;
```

**Spell check:**
```bash
# Install cspell
npm install -g cspell

# Run spell check
cspell "**/*.md"
```

**Validate structure:**
```bash
# Run the validation script
bash .github/workflows/docs-validation.yml # (extract the bash script portion)
```

## 🚦 Workflow Triggers Reference

| Workflow | Push | PR | Schedule | Manual |
|----------|------|-----|----------|--------|
| Markdown Lint | ✅ | ✅ | ❌ | ✅ |
| Link Checker | ✅ | ✅ | ✅ Weekly | ✅ |
| Docs Validation | ✅ | ✅ | ❌ | ✅ |
| Spell Check | ✅ | ✅ | ❌ | ✅ |
| Auto Label PRs | ❌ | ✅ | ❌ | ❌ |
| PR Size Labeler | ❌ | ✅ | ❌ | ❌ |
| Generate TOC | ✅ | ❌ | ❌ | ✅ |
| Welcome First-Time | ❌ | ✅ | ❌ | ❌ |

## 📝 Adding New Workflows

1. Create a new `.yml` file in `.github/workflows/`
2. Define the workflow structure:
```yaml
name: My Workflow
on: [push, pull_request]
jobs:
  my-job:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: My step
        run: echo "Hello, World!"
```
3. Test locally with `act` if possible
4. Commit and push
5. Update this README with workflow details

## 🔒 Required Permissions

Most workflows use default `GITHUB_TOKEN` permissions. Some workflows require specific permissions:

- **Auto Label PRs**: `pull-requests: write`
- **PR Size Labeler**: `pull-requests: write`
- **Welcome First-Time**: `issues: write`, `pull-requests: write`
- **Generate TOC**: Requires push access (writes commits)

## 🐛 Troubleshooting

### Workflow Failed?
1. Check the Actions tab for detailed logs
2. Look for the specific step that failed
3. Check if required secrets/permissions are set
4. Verify file paths and configurations

### Common Issues

**Permission Denied:**
- Check workflow permissions in the YAML
- Verify repository settings allow Actions to create PRs/issues

**Action Not Found:**
- Verify action exists and version is correct
- Check for typos in `uses:` statements

**Timeout:**
- Some external link checks may timeout
- Adjust timeout in configuration files

## 📚 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)
- [act - Local GitHub Actions](https://github.com/nektos/act)

## 🤝 Contributing

To improve workflows:
1. Test changes locally with `act`
2. Create a PR with clear description
3. Ensure workflows pass on your branch
4. Document any new workflows in this README

---

**Last Updated:** December 2024  
**Maintainers:** Repository maintainers  
**Questions?** Open an issue or discussion

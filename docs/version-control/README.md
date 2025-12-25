# Version Control

## Purpose

Comprehensive guide to Git version control system - from basic commands to advanced workflows, branching strategies, and team collaboration patterns. Essential for modern software development.

## Technologies Covered

### Version Control Systems
- **[Git](./git/git-guide.md)** - Distributed version control system for tracking code changes

### Text Processing & Pattern Matching
- **[Regular Expressions (Regex)](./regex/regex-guide.md)** - Pattern matching for .gitignore, git grep, log parsing, and data validation

## Prerequisites

### Basic Requirements
- Command-line proficiency
- Text editor familiarity
- Understanding of file systems
- Basic programming concepts

### Recommended Knowledge
- Software development lifecycle
- Collaborative development workflows
- Code review practices
- CI/CD fundamentals

## Common Use Cases

### Individual Development
- ✅ Track code changes over time
- ✅ Experiment with features safely (branching)
- ✅ Revert to previous versions
- ✅ Understand code history
- ✅ Maintain multiple project versions

### Team Collaboration
- ✅ Collaborate on codebases
- ✅ Code review workflows
- ✅ Merge team contributions
- ✅ Resolve conflicts
- ✅ Track who changed what and why

### Release Management
- ✅ Tag releases and versions
- ✅ Maintain release branches
- ✅ Cherry-pick hotfixes
- ✅ Manage semantic versioning
- ✅ Generate changelogs

### CI/CD Integration
- ✅ Trigger automated builds
- ✅ Deploy from specific branches
- ✅ Implement GitOps workflows
- ✅ Enforce quality gates
- ✅ Automate versioning

## Learning Path

### Beginner (1-2 weeks)
1. **Git Basics**
   - Install Git
   - Configure user settings
   - Initialize repositories
   - Add and commit changes
   - View history with `git log`

1a. **Regex Fundamentals** (Optional)
   - Basic syntax and metacharacters
   - Character classes and quantifiers
   - .gitignore patterns
   - git grep for searching code

2. **Basic Workflow**
   - Clone repositories
   - Pull latest changes
   - Create branches
   - Merge branches
   - Push to remote

3. **GitHub/GitLab Basics**
   - Create repositories
   - Fork projects
   - Create pull requests
   - Basic code review

### Intermediate (2-4 weeks)
4. **Branching Strategies**
   - Feature branches
   - GitFlow workflow
   - Trunk-based development
   - Release branches
   - Hotfix workflow

5. **Advanced Operations**
   - Interactive rebase
   - Cherry-pick commits
   - Stash changes
   - Reset and revert
   - Reflog recovery

6. **Collaboration**
   - Resolve merge conflicts
   - Code review best practices
   - Commit message conventions
   - PR templates
   - Branch protection rules

### Advanced (1-2 months)
7. **Git Internals**
   - Understand Git objects (blob, tree, commit)
   - References and HEAD
   - Git hooks
   - Submodules and subtrees
   - Git attributes

8. **Enterprise Workflows**
   - Monorepo strategies
   - Multi-repository coordination
   - Security and signing
   - Large file storage (Git LFS)
   - Migration strategies

## Git Architecture

```
Working Directory
       ↓ (git add)
Staging Area (Index)
       ↓ (git commit)
Local Repository (.git)
       ↓ (git push)
Remote Repository (GitHub/GitLab)
```

## Common Git Workflows

### Feature Branch Workflow
```
main
 ├─→ feature/user-auth
 │    ├─ commits
 │    └─→ (merge) → main
 └─→ feature/payment
      ├─ commits
      └─→ (merge) → main
```

### GitFlow Workflow
```
main (production)
 ↓
develop
 ├─→ feature/new-feature
 │    └─→ (merge) → develop
 ├─→ release/v1.0
 │    └─→ (merge) → main & develop
 └─→ hotfix/critical-bug
      └─→ (merge) → main & develop
```

### Trunk-Based Development
```
main (always deployable)
 ├─→ short-lived feature branch (1-2 days)
 │    └─→ (merge) → main
 └─→ feature flags for incomplete features
```

## Related Categories

- 🔄 **[CI/CD Automation](../cicd-automation/README.md)** - Automate workflows with Git hooks
- 🏗️ **[Infrastructure & DevOps](../infrastructure-devops/README.md)** - GitOps for infrastructure
- 🔧 **[Data Engineering](../data-engineering/README.md)** - Version control for data pipelines
- 📚 **[Project Meta](../project-meta/README.md)** - Contributing guidelines

## Quick Start Examples

### Basic Git Commands
```bash
# Initialize repository
git init
git clone https://github.com/user/repo.git

# Basic workflow
git status                    # Check status
git add file.txt             # Stage file
git add .                    # Stage all changes
git commit -m "Add feature"  # Commit changes
git push origin main         # Push to remote

# Branching
git branch feature-x         # Create branch
git checkout feature-x       # Switch to branch
git checkout -b feature-y    # Create and switch
git merge feature-x          # Merge branch
git branch -d feature-x      # Delete branch

# Remote operations
git remote add origin <url>  # Add remote
git fetch origin             # Fetch changes
git pull origin main         # Pull and merge
git push origin feature-x    # Push branch
```

### Advanced Git Commands
```bash
# Interactive rebase
git rebase -i HEAD~3         # Rebase last 3 commits
# Commands: pick, reword, edit, squash, drop

# Stash changes
git stash                    # Save changes temporarily
git stash list               # List stashes
git stash apply              # Reapply stash
git stash pop                # Apply and remove stash

# Cherry-pick
git cherry-pick abc123       # Apply specific commit

# Reset
git reset --soft HEAD~1      # Undo commit, keep changes staged
git reset --mixed HEAD~1     # Undo commit, unstage changes
git reset --hard HEAD~1      # Undo commit, discard changes

# Revert
git revert abc123            # Create new commit that undoes abc123

# History
git log --oneline            # Compact history
git log --graph --all        # Visual branch history
git blame file.txt           # See who changed each line
git reflog                   # History of HEAD movements
```

### Conflict Resolution
```bash
# When merge conflict occurs
git status                   # See conflicting files

# Edit files to resolve conflicts
# <<<<<<< HEAD
# Your changes
# =======
# Their changes
# >>>>>>> branch-name

git add resolved-file.txt    # Mark as resolved
git commit                   # Complete merge
```

### Git Configuration
```bash
# User configuration
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Editor
git config --global core.editor "vim"

# Aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.lg "log --graph --oneline --all"

# Default branch
git config --global init.defaultBranch main

# Line endings
git config --global core.autocrlf input  # Linux/Mac
git config --global core.autocrlf true   # Windows
```

### .gitignore Examples
```gitignore
# Python
__pycache__/
*.py[cod]
*.so
.env
venv/
.venv/

# Node.js
node_modules/
npm-debug.log
.env
dist/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Build artifacts
build/
*.log
*.tmp
```

## Best Practices

### Commit Messages
1. ✅ **Use Present Tense** - "Add feature" not "Added feature"
2. ✅ **Be Descriptive** - Explain what and why, not how
3. ✅ **Keep First Line Short** - 50 characters or less
4. ✅ **Use Body for Details** - Separate with blank line
5. ✅ **Reference Issues** - Link to tickets (#123)

**Example:**
```
Add user authentication with JWT

Implement JWT-based authentication system to replace
session-based auth. This improves scalability and enables
API usage from mobile clients.

- Add JWT middleware
- Create login/logout endpoints
- Add token refresh mechanism

Fixes #123
```

### Branching Strategy
1. ✅ **Descriptive Names** - `feature/user-auth` not `fix1`
2. ✅ **Delete Merged Branches** - Keep repository clean
3. ✅ **Short-Lived Branches** - Merge frequently
4. ✅ **Protect Main Branch** - Require reviews
5. ✅ **Consistent Naming** - Use prefixes (feature/, bugfix/, hotfix/)

### Code Review
1. ✅ **Small PRs** - Easier to review and merge
2. ✅ **Clear Description** - Explain changes and reasoning
3. ✅ **Link Issues** - Connect to project management
4. ✅ **Tests Included** - Ensure quality
5. ✅ **Respond to Feedback** - Engage constructively

### Security
1. ✅ **Never Commit Secrets** - Use environment variables
2. ✅ **Sign Commits** - Use GPG signatures
3. ✅ **Review Dependencies** - Check for vulnerabilities
4. ✅ **Use .gitignore** - Prevent accidental commits
5. ✅ **Rotate Compromised Keys** - Act quickly if exposed

## Common Patterns

### Commit Message Conventions
```
feat: Add new feature
fix: Bug fix
docs: Documentation changes
style: Code style changes (formatting)
refactor: Code refactoring
test: Add or update tests
chore: Maintenance tasks
perf: Performance improvements
ci: CI/CD changes
build: Build system changes
```

### Git Hooks
```bash
# .git/hooks/pre-commit
#!/bin/bash
# Run tests before commit
npm test
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi

# .git/hooks/commit-msg
#!/bin/bash
# Enforce commit message format
commit_msg=$(cat $1)
pattern="^(feat|fix|docs|style|refactor|test|chore):"
if ! echo "$commit_msg" | grep -qE "$pattern"; then
    echo "Invalid commit message format"
    echo "Use: type: description"
    exit 1
fi
```

### Squash Commits
```bash
# Squash last 3 commits into one
git rebase -i HEAD~3

# In editor, change 'pick' to 'squash' for commits to combine
pick abc123 First commit
squash def456 Second commit
squash ghi789 Third commit

# Edit combined commit message
```

## Troubleshooting

### Undo Last Commit (Keep Changes)
```bash
git reset --soft HEAD~1
```

### Undo Last Commit (Discard Changes)
```bash
git reset --hard HEAD~1
```

### Recover Deleted Branch
```bash
git reflog               # Find commit hash
git checkout -b branch-name abc123
```

### Remove File from History
```bash
git filter-branch --tree-filter 'rm -f passwords.txt' HEAD
# Or use BFG Repo-Cleaner (faster)
```

## Navigation

- [← Back to Main Documentation](../../README.md)
- [→ Next: AI/ML Frameworks](../ai-ml-frameworks/README.md)

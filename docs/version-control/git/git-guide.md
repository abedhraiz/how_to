# Git Guide

## What is Git?

Git is a distributed version control system designed to handle everything from small to very large projects with speed and efficiency. Created by Linus Torvalds in 2005, it's now the most widely used version control system in the world.

**Key Features:**
- Distributed architecture
- Branching and merging
- Fast performance
- Data integrity
- Staging area
- Free and open source

## Prerequisites

- Basic command line knowledge
- Text editor familiarity
- Understanding of file systems

## Installation

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install git

# Verify installation
git --version
```

### macOS

```bash
# Using Homebrew
brew install git

# Or download from git-scm.com
```

### Windows

Download from https://git-scm.com/download/win or use:

```bash
# Using Chocolatey
choco install git

# Using Winget
winget install Git.Git
```

## Initial Setup

### Configure User Information

```bash
# Set name and email (required)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set default branch name
git config --global init.defaultBranch main

# Set default editor
git config --global core.editor "code --wait"  # VS Code
git config --global core.editor "vim"  # Vim
git config --global core.editor "nano"  # Nano

# Enable colors
git config --global color.ui auto

# View configuration
git config --list
git config user.name
git config user.email

# Edit config file directly
git config --global --edit
```

### SSH Key Setup

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Start ssh-agent
eval "$(ssh-agent -s)"

# Add key to ssh-agent
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH Keys → New SSH Key
```

## Basic Commands

### Initialize Repository

```bash
# Create new repository
git init

# Initialize with specific branch name
git init -b main

# Clone existing repository
git clone https://github.com/user/repo.git

# Clone specific branch
git clone -b branch-name https://github.com/user/repo.git

# Clone with SSH
git clone git@github.com:user/repo.git

# Clone to specific directory
git clone https://github.com/user/repo.git my-folder
```

### Check Status

```bash
# Show working tree status
git status

# Short format
git status -s

# Show branch information
git status -sb
```

### Add Files

```bash
# Add specific file
git add filename.txt

# Add all files
git add .

# Add all files in current directory
git add *

# Add by pattern
git add *.js

# Add directory
git add src/

# Interactive add
git add -i

# Add parts of file
git add -p filename.txt
```

### Commit Changes

```bash
# Commit with message
git commit -m "Add new feature"

# Commit with detailed message
git commit -m "Add user authentication" -m "Implemented JWT-based authentication with refresh tokens"

# Add and commit in one step
git commit -am "Update README"

# Amend last commit
git commit --amend -m "Updated commit message"

# Amend without changing message
git commit --amend --no-edit

# Empty commit (useful for CI triggers)
git commit --allow-empty -m "Trigger CI"
```

### View History

```bash
# Show commit logs
git log

# One line per commit
git log --oneline

# Show last N commits
git log -n 5

# Show with graph
git log --graph --oneline --all

# Show with details
git log --stat

# Show commits by author
git log --author="John"

# Show commits in date range
git log --since="2 weeks ago"
git log --after="2024-01-01" --before="2024-01-31"

# Show commits for specific file
git log -- filename.txt

# Show commits with specific message
git log --grep="bug fix"

# Pretty format
git log --pretty=format:"%h - %an, %ar : %s"

# Show who changed what in file
git blame filename.txt
```

### View Changes

```bash
# Show unstaged changes
git diff

# Show staged changes
git diff --staged
git diff --cached

# Show changes in specific file
git diff filename.txt

# Show changes between commits
git diff commit1 commit2

# Show changes between branches
git diff main feature-branch

# Show only file names
git diff --name-only

# Show statistics
git diff --stat

# Show specific commit
git show commit-hash
```

## Branching

### Basic Branch Operations

```bash
# List branches
git branch

# List all branches (including remote)
git branch -a

# Create new branch
git branch feature-branch

# Switch to branch
git checkout feature-branch

# Create and switch to branch
git checkout -b feature-branch

# Modern syntax (Git 2.23+)
git switch feature-branch
git switch -c feature-branch

# Rename branch
git branch -m old-name new-name

# Rename current branch
git branch -m new-name

# Delete branch
git branch -d feature-branch

# Force delete branch
git branch -D feature-branch

# Delete remote branch
git push origin --delete feature-branch
```

### Merging

```bash
# Merge branch into current branch
git merge feature-branch

# Merge with commit message
git merge feature-branch -m "Merge feature branch"

# Merge without fast-forward
git merge --no-ff feature-branch

# Abort merge
git merge --abort

# Continue merge after resolving conflicts
git merge --continue

# Show merged branches
git branch --merged

# Show unmerged branches
git branch --no-merged
```

### Rebasing

```bash
# Rebase current branch onto main
git rebase main

# Interactive rebase (last 3 commits)
git rebase -i HEAD~3

# Continue after resolving conflicts
git rebase --continue

# Skip current commit
git rebase --skip

# Abort rebase
git rebase --abort

# Rebase and preserve merges
git rebase -p main
```

## Remote Repositories

### Remote Management

```bash
# Show remotes
git remote

# Show remote URLs
git remote -v

# Add remote
git remote add origin https://github.com/user/repo.git

# Change remote URL
git remote set-url origin https://github.com/user/new-repo.git

# Rename remote
git remote rename origin upstream

# Remove remote
git remote remove origin

# Show remote details
git remote show origin
```

### Fetch, Pull, Push

```bash
# Fetch from remote
git fetch origin

# Fetch all remotes
git fetch --all

# Pull from remote (fetch + merge)
git pull origin main

# Pull with rebase
git pull --rebase origin main

# Push to remote
git push origin main

# Push all branches
git push --all origin

# Push tags
git push --tags

# Force push (dangerous!)
git push --force origin main

# Force push with lease (safer)
git push --force-with-lease origin main

# Set upstream and push
git push -u origin main

# Delete remote branch
git push origin --delete feature-branch
```

## Undoing Changes

### Discard Changes

```bash
# Discard changes in working directory
git checkout -- filename.txt

# Modern syntax
git restore filename.txt

# Discard all changes
git checkout -- .
git restore .

# Unstage file
git reset HEAD filename.txt

# Modern syntax
git restore --staged filename.txt

# Remove untracked files
git clean -n  # Dry run
git clean -f  # Force remove
git clean -fd  # Remove files and directories
```

### Reset

```bash
# Soft reset (keep changes staged)
git reset --soft HEAD~1

# Mixed reset (keep changes unstaged) - default
git reset HEAD~1
git reset --mixed HEAD~1

# Hard reset (discard all changes)
git reset --hard HEAD~1

# Reset to specific commit
git reset --hard commit-hash

# Reset file to specific commit
git checkout commit-hash -- filename.txt
```

### Revert

```bash
# Create new commit that undoes changes
git revert commit-hash

# Revert without committing
git revert -n commit-hash

# Revert multiple commits
git revert commit1..commit3

# Revert merge commit
git revert -m 1 merge-commit-hash
```

## Stashing

```bash
# Stash changes
git stash

# Stash with message
git stash save "Work in progress"

# Include untracked files
git stash -u

# List stashes
git stash list

# Show stash contents
git stash show
git stash show -p  # Show diff

# Apply stash (keep in stash list)
git stash apply
git stash apply stash@{2}

# Pop stash (apply and remove)
git stash pop

# Drop stash
git stash drop stash@{0}

# Clear all stashes
git stash clear

# Create branch from stash
git stash branch new-branch stash@{0}
```

## Tags

```bash
# List tags
git tag

# Create lightweight tag
git tag v1.0.0

# Create annotated tag
git tag -a v1.0.0 -m "Version 1.0.0"

# Tag specific commit
git tag -a v1.0.0 commit-hash -m "Release 1.0.0"

# Show tag details
git show v1.0.0

# Push tag to remote
git push origin v1.0.0

# Push all tags
git push origin --tags

# Delete local tag
git tag -d v1.0.0

# Delete remote tag
git push origin --delete v1.0.0

# Checkout tag
git checkout v1.0.0
```

## Advanced Operations

### Cherry-pick

```bash
# Apply specific commit to current branch
git cherry-pick commit-hash

# Cherry-pick multiple commits
git cherry-pick commit1 commit2

# Cherry-pick without committing
git cherry-pick -n commit-hash

# Continue after resolving conflicts
git cherry-pick --continue

# Abort cherry-pick
git cherry-pick --abort
```

### Submodules

```bash
# Add submodule
git submodule add https://github.com/user/repo.git path/to/submodule

# Initialize submodules
git submodule init

# Update submodules
git submodule update

# Clone with submodules
git clone --recursive https://github.com/user/repo.git

# Update submodules to latest
git submodule update --remote

# Remove submodule
git submodule deinit path/to/submodule
git rm path/to/submodule
```

### Bisect

```bash
# Start bisect
git bisect start

# Mark current commit as bad
git bisect bad

# Mark known good commit
git bisect good commit-hash

# Git will checkout middle commit
# Test and mark as good or bad
git bisect good
# or
git bisect bad

# Continue until bad commit is found

# End bisect
git bisect reset
```

### Worktrees

```bash
# List worktrees
git worktree list

# Add worktree
git worktree add ../my-feature feature-branch

# Remove worktree
git worktree remove ../my-feature

# Prune worktrees
git worktree prune
```

## .gitignore

### Creating .gitignore

```bash
# Create .gitignore file
touch .gitignore
```

### Common Patterns

```gitignore
# Node.js
node_modules/
npm-debug.log
yarn-error.log
.env

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
*.egg-info/

# Java
*.class
*.jar
*.war
target/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Build outputs
dist/
build/
*.o
*.so

# Secrets
*.key
*.pem
secrets.yml
```

### Global .gitignore

```bash
# Create global gitignore
touch ~/.gitignore_global

# Configure Git to use it
git config --global core.excludesfile ~/.gitignore_global
```

## Git Workflows

### Feature Branch Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature main

# Make changes
git add .
git commit -m "Add new feature"

# Push to remote
git push -u origin feature/new-feature

# Create pull request on GitHub/GitLab

# After PR approval, merge and delete branch
git checkout main
git pull origin main
git branch -d feature/new-feature
```

### Gitflow Workflow

```bash
# Main branches: main (production), develop (integration)

# Start new feature
git checkout -b feature/my-feature develop

# Finish feature
git checkout develop
git merge --no-ff feature/my-feature
git branch -d feature/my-feature
git push origin develop

# Start release
git checkout -b release/1.0.0 develop

# Finish release
git checkout main
git merge --no-ff release/1.0.0
git tag -a v1.0.0
git checkout develop
git merge --no-ff release/1.0.0
git branch -d release/1.0.0

# Hotfix
git checkout -b hotfix/1.0.1 main
# Fix bug
git checkout main
git merge --no-ff hotfix/1.0.1
git tag -a v1.0.1
git checkout develop
git merge --no-ff hotfix/1.0.1
git branch -d hotfix/1.0.1
```

### Forking Workflow

```bash
# Fork repository on GitHub

# Clone your fork
git clone https://github.com/your-username/repo.git

# Add upstream remote
git remote add upstream https://github.com/original-owner/repo.git

# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/my-contribution

# Make changes and push
git push origin feature/my-contribution

# Create pull request on GitHub
```

## GitHub/GitLab Specific

### GitHub CLI

```bash
# Install GitHub CLI
# See: https://cli.github.com/

# Login
gh auth login

# Create repository
gh repo create my-repo --public

# Clone repository
gh repo clone user/repo

# Create pull request
gh pr create --title "Add feature" --body "Description"

# List pull requests
gh pr list

# Check out pull request
gh pr checkout 123

# View pull request
gh pr view 123

# Merge pull request
gh pr merge 123

# Create issue
gh issue create --title "Bug report" --body "Description"

# List issues
gh issue list
```

### Pull Requests

```bash
# Create PR from command line
git push origin feature-branch
# Then create PR on web interface

# Or use hub (GitHub CLI predecessor)
hub pull-request -m "PR title" -m "PR description"

# Update PR
git push origin feature-branch

# Squash commits before merging
git rebase -i HEAD~3
# Mark commits as 'squash' or 'fixup'
git push --force-with-lease origin feature-branch
```

## Aliases

### Set Up Aliases

```bash
# Basic aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'

# Advanced aliases
git config --global alias.lg "log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
git config --global alias.undo 'reset --soft HEAD~1'
git config --global alias.amend 'commit --amend --no-edit'
git config --global alias.aliases "config --get-regexp '^alias\.'"

# Now use them
git st
git co main
git lg
```

### Shell Aliases (Bash/Zsh)

```bash
# Add to ~/.bashrc or ~/.zshrc
alias g='git'
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git pull'
alias gco='git checkout'
alias gb='git branch'
alias gd='git diff'
alias glg='git log --graph --oneline --all'
```

## Best Practices

### Commit Messages

```bash
# Good commit message format
<type>: <subject>

<body>

<footer>

# Types: feat, fix, docs, style, refactor, test, chore

# Examples:
feat: add user authentication
fix: resolve memory leak in data processing
docs: update API documentation
refactor: simplify database query logic
test: add unit tests for user service
chore: update dependencies
```

### Branching Strategy

```bash
# Use descriptive branch names
feature/user-authentication
bugfix/login-error
hotfix/critical-security-patch
release/v1.2.0

# Keep branches short-lived
# Merge frequently
# Delete merged branches
```

### Commit Practices

```bash
# Commit early and often
# Make atomic commits (one logical change)
# Don't commit broken code
# Review changes before committing

# Use staging area effectively
git add -p  # Review each change

# Write meaningful commit messages
# Use imperative mood: "Add feature" not "Added feature"
```

## Troubleshooting

### Undo Last Commit

```bash
# Keep changes
git reset --soft HEAD~1

# Discard changes
git reset --hard HEAD~1
```

### Fix Commit Message

```bash
# Last commit
git commit --amend -m "New message"

# Older commit (interactive rebase)
git rebase -i HEAD~3
# Change 'pick' to 'reword' for commits to change
```

### Resolve Merge Conflicts

```bash
# During merge
git merge feature-branch
# CONFLICT appears

# View conflicts
git status

# Edit files to resolve conflicts
# Remove conflict markers (<<<<<<<, =======, >>>>>>>)

# Mark as resolved
git add conflicted-file.txt

# Complete merge
git commit

# Or abort
git merge --abort
```

### Recover Lost Commits

```bash
# View reflog
git reflog

# Restore commit
git checkout commit-hash

# Create branch from lost commit
git branch recovered-branch commit-hash
```

### Fix Wrong Branch

```bash
# Made commits on wrong branch
git stash
git checkout correct-branch
git stash pop

# Or move commits to new branch
git branch new-branch
git reset --hard HEAD~3  # Remove last 3 commits from current branch
git checkout new-branch
```

## Git Hooks

### Client-side Hooks

```bash
# Located in .git/hooks/

# Pre-commit hook
# .git/hooks/pre-commit
#!/bin/bash
npm run lint
if [ $? -ne 0 ]; then
  echo "Linting failed"
  exit 1
fi

# Make executable
chmod +x .git/hooks/pre-commit

# Pre-push hook
# .git/hooks/pre-push
#!/bin/bash
npm test
if [ $? -ne 0 ]; then
  echo "Tests failed"
  exit 1
fi

# Commit message hook
# .git/hooks/commit-msg
#!/bin/bash
commit_msg=$(cat $1)
if ! echo "$commit_msg" | grep -qE "^(feat|fix|docs|style|refactor|test|chore):"; then
  echo "Invalid commit message format"
  exit 1
fi
```

### Using Husky

```bash
# Install husky
npm install husky --save-dev

# Initialize
npx husky install

# Add pre-commit hook
npx husky add .husky/pre-commit "npm test"

# Add commit-msg hook
npx husky add .husky/commit-msg 'npx --no -- commitlint --edit "$1"'
```

## Workflow Examples

### GitFlow Workflow

```bash
# Initialize GitFlow
git flow init

# Or manually:
git branch develop
git checkout develop

# Start a new feature
git flow feature start user-authentication
# Or: git checkout -b feature/user-authentication develop

# Work on feature
git add .
git commit -m "feat: add user authentication"

# Finish feature
git flow feature finish user-authentication
# Or manually:
git checkout develop
git merge --no-ff feature/user-authentication
git branch -d feature/user-authentication

# Start a release
git flow release start 1.0.0
# Or: git checkout -b release/1.0.0 develop

# Update version numbers, changelog
git add .
git commit -m "chore: prepare release 1.0.0"

# Finish release
git flow release finish 1.0.0
# Or manually:
git checkout main
git merge --no-ff release/1.0.0
git tag -a v1.0.0 -m "Release version 1.0.0"
git checkout develop
git merge --no-ff release/1.0.0
git branch -d release/1.0.0

# Hotfix for production
git flow hotfix start 1.0.1
# Or: git checkout -b hotfix/1.0.1 main

# Fix the bug
git add .
git commit -m "fix: critical security patch"

# Finish hotfix
git flow hotfix finish 1.0.1
# Merges to main and develop, creates tag

# Push everything
git push origin main develop --tags
```

### GitHub Flow

```bash
# 1. Always work from main branch
git checkout main
git pull origin main

# 2. Create a descriptive branch
git checkout -b add-payment-gateway

# 3. Make commits with clear messages
git add src/payment.js
git commit -m "feat: integrate Stripe payment gateway"

git add tests/payment.test.js
git commit -m "test: add payment gateway tests"

# 4. Push branch and open pull request
git push -u origin add-payment-gateway

# On GitHub, open Pull Request
# Add description, request reviews

# 5. After review and CI passes, merge to main
# (Done via GitHub UI with "Squash and merge" or "Merge commit")

# 6. Delete branch after merge
git checkout main
git pull origin main
git branch -d add-payment-gateway
git push origin --delete add-payment-gateway
```

### Trunk-Based Development

```bash
# Always work on main (trunk)
git checkout main
git pull origin main

# Create short-lived feature branch
git checkout -b feature-xyz

# Make small, incremental changes
git add .
git commit -m "feat: add user profile endpoint"
git push origin feature-xyz

# Immediately create PR and merge (within hours)
# After merge, delete branch
git checkout main
git pull origin main
git branch -d feature-xyz

# Or commit directly to main for small changes
git checkout main
git pull origin main
# Make changes
git add .
git commit -m "fix: typo in documentation"
git push origin main

# Feature flags for incomplete features
# config.js
const features = {
  newUI: process.env.ENABLE_NEW_UI === 'true',
  payments: process.env.ENABLE_PAYMENTS === 'true'
};

# Code
if (features.newUI) {
  // New UI code
} else {
  // Old UI code
}
```

### Advanced Rebase Examples

```bash
# Interactive rebase to clean history
git checkout feature-branch
git rebase -i HEAD~5

# In editor, you can:
# pick = keep commit
# reword = keep commit but edit message  
# edit = keep commit but stop for amending
# squash = combine with previous commit
# fixup = like squash but discard message
# drop = remove commit

# Example:
pick abc1234 feat: add login
squash def5678 fix: login validation
reword ghi9012 feat: add logout
drop jkl3456 wip: testing
pick mno7890 docs: update README

# Rebase onto another branch
git checkout feature-branch
git rebase main

# If conflicts occur:
git status  # See conflicting files
# Fix conflicts in files
git add <resolved-files>
git rebase --continue

# Or skip a commit
git rebase --skip

# Or abort rebase
git rebase --abort

# Rebase and update remote
git push origin feature-branch --force-with-lease

# Autosquash commits
git commit --fixup=abc1234  # Creates fixup commit
git rebase -i --autosquash HEAD~10  # Automatically squashes fixups
```

### Cherry-Pick Examples

```bash
# Pick single commit from another branch
git checkout main
git cherry-pick abc1234

# Pick multiple commits
git cherry-pick abc1234 def5678 ghi9012

# Pick a range of commits
git cherry-pick abc1234^..ghi9012

# Cherry-pick without committing (for review)
git cherry-pick -n abc1234

# Cherry-pick and edit commit message
git cherry-pick -e abc1234

# Resolve conflicts during cherry-pick
# Fix conflicts
git add <files>
git cherry-pick --continue

# Or abort
git cherry-pick --abort

# Example: backport fix to release branch
git checkout release/1.0
git cherry-pick abc1234  # Commit from main
git push origin release/1.0
```

### Git Bisect Examples

```bash
# Find the commit that introduced a bug
git bisect start

# Mark current commit as bad
git bisect bad

# Mark a known good commit
git bisect good v1.0.0

# Git will checkout a commit in between
# Test if bug exists

# If bug exists:
git bisect bad

# If bug doesn't exist:
git bisect good

# Continue until Git finds the problematic commit

# Automate bisect with a script
git bisect start HEAD v1.0.0
git bisect run npm test

# Git will automatically test each commit
# and find the first bad commit

# When done
git bisect reset

# Example: find when tests started failing
git bisect start
git bisect bad HEAD
git bisect good v2.0.0
git bisect run sh -c "npm install && npm test"
```

### Submodules Workflow

```bash
# Add a submodule
git submodule add https://github.com/user/library.git libs/library

# This creates .gitmodules file
# And adds the submodule

# Clone repository with submodules
git clone --recursive https://github.com/user/repo.git

# Or if already cloned:
git submodule init
git submodule update

# Update submodule to latest
cd libs/library
git fetch
git checkout main
git pull
cd ../..
git add libs/library
git commit -m "chore: update library submodule"

# Update all submodules
git submodule update --remote --merge

# Remove a submodule
git submodule deinit libs/library
git rm libs/library
rm -rf .git/modules/libs/library
git commit -m "chore: remove library submodule"
```

### Subtree Workflow

```bash
# Add a subtree (better than submodules for most cases)
git subtree add --prefix=libs/library \
  https://github.com/user/library.git main --squash

# Pull updates from subtree
git subtree pull --prefix=libs/library \
  https://github.com/user/library.git main --squash

# Push changes back to subtree repository
git subtree push --prefix=libs/library \
  https://github.com/user/library.git main

# Split subtree into separate repository
git subtree split --prefix=libs/library -b library-only

# Create new repository with the subtree
git push https://github.com/user/new-library.git library-only:main
```

### Worktree Examples

```bash
# Work on multiple branches simultaneously
git worktree add ../project-feature feature-branch

# This creates a new working directory
# You can work on feature-branch there while
# main directory stays on main branch

# List worktrees
git worktree list

# Remove worktree when done
git worktree remove ../project-feature

# Or prune deleted worktrees
git worktree prune

# Example: hotfix while working on feature
# In main project directory
git worktree add ../project-hotfix hotfix/1.0.1

# Work on hotfix in ../project-hotfix
cd ../project-hotfix
# Make fixes
git add .
git commit -m "fix: critical bug"
git push origin hotfix/1.0.1

# Return to feature work
cd ../project
# Your feature branch is unchanged
```

### Reflog Examples

```bash
# View reflog (history of HEAD)
git reflog

# Output like:
# abc1234 HEAD@{0}: commit: feat: add feature
# def5678 HEAD@{1}: checkout: moving from main to feature
# ghi9012 HEAD@{2}: pull: Fast-forward

# Recover deleted branch
git branch feature-branch HEAD@{1}

# Undo a reset
git reset --hard abc1234
# Oops, that was wrong
git reflog
git reset --hard HEAD@{1}

# Find lost commits
git reflog show --all
git cherry-pick <lost-commit-hash>

# Recover after accidental branch delete
git reflog
# Find the commit where branch was
git checkout -b recovered-branch <commit-hash>

# Reflog expires after 90 days by default
# View reflog expiry settings
git config --get gc.reflogExpire
```

### Large File Management (Git LFS)

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.psd"
git lfs track "*.mp4"
git lfs track "design/**"

# This creates/updates .gitattributes
git add .gitattributes

# Add and commit large files normally
git add design/mockup.psd
git commit -m "docs: add design mockup"
git push origin main

# Clone repository with LFS files
git lfs clone https://github.com/user/repo.git

# Pull LFS files
git lfs pull

# View LFS files
git lfs ls-files

# Fetch LFS files for specific branch
git lfs fetch origin main

# Migrate existing files to LFS
git lfs migrate import --include="*.psd"

# Untrack LFS files
git lfs untrack "*.psd"
```

### Monorepo Strategies

```bash
# Sparse checkout - only checkout specific directories
git clone --no-checkout https://github.com/user/monorepo.git
cd monorepo

git sparse-checkout init --cone
git sparse-checkout set packages/api packages/shared

git checkout main

# Now only packages/api and packages/shared are checked out

# Add more paths
git sparse-checkout add packages/frontend

# Disable sparse checkout
git sparse-checkout disable

# Partial clone - download objects on demand
git clone --filter=blob:none https://github.com/user/monorepo.git
```

### Multi-Remote Workflow

```bash
# Add multiple remotes
git remote add origin https://github.com/user/repo.git
git remote add upstream https://github.com/original/repo.git
git remote add staging https://github.com/company/staging.git

# Fetch from all remotes
git fetch --all

# Push to multiple remotes
git push origin main
git push staging main

# Set up to push to multiple remotes at once
git remote set-url --add --push origin https://github.com/user/repo.git
git remote set-url --add --push origin https://gitlab.com/user/repo.git

# Now git push origin main pushes to both

# Pull from upstream, push to origin
git fetch upstream
git merge upstream/main
git push origin main

# Or rebase
git pull --rebase upstream main
git push origin main
```

### Release Management

```bash
# Semantic versioning tags
git tag -a v1.0.0 -m "Release version 1.0.0"
git tag -a v1.1.0 -m "Release version 1.1.0 - Added feature X"
git tag -a v1.1.1 -m "Release version 1.1.1 - Bugfix"

# Push tags
git push origin --tags

# List tags
git tag -l

# List tags matching pattern
git tag -l "v1.*"

# Checkout specific tag
git checkout v1.0.0

# Create branch from tag
git checkout -b hotfix/1.0.1 v1.0.0

# Delete tag locally
git tag -d v1.0.0

# Delete tag remotely
git push origin :refs/tags/v1.0.0

# Signed tags (GPG)
git tag -s v1.0.0 -m "Signed release"

# Verify signed tag
git tag -v v1.0.0

# Generate changelog between tags
git log v1.0.0..v1.1.0 --oneline --decorate

# Or use conventional commits
git log v1.0.0..HEAD --pretty=format:"%s" | \
  grep "^feat:" | sed 's/^feat: /- /'
```

### Git Hooks Advanced

```bash
# Server-side pre-receive hook
# .git/hooks/pre-receive
#!/bin/bash

while read oldrev newrev refname; do
  # Prevent force push to main
  if [ "$refname" = "refs/heads/main" ]; then
    if [ "$oldrev" != "0000000000000000000000000000000000000000" ]; then
      merge_base=$(git merge-base $oldrev $newrev)
      if [ "$merge_base" != "$oldrev" ]; then
        echo "Force push to main is not allowed"
        exit 1
      fi
    fi
  fi
  
  # Check commit messages
  commits=$(git rev-list $oldrev..$newrev)
  for commit in $commits; do
    message=$(git log --format=%B -n 1 $commit)
    if ! echo "$message" | grep -qE "^(feat|fix|docs|style|refactor|test|chore):"; then
      echo "Invalid commit message format in $commit"
      exit 1
    fi
  done
done

# Pre-commit: run tests and linting
# .git/hooks/pre-commit
#!/bin/bash

echo "Running tests..."
npm test
if [ $? -ne 0 ]; then
  echo "Tests failed. Commit aborted."
  exit 1
fi

echo "Running linter..."
npm run lint
if [ $? -ne 0 ]; then
  echo "Linting failed. Commit aborted."
  exit 1
fi

# Check for secrets
if git diff --cached | grep -iE "(password|api_key|secret|token)\s*="; then
  echo "Possible secret detected. Commit aborted."
  exit 1
fi

echo "Pre-commit checks passed!"
exit 0

# Commit-msg: enforce commit message format
# .git/hooks/commit-msg
#!/bin/bash

commit_msg=$(cat "$1")

# Check conventional commits format
if ! echo "$commit_msg" | grep -qE "^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .+"; then
  echo "Error: Commit message must follow Conventional Commits format"
  echo "Format: type(scope?): description"
  echo "Example: feat(auth): add OAuth2 support"
  exit 1
fi

# Check message length
if [ ${#commit_msg} -gt 72 ]; then
  echo "Error: Commit message must be 72 characters or less"
  exit 1
fi

exit 0
```

### Conflict Resolution Strategies

```bash
# When merge conflict occurs
git status  # Shows conflicting files

# Open conflicting file:
<<<<<<< HEAD
Your changes
=======
Their changes
>>>>>>> branch-name

# Strategy 1: Manual resolution
# Edit file to resolve conflicts
git add <resolved-file>
git commit

# Strategy 2: Use ours or theirs
git checkout --ours <file>   # Keep your version
git checkout --theirs <file>  # Use their version
git add <file>

# Strategy 3: Use merge tool
git mergetool

# Strategy 4: Accept all from one side
git merge -X ours branch-name    # Prefer your changes
git merge -X theirs branch-name  # Prefer their changes

# View conflict in different layouts
git diff --name-only --diff-filter=U  # List conflicting files
git diff --check  # Show conflict markers

# Abort merge if too complex
git merge --abort

# For rebase conflicts
git rebase --abort

# Continue after resolving
git rebase --continue

# Skip problematic commit
git rebase --skip
```

## Performance Tips

```bash
# Shallow clone (faster, less history)
git clone --depth 1 https://github.com/user/repo.git

# Partial clone (fetch objects on demand)
git clone --filter=blob:none https://github.com/user/repo.git

# Speed up status
git config --global core.fsmonitor true

# Optimize repository
git gc
git prune

# Reduce repository size
git filter-branch --tree-filter 'rm -f large-file.bin' HEAD

# Or use git-filter-repo (recommended)
git filter-repo --path large-file.bin --invert-paths
```

## Resources

- **Official Documentation**: https://git-scm.com/doc
- **Pro Git Book**: https://git-scm.com/book/en/v2
- **GitHub Docs**: https://docs.github.com/
- **GitLab Docs**: https://docs.gitlab.com/
- **Interactive Learning**: https://learngitbranching.js.org/

## Quick Reference

### Most Common Commands

```bash
# Setup
git config --global user.name "Name"
git config --global user.email "email@example.com"

# Create/Clone
git init
git clone <url>

# Basic workflow
git status
git add <file>
git commit -m "message"
git push origin main

# Branching
git branch
git checkout -b <branch>
git merge <branch>

# Undo
git reset HEAD <file>
git checkout -- <file>
git revert <commit>

# History
git log
git log --oneline --graph --all
git diff

# Remote
git remote -v
git fetch
git pull
git push
```

### Git Cheat Sheet

| Command | Description |
|---------|-------------|
| `git init` | Initialize repository |
| `git clone <url>` | Clone repository |
| `git status` | Show status |
| `git add <file>` | Stage file |
| `git commit -m "msg"` | Commit changes |
| `git push` | Push to remote |
| `git pull` | Pull from remote |
| `git branch` | List branches |
| `git checkout <branch>` | Switch branch |
| `git merge <branch>` | Merge branch |
| `git log` | View history |
| `git diff` | Show changes |
| `git reset` | Undo changes |
| `git stash` | Stash changes |
| `git tag` | Create tag |

---

*This guide covers Git fundamentals and advanced workflows. Practice regularly to master version control!*

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

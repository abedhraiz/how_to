# GitHub Actions Guide

## What is GitHub Actions?

GitHub Actions is a continuous integration and continuous deployment (CI/CD) platform that allows you to automate your build, test, and deployment pipeline directly in your GitHub repository.

## Core Concepts

### 1. **Workflows**
Automated processes defined in YAML files stored in `.github/workflows/` directory.

### 2. **Events**
Triggers that cause a workflow to run (push, pull request, schedule, etc.).

### 3. **Jobs**
A set of steps that execute on the same runner.

### 4. **Steps**
Individual tasks that run commands or actions.

### 5. **Actions**
Reusable units of code that can be shared across workflows.

### 6. **Runners**
Servers that run your workflows (GitHub-hosted or self-hosted).

## Getting Started

### Basic Workflow Structure

Create a file at `.github/workflows/main.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    - name: Run a script
      run: echo "Hello, GitHub Actions!"
```

## Workflow Syntax

### Workflow Name
```yaml
name: My Workflow
```

### Triggers (Events)

#### Push Event
```yaml
on:
  push:
    branches:
      - main
      - develop
    paths:
      - 'src/**'
      - '**.js'
```

#### Pull Request Event
```yaml
on:
  pull_request:
    branches:
      - main
    types:
      - opened
      - synchronize
      - reopened
```

#### Schedule (Cron)
```yaml
on:
  schedule:
    # Run at 00:00 UTC every day
    - cron: '0 0 * * *'
```

#### Manual Trigger (workflow_dispatch)
```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy'
        required: true
        default: 'staging'
```

#### Multiple Events
```yaml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 1'
```

### Jobs

#### Basic Job
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm install
      - run: npm test
```

#### Multiple Jobs
```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm run build
  
  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - run: npm test
  
  deploy:
    runs-on: ubuntu-latest
    needs: [build, test]
    steps:
      - uses: actions/checkout@v4
      - run: npm run deploy
```

#### Matrix Strategy
```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        node-version: [14, 16, 18, 20]
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm test
```

### Runners

```yaml
runs-on: ubuntu-latest    # Ubuntu
runs-on: windows-latest   # Windows
runs-on: macos-latest     # macOS
runs-on: self-hosted      # Self-hosted runner
```

## Common Workflow Examples

### Node.js CI/CD

```yaml
name: Node.js CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        node-version: [16.x, 18.x, 20.x]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run linter
      run: npm run lint
    
    - name: Run tests
      run: npm test
    
    - name: Build
      run: npm run build
```

### Python CI/CD

```yaml
name: Python CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov=./ --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### Docker Build and Push

```yaml
name: Docker Build

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Log in to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: user/app:latest
```

### Deploy to AWS

```yaml
name: Deploy to AWS

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Deploy to S3
      run: aws s3 sync ./build s3://my-bucket --delete
```

## Environment Variables and Secrets

### Using Environment Variables

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    env:
      NODE_ENV: production
      API_URL: https://api.example.com
    steps:
      - run: echo "Environment is $NODE_ENV"
      - run: echo "API URL is $API_URL"
```

### Using Secrets

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy
        env:
          API_KEY: ${{ secrets.API_KEY }}
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: ./deploy.sh
```

### GitHub Context Variables

```yaml
steps:
  - name: Print context
    run: |
      echo "Repository: ${{ github.repository }}"
      echo "Branch: ${{ github.ref }}"
      echo "Commit SHA: ${{ github.sha }}"
      echo "Actor: ${{ github.actor }}"
      echo "Event: ${{ github.event_name }}"
```

## Artifacts and Caching

### Upload Artifacts

```yaml
steps:
  - name: Build
    run: npm run build
  
  - name: Upload artifact
    uses: actions/upload-artifact@v4
    with:
      name: build-files
      path: dist/
```

### Download Artifacts

```yaml
steps:
  - name: Download artifact
    uses: actions/download-artifact@v4
    with:
      name: build-files
      path: dist/
```

### Caching Dependencies

```yaml
steps:
  - uses: actions/checkout@v4
  
  - name: Cache node modules
    uses: actions/cache@v3
    with:
      path: ~/.npm
      key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
      restore-keys: |
        ${{ runner.os }}-node-
  
  - run: npm install
```

## Conditional Execution

### If Conditions

```yaml
steps:
  - name: Deploy to production
    if: github.ref == 'refs/heads/main'
    run: ./deploy.sh
  
  - name: Deploy to staging
    if: github.ref == 'refs/heads/develop'
    run: ./deploy-staging.sh
```

### Multiple Conditions

```yaml
jobs:
  deploy:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploying to production"
```

### Success/Failure Conditions

```yaml
steps:
  - name: Run tests
    id: tests
    run: npm test
  
  - name: On success
    if: success()
    run: echo "Tests passed!"
  
  - name: On failure
    if: failure()
    run: echo "Tests failed!"
  
  - name: Always run
    if: always()
    run: echo "This always runs"
```

## Reusable Workflows

### Define Reusable Workflow (`.github/workflows/reusable.yml`)

```yaml
name: Reusable Workflow

on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
    secrets:
      api-key:
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy to ${{ inputs.environment }}
        env:
          API_KEY: ${{ secrets.api-key }}
        run: ./deploy.sh ${{ inputs.environment }}
```

### Call Reusable Workflow

```yaml
name: Deploy

on: [push]

jobs:
  call-workflow:
    uses: ./.github/workflows/reusable.yml
    with:
      environment: production
    secrets:
      api-key: ${{ secrets.API_KEY }}
```

## Custom Actions

### JavaScript Action

Create `action.yml`:
```yaml
name: 'My Custom Action'
description: 'Does something cool'
inputs:
  name:
    description: 'Name to greet'
    required: true
runs:
  using: 'node20'
  main: 'index.js'
```

### Use Custom Action

```yaml
steps:
  - uses: ./path/to/action
    with:
      name: 'World'
```

## Debugging

### Enable Debug Logging

Set repository secrets:
- `ACTIONS_STEP_DEBUG`: `true`
- `ACTIONS_RUNNER_DEBUG`: `true`

### Debug Step

```yaml
steps:
  - name: Debug
    run: |
      echo "Event: ${{ github.event_name }}"
      echo "Ref: ${{ github.ref }}"
      echo "SHA: ${{ github.sha }}"
      env
```

### Use tmate for SSH Debugging

```yaml
steps:
  - name: Setup tmate session
    if: failure()
    uses: mxschmitt/action-tmate@v3
```

## Security Best Practices

1. **Use Secrets** - Never hardcode sensitive data
2. **Pin Action Versions** - Use specific SHA instead of tags
3. **Limit Permissions** - Use minimum required permissions
4. **Review Third-Party Actions** - Check action source code
5. **Use CODEOWNERS** - Require reviews for workflow changes
6. **Enable Branch Protection** - Protect main branches
7. **Audit Logs** - Review workflow runs regularly
8. **Use Environment Protection** - Add manual approval steps

### Minimal Permissions

```yaml
permissions:
  contents: read
  pull-requests: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
```

## Monitoring and Notifications

### Slack Notification

```yaml
steps:
  - name: Slack notification
    if: always()
    uses: 8398a7/action-slack@v3
    with:
      status: ${{ job.status }}
      webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Email Notification

```yaml
steps:
  - name: Send email
    if: failure()
    uses: dawidd6/action-send-mail@v3
    with:
      server_address: smtp.gmail.com
      server_port: 465
      username: ${{ secrets.EMAIL_USERNAME }}
      password: ${{ secrets.EMAIL_PASSWORD }}
      subject: Build failed
      body: Build failed for ${{ github.repository }}
      to: admin@example.com
```

## GitHub Actions Marketplace

Browse and use community actions: https://github.com/marketplace?type=actions

Popular actions:
- `actions/checkout` - Check out repository
- `actions/setup-node` - Setup Node.js
- `actions/setup-python` - Setup Python
- `actions/cache` - Cache dependencies
- `actions/upload-artifact` - Upload artifacts
- `docker/build-push-action` - Build and push Docker images

## Cost Optimization

1. **Use Caching** - Cache dependencies
2. **Optimize Matrix Builds** - Only test necessary combinations
3. **Cancel Redundant Runs** - Use concurrency settings
4. **Self-Hosted Runners** - For high-volume workflows
5. **Conditional Jobs** - Skip unnecessary jobs

### Concurrency Control

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

## Useful Commands

### GitHub CLI

```bash
# List workflows
gh workflow list

# View workflow runs
gh run list

# View specific run
gh run view <run-id>

# Rerun a workflow
gh run rerun <run-id>

# Watch a workflow run
gh run watch
```

## Common Issues and Solutions

### Issue: Workflow not triggering
- Check branch name in trigger
- Verify YAML syntax
- Check if Actions are enabled in repository settings

### Issue: Permission denied
- Add required permissions to workflow
- Check token permissions
- Verify secret access

### Issue: Timeout
- Increase timeout: `timeout-minutes: 30`
- Optimize slow steps
- Use caching

## Quick Reference

| Syntax | Description |
|--------|-------------|
| `on: push` | Trigger on push |
| `runs-on: ubuntu-latest` | Runner OS |
| `uses: actions/checkout@v4` | Use an action |
| `run: npm test` | Run shell command |
| `if: success()` | Conditional step |
| `needs: [build]` | Job dependency |
| `strategy.matrix` | Matrix build |
| `${{ secrets.API_KEY }}` | Use secret |
| `${{ github.sha }}` | Context variable |
| `timeout-minutes: 10` | Job timeout |

## Resources

- Official Documentation: https://docs.github.com/en/actions
- Workflow Syntax: https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions
- Actions Marketplace: https://github.com/marketplace?type=actions
- Awesome Actions: https://github.com/sdras/awesome-actions
- GitHub Actions Toolkit: https://github.com/actions/toolkit

---

*This guide covers GitHub Actions fundamentals. For advanced scenarios, refer to the official documentation and explore community actions in the marketplace.*

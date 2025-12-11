# Jenkins Guide

## What is Jenkins?

Jenkins is an open-source automation server that enables developers to build, test, and deploy their software reliably. It's a leading CI/CD tool that supports hundreds of plugins to integrate with virtually any tool in the software development lifecycle.

## Prerequisites

- Java 11 or Java 17 installed
- Basic understanding of CI/CD concepts
- Server or local machine for installation
- Git (for version control integration)

## Installation

### Linux (Ubuntu/Debian)

```bash
# Install Java
sudo apt update
sudo apt install openjdk-11-jdk -y

# Add Jenkins repository
curl -fsSL https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key | sudo tee \
  /usr/share/keyrings/jenkins-keyring.asc > /dev/null

echo deb [signed-by=/usr/share/keyrings/jenkins-keyring.asc] \
  https://pkg.jenkins.io/debian-stable binary/ | sudo tee \
  /etc/apt/sources.list.d/jenkins.list > /dev/null

# Install Jenkins
sudo apt update
sudo apt install jenkins -y

# Start Jenkins
sudo systemctl start jenkins
sudo systemctl enable jenkins

# Check status
sudo systemctl status jenkins
```

### macOS

```bash
# Using Homebrew
brew install jenkins-lts

# Start Jenkins
brew services start jenkins-lts
```

### Docker

```bash
# Run Jenkins in Docker
docker run -d \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  --name jenkins \
  jenkins/jenkins:lts

# Get initial admin password
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

### Initial Setup

1. Access Jenkins at `http://localhost:8080`
2. Retrieve initial admin password:
   ```bash
   sudo cat /var/jenkins_home/secrets/initialAdminPassword
   ```
3. Install suggested plugins
4. Create first admin user
5. Configure Jenkins URL

## Core Concepts

### 1. **Jobs/Projects**
Tasks that Jenkins executes (build, test, deploy).

### 2. **Pipeline**
A suite of plugins that supports implementing and integrating CI/CD pipelines.

### 3. **Nodes/Agents**
Machines where Jenkins executes jobs (master and slave nodes).

### 4. **Builds**
A single execution of a job.

### 5. **Plugins**
Extensions that add functionality to Jenkins.

### 6. **Workspace**
Directory where Jenkins stores files for a job.

### 7. **Jenkinsfile**
A text file containing Pipeline definition.

## Creating Jobs

### Freestyle Project

1. Click "New Item"
2. Enter job name
3. Select "Freestyle project"
4. Configure:
   - Source Code Management (Git)
   - Build Triggers
   - Build Steps
   - Post-build Actions

### Example Configuration

**Source Code Management:**
- Repository URL: `https://github.com/user/repo.git`
- Credentials: Add GitHub credentials
- Branch: `*/main`

**Build Triggers:**
- Poll SCM: `H/5 * * * *` (every 5 minutes)
- GitHub hook trigger

**Build Steps:**
```bash
npm install
npm test
npm run build
```

**Post-build Actions:**
- Archive artifacts: `dist/**`
- Publish test results: `test-reports/*.xml`

## Jenkins Pipeline

### Declarative Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        NODE_ENV = 'production'
        API_URL = 'https://api.example.com'
    }
    
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main',
                    url: 'https://github.com/user/repo.git'
            }
        }
        
        stage('Install Dependencies') {
            steps {
                sh 'npm install'
            }
        }
        
        stage('Test') {
            steps {
                sh 'npm test'
            }
        }
        
        stage('Build') {
            steps {
                sh 'npm run build'
            }
        }
        
        stage('Deploy') {
            steps {
                sh './deploy.sh'
            }
        }
    }
    
    post {
        success {
            echo 'Pipeline succeeded!'
            mail to: 'team@example.com',
                 subject: "SUCCESS: ${env.JOB_NAME}",
                 body: "Build succeeded: ${env.BUILD_URL}"
        }
        failure {
            echo 'Pipeline failed!'
            mail to: 'team@example.com',
                 subject: "FAILURE: ${env.JOB_NAME}",
                 body: "Build failed: ${env.BUILD_URL}"
        }
        always {
            cleanWs()
        }
    }
}
```

### Scripted Pipeline

```groovy
// Jenkinsfile
node {
    try {
        stage('Checkout') {
            checkout scm
        }
        
        stage('Build') {
            sh 'npm install'
            sh 'npm run build'
        }
        
        stage('Test') {
            sh 'npm test'
        }
        
        stage('Deploy') {
            if (env.BRANCH_NAME == 'main') {
                sh './deploy.sh production'
            } else {
                sh './deploy.sh staging'
            }
        }
        
        currentBuild.result = 'SUCCESS'
    } catch (Exception e) {
        currentBuild.result = 'FAILURE'
        throw e
    } finally {
        cleanWs()
    }
}
```

## Pipeline Syntax

### Agent

```groovy
// Run on any available agent
agent any

// Run on specific agent
agent { label 'linux' }

// Run in Docker container
agent {
    docker {
        image 'node:18-alpine'
        args '-v /tmp:/tmp'
    }
}

// Run on Kubernetes
agent {
    kubernetes {
        yaml '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: node
    image: node:18
'''
    }
}

// No agent at pipeline level (specify per stage)
agent none
```

### Stages and Steps

```groovy
stages {
    stage('Build') {
        steps {
            // Execute shell command
            sh 'make build'
            
            // Execute batch command (Windows)
            bat 'build.bat'
            
            // Execute Groovy script
            script {
                def version = sh(returnStdout: true, script: 'git describe --tags').trim()
                echo "Version: ${version}"
            }
            
            // Retry steps
            retry(3) {
                sh 'npm install'
            }
            
            // Timeout
            timeout(time: 10, unit: 'MINUTES') {
                sh 'npm test'
            }
        }
    }
}
```

### Environment Variables

```groovy
pipeline {
    agent any
    
    environment {
        // Pipeline-level variable
        APP_NAME = 'myapp'
        
        // Credential binding
        AWS_CREDENTIALS = credentials('aws-creds')
        
        // Dynamic variable
        BUILD_VERSION = "${env.BUILD_NUMBER}"
    }
    
    stages {
        stage('Example') {
            environment {
                // Stage-level variable
                STAGE_VAR = 'stage-value'
            }
            steps {
                echo "App: ${APP_NAME}"
                echo "Build: ${BUILD_VERSION}"
                echo "Job: ${env.JOB_NAME}"
                echo "Branch: ${env.BRANCH_NAME}"
            }
        }
    }
}
```

### Parameters

```groovy
pipeline {
    agent any
    
    parameters {
        string(name: 'ENVIRONMENT', defaultValue: 'staging', description: 'Environment to deploy')
        choice(name: 'REGION', choices: ['us-east-1', 'us-west-2', 'eu-west-1'], description: 'AWS Region')
        booleanParam(name: 'RUN_TESTS', defaultValue: true, description: 'Run tests?')
        text(name: 'DEPLOY_NOTES', defaultValue: '', description: 'Deployment notes')
        password(name: 'API_KEY', defaultValue: '', description: 'API Key')
    }
    
    stages {
        stage('Deploy') {
            steps {
                script {
                    echo "Deploying to ${params.ENVIRONMENT} in ${params.REGION}"
                    if (params.RUN_TESTS) {
                        sh 'npm test'
                    }
                }
            }
        }
    }
}
```

### Triggers

```groovy
pipeline {
    agent any
    
    triggers {
        // Poll SCM every 5 minutes
        pollSCM('H/5 * * * *')
        
        // Cron schedule (daily at 2 AM)
        cron('0 2 * * *')
        
        // Upstream job trigger
        upstream(upstreamProjects: 'job1,job2', threshold: hudson.model.Result.SUCCESS)
        
        // GitHub webhook
        githubPush()
    }
    
    stages {
        stage('Build') {
            steps {
                sh 'make build'
            }
        }
    }
}
```

### When Conditions

```groovy
stages {
    stage('Deploy to Production') {
        when {
            branch 'main'
        }
        steps {
            sh './deploy-prod.sh'
        }
    }
    
    stage('Deploy to Staging') {
        when {
            not {
                branch 'main'
            }
        }
        steps {
            sh './deploy-staging.sh'
        }
    }
    
    stage('Only on Weekdays') {
        when {
            expression { 
                return (new Date().getDay() > 0 && new Date().getDay() < 6)
            }
        }
        steps {
            echo 'Running on weekday'
        }
    }
    
    stage('Conditional') {
        when {
            allOf {
                branch 'main'
                environment name: 'DEPLOY', value: 'true'
            }
        }
        steps {
            sh './deploy.sh'
        }
    }
}
```

### Parallel Stages

```groovy
stages {
    stage('Parallel Tests') {
        parallel {
            stage('Unit Tests') {
                steps {
                    sh 'npm run test:unit'
                }
            }
            stage('Integration Tests') {
                steps {
                    sh 'npm run test:integration'
                }
            }
            stage('E2E Tests') {
                steps {
                    sh 'npm run test:e2e'
                }
            }
        }
    }
}
```

### Post Actions

```groovy
post {
    always {
        // Always run
        cleanWs()
        echo 'Pipeline completed'
    }
    
    success {
        // Run on success
        echo 'Build succeeded!'
        slackSend channel: '#builds',
                  color: 'good',
                  message: "Build succeeded: ${env.JOB_NAME} ${env.BUILD_NUMBER}"
    }
    
    failure {
        // Run on failure
        echo 'Build failed!'
        mail to: 'team@example.com',
             subject: "Failed: ${env.JOB_NAME}",
             body: "Build failed: ${env.BUILD_URL}"
    }
    
    unstable {
        // Run if marked unstable
        echo 'Build is unstable'
    }
    
    changed {
        // Run if build status changed
        echo 'Build status changed'
    }
    
    cleanup {
        // Always run after all other post conditions
        deleteDir()
    }
}
```

## Complete Pipeline Examples

### Node.js Application

```groovy
pipeline {
    agent {
        docker {
            image 'node:18-alpine'
        }
    }
    
    environment {
        NODE_ENV = 'production'
        NPM_CONFIG_CACHE = "${WORKSPACE}/.npm"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Install') {
            steps {
                sh 'npm ci'
            }
        }
        
        stage('Lint') {
            steps {
                sh 'npm run lint'
            }
        }
        
        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'npm run test:unit'
                    }
                }
                stage('Integration Tests') {
                    steps {
                        sh 'npm run test:integration'
                    }
                }
            }
        }
        
        stage('Build') {
            steps {
                sh 'npm run build'
            }
        }
        
        stage('Docker Build') {
            steps {
                script {
                    docker.build("myapp:${env.BUILD_NUMBER}")
                }
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                script {
                    docker.withRegistry('https://registry.example.com', 'docker-credentials') {
                        docker.image("myapp:${env.BUILD_NUMBER}").push()
                        docker.image("myapp:${env.BUILD_NUMBER}").push('latest')
                    }
                }
                sh './deploy.sh'
            }
        }
    }
    
    post {
        always {
            junit 'test-results/**/*.xml'
            publishHTML([
                reportDir: 'coverage',
                reportFiles: 'index.html',
                reportName: 'Coverage Report'
            ])
        }
        success {
            slackSend color: 'good', message: "Build succeeded: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
        }
        failure {
            slackSend color: 'danger', message: "Build failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}"
        }
        cleanup {
            cleanWs()
        }
    }
}
```

### Python Application

```groovy
pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.11'
        VIRTUAL_ENV = "${WORKSPACE}/venv"
    }
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    python${PYTHON_VERSION} -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }
        
        stage('Lint') {
            steps {
                sh '''
                    . venv/bin/activate
                    flake8 src/
                    black --check src/
                '''
            }
        }
        
        stage('Test') {
            steps {
                sh '''
                    . venv/bin/activate
                    pytest --cov=src --cov-report=xml --cov-report=html
                '''
            }
        }
        
        stage('Security Scan') {
            steps {
                sh '''
                    . venv/bin/activate
                    bandit -r src/
                    safety check
                '''
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh './deploy.sh'
            }
        }
    }
    
    post {
        always {
            junit 'test-results/*.xml'
            cobertura coberturaReportFile: 'coverage.xml'
        }
    }
}
```

### Multi-Branch Pipeline

```groovy
pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                echo "Building branch: ${env.BRANCH_NAME}"
                sh 'make build'
            }
        }
        
        stage('Test') {
            steps {
                sh 'make test'
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                sh './deploy.sh staging'
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to production?', ok: 'Deploy'
                sh './deploy.sh production'
            }
        }
    }
}
```

## Jenkins Configuration

### Installing Plugins

**Via UI:**
1. Go to "Manage Jenkins" → "Manage Plugins"
2. Click "Available" tab
3. Search and select plugins
4. Click "Install without restart"

**Via Jenkins CLI:**
```bash
java -jar jenkins-cli.jar -s http://localhost:8080/ install-plugin <plugin-name>
```

### Essential Plugins

- **Git Plugin** - Git integration
- **Pipeline** - Pipeline support
- **Docker Pipeline** - Docker integration
- **Blue Ocean** - Modern UI
- **Credentials Binding** - Credential management
- **GitHub** - GitHub integration
- **Slack Notification** - Slack integration
- **AWS Steps** - AWS integration
- **Kubernetes** - Kubernetes integration
- **SonarQube Scanner** - Code quality
- **JUnit** - Test results
- **HTML Publisher** - HTML reports

### Credentials Management

```groovy
// Using credentials in pipeline
pipeline {
    agent any
    
    stages {
        stage('Deploy') {
            steps {
                // Username/password
                withCredentials([usernamePassword(
                    credentialsId: 'docker-hub',
                    usernameVariable: 'USERNAME',
                    passwordVariable: 'PASSWORD'
                )]) {
                    sh 'docker login -u $USERNAME -p $PASSWORD'
                }
                
                // Secret text
                withCredentials([string(
                    credentialsId: 'api-key',
                    variable: 'API_KEY'
                )]) {
                    sh 'curl -H "Authorization: Bearer $API_KEY" https://api.example.com'
                }
                
                // SSH key
                withCredentials([sshUserPrivateKey(
                    credentialsId: 'ssh-key',
                    keyFileVariable: 'SSH_KEY'
                )]) {
                    sh 'ssh -i $SSH_KEY user@server "deploy.sh"'
                }
                
                // AWS credentials
                withAWS(credentials: 'aws-creds', region: 'us-east-1') {
                    sh 'aws s3 ls'
                }
            }
        }
    }
}
```

### Shared Libraries

```groovy
// Load shared library
@Library('my-shared-library@main') _

pipeline {
    agent any
    
    stages {
        stage('Build') {
            steps {
                // Use function from shared library
                buildApp(language: 'node', version: '18')
            }
        }
    }
}
```

## Jenkins CLI

```bash
# Download CLI
wget http://localhost:8080/jnlpJars/jenkins-cli.jar

# Build a job
java -jar jenkins-cli.jar -s http://localhost:8080/ build my-job

# Build with parameters
java -jar jenkins-cli.jar -s http://localhost:8080/ build my-job -p ENV=prod

# Get job status
java -jar jenkins-cli.jar -s http://localhost:8080/ get-job my-job

# List jobs
java -jar jenkins-cli.jar -s http://localhost:8080/ list-jobs

# Stop a build
java -jar jenkins-cli.jar -s http://localhost:8080/ stop-builds my-job 42

# Console output
java -jar jenkins-cli.jar -s http://localhost:8080/ console my-job 42
```

## Best Practices

### 1. Use Jenkinsfile
Store pipeline definition in version control.

### 2. Use Declarative Pipeline
Prefer declarative over scripted for readability.

### 3. Use Shared Libraries
Reuse common pipeline code.

### 4. Parameterize Builds
Make pipelines flexible with parameters.

### 5. Use Credentials Plugin
Never hardcode credentials.

### 6. Clean Workspace
Always clean up after builds.

```groovy
post {
    always {
        cleanWs()
    }
}
```

### 7. Use Docker Agents
Ensure consistent build environments.

### 8. Implement Proper Error Handling
```groovy
try {
    // build steps
} catch (Exception e) {
    currentBuild.result = 'FAILURE'
    throw e
} finally {
    // cleanup
}
```

### 9. Archive Artifacts
```groovy
post {
    always {
        archiveArtifacts artifacts: 'dist/**', fingerprint: true
    }
}
```

### 10. Parallel Execution
Speed up builds with parallel stages.

## Monitoring and Maintenance

### Backup Jenkins

```bash
# Backup Jenkins home
tar -czf jenkins-backup.tar.gz /var/lib/jenkins

# Backup specific job
tar -czf job-backup.tar.gz /var/lib/jenkins/jobs/my-job
```

### Monitoring

```groovy
// Monitor disk space
node {
    stage('Check Disk Space') {
        sh 'df -h'
        def diskSpace = sh(returnStdout: true, script: "df -h / | tail -1 | awk '{print \$5}' | sed 's/%//'").trim().toInteger()
        if (diskSpace > 80) {
            error("Disk space critical: ${diskSpace}%")
        }
    }
}
```

### Log Management

```bash
# View Jenkins logs
tail -f /var/log/jenkins/jenkins.log

# Or in Docker
docker logs -f jenkins
```

## Troubleshooting

### Check System Log
Manage Jenkins → System Log

### Increase Memory
```bash
# Edit /etc/default/jenkins
JAVA_ARGS="-Xmx2048m -Xms512m"

# Restart
sudo systemctl restart jenkins
```

### Debug Pipeline
```groovy
pipeline {
    agent any
    options {
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }
    stages {
        stage('Debug') {
            steps {
                echo "WORKSPACE: ${WORKSPACE}"
                echo "BUILD_NUMBER: ${BUILD_NUMBER}"
                echo "JOB_NAME: ${JOB_NAME}"
                sh 'env | sort'
            }
        }
    }
}
```

## Useful Resources

- Official Documentation: https://www.jenkins.io/doc/
- Pipeline Syntax: https://www.jenkins.io/doc/book/pipeline/syntax/
- Plugin Index: https://plugins.jenkins.io/
- Community: https://community.jenkins.io/
- GitHub: https://github.com/jenkinsci/jenkins

## Quick Reference

| Task | Command/Location |
|------|------------------|
| Access Jenkins | `http://localhost:8080` |
| Initial password | `/var/jenkins_home/secrets/initialAdminPassword` |
| Configuration | Manage Jenkins → Configure System |
| Plugins | Manage Jenkins → Manage Plugins |
| Credentials | Manage Jenkins → Manage Credentials |
| System log | Manage Jenkins → System Log |
| Script console | Manage Jenkins → Script Console |
| Restart | `http://localhost:8080/restart` |

---

*This guide covers Jenkins fundamentals. For production use, consider implementing security best practices, distributed builds, monitoring, and integration with your specific tech stack.*

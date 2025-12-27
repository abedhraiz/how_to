# DevSecOps Guide

Automating security controls within the CI/CD pipeline.

## The Pipeline Security Gates

### 1. IDE / Pre-Commit (Local)
- **Linting**: ESLint (security plugins), Bandit (Python).
- **Secret Scanning**: Gitleaks.

### 2. Build / Commit Stage (CI)
- **SAST (Static Application Security Testing)**: Analyzes source code for vulnerabilities without running it.
    - *Tools*: SonarQube, CodeQL, Semgrep.
- **SCA (Software Composition Analysis)**: Checks dependencies for known vulnerabilities (CVEs).
    - *Tools*: OWASP Dependency-Check, Snyk, npm audit, pip-audit.

### 3. Test / Deploy Stage (CD)
- **DAST (Dynamic Application Security Testing)**: Attacks the running application from the outside.
    - *Tools*: OWASP ZAP, Burp Suite Enterprise.
- **Container Scanning**: Checks Docker images for OS-level vulnerabilities.
    - *Tools*: Trivy, Clair, Docker Scout.

### 4. Runtime / Monitor (Production)
- **RASP (Runtime Application Self-Protection)**: Instruments the app to detect attacks in real-time.
- **WAF (Web Application Firewall)**: Filters malicious HTTP traffic.

## Example GitHub Actions Workflow

```yaml
name: DevSecOps Pipeline

on: [push]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run SAST (Semgrep)
        uses: returntocorp/semgrep-action@v1

      - name: Run SCA (Trivy fs)
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          ignore-unfixed: true
          format: 'table'
          exit-code: '1' # Fail build on high severity
          severity: 'CRITICAL,HIGH'
```

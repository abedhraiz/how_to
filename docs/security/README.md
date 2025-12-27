# Security & DevSecOps

Integrating security into every stage of the software development lifecycle.

## Purpose

Security is not an afterthought; it is a continuous process. This section covers application security, infrastructure protection, and the "Shift Left" philosophy.

## Guides

- **[Application Security (AppSec)](./appsec-guide.md)** - OWASP Top 10, Input Validation, and Auth.
- **[Secrets Management](./secrets-management.md)** - Handling API keys, passwords, and certificates safely.
- **[DevSecOps Pipelines](./devsecops-guide.md)** - Automating security scans (SAST/DAST) in CI/CD.

## Key Concepts

### The CIA Triad
- **Confidentiality**: Only authorized users can access data.
- **Integrity**: Data is accurate and trustworthy.
- **Availability**: Data is accessible when needed.

### Shift Left
Moving security testing earlier in the development process (e.g., IDE plugins, pre-commit hooks) rather than waiting for a penetration test before production.

### Zero Trust
"Never trust, always verify." Assume the network is hostile. Authenticate and authorize every request, even inside the firewall.

## Related Categories

- 🤖 **[AI Security](../ai-ml-frameworks/ai-security-guide.md)** - LLM specific vulnerabilities.
- ☁️ **[Cloud Platforms](../cloud-platforms/README.md)** - IAM and Cloud Security Posture Management (CSPM).
- 🔄 **[CI/CD Automation](../cicd-automation/README.md)** - Where security tools run.

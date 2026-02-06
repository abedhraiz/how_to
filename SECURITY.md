# Security Policy

## Reporting a Vulnerability

We take the security of this documentation repository seriously. While this repository primarily contains documentation and educational content, we recognize that security concerns can arise in various forms.

### What to Report

Please report any of the following:

- **Malicious code examples** in documentation that could harm users
- **Vulnerable dependencies** in example code or workflows
- **Exposed credentials** or sensitive information
- **XSS vulnerabilities** in markdown or embedded HTML
- **Broken authentication examples** that could mislead users
- **Insecure configuration examples** that could lead to security issues

### How to Report

**DO NOT** open a public issue for security vulnerabilities.

Instead, please report security concerns by:

1. **Email**: Contact the maintainer via [LinkedIn](https://www.linkedin.com/in/abed-elalim-hraiz-25bb90113/)
2. **GitHub Security**: Use GitHub's private vulnerability reporting feature if available

### What to Include

When reporting a vulnerability, please include:

- **Description**: Clear description of the security concern
- **Location**: File path and line number (if applicable)
- **Impact**: Potential impact if exploited
- **Reproduction**: Steps to verify the issue
- **Suggested fix**: If you have a recommendation

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Depends on severity and complexity

### Security Best Practices

This repository follows these security practices:

#### For Documentation
- ✅ No hardcoded credentials in examples
- ✅ All examples use placeholder values (e.g., `YOUR_API_KEY`)
- ✅ Security warnings included in sensitive topics
- ✅ Regular review of code examples for security issues

#### For Repository
- ✅ Branch protection on main branch
- ✅ Required reviews for pull requests
- ✅ Automated security scanning via GitHub Actions
- ✅ Regular dependency updates for workflows

#### For Examples
- ✅ Docker images use specific versions (not `latest`)
- ✅ Secrets management examples follow best practices
- ✅ Network configurations show security-first approaches
- ✅ Authentication examples use modern, secure methods

### Scope

This security policy covers:

- ✅ All markdown documentation files
- ✅ Code examples within documentation
- ✅ GitHub Actions workflows
- ✅ Configuration files (.yml, .json, etc.)
- ❌ Third-party tools documented (report to their maintainers)

### Supported Versions

We support the latest version of the documentation. When security issues are found:

- **Critical**: Fixed immediately with emergency release
- **High**: Fixed within 7 days
- **Medium**: Fixed in next regular update
- **Low**: Tracked for future update

### Security Updates

Security fixes will be:

1. Applied to the main branch
2. Documented in CHANGELOG.md
3. Announced in commit messages
4. Credited to reporter (unless anonymity requested)

### Additional Resources

For security topics covered in this repository:

- [Security & DevSecOps Guide](docs/security/README.md)
- [Secrets Management](docs/security/secrets-management.md)
- [DevSecOps Best Practices](docs/security/devsecops-guide.md)
- [Application Security](docs/security/appsec-guide.md)

### Responsible Disclosure

We believe in responsible disclosure:

- We will work with you to understand and address the issue
- We will keep you informed of our progress
- We will credit you for the discovery (unless you prefer to remain anonymous)
- We ask that you do not publicly disclose the vulnerability until we've addressed it

### Contact

For all security-related matters:

- **Primary Contact**: [Abed Elalim Hraiz on LinkedIn](https://www.linkedin.com/in/abed-elalim-hraiz-25bb90113/)
- **Repository Issues**: Use private reporting through GitHub (for non-critical issues)

---

**Thank you for helping keep this repository and its users secure!**

Last updated: February 2026

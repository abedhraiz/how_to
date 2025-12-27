# Application Security (AppSec) Guide

Building secure software by design and defending against common attacks.

## OWASP Top 10 (2021)

The standard awareness document for developers and web application security.

1.  **Broken Access Control**: Users acting outside of their intended permissions.
    *   *Fix*: Deny by default. Implement Role-Based Access Control (RBAC).
2.  **Cryptographic Failures**: Exposure of sensitive data (passwords, health records).
    *   *Fix*: Encrypt data at rest and in transit (TLS). Don't roll your own crypto.
3.  **Injection**: SQL, NoSQL, OS command injection.
    *   *Fix*: Use parameterized queries (Prepared Statements). Validate all input.
4.  **Insecure Design**: Missing security controls in the architecture.
    *   *Fix*: Threat modeling during the design phase.
5.  **Security Misconfiguration**: Default passwords, open cloud storage, verbose error messages.
    *   *Fix*: Hardening guides, automated configuration audits.
6.  **Vulnerable and Outdated Components**: Using libraries with known CVEs.
    *   *Fix*: Software Composition Analysis (SCA) tools like Dependabot or Snyk.
7.  **Identification and Authentication Failures**: Weak passwords, missing MFA.
    *   *Fix*: Multi-Factor Authentication, strong password policies, rate limiting.
8.  **Software and Data Integrity Failures**: Code from untrusted sources, insecure CI/CD.
    *   *Fix*: Code signing, verifying checksums.
9.  **Security Logging and Monitoring Failures**: Not detecting breaches in time.
    *   *Fix*: Centralized logging, alerting on suspicious activities.
10. **Server-Side Request Forgery (SSRF)**: Fetching a remote resource without validating the user-supplied URL.
    *   *Fix*: Allow-lists for outgoing requests.

## Input Validation

- **Sanitization**: Cleaning input (removing dangerous characters).
- **Validation**: Checking if input meets criteria (type, length, format).
- **Encoding**: Converting data into a safe format for the context (HTML encoding, URL encoding).

## Authentication vs. Authorization

- **Authentication (AuthN)**: Who are you? (Login, OIDC, SAML).
- **Authorization (AuthZ)**: What can you do? (RBAC, ABAC, OAuth Scopes).

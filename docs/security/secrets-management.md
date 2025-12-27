# Secrets Management Guide

Best practices for handling sensitive credentials, API keys, and certificates.

## The Golden Rule
**NEVER commit secrets to version control (Git).**

## Where to Store Secrets

### 1. Environment Variables
The standard 12-factor app approach.
- **Pros**: Simple, language-agnostic.
- **Cons**: Can leak in logs or process listings if not careful.

### 2. Cloud Secret Managers
Dedicated services for storage, rotation, and access control.
- **AWS Secrets Manager** / **Parameter Store**
- **Azure Key Vault**
- **Google Secret Manager**
- **HashiCorp Vault** (Cloud-agnostic, enterprise standard)

### 3. Kubernetes Secrets
Native K8s object for storing sensitive data.
- **Warning**: By default, they are just base64 encoded, not encrypted. Enable *Encryption at Rest* in etcd.

## Detection & Prevention

### Pre-Commit Hooks
Tools that scan your staged files for secrets *before* you commit.
- **Talisman**
- **Gitleaks**

### CI/CD Scanning
Scanning the entire history of the repository.
- **TruffleHog**
- **GitHub Secret Scanning** (Built-in for public repos).

## What to do if you leak a secret?

1.  **Revoke** the secret immediately.
2.  **Rotate** (Generate a new one).
3.  **Update** applications with the new secret.
4.  **Rewrite** Git history (BFG Repo-Cleaner) if necessary to remove the trace (or delete the repo if possible).

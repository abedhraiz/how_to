# Python Automation

Practical, production-minded patterns for automating work with Python: scripts, CLIs, scheduled jobs, API integrations, and safe task execution.

## What You'll Learn

- How to structure automation scripts so they stay maintainable
- Building CLIs with good UX (help text, exit codes, dry-run)
- Logging, configuration, secrets, and error handling
- File and data automation (CSV/JSON/Parquet basics)
- HTTP automation (REST calls, retries, rate limits)
- Running shell commands safely (subprocess)
- Scheduling (cron) and containerizing jobs

## Guides

- **[Python Automation Guide](./python-automation-guide.md)** - End-to-end patterns and examples

## When to Use Python for Automation

Use Python when you need:

- Cross-platform scripts (Linux/macOS/Windows)
- Rich parsing and data transforms
- Reliable HTTP integrations with retries/backoff
- Better testability than ad-hoc shell scripts

If the task is a one-liner or purely file plumbing, Bash can still be the simplest tool.

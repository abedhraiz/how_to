# Python Automation Guide

Automation with Python is most effective when you treat scripts like small products: predictable inputs/outputs, solid logging, safe defaults, and easy troubleshooting.

## Table of Contents

- [Principles](#principles)
- [Project Structure for Automation](#project-structure-for-automation)
- [CLI Basics (argparse)](#cli-basics-argparse)
- [Logging](#logging)
- [Configuration & Secrets](#configuration--secrets)
- [File Automation (Paths, CSV, JSON)](#file-automation-paths-csv-json)
- [HTTP Automation (Requests, Retries)](#http-automation-requests-retries)
- [Running Shell Commands Safely](#running-shell-commands-safely)
- [Scheduling (cron)](#scheduling-cron)
- [Containerizing a Job (Docker)](#containerizing-a-job-docker)
- [Testing Automation Code](#testing-automation-code)
- [Operational Checklist](#operational-checklist)

---

## Principles

- **Idempotent by default**: re-running should not cause harm.
- **Dry-run mode**: show what would change.
- **Clear exit codes**: `0` success, non-zero on failure.
- **Observable**: structured logs and useful error messages.
- **Safe by default**: avoid destructive actions unless explicitly requested.

---

## Project Structure for Automation

For a single script, one file is fine. For repeated or shared automation, prefer a small structure:

```
automation/
  README.md
  pyproject.toml  (or requirements.txt)
  src/
    automation/
      __init__.py
      main.py
      clients/
      utils/
  tests/
```

Key idea: keep **pure logic** in importable modules, and keep the **entrypoint** thin.

---

## CLI Basics (argparse)

A good CLI should support `--help`, `--dry-run`, and explicit inputs.

```python
#!/usr/bin/env python3

import argparse
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sync-things",
        description="Example automation CLI with safe defaults",
    )
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.dry_run:
        print(f"[DRY RUN] Would process: {args.input}")
        return 0

    # Real work here...
    print(f"Processing: {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

---

## Logging

Prefer Python's standard `logging` over `print()` for anything that runs unattended.

```python
import logging

logger = logging.getLogger("automation")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def run_job() -> None:
    logger.info("Job started")
```

Tips:
- Use `INFO` for normal progress, `WARNING` for unusual but recoverable issues, `ERROR` for failures.
- Include identifiers (record ids, filenames) in log lines.

---

## Configuration & Secrets

### Configuration

Use environment variables for deploy-time configuration.

```python
import os

API_BASE_URL = os.environ.get("API_BASE_URL", "https://example.com")
TIMEOUT_SECONDS = int(os.environ.get("TIMEOUT_SECONDS", "30"))
```

### Secrets

- Don’t commit secrets.
- Read secrets from environment variables or your secret manager.
- Avoid printing secrets in logs.

---

## File Automation (Paths, CSV, JSON)

Use `pathlib` for OS-safe path handling.

```python
from pathlib import Path
import json


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
```

For CSV, prefer `csv` module for simple tasks and `pandas` for heavier transforms.

---

## HTTP Automation (Requests, Retries)

For simple REST calls, `requests` is the common choice (third-party). If you don’t want extra deps, use `urllib`.

A minimal, dependency-free pattern using `urllib`:

```python
import json
import urllib.request


def get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)
```

Retry guidance:
- Retry **timeouts** and **5xx** responses.
- Do not blindly retry **4xx**.
- Use exponential backoff and a max retry count.

---

## Running Shell Commands Safely

Use `subprocess.run()` with argument lists (not a single string) to avoid quoting issues.

```python
import subprocess


def run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {cmd}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
```

Avoid `shell=True` unless you truly need shell features.

---

## Scheduling (cron)

Example cron entry to run every day at 02:30:

```cron
30 2 * * * /usr/bin/python3 /opt/jobs/sync_things.py --input /data/input.json >> /var/log/sync_things.log 2>&1
```

Recommendations:
- Always log to a file (or stdout in containers).
- Make scripts exit non-zero on failure so schedulers can alert.

---

## Containerizing a Job (Docker)

A minimal job container (no extra dependencies):

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY sync_things.py /app/sync_things.py

ENTRYPOINT ["python", "/app/sync_things.py"]
```

Run it:

```bash
docker build -t sync-things:latest .
docker run --rm -e API_BASE_URL=https://example.com sync-things:latest --help
```

---

## Testing Automation Code

Even small automation benefits from tests:

- Unit-test parsing and transformation logic
- Use temporary directories (`tempfile`) for file-based tests
- Mock network calls

If you already use `pytest` elsewhere, prefer it for readability.

---

## Operational Checklist

Before you rely on a Python automation job in production:

- Logs are readable and include context
- Script is idempotent or protected by locks
- Supports `--dry-run` (when appropriate)
- Has timeouts for network calls
- Retries only safe-to-retry failures
- Fails fast with non-zero exit codes
- Has documentation for inputs/outputs and expected runtime

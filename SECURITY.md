# Security policy

## Supported versions

Chronicle is an alpha research framework. Security fixes are applied to the
latest `0.1.x` release and to `main`.

## Reporting a vulnerability

Please **do not open a public issue** for a suspected vulnerability.

Use GitHub's private vulnerability reporting:

<https://github.com/g1gowda123/chronicle/security/advisories/new>

Include the affected version or commit, a minimal reproduction, the impact, and
any suggested fix. You can expect an acknowledgement within a few business days.

## Scope

Chronicle processes local video and SQLite memories. It does not run a public
service by default. Relevant reports include:

- unsafe deserialization or path traversal in import/export paths;
- command injection through CLI or dashboard inputs;
- leakage of local paths, environment variables, or credentials in exports;
- dependency vulnerabilities with a practical exploit path.

Model weights, third-party datasets, and the optional Ollama/NiceGUI stack are
outside this repository's control; report upstream issues to their maintainers.

## Secrets

Never commit `.env` files, API keys, cloud credentials, or private dataset URLs.
If a secret is committed, rotate it first, then contact the maintainers so the
history can be cleaned up.

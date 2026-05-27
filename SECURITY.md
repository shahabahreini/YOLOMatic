# Security Policy

## Supported Versions

YOLOmatic is distributed as source and Python package metadata. Security fixes
target the latest minor release line.

| Version | Supported |
| --- | --- |
| 4.4.x | Yes |
| 4.3.x | Critical fixes only |
| < 4.3 | No |

## Reporting a Vulnerability

Please do not open a public GitHub issue for security reports.

Email private vulnerability details to shahabahreini@hotmail.com with:

- Affected version or commit
- Operating system and Python version
- Minimal reproduction steps
- Impact assessment, if known
- Whether the issue involves credentials, local files, model artifacts, or
  remote downloads

You can expect an acknowledgement within 72 hours and a status update within 7
days. Confirmed vulnerabilities will be fixed privately first, then disclosed in
the changelog and GitHub release notes once a patch is available.

## Scope

In scope:

- Unsafe handling of credentials or environment variables
- Path traversal or arbitrary file writes from dataset conversion or uploads
- Unsafe deserialization of project configuration
- Dependency vulnerabilities that are exploitable through YOLOmatic workflows

Out of scope:

- Vulnerabilities requiring already-compromised local machines
- Issues in upstream ML frameworks unless YOLOmatic introduces the exploit path
- Denial-of-service reports based only on intentionally huge local datasets

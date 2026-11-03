# Security policy

Report vulnerabilities affecting WSPBench's current code, artifact export and replay, provider execution, or benchmark integrity through [GitHub private vulnerability reporting](https://github.com/alexcarterhq/wspbench/security/advisories/new).

Include the affected source commit or release, relevant component, expected impact, and a minimal reproduction using synthetic data. Keep active held-out instances, answer keys, generation seeds, private checkpoints, credentials, account identifiers, and private run logs out of public issues, pull requests, and attachments. Share only the material needed to reproduce the problem through the private channel.

If private reporting is unavailable, open an issue requesting a private reporting channel without including sensitive details. Ordinary scoring questions or methodological bugs that do not expose protected material can use public issues.

If a credential has been exposed, revoke it with its provider promptly. Deleting it from a file or a Git commit does not invalidate the credential. Report the exposure privately without reposting the credential.

Public development examples and their released grading fixtures are intentionally inspectable. Active held-out material remains private. A protocol or access-control bypass, unintended held-out disclosure, or executable content escaping the intended evaluation boundary is a security concern even when it also affects benchmark scores.

Maintainers will coordinate investigation, fixes, and disclosure through the private report. Please allow that coordination before publishing exploit details or protected benchmark material. This research preview does not operate a public submission service or promise a fixed response deadline.

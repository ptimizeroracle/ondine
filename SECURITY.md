# Security Policy

The Ondine team takes the security of our LLM batch processing SDK seriously. We appreciate your efforts to responsibly disclose your findings, and will make every effort to acknowledge your contributions.

## Supported Versions

We currently provide security updates for the following versions of Ondine:

| Version | Supported          |
| ------- | ------------------ |
| v1.7.0  | :white_check_mark: |
| < 1.7.0 | :x:                |

If you are using an unsupported version, we strongly recommend upgrading to the latest version.

## Reporting a Vulnerability

If you discover a security vulnerability within Ondine, please report it to us privately. **Do not disclose the vulnerability publicly until a fix has been released.**

You can report vulnerabilities using one of the following methods:

1. **GitHub Private Security Advisory (Preferred):**
   Navigate to the [Security tab](https://github.com/ptimizeroracle/ondine/security) of our repository and click the **"Report a vulnerability"** button. This allows you to securely disclose the issue directly to the maintainers.

2. **Email:**
   Send an email to **git@binblok.com**. Please encrypt your message if possible and include "Security Vulnerability" in the subject line.

### What to Include in Your Report

To help us understand and resolve the issue quickly, please include the following information in your report:

* **Description:** A clear and detailed description of the vulnerability.
* **Steps to Reproduce:** Step-by-step instructions to reproduce the issue. Include any necessary configuration, code snippets, or sample inputs.
* **Impact:** The potential impact of the vulnerability (e.g., what an attacker could achieve).
* **Environment:** The version of Ondine, Python version, and operating system where the vulnerability was observed.
* **Suggested Fix (Optional):** If you have a proposed solution or patch, please include it.

### Response Service Level Agreement (SLA)

We are committed to resolving security issues in a timely manner. Our target SLAs are as follows:

* **Acknowledgment:** We will acknowledge receipt of your vulnerability report within **48 hours**.
* **Resolution/Patch:** We aim to provide a fix or mitigation for confirmed vulnerabilities within **90 days** of the initial report.

We will keep you updated on the progress of the investigation and the timeline for a fix.

## Out of Scope

The following types of reports are generally considered out of scope and will not be accepted:

* Vulnerabilities in third-party dependencies (unless they can be uniquely exploited through Ondine's specific usage). Please report these directly to the respective upstream projects.
* Denial of Service (DoS) attacks against our infrastructure or website.
* Theoretical vulnerabilities without a proven exploit or clear impact.
* Issues related to the user's own infrastructure or misconfiguration of the SDK.
* Spam, social engineering, or phishing attacks against Ondine contributors.

Thank you for helping keep Ondine and its users safe!

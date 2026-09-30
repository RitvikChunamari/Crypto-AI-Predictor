# Security Policy

## Supported Versions

Currently, the `master` branch is supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

If you believe you have found a security vulnerability in Kinetix Quant, please report it to us by email. We will investigate all reports and do our best to quickly fix the problem.

## Secrets and API Keys
This repository has been scrubbed for hardcoded secrets. Please ensure that if you are forking this repo and adding your own keys (e.g. OpenAI, Binance Premium), you use environment variables `.env` and never commit them to the repository. The `.gitignore` is already configured to ignore `.env` files.

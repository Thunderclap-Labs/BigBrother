# Deployment and Security Hardening

## Deployment Overview

This application is designed to run as a local desktop service on a Windows or Linux workstation. It is not intended to be exposed to the public internet.

## Prerequisites

- Python 3.10+
- All dependencies installed: `pip install -r requirements.txt`
- `config/settings.yaml` configured for the target machine

## Running as a Background Service

### Windows (Task Scheduler)

1. Open **Task Scheduler** → Create Basic Task.
2. Set trigger: **At log on**.
3. Action: **Start a program** → `pythonw.exe` with argument `main.py` and working directory set to the project root.
4. Enable **Run only when user is logged on**.

### Linux (systemd)

Create `/etc/systemd/system/workmonitor.service`:

```ini
[Unit]
Description=WorkSimulation Monitor
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=/opt/backub-web
ExecStart=/usr/bin/python3 main.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable workmonitor
sudo systemctl start workmonitor
```

## Security Hardening

### Network Exposure

- The Flask server binds to `127.0.0.1` only by default. Do **not** change this to `0.0.0.0` unless the host is behind a firewall.
- No authentication is implemented — the dashboard is accessible to any process on localhost.

### Credentials and Secrets

- **Never commit** `config/token.json` or `config/local.yaml` to version control. Both are listed in `.gitignore`.
- Rotate Google OAuth tokens regularly. Revoke tokens via the [Google Security page](https://myaccount.google.com/permissions) if the machine is compromised.
- Store any API keys in `config/local.yaml`, not in `config/settings.yaml`.

### File Permissions

Restrict access to config files containing credentials:

```bash
chmod 600 config/token.json
chmod 600 config/local.yaml
```

### Dependency Auditing

Periodically audit dependencies for known vulnerabilities:

```bash
pip install pip-audit
pip-audit
```

### Logging

Application logs are written to stdout only. Redirect to a file with restricted permissions if long-term logging is required:

```bash
python main.py >> /var/log/workmonitor.log 2>&1
chmod 640 /var/log/workmonitor.log
```

## Updating

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

Restart the service after updating.

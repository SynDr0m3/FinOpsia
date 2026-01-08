import os

EMAIL_CONFIG = {
    "from_email": os.environ.get("ALERT_FROM_EMAIL"),
    "to_email": os.environ.get("ALERT_TO_EMAIL"),
    "smtp_server": os.environ.get("ALERT_SMTP_SERVER"),
    "smtp_port": int(os.environ.get("ALERT_SMTP_PORT", 587)),
    "smtp_user": os.environ.get("ALERT_SMTP_USER"),
    "smtp_password": os.environ.get("ALERT_SMTP_PASSWORD"),
}

SLACK_WEBHOOK_URL = os.environ.get("ALERT_SLACK_WEBHOOK")

from .config import EMAIL_CONFIG, SLACK_WEBHOOK_URL
"""
Alerting utilities for FinOpsia monitoring.
Supports email and Slack/webhook notifications.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from .logger import logger

def send_email_alert(subject: str, body: str, to_email: str, from_email: str, smtp_server: str, smtp_port: int, smtp_user: str, smtp_password: str):
	"""
	Send an email alert using SMTP.
	"""
	try:
		msg = MIMEMultipart()
		msg['From'] = from_email
		msg['To'] = to_email
		msg['Subject'] = subject
		msg.attach(MIMEText(body, 'plain'))

		with smtplib.SMTP(smtp_server, smtp_port) as server:
			server.starttls()
			server.login(smtp_user, smtp_password)
			server.send_message(msg)
		logger.success(f"Email alert sent to {to_email}", extra={"account_id": None, "user_id": None})
	except Exception as e:
		logger.error(f"Failed to send email alert: {e}", extra={"account_id": None, "user_id": None})


def send_slack_alert(message: str, webhook_url: str):
	"""
	Send an alert to Slack (or compatible webhook).
	"""
	try:
		payload = {"text": message}
		response = requests.post(webhook_url, json=payload)
		if response.status_code == 200:
			logger.success("Slack alert sent successfully", extra={"account_id": None, "user_id": None})
		else:
			logger.error(f"Slack alert failed: {response.status_code} {response.text}", extra={"account_id": None, "user_id": None})
	except Exception as e:
		logger.error(f"Failed to send Slack alert: {e}", extra={"account_id": None, "user_id": None})


def send_alert(message: str, email_config: dict = None, slack_webhook: str = None):
	"""
	Send alert via all configured channels.
	"""
	# Use config defaults if not provided
	if email_config is None:
		email_config = EMAIL_CONFIG
	if slack_webhook is None:
		slack_webhook = SLACK_WEBHOOK_URL

	if email_config and all(email_config.values()):
		send_email_alert(
			subject=email_config.get('subject', 'FinOpsia Alert'),
			body=message,
			to_email=email_config['to_email'],
			from_email=email_config['from_email'],
			smtp_server=email_config['smtp_server'],
			smtp_port=email_config['smtp_port'],
			smtp_user=email_config['smtp_user'],
			smtp_password=email_config['smtp_password'],
		)
	if slack_webhook:
		send_slack_alert(message, slack_webhook)

import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


class EmailService:
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender_email = username
        self.app_name = "AI Productivity App"

    async def send_password_reset_email(self, to_email: str, reset_token: str, base_url: str):
        """Send password reset email with secure token"""
        reset_link = f"{base_url}/reset-password?token={reset_token}"

        subject = f"{self.app_name} - Password Reset Request"
        body = f"""
        <html>
            <body>
                <h2>Password Reset Request</h2>
                <p>You requested a password reset for your {self.app_name} account.</p>
                <p>Click the link below to reset your password:</p>
                <p><a href="{reset_link}">Reset Password</a></p>
                <p>This link will expire in 1 hour.</p>
                <p>If you didn't request this, please ignore this email.</p>
            </body>
        </html>
        """

        await self._send_email(to_email, subject, body)

    async def _send_email(self, to_email: str, subject: str, html_body: str):
        """Internal method to send emails via SMTP"""
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{self.app_name} <{self.sender_email}>"
        message["To"] = to_email

        html_part = MIMEText(html_body, "html")
        message.attach(html_part)

        async with aiosmtplib.SMTP(hostname=self.smtp_host, port=self.smtp_port) as server:
            await server.starttls()
            await server.login(self.username, self.password)
            await server.send_message(message)
# This file only help send OTP, it doesn't generate, validate etc.

from html import escape
import httpx
from app.core.config import settings

BREVO_EMAIL_API_URL = "https://api.brevo.com/v3/smtp/email"


class BrevoEmailError(Exception):
    """Raised when Brevo fails to send a transactional email."""


async def send_password_reset_otp(
    recipient_email: str,
    otp: str,
) -> None:
    """Send a password-reset OTP through Brevo."""

    expires_in = settings.password_reset_otp_expire_minutes
    safe_otp = escape(otp)

    payload = {
        "sender": {
            "name": settings.brevo_sender_name,
            "email": str(settings.brevo_sender_email),
        },
        "to": [
            {
                "email": recipient_email,
            }
        ],
        "subject": "Your ForgePath password reset code",
        "textContent": (
            f"Your ForgePath password reset code is {otp}.\n\n"
            f"This code expires in {expires_in} minutes.\n"
            "If you did not request a password reset, ignore this email."
        ),
        "htmlContent": f"""
            <!DOCTYPE html>
            <html lang="en">
                <body style="font-family: Arial, sans-serif; color: #222;">
                    <h2>Reset your ForgePath password</h2>
                    <p>Use the following code to reset your password:</p>
                    <div style="
                        font-size: 32px;
                        font-weight: bold;
                        letter-spacing: 8px;
                        padding: 16px 0;
                    ">{safe_otp}</div>
                    <p>
                        This code expires in
                        <strong>{expires_in} minutes</strong>.
                    </p>
                    <p>
                        If you did not request a password reset,
                        you can safely ignore this email.
                    </p>
                </body>
            </html>
        """,
    }

    headers = {
        "api-key": settings.brevo_api_key.get_secret_value(),
        "accept": "application/json",
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                BREVO_EMAIL_API_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        raise BrevoEmailError(
            f"Brevo rejected the email request "
            f"(HTTP {exc.response.status_code})."
        ) from exc

    except httpx.RequestError as exc:
        raise BrevoEmailError(
            "Could not connect to Brevo to send the email."
        ) from exc



async def send_email_verification_otp(
    recipient_email: str,
    otp: str,
) -> None:
    """Send an email-verification OTP through Brevo."""

    expires_in = settings.password_reset_otp_expire_minutes
    safe_otp = escape(otp)

    payload = {
        "sender": {
            "name": settings.brevo_sender_name,
            "email": str(settings.brevo_sender_email),
        },
        "to": [
            {
                "email": recipient_email,
            }
        ],
        "subject": "Verify your ForgePath email address",
        "textContent": (
            f"Your ForgePath email verification code is {otp}.\n\n"
            f"This code expires in {expires_in} minutes.\n"
            "If you did not request this code, you can ignore this email."
        ),
        "htmlContent": f"""
            <!DOCTYPE html>
            <html lang="en">
                <body style="font-family: Arial, sans-serif; color: #222;">
                    <h2>Verify your ForgePath email address</h2>
                    <p>Use this code to verify your email address:</p>
                    <div style="
                        font-size: 32px;
                        font-weight: bold;
                        letter-spacing: 8px;
                        padding: 16px 0;
                    ">{safe_otp}</div>
                    <p>
                        This code expires in
                        <strong>{expires_in} minutes</strong>.
                    </p>
                    <p>
                        If you did not request this code,
                        you can safely ignore this email.
                    </p>
                </body>
            </html>
        """,
    }

    headers = {
        "api-key": settings.brevo_api_key.get_secret_value(),
        "accept": "application/json",
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                BREVO_EMAIL_API_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        raise BrevoEmailError(
            f"Brevo rejected the email request "
            f"(HTTP {exc.response.status_code})."
        ) from exc

    except httpx.RequestError as exc:
        raise BrevoEmailError(
            "Could not connect to Brevo to send the email."
        ) from exc
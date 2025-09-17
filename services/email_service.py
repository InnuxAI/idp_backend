import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

class EmailService:
    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("FROM_EMAIL")
        
    def _send_email_sync(self, to_email: str, subject: str, body: str, is_html: bool = False):
        """Send email synchronously"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html' if is_html else 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
                
            return True
        except Exception as e:
            print(f"Failed to send email to {to_email}: {e}")
            return False
    
    async def send_email(self, to_email: str, subject: str, body: str, is_html: bool = False) -> bool:
        """Send email asynchronously"""
        if not all([self.smtp_username, self.smtp_password, self.from_email]):
            print("Email configuration not complete. Please set SMTP_USERNAME, SMTP_PASSWORD, and FROM_EMAIL")
            return False
            
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, 
                self._send_email_sync, 
                to_email, 
                subject, 
                body, 
                is_html
            )
    
    async def send_otp_email(self, email: str, otp: str) -> bool:
        """Send OTP verification email"""
        subject = "Your IDP Verification Code"
        body = f"""
        <html>
        <body>
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; font-family: Arial, sans-serif;">
                <h2 style="color: #333; text-align: center;">Email Verification</h2>
                <p>Hello,</p>
                <p>You have requested to verify your email address for IDP (Intelligent Document Processing) platform.</p>
                <div style="background-color: #f5f5f5; padding: 20px; text-align: center; margin: 20px 0; border-radius: 5px;">
                    <h3 style="color: #333; margin: 0;">Your verification code is:</h3>
                    <h1 style="color: #007bff; font-size: 36px; margin: 10px 0; letter-spacing: 5px;">{otp}</h1>
                </div>
                <p>This code will expire in 10 minutes for security reasons.</p>
                <p>If you didn't request this verification, please ignore this email.</p>
                <hr style="border: 1px solid #eee; margin: 30px 0;">
                <p style="color: #666; font-size: 12px;">
                    This is an automated message from IDP Platform. Please do not reply to this email.
                </p>
            </div>
        </body>
        </html>
        """
        return await self.send_email(email, subject, body, is_html=True)
    
    async def send_password_reset_email(self, email: str, reset_link: str) -> bool:
        """Send password reset email"""
        subject = "Password Reset Request"
        body = f"""
        <html>
        <body>
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; font-family: Arial, sans-serif;">
                <h2 style="color: #333; text-align: center;">Password Reset Request</h2>
                <p>Hello,</p>
                <p>You have requested to reset your password for IDP (Intelligent Document Processing) platform.</p>
                <p>Click the button below to reset your password:</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{reset_link}" 
                       style="background-color: #007bff; color: white; padding: 12px 30px; 
                              text-decoration: none; border-radius: 5px; display: inline-block;
                              font-weight: bold;">
                        Reset Password
                    </a>
                </div>
                <p>Or copy and paste this link in your browser:</p>
                <p style="word-break: break-all; color: #007bff;">{reset_link}</p>
                <p><strong>This link will expire in 1 hour for security reasons.</strong></p>
                <p>If you didn't request this password reset, please ignore this email. Your password will remain unchanged.</p>
                <hr style="border: 1px solid #eee; margin: 30px 0;">
                <p style="color: #666; font-size: 12px;">
                    This is an automated message from IDP Platform. Please do not reply to this email.
                </p>
            </div>
        </body>
        </html>
        """
        return await self.send_email(email, subject, body, is_html=True)
    
    async def send_welcome_email(self, email: str, signup_method: str) -> bool:
        """Send welcome email after successful registration"""
        subject = "Welcome to IDP Platform"
        method_text = "traditional signup" if signup_method == "traditional" else "email verification"
        
        body = f"""
        <html>
        <body>
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; font-family: Arial, sans-serif;">
                <h2 style="color: #333; text-align: center;">Welcome to IDP Platform!</h2>
                <p>Hello,</p>
                <p>Thank you for joining IDP (Intelligent Document Processing) platform through {method_text}.</p>
                <p>Your account has been successfully created and you can now start using our services:</p>
                <ul style="line-height: 1.6;">
                    <li>Create custom extraction schemas</li>
                    <li>Upload and process documents</li>
                    <li>Extract structured data from PDFs</li>
                    <li>Perform two-way matching between documents</li>
                </ul>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="http://localhost:3000/login" 
                       style="background-color: #28a745; color: white; padding: 12px 30px; 
                              text-decoration: none; border-radius: 5px; display: inline-block;
                              font-weight: bold;">
                        Login to Your Account
                    </a>
                </div>
                <p>If you have any questions or need assistance, please don't hesitate to contact our support team.</p>
                <hr style="border: 1px solid #eee; margin: 30px 0;">
                <p style="color: #666; font-size: 12px;">
                    This is an automated message from IDP Platform. Please do not reply to this email.
                </p>
            </div>
        </body>
        </html>
        """
        return await self.send_email(email, subject, body, is_html=True)

    async def send_account_pending_email(self, email: str, first_name: str) -> bool:
        """Send account pending approval email"""
        subject = "Account Created - Pending Approval"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2563eb;">Account Created Successfully!</h2>
                <p>Hi {first_name},</p>
                <p>Thank you for signing up for our Intelligent Document Processing platform.</p>
                <p>Your account has been created and is currently <strong>pending approval</strong> from our administrators.</p>
                
                <div style="background-color: #f3f4f6; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <h3 style="margin-top: 0; color: #374151;">What happens next?</h3>
                    <ul style="margin-bottom: 0;">
                        <li>Our admin team will review your account</li>
                        <li>You'll receive an email once your account is approved</li>
                        <li>After approval, you can log in and start using the platform</li>
                    </ul>
                </div>
                
                <p>If you have any questions, please contact our support team.</p>
                <p>Best regards,<br>InnuxAI Team</p>
            </div>
        </body>
        </html>
        """
        return await self.send_email(email, subject, body, is_html=True)

    async def send_account_approved_email(self, email: str, first_name: str):
        """Send account approval notification email"""
        subject = "Account Approved - Welcome to InnuxAI IDP Platform"
        
        body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Account Approved</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2c5aa0;">🎉 Welcome to InnuxAI IDP Platform!</h2>
                
                <p>Dear {first_name},</p>
                
                <p>Great news! Your account has been approved and you can now access the InnuxAI Intelligent Document Processing platform.</p>
                
                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p><strong>What you can do now:</strong></p>
                    <ul>
                        <li>Log in to your account using your email and password</li>
                        <li>Upload and process documents</li>
                        <li>Create and manage extraction schemas</li>
                        <li>Access all platform features based on your role</li>
                    </ul>
                </div>
                
                <p>If you have any questions or need assistance, please contact our support team.</p>
                <p>Welcome aboard!<br>InnuxAI Team</p>
            </div>
        </body>
        </html>
        """
        return await self.send_email(email, subject, body, is_html=True)

    async def send_account_rejected_email(self, email: str, first_name: str):
        """Send account rejection notification email"""
        subject = "Account Application Update - InnuxAI IDP Platform"
        
        body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Account Application Update</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #2c5aa0;">Account Application Update</h2>
                
                <p>Dear {first_name},</p>
                
                <p>Thank you for your interest in the InnuxAI Intelligent Document Processing platform.</p>
                
                <p>After reviewing your account application, we are unable to approve your request at this time. This decision may be based on various factors including capacity limitations or specific access requirements.</p>
                
                <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
                    <p><strong>Next steps:</strong></p>
                    <ul>
                        <li>If you believe this is an error, please contact our support team</li>
                        <li>You may reapply in the future when circumstances change</li>
                        <li>For enterprise inquiries, please reach out to our sales team</li>
                    </ul>
                </div>
                
                <p>If you have any questions, please contact our support team.</p>
                <p>Best regards,<br>InnuxAI Team</p>
            </div>
        </body>
        </html>
        """
        return await self.send_email(email, subject, body, is_html=True)

# Global email service instance
email_service = EmailService()

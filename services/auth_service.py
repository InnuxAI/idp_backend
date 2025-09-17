from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from bson import ObjectId
from fastapi import HTTPException, status
from pymongo.errors import DuplicateKeyError

from db.mongodb import get_database
from models.auth_models import (
    UserCreate, UserTraditionalSignup, UserOTPSignup, UserOTPVerification,
    UserLogin, ForgotPassword, ResetPassword, UserInDB, OTPRecord, 
    PasswordResetToken, UserSignupMethod
)
from utils.auth_utils import (
    get_password_hash, verify_password, create_access_token, generate_otp,
    create_password_reset_token, verify_password_reset_token, 
    is_account_locked, should_lock_account, get_lock_duration,
    create_user_response_data
)
from services.email_service import email_service

class AuthService:
    def __init__(self):
        self.db = None
        
    def get_db(self):
        if self.db is None:
            self.db = get_database()
            if self.db is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database connection not available"
                )
        return self.db

    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email"""
        db = self.get_db()
        user_data = await db.users.find_one({"email": email})
        if user_data:
            user_data["id"] = str(user_data["_id"])
            del user_data["_id"]
            return UserInDB(**user_data)
        return None

    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID"""
        db = self.get_db()
        try:
            user_data = await db.users.find_one({"_id": ObjectId(user_id)})
            if user_data:
                user_data["id"] = str(user_data["_id"])
                del user_data["_id"]
                return UserInDB(**user_data)
        except Exception:
            pass
        return None

    async def create_user(self, user_create: UserCreate, signup_method: UserSignupMethod) -> UserInDB:
        """Create a new user"""
        db = self.get_db()
        
        # Check if user already exists
        existing_user = await self.get_user_by_email(user_create.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user document
        user_doc = {
            "email": user_create.email,
            "hashed_password": get_password_hash(user_create.password),
            "role": user_create.role,
            "is_active": True,
            "email_verified": signup_method == UserSignupMethod.traditional,
            "created_at": datetime.utcnow(),
            "failed_login_attempts": 0,
            "locked_until": None
        }
        
        try:
            result = await db.users.insert_one(user_doc)
            user_doc["id"] = str(result.inserted_id)
            del user_doc["_id"]
            
            # Send welcome email
            await email_service.send_welcome_email(user_create.email, signup_method.value)
            
            return UserInDB(**user_doc)
        except DuplicateKeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

    async def authenticate_user(self, email: str, password: str) -> Optional[UserInDB]:
        """Authenticate user with email and password"""
        user = await self.get_user_by_email(email)
        if not user:
            return None
        
        # Check if account is locked
        if is_account_locked(user):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account temporarily locked due to multiple failed login attempts"
            )
        
        if not verify_password(password, user.hashed_password):
            # Increment failed login attempts
            await self.increment_failed_login_attempts(user.email)
            return None
        
        # Reset failed login attempts on successful login
        await self.reset_failed_login_attempts(user.email)
        return user

    async def increment_failed_login_attempts(self, email: str):
        """Increment failed login attempts for a user"""
        db = self.get_db()
        user = await self.get_user_by_email(email)
        if user:
            new_attempts = user.failed_login_attempts + 1
            update_data = {"failed_login_attempts": new_attempts}
            
            if should_lock_account(new_attempts):
                lock_duration = get_lock_duration(new_attempts)
                update_data["locked_until"] = datetime.utcnow() + lock_duration
            
            await db.users.update_one(
                {"email": email},
                {"$set": update_data}
            )

    async def reset_failed_login_attempts(self, email: str):
        """Reset failed login attempts for a user"""
        db = self.get_db()
        await db.users.update_one(
            {"email": email},
            {"$set": {"failed_login_attempts": 0, "locked_until": None}}
        )

    async def initiate_otp_signup(self, signup_data: UserOTPSignup) -> Dict[str, str]:
        """Initiate OTP-based signup"""
        # Check if user already exists
        existing_user = await self.get_user_by_email(signup_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Generate OTP
        otp = generate_otp()
        
        # Store OTP in database
        db = self.get_db()
        otp_record = OTPRecord(
            email=signup_data.email,
            otp=otp,
            purpose="signup",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=10),
            attempts=0,
            verified=False,
            used=False
        )
        
        await db.otp_records.insert_one(otp_record.dict())
        
        # Send OTP email
        email_sent = await email_service.send_otp_email(signup_data.email, otp)
        
        if not email_sent:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send OTP email"
            )
        
        return {"message": "OTP sent to your email address"}

    async def verify_otp_and_signup(self, verification_data: UserOTPVerification) -> Dict[str, Any]:
        """Verify OTP and complete signup"""
        db = self.get_db()
        
        # Find valid OTP record
        otp_record = await db.otp_records.find_one({
            "email": verification_data.email,
            "otp": verification_data.otp,
            "used": False,
            "expires_at": {"$gt": datetime.utcnow()}
        })
        
        if not otp_record:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired OTP"
            )
        
        # Mark OTP as used
        await db.otp_records.update_one(
            {"_id": otp_record["_id"]},
            {"$set": {"used": True}}
        )
        
        # Create user
        user_create = UserCreate(
            email=verification_data.email,
            password=verification_data.password,
            role=""  # No default role, admin will assign
        )
        
        user = await self.create_user(user_create, UserSignupMethod.otp)
        
        # Create access token
        access_token = create_access_token(data={"sub": user.email})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": create_user_response_data(user)
        }

    async def login(self, login_data: UserLogin) -> Dict[str, Any]:
        """Handle user login"""
        user = await self.authenticate_user(login_data.email, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Account is deactivated"
            )
        
        # Create access token with longer expiry if remember_me is checked
        expires_delta = timedelta(days=30) if login_data.remember_me else None
        access_token = create_access_token(
            data={"sub": user.email}, 
            expires_delta=expires_delta
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": create_user_response_data(user)
        }

    async def forgot_password(self, forgot_data: ForgotPassword) -> Dict[str, str]:
        """Handle forgot password request"""
        user = await self.get_user_by_email(forgot_data.email)
        if not user:
            # Don't reveal if email exists or not
            return {"message": "If the email exists, a reset link has been sent"}
        
        # Generate reset token
        reset_token = create_password_reset_token(forgot_data.email)
        
        # Store reset token in database
        db = self.get_db()
        reset_record = {
            "email": forgot_data.email,
            "token": reset_token,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=1),
            "used": False
        }
        
        await db.password_reset_tokens.insert_one(reset_record)
        
        # Send reset email
        reset_link = f"http://localhost:3000/reset-password?token={reset_token}"
        email_sent = await email_service.send_password_reset_email(forgot_data.email, reset_link)
        
        if not email_sent:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send reset email"
            )
        
        return {"message": "If the email exists, a reset link has been sent"}

    async def reset_password(self, reset_data: ResetPassword) -> Dict[str, str]:
        """Handle password reset"""
        # Verify reset token
        email = verify_password_reset_token(reset_data.token)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Check if token exists in database and is not used
        db = self.get_db()
        reset_record = await db.password_reset_tokens.find_one({
            "email": email,
            "token": reset_data.token,
            "used": False,
            "expires_at": {"$gt": datetime.utcnow()}
        })
        
        if not reset_record:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Update user password
        hashed_password = get_password_hash(reset_data.new_password)
        await db.users.update_one(
            {"email": email},
            {"$set": {"hashed_password": hashed_password, "failed_login_attempts": 0, "locked_until": None}}
        )
        
        # Mark reset token as used
        await db.password_reset_tokens.update_one(
            {"_id": reset_record["_id"]},
            {"$set": {"used": True}}
        )
        
        return {"message": "Password has been reset successfully"}

    async def get_current_user_data(self, email: str) -> Dict[str, Any]:
        """Get current user data for /me endpoint"""
        user = await self.get_user_by_email(email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return create_user_response_data(user)

    async def traditional_signup(self, signup_data: UserTraditionalSignup) -> Dict[str, Any]:
        """Handle traditional form-based signup"""
        # Create user with traditional signup method
        user_create = UserCreate(
            email=signup_data.email,
            password=signup_data.password,
            role=signup_data.role
        )
        
        # For traditional signup, account is pending approval by default
        # We'll set is_active to False until admin approves
        db = self.get_db()
        
        # Check if user already exists
        existing_user = await self.get_user_by_email(signup_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user document with pending status
        user_doc = {
            "email": signup_data.email,
            "hashed_password": get_password_hash(signup_data.password),
            "first_name": signup_data.first_name,
            "last_name": signup_data.last_name,
            "role": signup_data.role,
            "is_active": False,  # Pending admin approval
            "email_verified": True,  # Email doesn't need verification for traditional signup
            "signup_method": "traditional",
            "created_at": datetime.utcnow(),
            "failed_login_attempts": 0,
            "locked_until": None,
            "approval_status": "pending"  # pending, approved, rejected
        }
        
        try:
            result = await db.users.insert_one(user_doc)
            user_doc["id"] = str(result.inserted_id)
            del user_doc["_id"]
            
            # Send notification email to user
            await email_service.send_account_pending_email(signup_data.email, signup_data.first_name)
            
            # Note: Don't create access token since account needs approval
            return {
                "message": "Account created successfully. Please wait for admin approval.",
                "status": "pending_approval",
                "user_id": user_doc["id"]
            }
        except DuplicateKeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

    async def initiate_otp_signup(self, signup_data: UserOTPSignup) -> Dict[str, str]:
        """Initiate OTP-based signup by sending verification code"""
        # Check if user already exists
        existing_user = await self.get_user_by_email(signup_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Generate OTP
        otp_code = generate_otp()
        
        # Store OTP in database
        db = self.get_db()
        otp_record = OTPRecord(
            email=signup_data.email,
            otp=otp_code,
            purpose="signup",
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=10),
            attempts=0,
            verified=False
        )
        
        await db.otp_records.insert_one(otp_record.dict())
        
        # Send OTP email
        email_sent = await email_service.send_otp_email(signup_data.email, otp_code)
        
        if not email_sent:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send OTP email"
            )
        
        return {"message": "OTP sent to your email address"}

    async def verify_otp_and_signup(self, verification_data: UserOTPVerification) -> Dict[str, Any]:
        """Verify OTP and complete signup process"""
        db = self.get_db()
        
        # Find valid OTP record
        otp_record = await db.otp_records.find_one({
            "email": verification_data.email,
            "otp": verification_data.otp,
            "purpose": "signup",
            "verified": False,
            "expires_at": {"$gt": datetime.utcnow()}
        })
        
        if not otp_record:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired OTP"
            )
        
        # Mark OTP as verified
        await db.otp_records.update_one(
            {"_id": otp_record["_id"]},
            {"$set": {"verified": True}}
        )
        
        # Create user account (also pending approval for OTP signup)
        user_doc = {
            "email": verification_data.email,
            "hashed_password": get_password_hash(verification_data.password),
            "first_name": verification_data.first_name,
            "last_name": verification_data.last_name,
            "role": verification_data.role,
            "is_active": False,  # Pending admin approval
            "email_verified": True,  # Email verified via OTP
            "signup_method": "otp",
            "created_at": datetime.utcnow(),
            "failed_login_attempts": 0,
            "locked_until": None,
            "approval_status": "pending"
        }
        
        try:
            result = await db.users.insert_one(user_doc)
            user_doc["id"] = str(result.inserted_id)
            del user_doc["_id"]
            
            # Send notification email to user
            await email_service.send_account_pending_email(verification_data.email, verification_data.first_name)
            
            return {
                "message": "Account created successfully. Please wait for admin approval.",
                "status": "pending_approval",
                "user_id": user_doc["id"]
            }
        except DuplicateKeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

    async def get_current_user_data(self, email: str):
        """Get current user data by email"""
        db = self.get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection error")
        
        users_collection = db.users
        user = await users_collection.find_one({"email": email})
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "id": str(user["_id"]),
            "email": user["email"],
            "first_name": user.get("first_name", ""),
            "last_name": user.get("last_name", ""),
            "role": user.get("role", "user"),
            "is_active": user.get("is_active", False),
            "approval_status": user.get("approval_status", "pending"),
            "created_at": user.get("created_at"),
            "permissions": user.get("permissions", [])
        }

    # Admin methods
    async def get_pending_users(self):
        """Get all users pending approval"""
        db = self.get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection error")
        
        users_collection = db.users
        pending_users = await users_collection.find({
            "approval_status": "pending"
        }).to_list(length=None)
        
        # Convert ObjectId to string and format response
        formatted_users = []
        for user in pending_users:
            formatted_users.append({
                "id": str(user["_id"]),
                "email": user["email"],
                "first_name": user.get("first_name", ""),
                "last_name": user.get("last_name", ""),
                "created_at": user.get("created_at"),
                "role": user.get("role", "user")
            })
        
        return formatted_users

    async def get_all_users(self):
        """Get all users with their details"""
        db = self.get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection error")
        
        users_collection = db.users
        all_users = await users_collection.find({}).to_list(length=None)
        
        # Convert ObjectId to string and format response
        formatted_users = []
        for user in all_users:
            formatted_users.append({
                "id": str(user["_id"]),
                "email": user["email"],
                "first_name": user.get("first_name", ""),
                "last_name": user.get("last_name", ""),
                "role": user.get("role", "user"),
                "approval_status": user.get("approval_status", "pending"),
                "signup_type": user.get("signup_type", "traditional"),
                "is_active": user.get("is_active", True),
                "created_at": user.get("created_at")
            })
        
        return formatted_users

    async def approve_user(self, user_id: str):
        """Approve a pending user"""
        from bson import ObjectId
        
        db = self.get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection error")
        
        users_collection = db.users
        
        # Update user status
        result = await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "approval_status": "approved",
                    "is_active": True,
                    "approved_at": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get user info for email notification
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if user:
            # Send approval email
            await email_service.send_account_approved_email(
                user["email"],
                user.get("first_name", "")
            )
        
        return {"message": "User approved successfully"}

    async def reject_user(self, user_id: str):
        """Reject a pending user"""
        from bson import ObjectId
        
        db = self.get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection error")
        
        users_collection = db.users
        
        # Get user info before deletion for email notification
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Delete the user account
        result = await users_collection.delete_one({"_id": ObjectId(user_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Send rejection email
        await email_service.send_account_rejected_email(
            user["email"],
            user.get("first_name", "")
        )
        
        return {"message": "User rejected and account deleted"}

    async def update_user_role(self, user_id: str, new_role: str):
        """Update a user's role (admin only)"""
        from bson import ObjectId
        
        db = self.get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection error")
        
        users_collection = db.users
        
        # Update user role
        result = await users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "role": new_role,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": f"User role updated to {new_role} successfully"}

    async def delete_user(self, user_id: str):
        """Permanently delete a user (admin only)"""
        from bson import ObjectId
        
        db = self.get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection error")
        
        users_collection = db.users
        
        # Get user info before deletion
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Don't allow deletion of admin users (safety measure)
        if user.get("role") == "admin":
            raise HTTPException(status_code=400, detail="Cannot delete admin users")
        
        # Delete user
        result = await users_collection.delete_one({"_id": ObjectId(user_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {"message": f"User {user['email']} deleted successfully"}

    async def create_first_admin(self):
        """Create the first admin user if no admins exist"""
        db = self.get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection error")
        
        users_collection = db.users
        
        # Check if any admin users exist
        admin_exists = await users_collection.find_one({"role": "admin"})
        if admin_exists:
            raise HTTPException(status_code=400, detail="Admin user already exists")
        
        # Create default admin user
        admin_data = {
            "email": "admin@innuxai.com",
            "hashed_password": get_password_hash("AdminPass123!"),
            "first_name": "Admin",
            "last_name": "User",
            "role": "admin",
            "is_active": True,
            "approval_status": "approved",
            "created_at": datetime.utcnow(),
            "approved_at": datetime.utcnow(),
            "permissions": [
                "view_dashboard", "view_schemas", "create_schemas", "edit_schemas", "delete_schemas",
                "view_extractions", "create_extractions", "edit_extractions", "delete_extractions",
                "view_data_library", "manage_users", "view_analytics", "upload_documents",
                "download_documents", "approve_extractions", "perform_two_way_match", "system_admin"
            ]
        }
        
        result = await users_collection.insert_one(admin_data)
        
        return {
            "message": "Admin user created successfully",
            "data": {
                "email": "admin@innuxai.com",
                "password": "AdminPass123!",
                "note": "Please change the password after first login"
            }
        }

# Global auth service instance
auth_service = AuthService()

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum
import os

# Get roles from environment variable
def get_user_roles() -> List[str]:
    roles_env = os.getenv("USER_ROLES", "admin")
    return [role.strip() for role in roles_env.split(",")]

class UserRole(str, Enum):
    @classmethod
    def _missing_(cls, value):
        # This allows dynamic roles from environment
        return value

# Authentication Models
class UserSignupMethod(str, Enum):
    traditional = "traditional"
    otp = "otp"

class TokenData(BaseModel):
    email: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class UserBase(BaseModel):
    email: EmailStr
    is_active: bool = True
    role: str = Field(default="", description="User role to be assigned by admin")

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    signup_method: UserSignupMethod = UserSignupMethod.traditional
    role: str = Field(default="", description="User role to be assigned by admin")
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v
    
    @validator('role')
    def validate_role(cls, v):
        valid_roles = get_user_roles()
        if v not in valid_roles:
            raise ValueError(f'Role must be one of: {", ".join(valid_roles)}')
        return v

class UserTraditionalSignup(UserCreate):
    confirm_password: str
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class UserOTPSignup(BaseModel):
    email: EmailStr

class UserOTPVerification(BaseModel):
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)
    password: str = Field(..., min_length=8)
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    role: str = Field(default="", description="User role to be assigned by admin")
    
    @validator('otp')
    def validate_otp(cls, v):
        if not v.isdigit():
            raise ValueError('OTP must contain only digits')
        return v
    
    @validator('role')
    def validate_role(cls, v):
        valid_roles = get_user_roles()
        if v not in valid_roles:
            raise ValueError(f'Role must be one of: {", ".join(valid_roles)}')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False

class ForgotPassword(BaseModel):
    email: EmailStr

class ResetPassword(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    permissions: List[str] = []

class UserInDB(UserBase):
    id: str
    hashed_password: str
    created_at: datetime
    email_verified: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

class OTPRecord(BaseModel):
    email: str
    otp: str
    purpose: str = "signup"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    attempts: int = 0
    verified: bool = False
    used: bool = False

class PasswordResetToken(BaseModel):
    email: str
    token: str
    created_at: datetime
    expires_at: datetime
    used: bool = False

# Permission system
class Permission(str, Enum):
    # Page permissions
    VIEW_DASHBOARD = "view_dashboard"
    VIEW_SCHEMAS = "view_schemas"
    CREATE_SCHEMAS = "create_schemas"
    EDIT_SCHEMAS = "edit_schemas"
    DELETE_SCHEMAS = "delete_schemas"
    VIEW_EXTRACTIONS = "view_extractions"
    CREATE_EXTRACTIONS = "create_extractions"
    EDIT_EXTRACTIONS = "edit_extractions"
    DELETE_EXTRACTIONS = "delete_extractions"
    VIEW_DATA_LIBRARY = "view_data_library"
    MANAGE_USERS = "manage_users"
    VIEW_ANALYTICS = "view_analytics"
    
    # Feature permissions
    UPLOAD_DOCUMENTS = "upload_documents"
    DOWNLOAD_DOCUMENTS = "download_documents"
    APPROVE_EXTRACTIONS = "approve_extractions"
    PERFORM_TWO_WAY_MATCH = "perform_two_way_match"
    
    # System permissions
    SYSTEM_ADMIN = "system_admin"

# Role-based permissions mapping
ROLE_PERMISSIONS = {
    "admin": [
        Permission.VIEW_DASHBOARD,
        Permission.VIEW_SCHEMAS,
        Permission.CREATE_SCHEMAS,
        Permission.EDIT_SCHEMAS,
        Permission.DELETE_SCHEMAS,
        Permission.VIEW_EXTRACTIONS,
        Permission.CREATE_EXTRACTIONS,
        Permission.EDIT_EXTRACTIONS,
        Permission.DELETE_EXTRACTIONS,
        Permission.VIEW_DATA_LIBRARY,
        Permission.MANAGE_USERS,
        Permission.VIEW_ANALYTICS,
        Permission.UPLOAD_DOCUMENTS,
        Permission.DOWNLOAD_DOCUMENTS,
        Permission.APPROVE_EXTRACTIONS,
        Permission.PERFORM_TWO_WAY_MATCH,
        Permission.SYSTEM_ADMIN,
    ]
}

def get_permissions_for_role(role: str) -> List[str]:
    """Get permissions for a specific role"""
    return [perm.value for perm in ROLE_PERMISSIONS.get(role, [])]

def has_permission(user_role: str, required_permission: str) -> bool:
    """Check if a user role has a specific permission"""
    user_permissions = get_permissions_for_role(user_role)
    return required_permission in user_permissions

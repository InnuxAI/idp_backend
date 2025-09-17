import os
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from fastapi import HTTPException, status
from passlib.context import CryptContext
import secrets
import string
from models.auth_models import TokenData, UserInDB
from dotenv import load_dotenv

load_dotenv()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_super_secret_jwt_key_here_change_in_production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> TokenData:
    """Verify and decode a JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    return token_data

def generate_otp() -> str:
    """Generate a 6-digit OTP"""
    return ''.join(secrets.choice(string.digits) for _ in range(6))

def generate_reset_token() -> str:
    """Generate a secure reset token"""
    return secrets.token_urlsafe(32)

def create_password_reset_token(email: str) -> str:
    """Create a password reset token"""
    data = {"sub": email, "type": "reset"}
    expires_delta = timedelta(hours=1)  # Reset tokens expire in 1 hour
    return create_access_token(data, expires_delta)

def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify a password reset token and return the email"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if email is None or token_type != "reset":
            return None
        
        return email
    except JWTError:
        return None

def is_account_locked(user: UserInDB) -> bool:
    """Check if user account is locked due to failed login attempts"""
    if user.locked_until and user.locked_until > datetime.utcnow():
        return True
    return False

def should_lock_account(failed_attempts: int) -> bool:
    """Determine if account should be locked after failed attempts"""
    return failed_attempts >= 5

def get_lock_duration(failed_attempts: int) -> timedelta:
    """Get lock duration based on failed attempts"""
    if failed_attempts >= 10:
        return timedelta(hours=1)
    elif failed_attempts >= 7:
        return timedelta(minutes=30)
    else:
        return timedelta(minutes=15)

def create_user_response_data(user: UserInDB) -> Dict[str, Any]:
    """Create user response data for API responses"""
    from models.auth_models import get_permissions_for_role
    
    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": user.created_at,
        "permissions": get_permissions_for_role(user.role)
    }

def get_user_roles():
    """Get list of valid user roles"""
    return ["admin", "manager", "analyst", "viewer"]

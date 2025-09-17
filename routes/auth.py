from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any

from models.auth_models import (
    UserTraditionalSignup, UserOTPSignup, UserOTPVerification,
    UserLogin, ForgotPassword, ResetPassword, Token, UserResponse
)
from services.auth_service import auth_service
from utils.auth_utils import verify_token

router = APIRouter(prefix="/auth", tags=["authentication"])

# Security scheme
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current authenticated user"""
    token_data = verify_token(credentials.credentials)
    return token_data.email

@router.post("/signup/form")
async def traditional_signup(signup_data: UserTraditionalSignup) -> Dict[str, Any]:
    """Traditional form-based signup with immediate account creation"""
    try:
        result = await auth_service.traditional_signup(signup_data)
        return {
            "success": True,
            "message": result["message"],
            "data": {
                "status": result["status"],
                "user_id": result["user_id"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signup failed: {str(e)}"
        )

@router.post("/send-otp")
async def otp_signup(signup_data: UserOTPSignup) -> Dict[str, Any]:
    """Initiate OTP-based signup by sending verification code"""
    try:
        result = await auth_service.initiate_otp_signup(signup_data)
        return {
            "success": True,
            "message": result["message"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OTP signup failed: {str(e)}"
        )

@router.post("/verify-otp")
async def verify_otp(verification_data: UserOTPVerification) -> Dict[str, Any]:
    """Verify OTP and complete signup process"""
    try:
        result = await auth_service.verify_otp_and_signup(verification_data)
        return {
            "success": True,
            "message": result["message"],
            "data": {
                "status": result["status"],
                "user_id": result["user_id"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OTP verification failed: {str(e)}"
        )

@router.post("/login")
async def login(login_data: UserLogin) -> Dict[str, Any]:
    """User login with email and password"""
    try:
        result = await auth_service.login(login_data)
        return {
            "success": True,
            "message": "Login successful",
            "data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@router.post("/forgot-password")
async def forgot_password(forgot_data: ForgotPassword) -> Dict[str, Any]:
    """Send password reset email"""
    try:
        result = await auth_service.forgot_password(forgot_data)
        return {
            "success": True,
            "message": result["message"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password reset request failed: {str(e)}"
        )

@router.post("/reset-password")
async def reset_password(reset_data: ResetPassword) -> Dict[str, Any]:
    """Reset password with token"""
    try:
        result = await auth_service.reset_password(reset_data)
        return {
            "success": True,
            "message": result["message"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Password reset failed: {str(e)}"
        )

@router.get("/me")
async def get_current_user_info(current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Get current user information"""
    try:
        user_data = await auth_service.get_current_user_data(current_user)
        return {
            "success": True,
            "data": user_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user info: {str(e)}"
        )

@router.post("/logout")
async def logout() -> Dict[str, Any]:
    """User logout (client-side token removal)"""
    return {
        "success": True,
        "message": "Logged out successfully"
    }

@router.get("/validate-token")
async def validate_token(current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Validate if current token is valid"""
    try:
        user_data = await auth_service.get_current_user_data(current_user)
        return {
            "success": True,
            "valid": True,
            "data": user_data
        }
    except HTTPException:
        return {
            "success": False,
            "valid": False,
            "message": "Invalid token"
        }

# Admin endpoints
@router.get("/admin/pending-users")
async def get_pending_users(current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Get all pending user accounts (admin only)"""
    try:
        # Check if current user is admin
        user_data = await auth_service.get_current_user_data(current_user)
        if user_data.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        pending_users = await auth_service.get_pending_users()
        return {
            "success": True,
            "data": pending_users
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pending users: {str(e)}"
        )

@router.get("/admin/users")
async def get_all_users(current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Get all users (admin only)"""
    try:
        # Check if current user is admin
        user_data = await auth_service.get_current_user_data(current_user)
        if user_data.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        all_users = await auth_service.get_all_users()
        return {
            "success": True,
            "data": all_users
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get users: {str(e)}"
        )

@router.post("/admin/approve-user/{user_id}")
async def approve_user(user_id: str, current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Approve a pending user account (admin only)"""
    try:
        # Check if current user is admin
        user_data = await auth_service.get_current_user_data(current_user)
        if user_data.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        result = await auth_service.approve_user(user_id)
        return {
            "success": True,
            "message": result["message"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve user: {str(e)}"
        )

@router.post("/admin/reject-user/{user_id}")
async def reject_user(user_id: str, current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Reject a pending user account (admin only)"""
    try:
        # Check if current user is admin
        user_data = await auth_service.get_current_user_data(current_user)
        if user_data.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        result = await auth_service.reject_user(user_id)
        return {
            "success": True,
            "message": result["message"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reject user: {str(e)}"
        )

@router.post("/admin/update-user-role/{user_id}")
async def update_user_role(user_id: str, request: Dict[str, str], current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Update a user's role (admin only)"""
    try:
        # Check if current user is admin
        user_data = await auth_service.get_current_user_data(current_user)
        if user_data.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        new_role = request.get("role")
        if not new_role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Role is required"
            )
        
        result = await auth_service.update_user_role(user_id, new_role)
        return {
            "success": True,
            "message": result["message"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user role: {str(e)}"
        )

@router.delete("/admin/delete-user/{user_id}")
async def delete_user(user_id: str, current_user: str = Depends(get_current_user)) -> Dict[str, Any]:
    """Delete a user permanently (admin only)"""
    try:
        # Check if current user is admin
        user_data = await auth_service.get_current_user_data(current_user)
        if user_data.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        result = await auth_service.delete_user(user_id)
        return {
            "success": True,
            "message": result["message"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )

@router.post("/admin/create-admin")
async def create_admin_user() -> Dict[str, Any]:
    """Create the first admin user (only works if no admins exist)"""
    try:
        result = await auth_service.create_first_admin()
        return {
            "success": True,
            "message": result["message"],
            "data": result.get("data")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create admin: {str(e)}"
        )

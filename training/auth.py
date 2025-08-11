import logging
from typing import Optional
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import auth, credentials
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Initialize Firebase Admin SDK
def initialize_firebase():
    """Initialize Firebase Admin SDK with service account credentials"""
    try:
        # Check if Firebase is already initialized
        firebase_admin.get_app()
        logger.info("Firebase Admin SDK already initialized")
    except ValueError:
        # Initialize Firebase with default service account credentials
        # Expects GOOGLE_APPLICATION_CREDENTIALS env var or default service account
        cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully")


class FirebaseUser(BaseModel):
    """Firebase user information"""

    uid: str
    email: Optional[str] = None
    name: Optional[str] = None
    email_verified: bool = False


# Security scheme for Bearer token
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> FirebaseUser:
    """
    Verify Firebase ID token and return user information.

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        FirebaseUser: Verified user information

    Raises:
        HTTPException: 401 if token is invalid or expired
    """
    try:
        # Verify the ID token
        decoded_token = auth.verify_id_token(credentials.credentials)

        # Extract user information
        user = FirebaseUser(
            uid=decoded_token["uid"],
            email=decoded_token.get("email"),
            name=decoded_token.get("name"),
            email_verified=decoded_token.get("email_verified", False),
        )

        logger.debug(f"Authenticated user: {user.uid}")
        return user

    except Exception as e:
        logger.warning(f"Firebase token verification failed: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user_id(user: FirebaseUser = Depends(get_current_user)) -> str:
    """Extract user ID from authenticated user"""
    return user.uid

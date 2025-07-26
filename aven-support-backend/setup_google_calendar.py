import os
import logging
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The SCOPES defines the level of access you are requesting.
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# File paths
CREDENTIALS_PATH = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH", "credentials.json")
TOKEN_PATH = "token.json"

def main():
    """
    Runs the OAuth 2.0 flow to get user consent and stores credentials.
    This script needs to be run once manually to generate the token.json file.
    """
    creds = None
    
    # Check if token.json exists and has valid credentials
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        except Exception as e:
            logger.warning(f"Could not load existing credentials from {TOKEN_PATH}: {e}")

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                logger.info("Credentials have expired, refreshing...")
                creds.refresh(Request())
            except Exception as e:
                logger.error(f"Failed to refresh credentials, re-authentication is required: {e}")
                creds = None # Force re-authentication
        
        if not creds:
            if not os.path.exists(CREDENTIALS_PATH):
                logger.error(f"'{CREDENTIALS_PATH}' not found. Please download your OAuth 2.0 credentials for a 'Desktop app' from Google Cloud Console and save it as '{CREDENTIALS_PATH}'.")
                return

            logger.info(f"Starting authentication flow using '{CREDENTIALS_PATH}'...")
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_PATH, SCOPES
            )
            # This will open a browser window for the user to authorize the application
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())
        logger.info(f"Credentials saved to '{TOKEN_PATH}' successfully.")
    else:
        logger.info(f"Valid credentials already exist in '{TOKEN_PATH}'. No action needed.")

if __name__ == "__main__":
    main() 
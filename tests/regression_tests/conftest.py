import logging
import os
import pytest

from dotenv import load_dotenv

PARENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def load_env_file():
    """Load the environment variables from the .env file."""

    if os.path.exists(os.path.join(PARENT_DIR_PATH, ".env")):
        logging.info(f"Loading variables from {os.path.join(PARENT_DIR_PATH, '.env')} file...")
        load_dotenv(dotenv_path=os.path.join(PARENT_DIR_PATH, ".env"), override=True)
    else:
        raise FileNotFoundError(f"No .env file found in {PARENT_DIR_PATH}")

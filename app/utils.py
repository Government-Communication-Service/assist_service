import logging
import os

logger = logging.getLogger(__name__)


# This wrapper raises an error if the environment variables are not as expected
def get_env_wrapper(env_variable_name: str):
    try:
        variable = os.getenv(env_variable_name)
    except Exception as e:
        logger.info(f"Could not load env variable {env_variable_name}: {e}")
    if variable is None:
        logger.info(f"Environment variable {env_variable_name} was loaded as a null value")
        # This doesn't seem to work?
        logger.error(f"Environment variable {env_variable_name} was loaded as a null value")
        raise ValueError

    return variable

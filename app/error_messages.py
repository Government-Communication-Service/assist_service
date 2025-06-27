from app.config import IS_DEV, env_variable


class ErrorMessages:
    @staticmethod
    def default(task: str, e: Exception):
        if IS_DEV and env_variable("SHOW_DETAILED_ERROR_MESSAGES"):
            return f"ERROR {task}: {str(e)}"

        return f"An error occurred during {task}. Please try again later."

    @staticmethod
    def invalid_or_expired(item: str, typ: str = ""):
        return f"'{item}' {typ} is invalid or expired."

    @staticmethod
    def not_provided(item: str, typ: str = ""):
        return f"'{item}' {typ} is not provided."

    @staticmethod
    def item_not_found(item_name: str, key: str, value: str):
        return f"{item_name} with {key} '{value}' not found."

    @staticmethod
    def invalid_input(input_name: str):
        return f"Invalid input: {input_name}."

    @staticmethod
    def access_denied(resource_name: str):
        return f"Access denied to {resource_name}."

    @staticmethod
    def operation_failed(operation: str):
        return f"Operation {operation} failed."

    @staticmethod
    def timeout_occurred(task: str):
        return f"Timeout occurred during {task}."

    @staticmethod
    def database_error(task: str):
        return f"Database error during {task}."

    @staticmethod
    def network_error(task: str):
        return f"Network error during {task}."

    @staticmethod
    def unauthorized_action(action: str):
        return f"Unauthorized action: {action}."

    @staticmethod
    def missing_env_variable(var_name: str):
        return f"Required environment variable '{var_name}' not found. Terminating now."

    ENV_VAR_NOT_FOUND = "Required environment variable not found. Terminating now."
    OLLAMA_NOT_LAUNCHED = "Ollama service is not running. Please ensure it is launched on your local machine."
    CHAT_TITLE_NOT_CREATED = "Unable to create chat title."
    CHAT_MESSAGE_NOT_CREATED = "Unable to create chat message."
    NO_RESPONSE_FROM_AI = "No response from the AI."

import logging
from io import StringIO
from typing import Any, Callable, Dict, List, Optional

import boto3

from app.services.bedrock.bedrock_types import BedrockError, BedrockErrorType
from app.my_types.chat_agent import AgentFinalResponse, FileInfo, HistoricalMessage

logger = logging.getLogger()


def _get_agent_id_by_name(agent_name: str, aws_region: str) -> Optional[str]:
    """
    Retrieves the AWS Bedrock Agent ID corresponding to a given agent name.
    """

    client = boto3.client("bedrock-agent", region_name=aws_region)
    paginator = client.get_paginator("list_agents")
    for page in paginator.paginate():
        for agent in page.get("agentSummaries", []):
            if agent.get("agentName") == agent_name:
                return agent.get("agentId")

    return None


def _get_alias_id_by_name(agent_id: str, alias_name: str, aws_region: str) -> Optional[str]:
    """
    Retrieves the AWS Bedrock Agent Alias ID by agent_id, alias_name and aws_region provided.
    """

    client = boto3.client("bedrock-agent", region_name=aws_region)
    paginator = client.get_paginator("list_agent_aliases")
    for page in paginator.paginate(agentId=agent_id):
        for alias in page.get("agentAliasSummaries", []):
            if alias.get("agentAliasName") == alias_name:
                return alias.get("agentAliasId")

    return None


class BedrockAgentClient:
    __clients = {}

    def __init__(self, agent_name: str, agent_id: str, agent_alias_id: str, aws_region: str):
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.agent_alias_id = agent_alias_id
        self.aws_region_name = aws_region
        self.boto_client = boto3.client("bedrock-agent-runtime", region_name=aws_region)

    @classmethod
    def get(cls, agent_name: str, aws_region: str) -> "BedrockAgentClient":
        client_key = f"agent_name:{agent_name},aws_region:{aws_region}"
        if cls.__clients.get(client_key) is None:
            agent_id = _get_agent_id_by_name(agent_name, aws_region)
            if agent_id is None:
                raise ValueError(f"Could not find agent id  by agent name:{agent_name} ")

            agent_alias_id = _get_alias_id_by_name(agent_id, agent_name, aws_region)
            if agent_alias_id is None:
                raise ValueError(f"Could not find agent alias id by agent name:{agent_name} ")

            cls.__clients[client_key] = cls(agent_name, agent_id, agent_alias_id, aws_region)
            return cls.__clients[client_key]

        return cls.__clients[client_key]

    async def call_agent(
        self,
        *,
        text: str,
        session_id: str,
        on_complete: Callable[[AgentFinalResponse], Any],
        file_info: FileInfo = None,
        message_history: List[HistoricalMessage] = None,
    ):
        """
        Invokes an AWS Bedrock Agent API to process input text, optionally handling files and message history,
        and streams the response. Once the full response is received, it calls the provided `on_complete` callback with
        the final response data and token usage info.

        Parameters:
        text (str): The input text to send to the agent.
        session_id (str): The unique session identifier.
        on_complete (Callable[[AgentFinalResponse], Any]): Callback function that processes the final response
            containing the full response, input tokens, and output tokens.
        file_info (FileInfo, optional): Information about the file to be processed by the agent. Default is None.
        message_history (List[HistoricalMessage], optional): A list of previous messages to provide context
            to the agent. Default is None.

        Yields:
        str: response text
        """

        session_state = {}
        if file_info:
            session_state["files"] = [
                {
                    "name": file_info.name,
                    "source": {
                        "byteContent": {"data": file_info.data, "mediaType": file_info.media_type},
                        "sourceType": "BYTE_CONTENT",
                    },
                    "useCase": "CODE_INTERPRETER",
                }
            ]
        if message_history:
            session_state["conversationHistory"] = {}
            session_state["conversationHistory"]["messages"] = [
                {"content": [{"text": m.content}], "role": m.role.value} for m in message_history
            ]

        streaming_configurations = {"applyGuardrailInterval": 150, "streamFinalResponse": True}

        response = self.boto_client.invoke_agent(
            agentId=self.agent_id,
            agentAliasId=self.agent_alias_id,
            sessionId=session_id,
            inputText=text,
            enableTrace=True,
            sessionState=session_state,
            streamingConfigurations=streaming_configurations,
        )

        # Process the streaming response
        total_input_tokens = 0
        total_output_tokens = 0
        final_response_io = StringIO()
        for event in response["completion"]:
            if "chunk" in event:
                chunk = event["chunk"]
                content_part = chunk["bytes"].decode("utf-8")
                final_response_io.write(content_part)
                yield content_part
            elif "trace" in event:
                trace = event["trace"]["trace"]
                if "orchestrationTrace" in trace:
                    orchestration_trace = trace["orchestrationTrace"]

                    if "modelInvocationOutput" in orchestration_trace:
                        model_invocation_output = orchestration_trace["modelInvocationOutput"]
                        if "metadata" in model_invocation_output:
                            metadata = model_invocation_output["metadata"]
                            if "usage" in metadata:
                                usage = metadata["usage"]
                                input_tokens = usage.get("inputTokens")
                                output_tokens = usage.get("outputTokens")
                                if input_tokens:
                                    total_input_tokens += input_tokens
                                if output_tokens:
                                    total_output_tokens += output_tokens

            else:
                self._check_errors(event)

        final_response = final_response_io.getvalue()
        final_response_io.close()

        final_response_data = AgentFinalResponse(
            final_response=final_response,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )
        await on_complete(final_response_data)

    def _check_errors(self, event: Dict):
        error_codes = [
            "internalServerException",
            "conflictException",
            "dependencyFailedException",
            "modelNotReadyException",
            "resourceNotFoundException",
        ]
        for error_code in error_codes:
            if error_code in event:
                error = event[error_code]
                error_message = error["message"]
                raise BedrockError(f"{error_code}: {error_message}", BedrockErrorType.BEDROCK_AGENT_EXCEPTION)

# ruff: noqa: E501
"""Compaction module configuration"""

from app import config

# Token threshold for triggering compaction (configurable via main config)
COMPACTION_TOKEN_THRESHOLD = config.COMPACTION_TOKEN_THRESHOLD

# LLM model used for message summarization
LLM_COMPACTION_SUMMARISATION_MODEL = config.LLM_COMPACTION_SUMMARISATION_MODEL

# System prompt for summarization
SUMMARISATION_SYSTEM_PROMPT = """You are a chat summarization assistant. Your task is to create concise, accurate summaries of chat messages while preserving the essential information and context.

Instructions:
- Summarize the message content while maintaining its key points and intent
- Preserve important technical details, names, dates, and specific requirements
- Keep summaries concise but informative (aim for 50-70% reduction in length)
- Maintain the conversational tone and context
- If the message contains code, preserve the key functionality described
- If the message asks questions, preserve the questions in the summary
- Do not add information that wasn't in the original message

Return only the summary without any prefixes or explanations."""

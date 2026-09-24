# ruff: noqa: E501
"""Prompt text and tool schema for the combined title + classification call."""

from app.database.models import ChatClassification

TOOL_NAME_TITLE_AND_CLASSIFICATION = "generate_title_and_classification"

SYSTEM_PROMPT_TITLE_AND_CLASSIFICATION_STATIC = """\
You are a title generator and topic classifier. You are given a message from a \
government communications professional and must:

1. Generate a short title (maximum 5 words) that identifies the subject of the message.
2. Classify the message into one of the provided communication categories.
3. Identify the type of task the user is asking for help with.
4. Infer the user's communications discipline from their job title if provided.

Title rules:
- Up to five words.
- Sentence case: capitalise the first letter, proper nouns, and acronyms only. Not title case.
- Do not use a full stop at the end.
- Do not enclose the title in quotes.
- Always generate a title even if the query is vague or appears to be missing information.
- It is not your job to respond to the human query, only to generate a title and classify it.

Classification rules:
- Choose the single best-matching category from the list provided.
- If the message does not clearly fit any category, use "Other".

Task type rules:
- Drafting: the user wants to write or produce a communications product (press release, briefing, update, copy, script, etc.).
- Summarising: the user wants existing content condensed, synthesised, or extracted.
- Planning/brainstorming: the user wants to develop a strategy, plan, framework, or think something through.
- Reviewing/feedback: the user wants critique, proofreading, checking against a standard, or improvement suggestions.
- Research: the user wants to find, gather, or verify facts, evidence, or background information.
- Unknown: the message is too short, vague, or ambiguous to confidently determine the type of task.

Discipline rules — infer from job title if provided, otherwise use "Unknown":
- Media: press officers, media officers, media managers, heads of press.
- External affairs: external affairs managers, public affairs roles, stakeholder engagement.
- Marketing: marketing managers/officers, campaign managers, brand roles.
- Strategic comms: communications managers/officers/directors, heads of communications, deputy heads, strategic communications roles, communications business partners.
- Internal comms: internal communications managers/officers, heads of internal communications, employee engagement roles.
- Digital: digital communications officers/managers, social media managers, digital engagement roles.
- Data & insights: insight managers, data analysts, research roles, behavioural scientists, evaluation officers.
- Unknown: job title not provided or does not map to a communications discipline.

Some useful context (do not include this in the title or topic):
- The user is a government communications professional for UK government.
- GCS is Government Communications Service.
- OASIS refers to a framework for planning comms strategy.
- MCOM is the Modern Communications Operating Model.

You may be provided with a list of names of uploaded documents — that is the only \
additional context you will have about document-based queries.

The human query is provided between XML tags:
<human-query>This is an example message from the human.</human-query>
"""


def build_title_and_classification_system_prompt(
    classifications: list[ChatClassification],
    document_names: list[str],
    job_title: str | None = None,
) -> str | list[dict]:
    """Build the system prompt as a list of content blocks."""
    category_lines = "\n".join(
        f"- {c.title}: {c.description}" if c.description else f"- {c.title}" for c in classifications
    )
    static_text = (
        SYSTEM_PROMPT_TITLE_AND_CLASSIFICATION_STATIC
        + f"\nAvailable categories:\n{category_lines}\n- Other: Does not fit any of the above categories\n"
        + "\nThe following message is the human query for which you need to generate a title and classification."
    )

    blocks: list[dict] = [{"type": "text", "text": static_text}]

    dynamic_parts = []
    if job_title:
        dynamic_parts.append(f"The user's job title is: {job_title}")
    if document_names:
        dynamic_parts.append(
            "The following documents are used in the chat:\n" + "".join(f"- {name}\n" for name in document_names)
        )

    if dynamic_parts:
        blocks.append({"type": "text", "text": "\n\n".join(dynamic_parts)})

    return blocks


def build_title_and_classification_tool(classifications: list[ChatClassification]) -> dict:
    """Build the forced-tool schema with the live classification list baked into the enum."""
    category_titles = [c.title for c in classifications] + ["Other"]
    return {
        "name": TOOL_NAME_TITLE_AND_CLASSIFICATION,
        "description": "Generate a short title and classify the communication category of the user's query",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "A short title of maximum 5 words for the user's query. "
                        "Sentence case (capitalise first letter, proper nouns, acronyms). "
                        "No full stop at the end. No enclosing quotes."
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": category_titles,
                    "description": (
                        "The single category that best matches the user's query. "
                        "Choose 'Other' if no category is a good fit."
                    ),
                },
                "task_type": {
                    "type": "string",
                    "enum": [
                        "Drafting",
                        "Summarising",
                        "Planning/brainstorming",
                        "Reviewing/feedback",
                        "Research",
                        "Unknown",
                    ],
                    "description": (
                        "The type of task the user is asking for help with. "
                        "Use 'Unknown' if the message doesn't give enough information to tell."
                    ),
                },
                "discipline": {
                    "type": "string",
                    "enum": [
                        "Media",
                        "External affairs",
                        "Marketing",
                        "Strategic comms",
                        "Internal comms",
                        "Digital",
                        "Data & insights",
                        "Unknown",
                    ],
                    "description": "The user's communications discipline, inferred from their job title. Use 'Unknown' if no job title is provided or it does not map to a discipline.",
                },
            },
            "required": ["title", "category", "task_type", "discipline"],
        },
    }

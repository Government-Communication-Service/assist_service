# ruff: noqa: E501

CENTRAL_RAG_INDEX_NAME = "central_guidance"
""" The index name for storing documents that are available to everyone"""

# =============================================================================
# INDEX RELEVANCE EVALUATION TOOL
# =============================================================================

SYSTEM_PROMPT_INDEX_RELEVANCE_EVALUATOR = (
    "You are an intelligent search index router for the Government Communication Service (GCS). "
    "Your task is to determine whether a search query requires information from a specific knowledge index. "
    "You will be provided with the query and information about the search index. "
    "Think carefully about whether the query could benefit from information in this index. "
    "Be thoughtful and analytical in your assessment."
)

TOOL_NAME_INDEX_RELEVANCE_EVALUATOR = "evaluate_index_relevance"

TOOL_INDEX_RELEVANCE_EVALUATOR = {
    "name": TOOL_NAME_INDEX_RELEVANCE_EVALUATOR,
    "description": "Evaluate whether a search index is relevant for answering the user's query",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Thinking about of why this search index is or isn't relevant to the user's query. "
                    "Consider the query topic, intent, and how well it matches the index's scope."
                ),
            },
            "requires_index": {
                "type": "boolean",
                "description": (
                    "True if this search index contains information that could help answer the user's query. "
                    "False if the index is not relevant to the query topic or intent."
                ),
            },
        },
        "required": ["reasoning", "requires_index"],
    },
}

# =============================================================================
# OPENSEARCH QUERY GENERATOR TOOL
# =============================================================================

SYSTEM_PROMPT_OPENSEARCH_QUERY_GENERATOR = (
    "You work for the Government Communication Service, which is part of the UK Civil Service. "
    "Your job is to rewrite the users query so it can be used in the query body for OpenSearch "
    "to retrieve relevant information. The next message you receive will be the users original message to a chatbot. "
    "When looking at the user message, try to understand their intention "
    "and use this insight to create your OpenSearch queries."
)

TOOL_NAME_OPENSEARCH_QUERY_GENERATOR = "query_rewriter"

TOOL_OPENSEARCH_QUERY_GENERATOR = {
    "name": TOOL_NAME_OPENSEARCH_QUERY_GENERATOR,
    "description": "Generate an array of OpenSearch queries",
    "input_schema": {
        "type": "object",
        "properties": {
            "keyword_queries": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "description": (
                    "An array of keyword queries to send to OpenSearch. "
                    "By default, create 3 queries. "
                    "You can create more or less queries if the user requests it, "
                    "or if you think it will help get better results. "
                    "When choosing words for your first query, "
                    "quote the most appropriate words from the user's message. "
                    "When choosing words for your additional queries, use synonyms "
                    "so that you cover a wider search space (we are using the BM25 algorithm here "
                    "so it's important to not use the same word too much). "
                    "Make sure all queries are written in a single array."
                ),
                "items": {"type": "string", "description": "An OpenSearch keyword query based on the user's message."},
            }
        },
    },
}

# =============================================================================
# CHUNK RELEVANCE EVALUATION TOOL
# =============================================================================

SYSTEM_PROMPT_CHUNK_RELEVANCE_EVALUATOR = (
    "You are a document relevance evaluator for the Government Communication Service (GCS). "
    "Your job is to determine if a document chunk is relevant to a user's query. "
    "You will be given a user's original query and a document chunk (with title and content). "
    "Evaluate whether the chunk contains information that would help answer the user's question. "
    "Be judicious, and think carefully before giving a final answer."
)

TOOL_NAME_CHUNK_RELEVANCE_EVALUATOR = "evaluate_chunk_relevance"

TOOL_CHUNK_RELEVANCE_EVALUATOR = {
    "name": TOOL_NAME_CHUNK_RELEVANCE_EVALUATOR,
    "description": "Evaluate whether a document chunk is relevant to the user's query",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Brief explanation of why the chunk is or isn't relevant to the user's query. "
                    "This helps when thinking step by step through the evaluation."
                ),
            },
            "is_relevant": {
                "type": "boolean",
                "description": (
                    "True if the document chunk contains information relevant to the user's query. "
                    "False if the chunk is not relevant or useful for answering the user's question. "
                ),
            },
        },
        "required": ["is_relevant", "reasoning"],
    },
}

# =============================================================================
# DEFAULT CONTENT FOR BOOTSTRAPPING
# =============================================================================

DEFAULT_CHUNKS = [
    {
        "document_name": "Modern Communications Operating Model 3.0",
        "document_url": "https://gcs.civilservice.gov.uk/modern-communications-operating-model-3-0/",
        "document_description": """The Modern Communications Operating Model (MCOM) 3.0 brings together all GCS policies and guidance needed to build and lead a team of governmnet communicators. It contains information on: GCS Strategy; team design principles; equality diversity and inclusion; recruitment; learning and development; propriety and ethics; generative AI; procurement and spend; data handling; data protection; accessible communications; His Majesty's Government brand guidelines; OASIS campaign planning; innovating ethically; crisis communication;strategic communication; behavioural science / COM-B;influencer marketing; communications disciplines; media monitoring unit.""",
        "chunk_name": "Introduction: how to use MCOM 3.0",
        "chunk_content": """"The purpose of this new Modern Communications Operating Model (MCOM) is to provide simplicity and clarity about the expectations of teams and leaders within the Government Communication Service (GCS).

MCOM brings together all the policies and guidance needed to build and lead a team that delivers the GCS vision of exceptional communications that make a difference.

This updated MCOM uses a *must*, *should*, *could* framework to provide complete clarity on: the policies teams must follow; those that we recommend they should follow; and guidance that is available to consult and apply where needed.

The GCS Strategy and Government Communications Plan set the overarching strategy framework for government communications. The MCOM 'house' sits underneath this with three pillars: People & Structure, Policies, and Guidance & Tools.

Whether you are new to GCS, or an established leader who wants an accessible guide to best practice, this MCOM is for you. It is a living document and will be updated regularly, so we welcome ongoing feedback to ensure it remains relevant to you.

We hope that this updated approach enables you to use the recommendations and supporting guidance within MCOM to its best and fullest effect. We look forward to working together to continue delivering world class communications.""",
    },
]

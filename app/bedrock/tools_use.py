SEARCH_API_SEARCH_TERMS = {
    "tools": [
        {
            "name": "call_gov_uk_search_api",
            "description": (
                "Calls GOV UK Search API with search parameters that best match the user query. "
                "An empty query will get results from all of Gov UK ordered by popularity. "
                "You can filter results by date range, content purpose, and other fields. "
                "The content_purpose_supergroup and content_purpose_subgroup should not be used generally "
                "as they generally lead to overfiltering. "
                "In response to queries about recent news, press releases, etc. "
                "set content_purpose_supergroup to 'news_and_communications'. "
                "When answering queries related to government announcements, "
                "don't use generic words like 'announcements', 'updates', 'statements', etc. "
                "When answering queries related to government announcements, sort results by 'popularity' "
                "in descending order. "
                "The 'order_by' parameter shuold NEVER be set to 'relevance' as this will cause an error; "
                "to order results by relevance you MUST leave it empty instead. "
                "The 'count' field applies to each search term individually (e.g. if you provide 2 search terms, "
                "and you set count to 10, you will get up to 20 results back in total). "
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string", "description": "a search query compatible with ElasticSearch"},
                        "description": "Optional list of search queries compatible with ElasticSearch. "
                        "Keep the search queries short and concise. "
                        "Create meaningfully different queries to prevent overlap when we eventually use these queries "
                        "in the Gov UK Search API. "
                        "By default, 5 queries are recommended. "
                        "If creating queries, make sure the first one is the most obvious query."
                        "This field can be left empty to get results ordered by page popularity "
                        "without any keyword matching.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results to return for each search term. Default is 10.",
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "date_range": {
                        "type": "object",
                        "properties": {
                            "from": {
                                "type": "string",
                                "format": "date",
                                "description": "Start date in YYYY-MM-DD format",
                            },
                            "to": {"type": "string", "format": "date", "description": "End date in YYYY-MM-DD format"},
                        },
                        "description": "Optional date range filter for public_timestamp field",
                    },
                    "content_purpose_supergroup": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "other",
                                "news_and_communications",
                                "services",
                                "guidance_and_regulation",
                                "policy_and_engagement",
                                "research_and_statistics",
                                "transparency",
                            ],
                            "description": "Top-level categorization of the page",
                        },
                        "description": "Filter by content purpose supergroup. "
                        "To avoid overfiltering only use one of content_purpose_supergroup "
                        "and content_purpose_subgroup.",
                    },
                    "content_purpose_subgroup": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "updates_and_alerts",
                                "news",
                                "decisions",
                                "speeches_and_statements",
                                "transactions",
                                "regulation",
                                "guidance",
                                "business_support",
                                "policy",
                                "consultations",
                                "research",
                                "statistics",
                                "transparency_data",
                                "freedom_of_information_releases",
                                "incidents",
                                "calls_for_evidence",
                            ],
                            "description": "Sub-categorization of the page.",
                        },
                        "description": "Filter by content purpose subgroup. To avoid overfiltering "
                        "only use one of content_purpose_supergroup and content_purpose_subgroup.",
                    },
                    "is_political": {
                        "type": "boolean",
                        "description": "Filter by whether the content is political or not",
                    },
                    "order_by": {
                        "type": "string",
                        "description": "Field to order results by (e.g., 'popularity', 'public_timestamp'). "
                        "By default the results are ordered by relevance. "
                        "If you want the results ordered by relevance you MUST leave this field empty.",
                    },
                    "descending_order": {
                        "type": "boolean",
                        "description": "Whether to order results in descending order. By default the results "
                        "are ordered in descending order; you do not need to specify this field if you want the "
                        "results ordered by descending order.",
                    },
                },
            },
        }
    ]
}

DOCUMENT_RELEVANCE_ASSESSMENT = {
    "tools": [
        {
            "name": "assess_document_relevance",
            "description": """
                This tool assesses document relevancy to the given query.
                I would like you analyse the title and content of the document. Answer 'True' if the document is
                highly relevant to the query I am going to give you below; otherwise answer 'False'.

                Be very strict with your assessment. In particular, pay attention to the title of the Gov UK page.
                Discard any pages that are clearly login pages,or pages that only exist to let you download document.
                """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "is_relevant": {
                        "type": "boolean",
                    }
                },
                "required": ["is_relevant"],
            },
        }
    ]
}

DOWNLOAD_URLS = {
    "tools": [
        {
            "name": "download_urls",
            "description": """
                This tool downloads documents using URLs found in the user query.
                """,
            "input_schema": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string", "description": "URL"},
                    }
                },
                "required": ["urls"],
            },
        }
    ]
}

NAME_USE_GOV_UK_SEARCH_ASSESSMENT = "assess_if_gov_uk_search_should_be_used"
PROPERTY_NAME_USE_GOV_UK_SEARCH_ASSESSMENT = "use_gov_uk_search"
USE_GOV_UK_SEARCH_ASSESSMENT = {
    "name": NAME_USE_GOV_UK_SEARCH_ASSESSMENT,
    "description": (
        "Determines if GOV.UK Search should be used to address the user's query. "
        "GOV.UK Search can be used to:\n"
        " - Get current information about UK government policy\n"
        " - Retrieve government announcements\n"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            PROPERTY_NAME_USE_GOV_UK_SEARCH_ASSESSMENT: {
                "type": "boolean",
                "description": "'true' if GOV.UK Search should be used. 'false' otherwise.",
            }
        },
        "required": [PROPERTY_NAME_USE_GOV_UK_SEARCH_ASSESSMENT],
    },
}

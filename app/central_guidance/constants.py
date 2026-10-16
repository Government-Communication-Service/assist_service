# ruff: noqa: E501

CENTRAL_RAG_INDEX_NAME = "central_guidance"
""" The index name for storing documents that are available to everyone"""

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

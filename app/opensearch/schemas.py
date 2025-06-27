from dataclasses import dataclass


@dataclass
class OpenSearchRecord:
    """An internal representation of the data we need to submit a record to OpenSearch to create a record."""

    document_name: str
    document_url: str
    chunk_name: str
    chunk_content: str
    document_uuid: str = None

    def to_opensearch_dict(self):
        return {
            "document_name": self.document_name,
            "document_url": self.document_url,
            "chunk_name": self.chunk_name,
            "chunk_content": self.chunk_content,
            "document_uuid": self.document_uuid,
        }

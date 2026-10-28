import uuid

_EMPTY_SEARCH = {"hits": {"hits": [], "total": {"value": 0}}}
_EMPTY_BULK = {"errors": False, "items": []}


class _MockIndices:
    def get_alias(self, *args, **kwargs):
        return {"_mock": {}}

    def delete(self, *args, **kwargs):
        return {"acknowledged": True}

    def create(self, *args, **kwargs):
        return {"acknowledged": True}


class _MockCluster:
    def health(self, *args, **kwargs):
        return {"status": "yellow"}


class MockOpenSearch:
    """A no-op stand-in for the sync OpenSearch client.

    Used when OPENSEARCH_MOCKED is set so the app can run without a cluster.
    Searches return empty hits; writes/deletes report success.
    """

    def search(self, *args, **kwargs):
        return dict(_EMPTY_SEARCH)

    def bulk(self, *args, **kwargs):
        return dict(_EMPTY_BULK)

    def index(self, *args, **kwargs):
        return {"result": "created", "_id": str(uuid.uuid4())}

    def delete(self, *args, **kwargs):
        return {"result": "deleted"}

    @property
    def indices(self):
        return _MockIndices()

    @property
    def cluster(self):
        return _MockCluster()


class MockAsyncOpenSearch:
    """Async counterpart to MockOpenSearch — coroutine methods so callers can
    `await` results exactly as they would with the real AsyncOpenSearch client."""

    async def search(self, *args, **kwargs):
        return dict(_EMPTY_SEARCH)

    async def bulk(self, *args, body=None, **kwargs):
        body = body or []
        n = len(body) // 2  # body alternates action-header / document pairs
        return {
            "errors": False,
            "items": [{"index": {"_id": str(uuid.uuid4()), "_index": "mock", "result": "created"}} for _ in range(n)],
        }

    async def index(self, *args, **kwargs):
        return {"result": "created"}

    async def delete(self, *args, **kwargs):
        return {"result": "deleted"}

    @property
    def indices(self):
        return _MockIndices()

    async def close(self):
        return None

"""
An in-memory stand-in for OpenSearch, so the test suite runs without a cluster.

It supports only what the app sends: indexing and bulk index/delete, and searches built from
`multi_match` queries and `terms` filters. Scoring is a simple count of query words found in the
searched fields, which is enough for "the uploaded document is found by a query about it" and
"documents outside the filter are never returned". It says nothing about real relevance.
"""

import re
import uuid
from collections import defaultdict

from opensearchpy import NotFoundError

_WORD = re.compile(r"\w+")


def _words(text) -> list[str]:
    return _WORD.findall(str(text).lower())


class _Indices:
    def __init__(self, store: "InMemoryOpenSearch"):
        self._store = store

    def create(self, index, *args, **kwargs):
        self._store.docs.setdefault(index, {})
        return {"acknowledged": True, "index": index}

    def delete(self, index, *args, **kwargs):
        if index not in self._store.docs:
            raise NotFoundError(404, "index_not_found_exception", {"index": index})
        del self._store.docs[index]
        return {"acknowledged": True}

    def exists(self, index, *args, **kwargs):
        return index in self._store.docs

    def refresh(self, *args, **kwargs):
        return {"_shards": {"failed": 0}}

    def get_alias(self, *args, **kwargs):
        return {name: {"aliases": {}} for name in self._store.docs}


class _Cluster:
    def health(self, *args, **kwargs):
        return {"status": "green"}


class InMemoryOpenSearch:
    """The shared document store, with the sync client's methods."""

    def __init__(self):
        self.docs: dict[str, dict[str, dict]] = defaultdict(dict)
        self.indices = _Indices(self)
        self.cluster = _Cluster()

    def index(self, index, body, id=None, *args, **kwargs):  # noqa: A002 - matches opensearch-py
        # Generated ids depend only on the content, as test workers share one Postgres: the central index sync
        # stores each chunk's id there, and every worker's sync must agree on it.
        doc_id = str(id) if id is not None else uuid.uuid5(uuid.NAMESPACE_OID, f"{index}:{sorted(body.items())}").hex
        result = "updated" if doc_id in self.docs[index] else "created"
        self.docs[index][doc_id] = dict(body)
        return {"_index": index, "_id": doc_id, "result": result}

    def get(self, index, id, ignore=(), *args, **kwargs):  # noqa: A002 - matches opensearch-py
        source = self.docs[index].get(str(id))
        if source is None:
            if 404 in ignore:
                return {"_index": index, "_id": str(id), "found": False}
            raise NotFoundError(404, "not_found", {"_id": id})
        return {"_index": index, "_id": str(id), "found": True, "_source": source}

    def delete(self, index, id, *args, **kwargs):  # noqa: A002 - matches opensearch-py
        if self.docs[index].pop(str(id), None) is None:
            raise NotFoundError(404, "not_found", {"_id": id})
        return {"_index": index, "_id": str(id), "result": "deleted"}

    def bulk(self, body, *args, **kwargs):
        items = []
        lines = iter(body)
        for action in lines:
            ((op, meta),) = action.items()
            index, doc_id = meta["_index"], meta.get("_id")
            if op == "index":
                response = self.index(index, next(lines), id=doc_id)
                items.append({"index": {**response, "status": 201}})
            elif op == "delete":
                found = self.docs[index].pop(str(doc_id), None) is not None
                result = "deleted" if found else "not_found"
                items.append({"delete": {"_index": index, "_id": doc_id, "result": result, "status": 200}})
            else:
                raise NotImplementedError(f"Bulk action {op!r} is not supported by the fake")
        return {"errors": False, "items": items}

    def search(self, body=None, index=None, *args, **kwargs):
        query = body.get("query", {"match_all": {}})
        hits = []
        for doc_id, source in self.docs.get(index, {}).items():
            score = _score(query, source)
            if score is not None:
                hits.append({"_index": index, "_id": doc_id, "_score": score, "_source": source})
        hits.sort(key=lambda hit: hit["_score"], reverse=True)
        hits = hits[: body.get("size", 10)]
        return {"timed_out": False, "hits": {"total": {"value": len(hits)}, "hits": hits}}


def _score(query: dict, source: dict) -> float | None:
    """The document's score for the query, or None when it doesn't match."""
    ((kind, clause),) = query.items()
    if kind == "match_all":
        return 1.0
    if kind == "terms":
        ((field, values),) = clause.items()
        return 1.0 if str(source.get(field.removesuffix(".keyword"))) in {str(v) for v in values} else None
    if kind == "match_phrase":
        ((field, phrase),) = clause.items()
        return 1.0 if " ".join(_words(phrase)) in " ".join(_words(source.get(field, ""))) else None
    if kind == "multi_match":
        searched = [word for field in clause["fields"] for word in _words(source.get(field, ""))]
        score = sum(searched.count(word) for word in set(_words(clause["query"])))
        return float(score) if score else None
    if kind == "bool":
        if any(_score(f, source) is None for f in clause.get("filter", [])):
            return None
        must = [_score(m, source) for m in clause.get("must", [])]
        if any(s is None for s in must):
            return None
        return sum(must) if must else 1.0
    raise NotImplementedError(f"Query {kind!r} is not supported by the fake")


class InMemoryAsyncOpenSearch:
    """The async client's methods over the same store."""

    def __init__(self, store: InMemoryOpenSearch):
        self._store = store
        self.indices = store.indices

    async def search(self, body=None, index=None, *args, **kwargs):
        return self._store.search(body, index=index)

    async def bulk(self, body, *args, **kwargs):
        return self._store.bulk(body)

    async def index(self, index, body, id=None, *args, **kwargs):  # noqa: A002 - matches opensearch-py
        return self._store.index(index, body, id=id)

    async def get(self, index, id, *args, **kwargs):  # noqa: A002 - matches opensearch-py
        return self._store.get(index, id, *args, **kwargs)

    async def delete(self, index, id, *args, **kwargs):  # noqa: A002 - matches opensearch-py
        return self._store.delete(index, id)

    async def close(self):
        return None

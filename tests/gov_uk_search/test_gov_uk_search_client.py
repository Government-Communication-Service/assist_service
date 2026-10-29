import pytest

from app.gov_uk_search.service import GovUKSearch

QUERY = "bin collection"


class TestGovUKSearch:
    @pytest.mark.asyncio
    async def test_gov_uk_search(self):
        resp = await GovUKSearch.search(QUERY)
        assert isinstance(resp, dict)
        assert len(resp.get("results")) > 0

    @pytest.mark.asyncio
    async def test_gov_uk_search_count(self):
        resp = await GovUKSearch.search(QUERY, count=5)
        assert isinstance(resp, dict)
        assert len(resp.get("results")) <= 5

    @pytest.mark.asyncio
    async def test_gov_uk_search_order_by_field_name(self):
        resp_order_ascending = await GovUKSearch.search(
            QUERY,
            count=2,
            order_by_field_name="title",
        )
        resp_order_descending = await GovUKSearch.search(
            QUERY,
            count=2,
            order_by_field_name="title",
            descending_order=True,
        )
        assert isinstance(resp_order_ascending, dict)
        assert isinstance(resp_order_descending, dict)
        resp_order_ascending_results = [d["title"] for d in resp_order_ascending["results"]]
        resp_order_descending_results = [d["title"] for d in resp_order_descending["results"]]
        resp_order_ascending_results.reverse()
        assert resp_order_ascending_results[0] != resp_order_descending_results[0]

    @pytest.mark.skip(reason="Flaky: calls the live GOV.UK search API, which sometimes returns no results")
    @pytest.mark.asyncio
    async def test_gov_uk_search_start(self):
        resp = await GovUKSearch.search(
            QUERY,
        )
        resp_offset = await GovUKSearch.search(
            QUERY,
            start=2,
        )
        assert isinstance(resp, dict)
        assert isinstance(resp_offset, dict)
        assert resp["results"][2]["_id"] == resp_offset["results"][0]["_id"]

    @pytest.mark.asyncio
    async def test_gov_uk_search_fields(self):
        resp = await GovUKSearch.search(
            QUERY,
            fields=["id", "title"],
        )
        assert isinstance(resp, dict)
        assert resp["results"][0].get("_id") is not None
        assert resp["results"][0].get("title") is not None
        assert resp["results"][0].get("description") is None

    @pytest.mark.asyncio
    async def test_gov_uk_search_filter_(self):
        resp = await GovUKSearch.search(
            QUERY,
            filter_by_field=[("organisations", "cabinet-office")],
        )
        assert isinstance(resp, dict)
        assert len(resp.get("results")) > 0
        assert len(resp["results"][0].get("organisations")) > 0
        organisations = resp["results"][0].get("organisations")
        assert (
            len([org["organisation_brand"] for org in organisations if org["organisation_brand"] == "cabinet-office"])
            > 0
        )

    @pytest.mark.asyncio
    async def test_gov_uk_search_multiple_filter_(self):
        resp = await GovUKSearch.search(
            QUERY,
            filter_by_field=[("organisations", "cabinet-office"), ("organisations", "ministry-of-defence")],
        )
        assert isinstance(resp, dict)
        assert len(resp["results"][0].get("organisations")) > 0
        organisations = set()
        for row in resp.get("results"):
            for org in row.get("organisations"):
                organisations.add(org.get("organisation_brand"))
        assert len(organisations) > 0
        assert "cabinet-office" in organisations
        assert "ministry-of-defence" in organisations

    @pytest.mark.asyncio
    async def test_gov_uk_search_taxes_count_1(self):
        # official example for
        # Simple search query:
        # https://www.gov.uk/api/search.json?q=taxes&count=1
        resp = await GovUKSearch.search(
            query="taxes",
            count=1,
        )
        assert isinstance(resp, dict)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_taxes_count_1_start_from_1(self):
        # official example for
        # Get the next result in the sequence by specifying start=1
        # https://www.gov.uk/api/search.json?q=taxes&count=1&start=1
        resp = await GovUKSearch.search(
            query="taxes",
            count=1,
            start=1,
        )
        assert isinstance(resp, dict)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_taxes_count_1_start_from_1_oldest(self):
        # official example for
        # get the oldest match:
        # https://www.gov.uk/api/search.json?q=taxes&count=1&order=public_timestamp
        resp = await GovUKSearch.search(
            query="taxes",
            count=1,
            start=1,
            order_by_field_name="public_timestamp",
        )
        assert isinstance(resp, dict)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_passport_with_two_fields(self):
        # official example for
        # Retrieve just the title and 'mainstream browse pages' of documents matching search term "passport":
        # https://www.gov.uk/api/search.json?q=passport&fields=mainstream_browse_pages&fields=title
        resp = await GovUKSearch.search(
            query="passport",
            fields=["mainstream_browse_pages", "title"],
        )
        assert isinstance(resp, dict)
        assert len(resp.get("results")) > 0

    @pytest.mark.asyncio
    async def test_gov_uk_search_test_count_1_filter_by_format_transaction(self):
        # Retrieve documents of a specific type:
        # https://www.gov.uk/api/search.json?q=test&count=1&filter_format=transaction
        resp = await GovUKSearch.search(
            query="test",
            count=1,
            filter_by_field=[("format", "transaction")],
        )
        assert isinstance(resp, dict)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_policy_count_1_filter_organisation_cabinet_office(self):
        # Retrieve documents for a given organisation:
        # https://www.gov.uk/api/search.json?q=policy&count=1&filter_organisations=cabinet-office
        resp = await GovUKSearch.search(
            query="policy",
            count=1,
            filter_by_field=[("organisations", "cabinet-office")],
        )
        assert isinstance(resp, dict)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_policy_count_1_filter_organisation_cabinet_office_home_office(self):
        # Retrieve documents for multiple organisations:
        # https://www.gov.uk/api/search.json?q=policy&count=1&filter_organisations=cabinet-office&filter_organisations=home-office
        resp = await GovUKSearch.search(
            query="policy",
            count=1,
            filter_by_field=[("organisations", "cabinet-office"), ("organisations", "home-office")],
        )
        assert isinstance(resp, dict)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_policy_count_1_filter_by_public_timestamp(self):
        # Find documents published within a certain date range:
        # https://www.gov.uk/api/search.json?q=pig&count=1&filter_public_timestamp=from:2020-01-01,to:2020-12-31
        resp = await GovUKSearch.search(
            query="pig",
            count=1,
            filter_by_field=[("public_timestamp", "from:2020-01-01,to:2020-12-31")],
        )
        assert isinstance(resp, dict)
        assert len(resp.get("results")) == 1

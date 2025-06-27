import pytest

from app.bedrock import BedrockHandler, RunMode
from app.config import LLM_DEFAULT_MODEL
from app.database.db_operations import DbOperations
from app.database.models import GovUkSearchQuery
from app.database.table import LLMTable
from app.gov_uk_search.service import GovUKSearch

QUERY = "bin collection"


class TestGovUKSearch:
    @pytest.mark.asyncio
    async def test_gov_uk_search(self, db_session):
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            QUERY, db_session=db_session, llm_internal_response_id_query=llm_internal_response_id_query
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert len(resp.get("results")) > 0

    @pytest.mark.asyncio
    async def test_gov_uk_search_count(self, db_session):
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            QUERY, db_session=db_session, count=5, llm_internal_response_id_query=llm_internal_response_id_query
        )
        assert isinstance(resp, dict)
        assert len(resp.get("results")) <= 5

    @pytest.mark.asyncio
    async def test_gov_uk_search_order_by_field_name(self, db_session):
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp_order_ascending = await GovUKSearch.simple_search(
            QUERY,
            db_session=db_session,
            count=2,
            order_by_field_name="title",
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        resp_order_descending = await GovUKSearch.simple_search(
            QUERY,
            db_session=db_session,
            count=2,
            order_by_field_name="title",
            descending_order=True,
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp_order_ascending, tuple)
        assert isinstance(resp_order_descending, tuple)
        resp_order_ascending_results = [d["title"] for d in resp_order_ascending[0]["results"]]
        resp_order_descending_results = [d["title"] for d in resp_order_descending[0]["results"]]
        resp_order_ascending_results.reverse()
        assert resp_order_ascending_results[0] != resp_order_descending_results[0]

    @pytest.mark.asyncio
    async def test_gov_uk_search_start(self, db_session):
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            QUERY,
            db_session=db_session,
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        resp_offset, gov_uk_search_query_offset = await GovUKSearch.simple_search(
            QUERY,
            db_session=db_session,
            start=2,
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(resp_offset, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert resp["results"][2]["_id"] == resp_offset["results"][0]["_id"]

    @pytest.mark.asyncio
    async def test_gov_uk_search_fields(self, db_session):
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            QUERY,
            db_session=db_session,
            fields=["id", "title"],
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert resp["results"][0].get("_id") is not None
        assert resp["results"][0].get("title") is not None
        assert resp["results"][0].get("description") is None

    @pytest.mark.asyncio
    async def test_gov_uk_search_filter_(self, db_session):
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            QUERY,
            db_session=db_session,
            filter_by_field=[("organisations", "cabinet-office")],
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert len(resp.get("results")) > 0
        assert len(resp["results"][0].get("organisations")) > 0
        organisations = resp["results"][0].get("organisations")
        assert (
            len([org["organisation_brand"] for org in organisations if org["organisation_brand"] == "cabinet-office"])
            > 0
        )

    @pytest.mark.asyncio
    async def test_gov_uk_search_multiple_filter_(self, db_session):
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            QUERY,
            db_session=db_session,
            filter_by_field=[("organisations", "cabinet-office"), ("organisations", "ministry-of-defence")],
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert len(resp["results"][0].get("organisations")) > 0
        organisations = set()
        for row in resp.get("results"):
            for org in row.get("organisations"):
                organisations.add(org.get("organisation_brand"))
        assert len(organisations) > 0
        assert "cabinet-office" in organisations
        assert "ministry-of-defence" in organisations

    @pytest.mark.asyncio
    async def test_gov_uk_search_taxes_count_1(self, db_session):
        # official example for
        # Simple search query:
        # https://www.gov.uk/api/search.json?q=taxes&count=1
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            query="taxes",
            db_session=db_session,
            count=1,
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_taxes_count_1_start_from_1(self, db_session):
        # official example for
        # Get the next result in the sequence by specifying start=1
        # https://www.gov.uk/api/search.json?q=taxes&count=1&start=1
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            query="taxes",
            db_session=db_session,
            count=1,
            start=1,
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_taxes_count_1_start_from_1_oldest(self, db_session):
        # official example for
        # get the oldest match:
        # https://www.gov.uk/api/search.json?q=taxes&count=1&order=public_timestamp
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            query="taxes",
            db_session=db_session,
            count=1,
            start=1,
            order_by_field_name="public_timestamp",
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_passport_with_two_fields(self, db_session):
        # official example for
        # Retrieve just the title and 'mainstream browse pages' of documents matching search term "passport":
        # https://www.gov.uk/api/search.json?q=passport&fields=mainstream_browse_pages&fields=title
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            query="passport",
            db_session=db_session,
            fields=["mainstream_browse_pages", "title"],
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert len(resp.get("results")) > 0

    @pytest.mark.asyncio
    async def test_gov_uk_search_test_count_1_filter_by_format_transaction(self, db_session):
        # Retrieve documents of a specific type:
        # https://www.gov.uk/api/search.json?q=test&count=1&filter_format=transaction
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            query="test",
            db_session=db_session,
            count=1,
            filter_by_field=[("format", "transaction")],
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_policy_count_1_filter_organisation_cabinet_office(self, db_session):
        # Retrieve documents for a given organisation:
        # https://www.gov.uk/api/search.json?q=policy&count=1&filter_organisations=cabinet-office
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            query="policy",
            db_session=db_session,
            count=1,
            filter_by_field=[("organisations", "cabinet-office")],
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_policy_count_1_filter_organisation_cabinet_office_home_office(self, db_session):
        # Retrieve documents for multiple organisations:
        # https://www.gov.uk/api/search.json?q=policy&count=1&filter_organisations=cabinet-office&filter_organisations=home-office
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            query="policy",
            db_session=db_session,
            count=1,
            filter_by_field=[("organisations", "cabinet-office"), ("organisations", "home-office")],
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert len(resp.get("results")) == 1

    @pytest.mark.asyncio
    async def test_gov_uk_search_policy_count_1_filter_by_public_timestamp(self, db_session):
        # Find documents published within a certain date range:
        # https://www.gov.uk/api/search.json?q=pig&count=1&filter_public_timestamp=from:2020-01-01,to:2020-12-31
        # setup
        llm_obj = LLMTable().get_by_model(LLM_DEFAULT_MODEL)
        web_browsing_llm = BedrockHandler(llm=llm_obj, mode=RunMode.ASYNC)

        response = await DbOperations.insert_llm_internal_response_id_query(
            db_session=db_session,
            web_browsing_llm=web_browsing_llm.llm,
            content="TEST",
            tokens_in=10,
            tokens_out=100,
            completion_cost=10 * web_browsing_llm.llm.input_cost_per_token
            + 100 * web_browsing_llm.llm.output_cost_per_token,
        )
        llm_internal_response_id_query = response.id
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            query="pig",
            db_session=db_session,
            count=1,
            filter_by_field=[("public_timestamp", "from:2020-01-01,to:2020-12-31")],
            llm_internal_response_id_query=llm_internal_response_id_query,
        )
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert len(resp.get("results")) == 1

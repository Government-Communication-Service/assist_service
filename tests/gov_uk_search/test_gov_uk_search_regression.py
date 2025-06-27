import pytest

from app.bedrock import BedrockHandler, RunMode
from app.config import LLM_DEFAULT_MODEL
from app.database.db_operations import DbOperations
from app.database.models import GovUkSearchQuery
from app.database.table import LLMTable
from app.gov_uk_search.service import GovUKSearch
from app.gov_uk_search.utils import build_search_url


class TestGovUKSearchRegression:
    """
    Regression tests for GOV UK Search API issues.

    This test class documents and prevents regression of specific bugs
    that have been encountered and fixed in the GOV UK Search functionality.
    """

    def test_relevance_order_parameter_issue(self):
        """
        Regression test for 422 error caused by invalid 'order=relevance' parameter.

        ISSUE DESCRIPTION:
        The GOV UK Search API was returning 422 Unprocessable Entity errors when
        the LLM set order_by to "relevance". This happened because:

        1. The LLM tool definition stated that results are "ordered by relevance" by default
        2. The LLM was explicitly setting order_by="relevance" in some cases
        3. The GOV UK Search API doesn't accept "relevance" as a valid order field value
        4. The API expects NO order parameter when you want relevance-based ordering

        EXAMPLE BROKEN URL:
        https://www.gov.uk/api/search.json?q=legal%20consequences%20rioting%20during%20trial&count=10&order=relevance&fields=title...

        ROOT CAUSE:
        The build_search_url function was blindly adding "&order=relevance" to the URL
        without checking if "relevance" is a valid API parameter.

        SOLUTION:
        Modified build_search_url to skip adding the order parameter when
        order_by_field_name is "relevance" (case-insensitive), since relevance
        is the default behavior when no order parameter is provided.

        FIXED URL:
        https://www.gov.uk/api/search.json?q=legal%20consequences%20rioting%20during%20trial&count=10&fields=title...
        (Notice the absence of &order=relevance)
        """

        # Test that "relevance" order_by doesn't add order parameter to URL
        url_with_relevance = build_search_url(
            query="legal consequences rioting during trial",
            count=10,
            order_by_field_name="relevance",
            descending_order=False,
            fields=["title", "description", "link", "primary_publishing_organisation", "public_timestamp"],
        )

        # The URL should NOT contain "&order=relevance"
        assert "&order=relevance" not in url_with_relevance
        assert "&order=" not in url_with_relevance  # No order parameter at all

        # Test case-insensitive handling
        url_with_relevance_caps = build_search_url(query="test query", order_by_field_name="RELEVANCE")
        assert "&order=RELEVANCE" not in url_with_relevance_caps
        assert "&order=" not in url_with_relevance_caps

        # Test that valid order fields still work
        url_with_popularity = build_search_url(
            query="test query", order_by_field_name="popularity", descending_order=True
        )
        assert "&order=-popularity" in url_with_popularity

        url_with_timestamp = build_search_url(
            query="test query", order_by_field_name="public_timestamp", descending_order=False
        )
        assert "&order=public_timestamp" in url_with_timestamp

    @pytest.mark.asyncio
    async def test_relevance_order_api_call(self, db_session):
        """
        Integration test to ensure API calls with relevance ordering work correctly.

        This test verifies that the fix for the relevance ordering issue works
        in practice by making an actual API call that would have previously failed.
        """
        # Setup
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

        # This call would have previously failed with 422 error when order_by_field_name="relevance"
        resp, gov_uk_search_query = await GovUKSearch.simple_search(
            query="legal consequences rioting during trial",
            db_session=db_session,
            count=10,
            order_by_field_name="relevance",  # This was the problematic parameter
            fields=["title", "description", "link", "primary_publishing_organisation", "public_timestamp"],
            llm_internal_response_id_query=llm_internal_response_id_query,
        )

        # Verify the call succeeded (no 422 error)
        assert isinstance(resp, dict)
        assert isinstance(gov_uk_search_query, GovUkSearchQuery)
        assert "results" in resp
        # Should have results or empty list, but not error
        assert isinstance(resp["results"], list)

    def test_url_building_edge_cases(self):
        """
        Test edge cases in URL building to prevent similar parameter issues.

        This test covers various edge cases that could cause similar API errors:
        - Empty order_by_field_name
        - None order_by_field_name
        - Mixed case variations
        - Other parameter combinations
        """

        # Test empty order_by_field_name
        url_empty_order = build_search_url(query="test", order_by_field_name="")
        assert "&order=" not in url_empty_order

        # Test None order_by_field_name
        url_none_order = build_search_url(query="test", order_by_field_name=None)
        assert "&order=" not in url_none_order

        # Test various case combinations of "relevance"
        relevance_variations = ["relevance", "RELEVANCE", "Relevance", "ReLeVaNcE"]
        for variation in relevance_variations:
            url = build_search_url(query="test", order_by_field_name=variation)
            assert "&order=" not in url, f"Failed for variation: {variation}"

        # Test that non-relevance values still work
        valid_order_fields = ["popularity", "public_timestamp", "title"]
        for field in valid_order_fields:
            url = build_search_url(query="test", order_by_field_name=field)
            assert f"&order={field}" in url, f"Failed for field: {field}"

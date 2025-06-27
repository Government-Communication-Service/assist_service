import logging

from app.api import ENDPOINTS

# Configure logging
logger = logging.getLogger(__name__)

api = ENDPOINTS("v1")


class TestThemes:
    def setup_method(self):
        """Set up prerequisites for each test method.

        This method is invoked before each test method to initialize any shared state.
        It sets the themes attribute to None and initializes use_cases as an empty dictionary.
        """
        logger.debug("Setting up for a test method.")
        self.themes = None
        self.use_cases = {}

    async def fetch_use_cases(self, async_client, headers, theme_uuid):
        """Fetch and cache use cases for a specified theme.

        Makes an API call to retrieve use cases for a given theme UUID and caches the results.
        Subsequent calls for the same theme UUID will return the cached results.

        Args:
            client: The HTTP client used to make requests.
            headers: The headers to include in the request.
            theme_uuid: The UUID of the theme for which to fetch use cases.

        Returns:
            A list of use cases associated with the specified theme UUID.

        Raises:
            AssertionError: If the response status code is not 200.
        """
        if theme_uuid not in self.use_cases:
            logger.debug(f"Fetching use cases for theme UUID {theme_uuid}.")
            response = await async_client.get(api.themes_use_cases(theme_uuid), headers=headers)
            logger.debug(
                f"API call to get use cases for theme {theme_uuid} returned status {response.status_code} "
                f"and body {response.text[:100]}",
            )
            assert response.status_code == 200, "Failed to fetch use cases."
            self.use_cases[theme_uuid] = response.json()["use_cases"]
        return self.use_cases[theme_uuid]

    async def fetch_themes(self, async_client, default_headers):
        """Fetch all themes from the API.

        Makes an API call to retrieve all available themes.

        Args:
            client: The HTTP client used to make requests.
            default_headers: The headers to include in the request.

        Returns:
            The response object containing the themes data.
        """
        logger.debug("Fetching themes from API.")
        themes_response = await async_client.get(api.get_themes(), headers=default_headers)
        logger.debug(
            "API call to get themes returned status"
            + f" {themes_response.status_code} and body {themes_response.text[:100]}"
        )
        return themes_response

    async def create_theme(
        self,
        async_client,
        auth_token_only_headers,
        title: str = "Test title: create theme",
        subtitle: str = "Test subtitle: create theme",
        position: int = 1,
    ):
        """Create a new theme using the API.

        Makes an API call to create a new theme with the specified parameters.

        Args:
            client: The HTTP client used to make requests.
            auth_token_only_headers: The headers containing authentication token.
            title: The title of the theme to be created. Defaults to "Test title: create theme".
            subtitle: The subtitle of the theme to be created. Defaults to "Test subtitle: create theme".
            position: The position of the theme to be created. Defaults to 1.

        Returns:
            The response object from the API call to create a theme.
        """
        response = await async_client.post(
            api.post_theme(),
            json={"title": title, "subtitle": subtitle, "position": position},
            headers=auth_token_only_headers,
        )
        return response

    async def create_use_case(
        self,
        theme_uuid,
        use_case_title: str,
        use_case_instruction: str,
        use_case_user_input_form: str,
        use_case_position: int,
        auth_token_only_headers,
        async_client,
    ):
        """Create a new use case for a specified theme.

        Makes an API call to create a new use case with the specified parameters.

        Args:
            theme_uuid: The UUID of the theme to associate with the use case.
            use_case_title: The title of the use case to be created.
            use_case_instruction: The instruction for the use case.
            use_case_user_input_form: The user input form for the use case.
            use_case_position: The position for the use case.
            auth_token_only_headers: The headers containing authentication token.
            client: The HTTP client used to make requests.

        Returns:
            The response object from the API call to create a use case.
        """
        response = await async_client.post(
            api.themes_use_cases(theme_uuid),
            json={
                "title": use_case_title,
                "instruction": use_case_instruction,
                "user_input_form": use_case_user_input_form,
                "position": use_case_position,
            },
            headers=auth_token_only_headers,
        )

        return response

    async def create_theme_and_use_case(
        self,
        async_client,
        auth_token_only_headers,
        theme_title: str = "Test title: create theme",
        theme_subtitle: str = "Test subtitle: create theme: ",
        theme_position: int = 1,
        use_case_title: str = "Test title: create use case",
        use_case_instruction: str = "Test instruction",
        use_case_user_input_form: str = "Test form",
        use_case_position: int = 1,
    ):
        """Create a new theme and an associated use case.

        Makes API calls to create a new theme and then creates a use case associated with it.

        Args:
            async_client: The HTTP client used to make requests.
            auth_token_only_headers: The headers containing authentication token.
            theme_title: The title of the theme to be created. Defaults to "Test title: create theme".
            theme_subtitle: The subtitle of the theme to be created. Defaults to "Test subtitle: create theme: ".
            theme_position: The position of the theme to be created. Defaults to 1.
            use_case_title: The title of the use case to be created. Defaults to "Test title: create use case".
            use_case_instruction: The instruction for the use case. Defaults to "Test instruction".
            use_case_user_input_form: The user input form for the use case. Defaults to "Test form".
            use_case_position: The position for the use case. Defaults to 1.

        Returns:
            A tuple containing the response objects from the API calls to create a theme and a use case.

        Raises:
            AssertionError: If either the theme or use case creation fails.
        """

        theme_response = await self.create_theme(
            async_client, auth_token_only_headers, title=theme_title, subtitle=theme_subtitle, position=theme_position
        )

        assert theme_response.status_code == 200, (
            f"Error creating theme: expected status code 200, got {theme_response.status_code}"
        )

        theme_uuid = theme_response.json()["uuid"]

        use_case_response = await self.create_use_case(
            theme_uuid,
            use_case_title,
            use_case_instruction,
            use_case_user_input_form,
            use_case_position,
            auth_token_only_headers,
            async_client,
        )

        assert use_case_response.status_code == 200, (
            f"Error creating use case: expected status code 200, got {use_case_response.status_code}"
        )

        return (theme_response, use_case_response)

    async def test_get_theme_use_cases_happy_path(self, async_client, default_headers, auth_token_only_headers):
        """Test fetching use cases of a theme (happy path).

        Creates a theme and use case, then verifies they can be successfully retrieved.

        Args:
            client: The HTTP client used to make requests.
            default_headers: The default headers for requests.
            auth_token_only_headers: The headers containing authentication token.

        Raises:
            AssertionError: If no use cases are found in the response.
        """
        logger.debug("Testing happy path for getting theme use cases.")

        # Start by creating a theme and use case
        theme_response, use_case_response = await self.create_theme_and_use_case(async_client, auth_token_only_headers)
        theme_uuid = theme_response.json()["uuid"]

        # Get the use cases
        use_cases = await self.fetch_use_cases(async_client, default_headers, theme_uuid=theme_uuid)

        # Check that there are some use cases present
        assert use_cases, "No use cases in the response."

    async def test_create_theme(self, async_client, default_headers, auth_token_only_headers):
        """Test the creation of a new theme.

        Creates multiple themes with different positions and verifies they are stored and ordered correctly.

        Args:
            db_session: sqlalchemy session.
            client: The HTTP client used to make requests.
            default_headers: The default headers for requests.
            auth_token_only_headers: The headers containing authentication token.

        Raises:
            AssertionError: If themes are not created or ordered correctly.
        """
        # Create a theme with unique title and subtitle
        first_theme_response = await self.create_theme(
            async_client, auth_token_only_headers, title="Theme 1", subtitle="Subtitle 1"
        )
        first_theme_uuid = first_theme_response.json()["uuid"]

        # Check the request was successful
        assert first_theme_response.status_code == 200, (
            f"Status code {first_theme_response.status_code} was not 200; Message = {first_theme_response.text}"
        )

        # Create a second theme with unique title and subtitle, position=2; make sure it comes after the first theme
        second_theme_response = await self.create_theme(
            async_client, auth_token_only_headers, title="Theme 2", subtitle="Subtitle 2", position=2
        )
        second_theme_uuid = second_theme_response.json()["uuid"]

        # Create a third theme with unique title and subtitle
        # with no explicit position (i.e. position=None).
        # Make sure it comes last when you get the themes
        third_theme_response = await self.create_theme(
            async_client, auth_token_only_headers, title="Theme 3", subtitle="Subtitle 3", position=None
        )
        third_theme_uuid = third_theme_response.json()["uuid"]

        # Create a fourth and fifth theme with unique titles and subtitles
        # with the same position as the first theme (position=1);
        # make sure these are all returned in insert order
        # but before the theme with position=2
        fourth_theme_response = await self.create_theme(
            async_client, auth_token_only_headers, title="Theme 4", subtitle="Subtitle 4", position=1
        )
        fourth_theme_uuid = fourth_theme_response.json()["uuid"]

        fifth_theme_response = await self.create_theme(
            async_client, auth_token_only_headers, title="Theme 5", subtitle="Subtitle 5", position=1
        )
        fifth_theme_uuid = fifth_theme_response.json()["uuid"]

        # Fetch all themes and verify the order
        themes_response = await self.fetch_themes(async_client, default_headers)
        themes = themes_response.json()["themes"]
        uuids_themes = [theme["uuid"] for theme in themes]

        # Check the database now contains this theme's uuid
        assert first_theme_uuid in uuids_themes, (
            f"First theme UUID {first_theme_uuid} not found in the list of theme UUIDs."
        )

        # Verify the order of themes
        assert uuids_themes.index(first_theme_uuid) < uuids_themes.index(second_theme_uuid), (
            "First theme is not before second theme in the list."
        )
        assert uuids_themes.index(second_theme_uuid) < uuids_themes.index(third_theme_uuid), (
            "Second theme is not before third theme in the list."
        )
        assert uuids_themes.index(first_theme_uuid) < uuids_themes.index(fourth_theme_uuid), (
            "First theme is not before fourth theme in the list."
        )
        assert uuids_themes.index(fourth_theme_uuid) < uuids_themes.index(fifth_theme_uuid), (
            "Fourth theme is not before fifth theme in the list."
        )
        assert uuids_themes.index(fifth_theme_uuid) < uuids_themes.index(second_theme_uuid), (
            "Fifth theme is not before second theme in the list."
        )

        # Test creating a theme with identical title and subtitle to the first theme
        duplicate_theme_response = await self.create_theme(
            async_client, auth_token_only_headers, title="Theme 1", subtitle="Subtitle 1"
        )
        duplicate_theme_uuid = duplicate_theme_response.json()["uuid"]

        # Verify that the same UUID is returned for the duplicate theme
        assert duplicate_theme_uuid == first_theme_uuid, (
            f"Duplicate theme UUID {duplicate_theme_uuid} does not match the original theme UUID {first_theme_uuid}."
        )

    async def test_get_theme(self, async_client, auth_token_only_headers):
        """Test fetching a specific theme by its UUID.

        Creates a theme and verifies it can be retrieved with all expected fields.

        Args:
            async_client: The HTTP client used to make requests.
            auth_token_only_headers: The headers containing authentication token.

        Raises:
            AssertionError: If theme retrieval fails or required fields are missing.
        """
        # Create a theme
        theme_response = await self.create_theme(async_client, auth_token_only_headers)
        theme_uuid = theme_response.json()["uuid"]

        # Get information about a specific theme
        response = await async_client.get(api.theme(theme_uuid), headers=auth_token_only_headers)
        body = response.json()

        # Check that the response is as we expect
        assert response.status_code == 200, f"Status code {response.status_code} was not 200; Message = {body}"
        assert body["uuid"], "No UUID provided when creating theme"
        assert body["created_at"], "No created_at provided when creating theme"
        assert body["title"], "No title provided when creating theme"
        assert body["subtitle"], "No subtitle provided when creating theme"
        assert body["position"], "No position provided when creating theme"

    async def test_get_themes_happy_path(self, async_client, default_headers, auth_token_only_headers):
        """Test fetching all themes (happy path).

        Creates a theme and verifies all themes can be successfully retrieved.

        Args:
            client: The HTTP client used to make requests.
            default_headers: The default headers for requests.
            auth_token_only_headers: The headers containing authentication token.

        Raises:
            AssertionError: If theme retrieval fails or no themes are found.
        """
        logger.debug("Testing happy path for getting themes.")

        # Create a theme
        await self.create_theme(async_client, auth_token_only_headers)

        # Get all themes in the database
        themes_response = await self.fetch_themes(async_client, default_headers)
        themes = themes_response.json()["themes"]

        # Check the fetch_themes fixture responded with a 200
        assert themes_response.status_code == 200, "Failed to fetch themes."

        # Check that there are themes in the response
        assert themes, "No themes in the response."

    async def test_put_theme(self, async_client, auth_token_only_headers):
        """Test updating a theme's title, subtitle, and position.

        Creates a theme, updates its attributes, and verifies the changes are saved correctly.

        Args:
            client: The HTTP client used to make requests.
            auth_token_only_headers: The headers containing authentication token.

        Raises:
            AssertionError: If theme update fails or changes are not reflected.
        """
        # Create a theme
        theme_response = await self.create_theme(async_client, auth_token_only_headers)
        theme_uuid = theme_response.json()["uuid"]

        # Edit the theme
        changed_title = "Changed title"
        changed_subtitle = "Changed subtitle"
        changed_position = 100

        await async_client.put(
            api.theme(theme_uuid),
            json={
                "title": changed_title,
                "subtitle": changed_subtitle,
                "position": changed_position,
            },
            headers=auth_token_only_headers,
        )

        # Check that the theme has been edited
        response = await async_client.get(api.theme(theme_uuid=theme_uuid), headers=auth_token_only_headers)
        assert response.json()["uuid"] == theme_uuid
        assert response.json()["title"] == changed_title
        assert response.json()["subtitle"] == changed_subtitle
        assert response.json()["position"] == changed_position

    async def test_delete_theme(self, async_client, auth_token_only_headers, default_headers):
        """Test deleting a theme.

        This test verifies that a theme can be deleted successfully and that it cannot
        be retrieved after deletion. It also checks that the deleted theme is not present
        in the list of all themes.

        Args:
            client: The HTTP client used to make requests.
            auth_token_only_headers: The headers to include in the request, containing authentication token.
            default_headers: The headers to include in the request.
        """
        # Create a theme
        theme_response = await self.create_theme(async_client, auth_token_only_headers)
        theme_uuid = theme_response.json()["uuid"]

        # Delete the theme
        response = await async_client.delete(api.theme(theme_uuid), headers=auth_token_only_headers)
        assert response.status_code == 200

        # Check that the theme cannot be retrieved
        response = await async_client.get(api.theme(theme_uuid=theme_uuid), headers=auth_token_only_headers)
        assert response.status_code == 404

        # Fetch all themes and ensure the deleted theme is not present
        themes_response = await self.fetch_themes(async_client, default_headers)
        themes = themes_response.json()["themes"]
        uuids_themes = [theme["uuid"] for theme in themes]
        assert theme_uuid not in uuids_themes, "Deleted theme is still present in the list of themes."

    async def test_create_use_case(self, async_client, auth_token_only_headers, default_headers):
        """Test the creation of new use cases for a theme with position checks.

        This test verifies that use cases can be created successfully for a theme,
        that their UUIDs are present in the database after creation, and that they
        are returned in the correct order based on their position.

        Args:
            client: The HTTP client used to make requests.
            auth_token_only_headers: The headers to include in the request, containing authentication token.
            default_headers: The headers to include in the request.
        """
        # Create a theme
        theme_response = await self.create_theme(async_client, auth_token_only_headers)
        theme_uuid = theme_response.json()["uuid"]

        # Create use cases with different positions and unique attributes
        first_use_case_response = await self.create_use_case(
            theme_uuid=theme_uuid,
            use_case_title="First Use Case Title",
            use_case_instruction="First Use Case Instruction",
            use_case_user_input_form="First Use Case Form",
            use_case_position=1,
            auth_token_only_headers=auth_token_only_headers,
            async_client=async_client,
        )
        first_use_case_uuid = first_use_case_response.json()["uuid"]

        second_use_case_response = await self.create_use_case(
            theme_uuid=theme_uuid,
            use_case_title="Second Use Case Title",
            use_case_instruction="Second Use Case Instruction",
            use_case_user_input_form="Second Use Case Form",
            use_case_position=2,
            auth_token_only_headers=auth_token_only_headers,
            async_client=async_client,
        )
        second_use_case_uuid = second_use_case_response.json()["uuid"]

        third_use_case_response = await self.create_use_case(
            theme_uuid=theme_uuid,
            use_case_title="Third Use Case Title",
            use_case_instruction="Third Use Case Instruction",
            use_case_user_input_form="Third Use Case Form",
            use_case_position=3,
            auth_token_only_headers=auth_token_only_headers,
            async_client=async_client,
        )
        third_use_case_uuid = third_use_case_response.json()["uuid"]

        # Add a use case with position=None
        first_none_position_use_case_response = await self.create_use_case(
            theme_uuid=theme_uuid,
            use_case_title="First None Position Use Case",
            use_case_instruction="First None Position Use Case Instruction",
            use_case_user_input_form="First None Position Use Case Form",
            use_case_position=None,
            auth_token_only_headers=auth_token_only_headers,
            async_client=async_client,
        )
        first_none_position_use_case_uuid = first_none_position_use_case_response.json()["uuid"]

        # Add another use case with position=None
        second_none_position_use_case_response = await self.create_use_case(
            theme_uuid=theme_uuid,
            use_case_title="Second None Position Use Case",
            use_case_instruction="Second None Position Use Case Instruction",
            use_case_user_input_form="Second None Position Use Case Form",
            use_case_position=None,
            auth_token_only_headers=auth_token_only_headers,
            async_client=async_client,
        )
        second_none_position_use_case_uuid = second_none_position_use_case_response.json()["uuid"]

        # Add a use case with the same position as position 1
        same_position_use_case_response = await self.create_use_case(
            theme_uuid=theme_uuid,
            use_case_title="Same Position Use Case",
            use_case_instruction="Same Position Use Case Instruction",
            use_case_user_input_form="Same Position Use Case Form",
            use_case_position=1,
            auth_token_only_headers=auth_token_only_headers,
            async_client=async_client,
        )
        same_position_use_case_uuid = same_position_use_case_response.json()["uuid"]

        # Fetch all use cases and verify the order
        use_cases = await self.fetch_use_cases(async_client, default_headers, theme_uuid)
        uuids_use_cases = [use_case["uuid"] for use_case in use_cases]

        # Check the database now contains these use cases' uuids
        assert first_use_case_uuid in uuids_use_cases, (
            f"First use case UUID {first_use_case_uuid} not found in the list of use case UUIDs."
        )
        assert second_use_case_uuid in uuids_use_cases, (
            f"Second use case UUID {second_use_case_uuid} not found in the list of use case UUIDs."
        )
        assert third_use_case_uuid in uuids_use_cases, (
            f"Third use case UUID {third_use_case_uuid} not found in the list of use case UUIDs."
        )
        assert first_none_position_use_case_uuid in uuids_use_cases, (
            f"First none position use case UUID {first_none_position_use_case_uuid} "
            + "not found in the list of use case UUIDs."
        )
        assert second_none_position_use_case_uuid in uuids_use_cases, (
            f"Second none position use case UUID {second_none_position_use_case_uuid}"
            + " not found in the list of use case UUIDs."
        )
        assert same_position_use_case_uuid in uuids_use_cases, (
            f"Same position use case UUID {same_position_use_case_uuid} not found in the list of use case UUIDs."
        )

        # Verify the order of use cases
        assert uuids_use_cases.index(first_use_case_uuid) < uuids_use_cases.index(same_position_use_case_uuid), (
            "First use case is not before same position use case in the list."
        )
        assert uuids_use_cases.index(same_position_use_case_uuid) < uuids_use_cases.index(second_use_case_uuid), (
            "Same position use case is not before second use case in the list."
        )
        assert uuids_use_cases.index(second_use_case_uuid) < uuids_use_cases.index(third_use_case_uuid), (
            "Second use case is not before third use case in the list."
        )
        assert uuids_use_cases.index(third_use_case_uuid) < uuids_use_cases.index(first_none_position_use_case_uuid), (
            "Third use case is not before first none position use case in the list."
        )
        assert uuids_use_cases.index(first_none_position_use_case_uuid) < uuids_use_cases.index(
            second_none_position_use_case_uuid
        ), "First none position use case is not before second none position use case in the list."

    async def test_get_use_case(self, async_client, default_headers, auth_token_only_headers):
        """Test fetching a specific use case by its UUID.

        This test verifies that after creating a use case, it can be fetched successfully
        and that the response contains the expected fields.

        Args:
            client: The HTTP client used to make requests.
            default_headers: The headers to include in the request.
            auth_token_only_headers: The headers to include in the request, containing authentication token.
        """
        logger.debug("Testing happy path for getting specific theme use cases.")

        # Create theme and use case
        theme_response, use_case_response = await self.create_theme_and_use_case(async_client, auth_token_only_headers)
        theme_uuid = theme_response.json()["uuid"]
        use_case_uuid = use_case_response.json()["uuid"]

        # Get the information for the selected use case
        response = await async_client.get(api.themes_use_case(theme_uuid, use_case_uuid), headers=default_headers)
        body = response.json()
        logger.debug(
            f"API call to get a specific theme use case returned status {response.status_code} and body {body}"
        )

        # Check that the response is as expected
        assert response.status_code == 200, f"The status code {response.status_code} was incorrect; it should be 200."
        assert body, "The response was empty."
        assert body["title"], "No title was found in the response."
        assert body["instruction"], "No instruction was found in the response."
        assert body["user_input_form"], "No user_input_form was found in the response."
        assert body["position"], "No position was found in the response."

    async def test_put_use_case(self, async_client, auth_token_only_headers, default_headers):
        """Test updating a use case's details.

        This test verifies that a use case's title, instruction, user input form, and position
        can be updated successfully and that the changes are reflected
        when fetching the use case.

        Args:
            client: The HTTP client used to make requests.
            auth_token_only_headers: The headers to include in the request, containing authentication token.
            default_headers: The headers to include in the request.
        """
        # Create a theme and use case
        theme_response, use_case_response = await self.create_theme_and_use_case(async_client, auth_token_only_headers)
        theme_uuid = theme_response.json()["uuid"]
        use_case_uuid = use_case_response.json()["uuid"]

        # Create a second theme (so we can swap the theme associated with the use case)
        second_theme_response = await self.create_theme(
            async_client, auth_token_only_headers, title="Title: Second theme", subtitle="Subtitle: Second theme"
        )
        second_theme_uuid = second_theme_response.json()["uuid"]

        # Edit the use case
        changed_title = "Changed use case title"
        changed_instruction = "Changed instruction"
        changed_user_input_form = "Changed user input form"
        changed_position = 123

        response = await async_client.put(
            api.themes_use_case(theme_uuid, use_case_uuid),
            json={
                "theme_uuid": second_theme_uuid,
                "title": changed_title,
                "instruction": changed_instruction,
                "user_input_form": changed_user_input_form,
                "position": changed_position,
            },
            headers=auth_token_only_headers,
        )
        assert response.status_code == 200, f"The status code {response.status_code} was incorrect; it should be 200."

        # Check that the use case has been edited
        response = await async_client.get(
            api.themes_use_case(theme_uuid=second_theme_uuid, use_case_uuid=use_case_uuid), headers=default_headers
        )
        assert response.status_code == 200, f"The status code {response.status_code} was incorrect; it should be 200."
        assert response.json()["uuid"] == use_case_uuid, (
            f"'uuid' field was incorrect; expected {use_case_uuid}, received {response.json()['uuid']}"
        )
        assert response.json()["theme_uuid"] == second_theme_uuid, (
            f"'theme_uuid' field was incorrect; expected {second_theme_uuid}, received {response.json()['theme_uuid']}"
        )
        assert response.json()["title"] == changed_title, (
            f"'title' field was incorrect; expected {changed_title}, received {response.json()['title']}"
        )
        assert response.json()["instruction"] == changed_instruction, (
            "'instruction' field was incorrect;"
            + f" expected {changed_instruction},"
            + f" received {response.json()['instruction']}"
        )
        assert response.json()["user_input_form"] == changed_user_input_form, (
            "'user_input_form' field was incorrect;"
            + f" expected {changed_user_input_form},"
            + f" received {response.json()['user_input_form']}"
        )
        assert response.json()["position"] == changed_position, (
            f"'position' field was incorrect; expected {changed_position}, received {response.json()['position']}"
        )

        # Check that the use case cannot be retrieved using the old theme
        response = await async_client.get(
            api.themes_use_case(theme_uuid=theme_uuid, use_case_uuid=use_case_uuid), headers=default_headers
        )
        assert response.status_code == 404, f"The status code {response.status_code} was incorrect; it should be 404."

    async def test_delete_use_case(self, async_client, auth_token_only_headers, default_headers):
        """Test deleting a use case.

        This test verifies that a use case can be deleted successfully and that it cannot
        be retrieved after deletion.

        Args:
            client: The HTTP client used to make requests.
            auth_token_only_headers: The headers to include in the request, containing authentication token.
            default_headers: The headers to include in the request.
        """
        # Create a theme and use case
        theme_response, use_case_response = await self.create_theme_and_use_case(async_client, auth_token_only_headers)
        theme_uuid = theme_response.json()["uuid"]
        use_case_uuid = use_case_response.json()["uuid"]

        # Delete the use case
        response = await async_client.delete(
            api.themes_use_case(theme_uuid, use_case_uuid), headers=auth_token_only_headers
        )
        assert response.status_code == 200

        # Check that the use case cannot be retrieved
        response = await async_client.get(api.themes_use_case(theme_uuid, use_case_uuid), headers=default_headers)

        assert response.status_code == 404

    async def test_fetch_all_use_cases_for_all_themes(self, async_client, default_headers):
        """Test fetching all use cases for all themes and ensure all respond with 200.

        This test verifies that for each theme, all associated use cases can be fetched
        successfully and that each use case responds with a 200 status code.

        Args:
            client: The HTTP client used to make requests.
            default_headers: The headers to include in the request.
        """
        logger.debug("Testing fetching all use cases for all themes.")

        # Fetch all themes
        themes_response = await self.fetch_themes(async_client, default_headers)
        themes = themes_response.json()["themes"]

        # Loop through each theme and fetch all use cases
        for theme in themes:
            theme_uuid = theme["uuid"]
            use_cases = await self.fetch_use_cases(async_client, default_headers, theme_uuid)

            # For each use case, send a get request to get all the use case information
            for use_case in use_cases:
                use_case_uuid = use_case["uuid"]
                response = await async_client.get(
                    api.themes_use_case(theme_uuid, use_case_uuid), headers=default_headers
                )
                assert response.status_code == 200, f"Failed to fetch use case {use_case_uuid} for theme {theme_uuid}."

    async def test_bulk_upload_happy_path(
        self, async_client, auth_token_only_headers, default_headers, prompts_for_upload
    ):
        """Test bulk uploading prompts (happy path).

        This test verifies that when prompts are bulk uploaded, the database only contains
        the uploaded themes and use cases, and all old themes and use cases are marked as deleted.

        Args:
            client: The HTTP client used to make requests.
            auth_token_only_headers: The headers to include in the request, containing authentication token.
            default_headers: The headers to include in the request.
            prompts_for_upload: The prompts data to be uploaded.
        """
        logger.debug("Testing happy path for bulk upload of pre-built prompts.")

        response = await async_client.post(api.prompts_bulk(), json=prompts_for_upload, headers=auth_token_only_headers)

        logger.debug(f"API call to bulk upload prebuilt prompts {response.status_code} and body {response.text[:100]}")

        assert response.status_code == 200, (
            f"The status code {response.status_code} was incorrect; it should be 200. Message: {response.text}"
        )

        # Get all themes in the database
        themes_response = await self.fetch_themes(async_client, default_headers)
        themes_in_db = themes_response.json()["themes"]

        # Check the database is in the appropriate state
        uploaded_prompts_theme_titles = [prompt["theme_title"] for prompt in prompts_for_upload]
        uploaded_prompts_theme_subtitles = [prompt["theme_subtitle"] for prompt in prompts_for_upload]
        [prompt["theme_position"] for prompt in prompts_for_upload]
        uploaded_prompts_use_case_titles = [prompt["use_case_title"] for prompt in prompts_for_upload]
        uploaded_prompts_use_case_instructions = [prompt["use_case_instruction"] for prompt in prompts_for_upload]
        uploaded_prompts_use_case_user_input_forms = [
            prompt["use_case_user_input_form"] for prompt in prompts_for_upload
        ]
        [prompt["use_case_position"] for prompt in prompts_for_upload]

        # Check the number of themes uploaded is correct
        # 1. The number of themes and use cases must be exactly the same as in the uploaded prompts
        # 2. Only the uploaded prompts should be returned from the database
        # (all old prompts should be marked as deleted)
        logger.info(f"{themes_in_db=}")
        assert len(themes_in_db) == len(set(uploaded_prompts_theme_titles))

        # Set to 1.
        # Don't set to 0, as this will evaluate to None if you assert it
        # which will mess with the test that checks that explicit positions
        # always come before null values.
        last_position_themes = 1
        for theme in themes_in_db:
            assert theme["title"] in uploaded_prompts_theme_titles
            assert theme["subtitle"] in uploaded_prompts_theme_subtitles

            # Check that the position of the theme in the list follows the position rules
            # 1. If an explicit position is provided, is it greater than or equal to the last position?
            # 2. If no explicit position is provided, does it come after all the other explicitly-positioned themes?
            if theme["position"]:
                # Check that explicit positions always come before any null values for position.
                assert last_position_themes, (
                    "Themes were not returned in the correct order; "
                    + "explicitly-positioned themes should always come before themes with a null value for 'position'."
                    + f" Current theme position: {theme['position']};"
                    + f" Last theme position: {last_position_themes}"
                )
                # Check that position always increases (or stays the same)
                assert theme["position"] >= last_position_themes, (
                    "Themes were not returned in the correct order as defined by the 'position' field."
                    + f" Current theme position: {theme['position']}; Last theme position: {last_position_themes}"
                )
                # Set the last_position_themes for the next iteration
                last_position_themes = theme["position"]
            # If no position was provided, set the last position to None.
            # This is important to test that explictly-positioned records
            # always come before records with null values for position.
            else:
                last_position_themes = None

            use_cases_in_db = await self.fetch_use_cases(async_client, default_headers, theme["uuid"])
            logger.info(f"{use_cases_in_db=}")
            assert len(prompts_for_upload) == len(uploaded_prompts_theme_titles)

            last_position_use_cases = 1
            for use_case in use_cases_in_db:
                assert use_case["title"] in uploaded_prompts_use_case_titles

                response = await async_client.get(
                    api.themes_use_case(theme["uuid"], use_case["uuid"]), headers=default_headers
                )
                assert response.status_code == 200, "Could not get use case"
                logger.info(f"{response=}")

                use_case_detailed = response.json()

                logger.info(f"{use_case_detailed=}")

                assert use_case_detailed["instruction"] in uploaded_prompts_use_case_instructions
                assert use_case_detailed["user_input_form"] in uploaded_prompts_use_case_user_input_forms

                # Check that the position of the use case in the list follows the position rules
                # 1. If an explicit position is provided, is it greater than or equal to the last position?
                # 2. If no explicit position is provided, does it come after all other explicitly-positioned use cases?
                if use_case["position"]:
                    # Check that explicit positions always come before any null values for position.
                    assert last_position_use_cases, (
                        "Use cases were not returned in the correct order;"
                        + "explicitly-positioned themes should always come before themes with a null 'position'."
                        + f" Current theme position: {use_case['position']};"
                        + f"Last theme position: {last_position_use_cases}"
                    )
                    # Check that position always increases (or stays the same)
                    assert use_case["position"] >= last_position_use_cases, (
                        "Use cases were not returned in the correct order as defined by the 'position' field."
                        + f"Current theme position: {use_case['position']};"
                        + f"Last theme position: {last_position_use_cases}"
                    )
                    # Set the last_position_use_cases for the next iteration
                    last_position_use_cases = use_case["position"]
                # If no position was provided, set the last position to None.
                # This is important to test that explictly-positioned records always come
                # before records with null values for position.
                else:
                    last_position_use_cases = None

    async def test_bulk_upload_happy_path_same_prompts_twice(
        self, async_client, auth_token_only_headers, default_headers, prompts_for_upload
    ):
        """Test bulk uploading the same prompts twice.

        This test checks that when the exact same prompts are uploaded twice in a row,
        no new records are created, and the UUIDs remain the same.

        Args:
            client: The HTTP client used to make requests.
            auth_token_only_headers: The headers to include in the request, containing authentication token.
            default_headers: The headers to include in the request.
            prompts_for_upload: The prompts data to be uploaded.
        """
        # Bulk-upload prompts for the first time
        await async_client.post(api.prompts_bulk(), json=prompts_for_upload, headers=auth_token_only_headers)

        # Get the themes and use cases uploaded
        themes_response = await self.fetch_themes(async_client, default_headers)
        themes_in_db_first_upload = themes_response.json()["themes"]

        for theme in themes_in_db_first_upload:
            use_cases_in_db_first_upload = await self.fetch_use_cases(async_client, default_headers, theme["uuid"])

        # Bulk-upload of the same prompts for the second time
        await async_client.post(api.prompts_bulk(), json=prompts_for_upload, headers=auth_token_only_headers)

        # Get the themes and use cases uploaded
        themes_response = await self.fetch_themes(async_client, default_headers)
        themes_in_db_second_upload = themes_response.json()["themes"]

        for theme in themes_in_db_second_upload:
            use_cases_in_db_second_upload = await self.fetch_use_cases(async_client, default_headers, theme["uuid"])

        # Check that UUIDs are the same
        uuids_themes_first = [theme["uuid"] for theme in themes_in_db_first_upload]
        uuids_themes_second = [theme["uuid"] for theme in themes_in_db_second_upload]

        assert uuids_themes_first == uuids_themes_second, "Theme UUIDs do not match after second upload."

        uuids_use_cases_first = [use_case["uuid"] for use_case in use_cases_in_db_first_upload]
        uuids_use_cases_second = [use_case["uuid"] for use_case in use_cases_in_db_second_upload]

        assert uuids_use_cases_first == uuids_use_cases_second, "Use Case UUIDs do not match after second upload."

    async def test_get_prompts_bulk(self, async_client, auth_token_only_headers):
        """Test fetching all prompts in bulk.

        This test verifies that after creating a theme and a use case, all prompts can be
        fetched in bulk and that the response contains the expected fields.

        Args:
            client: The HTTP client used to make requests.
            auth_token_only_headers: The headers to include in the request, containing authentication token.
        """
        # Create the theme and use case
        theme_response, use_case_response = await self.create_theme_and_use_case(async_client, auth_token_only_headers)
        theme_uuid = theme_response.json()["uuid"]
        use_case_response.json()["uuid"]

        # Get a bulk output
        bulk_get_response = await async_client.get(api.prompts_bulk(), headers=auth_token_only_headers)
        prompts = bulk_get_response.json()["prompts"]

        # Check that the response is as expected
        assert prompts, "No prompts in the bulk_get_prompts response."
        assert isinstance(prompts, list), "Prompts were not returned in a list."
        last_theme_position = -1
        last_use_case_position = {}

        for prompt in prompts:
            assert prompt["theme_title"] is not None, (
                f"No theme title present in one of the returned prompts: {prompt=}"
            )
            assert prompt["theme_subtitle"] is not None, (
                f"No theme subtitle present in one of the returned prompts: {prompt=}"
            )
            assert prompt["use_case_title"] is not None, (
                f"No use case title present in one of the returned prompts: {prompt=}"
            )
            assert prompt["use_case_instruction"] is not None, (
                f"No use case instruction present in one of the returned prompts: {prompt=}"
            )
            assert prompt["use_case_user_input_form"] is not None, (
                f"No user input form present in one of the returned prompts: {prompt=}"
            )

            # Check theme_position
            theme_position = prompt.get("theme_position")
            assert theme_position is not None, f"No theme position present in one of the returned prompts: {prompt=}"
            assert theme_position >= last_theme_position, (
                f"Theme position {theme_position} is not greater than or equal to the last {last_theme_position}."
            )
            last_theme_position = theme_position

            # Check use_case_position within the same theme
            theme_identifier = f"{prompt['theme_title']}_{theme_position}"
            use_case_position = prompt.get("use_case_position")
            assert use_case_position is not None, (
                f"No use case position present in one of the returned prompts: {prompt=}"
            )
            if theme_identifier not in last_use_case_position:
                last_use_case_position[theme_identifier] = -1
            assert use_case_position >= last_use_case_position[theme_identifier], (
                f"Use case position {use_case_position} is not greater than or equal to the last"
                + f"{last_use_case_position[theme_identifier]} within the same theme: {prompt=}"
            )
            last_use_case_position[theme_uuid] = use_case_position

    async def test_use_case_endpoint_without_theme(self, async_client, default_headers):
        """Test fetching a use case by its UUID without specifying a theme.

        This test verifies that a use case can be fetched successfully using its UUID
        without needing to specify the associated theme, and that the response contains
        the expected fields.

        Args:
            client: The HTTP client used to make requests.
            default_headers: The headers to include in the request.
        """
        # Create the theme and use case
        create_theme_response, create_use_case_response = await self.create_theme_and_use_case(
            async_client, auth_token_only_headers=default_headers
        )

        # Get all themes
        themes_response = await self.fetch_themes(async_client, default_headers)
        themes = themes_response.json()["themes"]
        theme_uuid = themes[0]["uuid"]

        # Get all use cases
        use_cases = await self.fetch_use_cases(async_client, default_headers, theme_uuid)
        use_case_uuid = use_cases[0]["uuid"]

        # Use the endpoint /prompts/use-cases/{use_case_uuid} to retrieve
        # information about the use case
        response = await async_client.get(api.get_use_case(use_case_uuid), headers=default_headers)

        logger.debug(
            "API call to get a specific theme use case returned status "
            + f"{response.status_code} and body {response.text[:100]}"
        )

        # Verify the request was successful.
        assert response.status_code == 200, f"The status code {response.status_code} was incorrect; it should be 200."

        # Verify the data returned is as expected.
        body = response.json()
        title = body["title"]
        instruction = body["instruction"]
        user_input_form = body["user_input_form"]

        assert body, "The response was empty."
        assert title, "No 'title' was present in the use case response."
        assert instruction, "No 'instruction' was present in the use case response."
        assert user_input_form, "No 'user_input_form' was present in the use case response."

    async def test_soft_deleted_use_cases_can_still_be_used(
        self, async_client, auth_token_only_headers, default_headers, user_id
    ):
        """Test that soft-deleted use cases can still be used.

        This test verifies that even if a use case is soft-deleted, it can still be used
        to create new chats, ensuring continuity for users who are already using the use case.

        Args:
            client: The HTTP client used to make requests.
            auth_token_only_headers: The headers to include in the request, containing authentication token.
            default_headers: The headers to include in the request.
        """
        # Create a theme and use case
        theme_response, use_case_response = await self.create_theme_and_use_case(async_client, auth_token_only_headers)
        theme_uuid = theme_response.json()["uuid"]
        use_case_uuid = use_case_response.json()["uuid"]

        # Soft-delete the use case
        response = await async_client.delete(
            api.themes_use_case(theme_uuid, use_case_uuid), headers=auth_token_only_headers
        )
        assert response.status_code == 200, "Failed to soft-delete the use case."

        # Attempt to use the soft-deleted use case
        # Simulate creating a chat with the soft-deleted use case
        chat_response = await async_client.post(
            api.chats(user_id), json={"query": "Test", "use_case_id": use_case_uuid}, headers=default_headers
        )
        assert chat_response.status_code == 200, (
            "Failed to create a chat with the soft-deleted use case:"
            + f"{chat_response.status_code=} {chat_response.json()=}"
        )
        logger.info(f"{chat_response.json()=}")

        chat_uuid = chat_response.json()["uuid"]

        chat_get_response = await async_client.get(
            api.get_chat_item(user_uuid=user_id, chat_uuid=chat_uuid), headers=default_headers
        )

        assert chat_get_response.status_code == 200, (
            "Failed to get chat with the soft-deleted use case:"
            + f"{chat_get_response.status_code=} {chat_get_response.json()=}"
        )

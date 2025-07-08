from logging import getLogger

from sqlalchemy.ext.asyncio import AsyncSession

from app.themes_use_cases.config import DEFAULT_THEMES_USE_CASES
from app.themes_use_cases.lib import upload_prompts_in_bulk
from app.themes_use_cases.schemas import PrebuiltPrompt

logger = getLogger(__name__)


async def sync_themes_use_cases(session: AsyncSession) -> None:
    try:
        parsed_prompts = await _parse_config_prompts_to_schema()
        await upload_prompts_in_bulk(session, parsed_prompts)
        logger.info("Succesfully syncronised themes and use cases")
    except Exception as e:
        logger.info(f"Failed to synchronise themes and use cases: {e}")
        raise e


async def _parse_config_prompts_to_schema() -> list[PrebuiltPrompt]:
    return [
        PrebuiltPrompt(
            theme_title=prompt["theme_title"],
            theme_subtitle=prompt["theme_subtitle"],
            theme_position=prompt["theme_position"],
            use_case_title=prompt["use_case_title"],
            use_case_instruction=prompt["use_case_instruction"],
            use_case_user_input_form=prompt["use_case_user_input_form"],
            use_case_position=prompt["use_case_position"],
        )
        for prompt in DEFAULT_THEMES_USE_CASES
    ]

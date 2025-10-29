from dataclasses import dataclass
from typing import Sequence

from fastapi import APIRouter

import app.auth.routes as auth
import app.central_guidance.routes as central_guidance
import app.chat.routes as chat
import app.feedback.routes as feedback
import app.healthcheck.routes as healthcheck
import app.personal_prompts.routes as personal_prompts
import app.routers.system as system
import app.themes_use_cases.themes_use_cases as themes_use_cases
import app.user.user as user


@dataclass
class RouterConfig:
    router: APIRouter
    prefix: str
    tags: Sequence[str]


routers = [
    RouterConfig(system.router, "", ["System"]),
    RouterConfig(healthcheck.router, "/healthcheck", ["Health Check"]),
    RouterConfig(auth.router, "/v1", ["Auth Sessions"]),
    RouterConfig(chat.router, "/v1", ["Chat Sessions"]),
    RouterConfig(feedback.router, "/v1", ["Message Feedback"]),
    RouterConfig(user.router, "/v1", ["User Data"]),
    RouterConfig(personal_prompts.router, "/v1", ["User Prompts"]),
    RouterConfig(themes_use_cases.router, "/v1", ["Themes / Use Cases"]),
    RouterConfig(central_guidance.router, "/v1", ["Central RAG"]),
]

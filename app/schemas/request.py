from pydantic import BaseModel


class RequestModel(BaseModel):
    def to_dict(self) -> dict:
        data = self.dict()
        return data

    class Config:
        # extra = Extra.forbid  # Deprecated
        # extra = Extra.allow  # Deprecated
        populate_by_name = True  # Equivalent to Extra.allow
        # allow_population_by_field_name = False  # Equivalent to Extra.forbid


class SessionRequest(BaseModel):
    auth_session_id: int


class UserRequest(BaseModel):
    user_id: int


class RequestStandard(RequestModel, SessionRequest, UserRequest):
    pass

from pydantic import BaseModel


class SessionRequest(BaseModel):
    id: int
    user_id: int

    # def __init__(self, id, user_id):
    #     self.user_id = user_id
    #     self.id = id

    def to_dict(self) -> dict:
        data = self.model_dump()

        return data

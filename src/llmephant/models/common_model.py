from pydantic import BaseModel
from typing import Optional
import datetime


class APIBase(BaseModel):
    """Common base_model with shared helpers/settings."""

    class Config:
        orm_mode = True
        validate_assignment = True
        json_encoders = {
            datetime.datetime: lambda dt: dt.isoformat() + "Z",
        }


class Timestamped(BaseModel):
    created_at: Optional[str] = None
    expires_at: Optional[str] = None

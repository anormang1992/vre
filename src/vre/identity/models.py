# Copyright 2026 Andrew Greene
# Licensed under the Apache License, Version 2.0

"""
Agent identity model for trace attribution.

An AgentIdentity binds a stable UUID to a registration key so that
traces from the same agent are attributable across restarts and sessions.
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentIdentity(BaseModel):
    """
    A registered agent identity with a stable, unique identifier.
    """

    agent_id: UUID = Field(default_factory=uuid4)
    registration_key: str
    name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

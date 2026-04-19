import asyncio
import logging
import os
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.schema import ChatRequest, ChatResponse, ChatStreamResponse, Error
from api.setting import DEFAULT_MODEL

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)


# Per-request deadline for non-streaming chat completions.  Configurable via
# REQUEST_DEADLINE_SECONDS env var (default 120s).  Applies ONLY to the
# non-streaming path — StreamingResponse returns immediately to FastAPI
# (it's a lazy generator), so wrapping it in a deadline would fire before
# the Bedrock call even starts; streaming connections stay alive naturally
# via SSE chunks arriving every few hundred ms.
#
# Rationale
# ---------
# Without this deadline, a stalled Bedrock call silently holds a
# thread-pool worker until some intermediate resource (ALB idle timer,
# VPC endpoint deadline, boto3 exhausting retries) severs the TCP
# connection — at which point the HTTP client sees
# "Empty reply from server" with zero bytes, no status line, and no
# actionable error.  A clean HTTP 504 with a descriptive message is
# strictly better: every OpenAI-compat client understands 504 as
# APITimeoutError and can retry or adjust parameters accordingly.
#
# 120s default balances two concerns:
#   - Too short: legitimate long-form responses (Opus 4.6 generating
#     64K tokens of a large document) get cut off.
#   - Too long: infrastructure-layer timers may fire first, making our
#     deadline irrelevant.
REQUEST_DEADLINE_SECONDS = int(os.getenv("REQUEST_DEADLINE_SECONDS", "120"))


@router.post(
    "/completions", response_model=ChatResponse | ChatStreamResponse | Error, response_model_exclude_unset=True
)
async def chat_completions(
    chat_request: Annotated[
        ChatRequest,
        Body(
            examples=[
                {
                    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                }
            ],
        ),
    ],
):
    if chat_request.model.lower().startswith("gpt-"):
        chat_request.model = DEFAULT_MODEL

    # Exception will be raised if model not supported.
    model = BedrockModel()
    model.validate(chat_request)
    if chat_request.stream:
        # Streaming path: SSE chunks keep the connection alive; no deadline.
        return StreamingResponse(content=model.chat_stream(chat_request), media_type="text/event-stream")

    # Non-streaming path: wrap in request-scoped deadline so stalled
    # requests surface as HTTP 504 instead of silent TCP resets.
    try:
        async with asyncio.timeout(REQUEST_DEADLINE_SECONDS):
            return await model.chat(chat_request)
    except asyncio.TimeoutError:
        logger.warning(
            "chat_completions deadline exceeded: model=%s deadline=%ds",
            chat_request.model, REQUEST_DEADLINE_SECONDS,
        )
        raise HTTPException(
            status_code=504,
            detail=(
                f"Bedrock request exceeded {REQUEST_DEADLINE_SECONDS}s deadline. "
                f"Try a smaller prompt, a lower max_tokens cap, or set "
                f'"stream": true to stream the response incrementally.'
            ),
        )

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path

from api.auth import api_key_auth
from api.models.bedrock import BedrockModel
from api.schema import Model, Models

router = APIRouter(
    prefix="/models",
    dependencies=[Depends(api_key_auth)],
    # responses={404: {"description": "Not found"}},
)

chat_model = BedrockModel()


async def validate_model_id(model_id: str):
    if model_id not in chat_model.list_models():
        raise HTTPException(status_code=500, detail="Unsupported Model Id")


def _build_model_response(model_id: str, metadata: dict) -> Model:
    """Build a Model response object from the proxy's internal metadata.

    Extracts capability fields (context_length, max_completion_tokens)
    if present; otherwise leaves them as None, which Pydantic serializes
    as JSON null.  Clients that don't care about capabilities (strict
    OpenAI-compat) can ignore the extra fields safely.
    """
    return Model(
        id=model_id,
        context_length=metadata.get("context_length"),
        max_completion_tokens=metadata.get("max_completion_tokens"),
    )


@router.get("", response_model=Models)
async def list_models():
    model_list = [
        _build_model_response(model_id, meta)
        for model_id, meta in chat_model.list_models_with_metadata().items()
    ]
    return Models(data=model_list)


@router.get(
    "/{model_id}",
    response_model=Model,
)
async def get_model(
    model_id: Annotated[
        str,
        Path(description="Model ID", example="anthropic.claude-3-sonnet-20240229-v1:0"),
    ],
):
    await validate_model_id(model_id)
    return _build_model_response(model_id, chat_model.get_model_metadata(model_id))

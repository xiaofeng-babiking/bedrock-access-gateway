"""
Per-model capability metadata for Bedrock models.

AWS Bedrock's `list_foundation_models` API does not return context-window
or max-output-token information, so the proxy maintains this small static
table and uses it to enrich the `/v1/models` response with capability
fields that downstream clients (Hermes, LangChain, Cline, LiteLLM, etc.)
can read to configure themselves automatically.

The convention follows OpenRouter's public model metadata shape:
    - context_length         : total context window (input + output)
    - max_completion_tokens  : max output tokens the model can produce

Matching is by substring: the proxy iterates this dict in declaration
order and picks the FIRST entry whose key appears as a substring in the
Bedrock model ID.  Because Bedrock ID conventions consistently include
the foundation-model fragment (e.g. `global.anthropic.claude-opus-4-6-v1`,
`us.anthropic.claude-opus-4-6-v1`, `apac.anthropic.claude-sonnet-4-...`),
substring matching cleanly handles cross-region inference profiles,
application profiles, and on-demand model IDs with one entry per family.

Ordering matters — put the more specific entries BEFORE the more general
ones.  For example `claude-sonnet-4-6` must come before `claude-sonnet-4`
so the latter doesn't shadow the former.

When a model has no matching entry the proxy returns null for both
capability fields; clients that rely on OpenRouter-style metadata should
treat missing fields as "unknown" rather than as an implicit default.
"""

# Ordered from most-specific to least-specific.  Substring match on the
# model ID string — any key that appears anywhere in the ID is a hit.
_MODEL_CAPABILITIES: tuple[tuple[str, dict], ...] = (
    # ── Claude 4.x family ─────────────────────────────────────────────
    ("claude-opus-4-6", {"context_length": 1_000_000, "max_completion_tokens": 128_000}),
    ("claude-sonnet-4-6", {"context_length": 1_000_000, "max_completion_tokens": 64_000}),
    ("claude-haiku-4-5", {"context_length": 200_000, "max_completion_tokens": 64_000}),
    ("claude-opus-4-5", {"context_length": 200_000, "max_completion_tokens": 32_000}),
    ("claude-sonnet-4-5", {"context_length": 200_000, "max_completion_tokens": 64_000}),
    ("claude-opus-4", {"context_length": 200_000, "max_completion_tokens": 32_000}),
    ("claude-sonnet-4", {"context_length": 200_000, "max_completion_tokens": 64_000}),

    # ── Claude 3.x family ─────────────────────────────────────────────
    ("claude-3-7-sonnet", {"context_length": 200_000, "max_completion_tokens": 64_000}),
    ("claude-3-5-sonnet", {"context_length": 200_000, "max_completion_tokens": 8_192}),
    ("claude-3-5-haiku", {"context_length": 200_000, "max_completion_tokens": 8_192}),
    ("claude-3-opus", {"context_length": 200_000, "max_completion_tokens": 4_096}),
    ("claude-3-sonnet", {"context_length": 200_000, "max_completion_tokens": 4_096}),
    ("claude-3-haiku", {"context_length": 200_000, "max_completion_tokens": 4_096}),

    # ── Amazon Nova family ────────────────────────────────────────────
    # Nova 2 ceilings are conservative estimates; update when AWS docs
    # confirm.  Stated values for Nova 1.x are from the public Bedrock
    # model catalog.
    ("nova-2-lite", {"context_length": 128_000, "max_completion_tokens": 5_000}),
    ("nova-2-multimodal-embeddings", {"context_length": 128_000, "max_completion_tokens": 5_000}),
    ("nova-premier", {"context_length": 1_000_000, "max_completion_tokens": 5_000}),
    ("nova-pro", {"context_length": 300_000, "max_completion_tokens": 5_000}),
    ("nova-lite", {"context_length": 300_000, "max_completion_tokens": 5_000}),
    ("nova-micro", {"context_length": 128_000, "max_completion_tokens": 5_000}),

    # ── DeepSeek / Llama / Mistral on Bedrock ─────────────────────────
    ("deepseek.v3", {"context_length": 131_072, "max_completion_tokens": 8_192}),
    ("deepseek-v3", {"context_length": 131_072, "max_completion_tokens": 8_192}),
    ("deepseek.r1", {"context_length": 128_000, "max_completion_tokens": 32_768}),
    ("llama3-3-70b", {"context_length": 128_000, "max_completion_tokens": 4_096}),
    ("llama3-2", {"context_length": 128_000, "max_completion_tokens": 4_096}),
    ("llama3-1", {"context_length": 128_000, "max_completion_tokens": 4_096}),
    ("mistral-large", {"context_length": 128_000, "max_completion_tokens": 8_192}),
    ("mistral-small", {"context_length": 32_000, "max_completion_tokens": 8_192}),
)


def lookup_capabilities(model_id: str) -> dict:
    """Return capability metadata for a Bedrock model ID.

    Looks up the model ID against the static capability table using
    substring matching.  Cross-region inference profiles (prefixed with
    ``us.``, ``global.``, ``apac.`` etc.) and application profiles all
    match the same foundation-model entry because the foundation-model
    fragment is always present in the ID.

    Returns a dict with ``context_length`` and ``max_completion_tokens``
    keys, both ``None`` if no entry matches.  Downstream code should
    treat ``None`` as "unknown" rather than as an implicit default.
    """
    if not model_id:
        return {"context_length": None, "max_completion_tokens": None}

    model_lower = model_id.lower()
    for key, caps in _MODEL_CAPABILITIES:
        if key in model_lower:
            return dict(caps)

    return {"context_length": None, "max_completion_tokens": None}

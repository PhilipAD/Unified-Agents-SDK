"""Shared utilities shared across multiple providers.

Import from here to avoid copy-paste across providers:
  CHAT_ROLE_MAP               — Role -> OpenAI/Groq/Mistral role string
  normalize_openai_usage      — prompt_tokens/completion_tokens -> input/output_tokens
  msg_to_openai_chat          — NormalizedMessage -> OpenAI chat-completions dict
  build_openai_chat_tools     — ToolDefinition list -> OpenAI tools list
  accumulate_tool_delta       — merge a streaming tool_calls delta into pending dict
  emit_pending_tool_calls     — finalise pending tool calls -> List[ToolCall]
  to_responses_input_items    — messages -> (instructions, input_items) for Responses API
  to_responses_tools          — build Responses API tools list
  parse_responses_output      — iterate resp.output -> (text, tool_calls, thinking, annots)
  normalize_responses_usage   — Responses API usage object -> normalised dict
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.types import NormalizedMessage, Role, ToolCall, ToolDefinition

# ---------------------------------------------------------------------------
# Role mapping (chat-completions style — OpenAI/Groq/Mistral/DeepSeek)
# ---------------------------------------------------------------------------

CHAT_ROLE_MAP: Dict[Role, str] = {
    Role.SYSTEM: "system",
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL: "tool",
}


# ---------------------------------------------------------------------------
# Usage normalisation
# ---------------------------------------------------------------------------


def normalize_openai_usage(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise an OpenAI-style usage dict to unified ``input_tokens`` /
    ``output_tokens`` keys (matching Anthropic, Gemini, etc.).

    Handles both the OpenAI naming convention (``prompt_tokens`` /
    ``completion_tokens``) and the Responses API naming (``input_tokens`` /
    ``output_tokens``).  Extra detail fields (reasoning, cache hit/miss,
    cached_tokens) are preserved when present.
    """
    out: Dict[str, Any] = {}
    if not raw:
        return out

    out["input_tokens"] = raw.get("input_tokens") or raw.get("prompt_tokens") or 0
    out["output_tokens"] = raw.get("output_tokens") or raw.get("completion_tokens") or 0
    total = raw.get("total_tokens")
    if total:
        out["total_tokens"] = total

    # Cached tokens (OpenAI prompt_tokens_details or Groq equivalent)
    ptd = raw.get("prompt_tokens_details") or {}
    cached = ptd.get("cached_tokens") or raw.get("prompt_cache_hit_tokens") or 0
    if cached:
        out["cached_tokens"] = cached
    cache_miss = raw.get("prompt_cache_miss_tokens")
    if cache_miss:
        out["cache_miss_tokens"] = cache_miss

    # Reasoning tokens
    ctd = raw.get("completion_tokens_details") or {}
    reasoning = ctd.get("reasoning_tokens") or 0
    if reasoning:
        out["reasoning_tokens"] = reasoning

    # finish_reason forwarded by callers separately; not in usage
    return out


def normalize_responses_usage(raw_usage: Any) -> Dict[str, Any]:
    """Normalise a Responses API usage object (attribute-based) to a dict."""
    out: Dict[str, Any] = {}
    if raw_usage is None:
        return out

    out["input_tokens"] = getattr(raw_usage, "input_tokens", 0) or 0
    out["output_tokens"] = getattr(raw_usage, "output_tokens", 0) or 0
    total = getattr(raw_usage, "total_tokens", 0)
    if total:
        out["total_tokens"] = total

    input_det = getattr(raw_usage, "input_tokens_details", None)
    if input_det:
        cached = getattr(input_det, "cached_tokens", 0) or 0
        if cached:
            out["cached_tokens"] = cached

    output_det = getattr(raw_usage, "output_tokens_details", None)
    if output_det:
        reasoning = getattr(output_det, "reasoning_tokens", 0) or 0
        if reasoning:
            out["reasoning_tokens"] = reasoning

    return out


# ---------------------------------------------------------------------------
# Chat-completions message conversion (OpenAI / Groq / Mistral / DeepSeek)
# ---------------------------------------------------------------------------


def msg_to_openai_chat(
    m: NormalizedMessage,
    *,
    include_reasoning: bool = False,
) -> Dict[str, Any]:
    """Convert a ``NormalizedMessage`` to OpenAI chat-completions message format.

    When ``include_reasoning`` is ``True`` (Groq reasoning models), a
    ``reasoning`` field is added to assistant messages that carry
    ``thinking_content``.  DeepSeek uses ``reasoning_content`` — callers
    should handle that separately by overriding or post-processing the dict.
    """
    msg: Dict[str, Any] = {
        "role": CHAT_ROLE_MAP[m.role],
        "content": m.content,
    }
    if m.role == Role.ASSISTANT and m.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in m.tool_calls
        ]
    if include_reasoning and m.role == Role.ASSISTANT and m.thinking_content:
        msg["reasoning"] = m.thinking_content
    if m.role == Role.TOOL and m.tool_call_id:
        msg["tool_call_id"] = m.tool_call_id
    if m.name:
        msg["name"] = m.name
    return msg


# ---------------------------------------------------------------------------
# Chat-completions tools list (OpenAI / Groq / Mistral)
# ---------------------------------------------------------------------------


def build_openai_chat_tools(
    tools: Optional[Sequence[ToolDefinition]],
) -> Optional[List[Dict[str, Any]]]:
    """Convert ``ToolDefinition`` list to the OpenAI function-tool format."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.json_schema,
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Streaming tool call accumulator (OpenAI / Groq / Mistral / DeepSeek)
# ---------------------------------------------------------------------------


def accumulate_tool_delta(
    pending: Dict[int, Dict[str, Any]],
    tc_delta: Dict[str, Any],
) -> None:
    """Merge a single tool-call delta chunk into the *pending* accumulator.

    *pending* is a ``{index: {"id", "name", "arguments"}}`` dict maintained
    across streaming chunks.  Mutates in place.
    """
    idx = tc_delta["index"]
    if idx not in pending:
        pending[idx] = {"id": tc_delta.get("id", ""), "name": "", "arguments": ""}
    entry = pending[idx]
    if tc_delta.get("id"):
        entry["id"] = tc_delta["id"]
    fn = tc_delta.get("function", {})
    if fn.get("name"):
        entry["name"] = fn["name"]
    if fn.get("arguments"):
        entry["arguments"] += fn["arguments"]


def emit_pending_tool_calls(
    pending: Dict[int, Dict[str, Any]],
) -> List[ToolCall]:
    """Finalise all accumulated tool calls, clear *pending*, return results."""
    result: List[ToolCall] = []
    for _idx in sorted(pending):
        e = pending[_idx]
        args = json.loads(e["arguments"]) if e["arguments"] else {}
        result.append(ToolCall(id=e["id"], name=e["name"], arguments=args))
    pending.clear()
    return result


# ---------------------------------------------------------------------------
# Responses API input conversion (OpenAI Responses / xAI)
# ---------------------------------------------------------------------------


def to_responses_input_items(
    messages: Sequence[NormalizedMessage],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Convert normalised messages to Responses API ``(instructions, input)``."""
    instructions: Optional[str] = None
    items: List[Dict[str, Any]] = []

    for m in messages:
        if m.role == Role.SYSTEM:
            instructions = m.content if isinstance(m.content, str) else str(m.content)
            continue

        if m.role == Role.USER:
            items.append({"role": "user", "content": m.content})

        elif m.role == Role.ASSISTANT:
            if m.content:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": m.content}],
                    }
                )
            for tc in m.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                )

        elif m.role == Role.TOOL:
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": m.tool_call_id or "",
                    "output": m.content,
                }
            )

    return instructions, items


# ---------------------------------------------------------------------------
# Responses API tools list (OpenAI Responses / xAI)
# ---------------------------------------------------------------------------


def to_responses_tools(
    tools: Optional[Sequence[ToolDefinition]],
    built_in_tools: Optional[List[Dict[str, Any]]] = None,
    mcp_servers: Optional[List[Dict[str, Any]]] = None,
    *,
    include_connector_id: bool = True,
    include_defer_loading: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """Build a Responses API tools list merging function tools, built-ins, and MCP.

    ``include_connector_id`` / ``include_defer_loading`` should be ``False``
    for xAI (which uses the Responses wire format but doesn't support those
    OpenAI-specific MCP fields).
    """
    result: List[Dict[str, Any]] = []

    if tools:
        for t in tools:
            result.append(
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.json_schema,
                    "strict": False,
                }
            )

    for bt in built_in_tools or []:
        result.append(bt)

    for mcp in mcp_servers or []:
        entry: Dict[str, Any] = {
            "type": "mcp",
            "server_label": mcp.get("server_label", "mcp"),
            "require_approval": mcp.get("require_approval", "never"),
        }
        for field in ("server_url", "server_description", "headers", "authorization",
                       "allowed_tools"):
            if field in mcp:
                entry[field] = mcp[field]
        if include_connector_id and "connector_id" in mcp:
            entry["connector_id"] = mcp["connector_id"]
        if include_defer_loading and "defer_loading" in mcp:
            entry["defer_loading"] = mcp["defer_loading"]
        result.append(entry)

    return result or None


# ---------------------------------------------------------------------------
# Responses API output parsing (OpenAI Responses / xAI)
# ---------------------------------------------------------------------------


def parse_responses_output(
    output_items: Any,
) -> Tuple[str, List[ToolCall], str, List[Dict[str, Any]]]:
    """Parse ``resp.output`` items into ``(content, tool_calls, thinking, annotations)``.

    Returns
    -------
    content       : str — concatenated text output
    tool_calls    : list of ToolCall
    thinking      : str — concatenated reasoning summary text
    annotations   : list of dicts (URL citations from output_text annotations)
    """
    content = ""
    tool_calls: List[ToolCall] = []
    thinking = ""
    annotations: List[Dict[str, Any]] = []

    for item in output_items:
        item_type = getattr(item, "type", None)

        if item_type == "message":
            for part in getattr(item, "content", []):
                if getattr(part, "type", None) == "output_text":
                    content += getattr(part, "text", "")
                    for ann in getattr(part, "annotations", None) or []:
                        annotations.append(
                            {
                                "type": getattr(ann, "type", ""),
                                "url": getattr(ann, "url", ""),
                                "title": getattr(ann, "title", ""),
                                "start_index": getattr(ann, "start_index", 0),
                                "end_index": getattr(ann, "end_index", 0),
                            }
                        )

        elif item_type == "function_call":
            args_raw = getattr(item, "arguments", "{}")
            arguments = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            tool_calls.append(
                ToolCall(
                    id=getattr(item, "call_id", ""),
                    name=getattr(item, "name", ""),
                    arguments=arguments,
                )
            )

        elif item_type == "reasoning":
            for part in getattr(item, "summary", []) or []:
                if getattr(part, "type", None) == "summary_text":
                    thinking += getattr(part, "text", "")

    return content, tool_calls, thinking, annotations

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from core.types import (
    GatewayError,
    NormalizedMessage,
    NormalizedResponse,
    Role,
    StreamEvent,
    ToolCall,
    ToolDefinition,
)
from providers._shared import (
    accumulate_tool_delta,
    build_openai_chat_tools,
    emit_pending_tool_calls,
    msg_to_openai_chat,
    normalize_openai_usage,
)
from providers.base import BaseProvider


class OpenAICompatibleProvider(BaseProvider):
    """Generic OpenAI chat-completions compatible provider.

    Any JSON field not listed below can be passed directly in ``kwargs``
    (e.g. ``temperature``, ``top_p``, ``tool_choice``, ``seed``,
    ``response_format``, ``parallel_tool_calls``, ``stream_options``,
    ``logprobs``, ``n``, ``stop``, ``presence_penalty``,
    ``frequency_penalty``, ``user``).
    """

    name = "openai_compatible"

    def _msg_to_api(self, m: NormalizedMessage) -> Dict[str, Any]:
        """Convert a NormalizedMessage to the OpenAI chat-completions message shape.

        Content lists (multimodal / vision) are forwarded as-is; OpenAI
        Chat Completions and all compatible endpoints accept
        ``[{"type": "text", ...}, {"type": "image_url", ...}]``.
        """
        return msg_to_openai_chat(m)

    def _build_payload(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]],
        **options: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [self._msg_to_api(m) for m in messages],
            **options,
        }
        tool_list = build_openai_chat_tools(tools)
        if tool_list:
            payload["tools"] = tool_list
        return payload

    @staticmethod
    def _parse_tool_calls(raw_calls: List[Dict[str, Any]]) -> List[ToolCall]:
        result: List[ToolCall] = []
        for tc in raw_calls:
            fn = tc["function"]
            args_raw = fn.get("arguments") or "{}"
            arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            result.append(ToolCall(id=tc["id"], name=fn["name"], arguments=arguments))
        return result

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = self._build_payload(messages, tools, **kwargs)

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise GatewayError(
                    f"OpenAI API error {exc.response.status_code}: {exc.response.text}",
                    provider=self.name,
                    status_code=exc.response.status_code,
                ) from exc

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        tool_calls = self._parse_tool_calls(msg["tool_calls"]) if "tool_calls" in msg else []

        out_msg = NormalizedMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )

        usage = normalize_openai_usage(data.get("usage") or {})
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            usage["finish_reason"] = finish_reason
        return NormalizedResponse(
            messages=[out_msg],
            usage=usage,
            provider=self.name,
            model=self.model,
            raw=data,
        )

    async def stream(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }
        payload = self._build_payload(messages, tools, stream=True, **kwargs)

        pending_tool_calls: Dict[int, Dict[str, Any]] = {}
        raw_usage: Dict[str, Any] = {}

        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk_str = line[len("data:"):].strip()
                        if chunk_str == "[DONE]":
                            break

                        chunk = json.loads(chunk_str)
                        if chunk.get("usage"):
                            raw_usage = chunk["usage"]

                        choice = chunk["choices"][0] if chunk.get("choices") else None
                        if not choice:
                            continue
                        delta = choice.get("delta", {})

                        if delta.get("content"):
                            yield StreamEvent(type="chunk", delta=delta["content"])

                        for tc_delta in delta.get("tool_calls") or []:
                            accumulate_tool_delta(pending_tool_calls, tc_delta)

                        if choice.get("finish_reason") == "tool_calls":
                            for tc in emit_pending_tool_calls(pending_tool_calls):
                                yield StreamEvent(type="tool_call", tool_call=tc)

            except httpx.HTTPStatusError as exc:
                raise GatewayError(
                    f"OpenAI streaming error {exc.response.status_code}",
                    provider=self.name,
                    status_code=exc.response.status_code,
                ) from exc

        if raw_usage:
            yield StreamEvent(type="usage", usage=normalize_openai_usage(raw_usage))
        yield StreamEvent(type="done")

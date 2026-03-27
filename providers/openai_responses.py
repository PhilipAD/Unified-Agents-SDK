from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

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
    normalize_responses_usage,
    parse_responses_output,
    to_responses_input_items,
    to_responses_tools,
)
from providers.base import BaseProvider

BUILT_IN_TOOL_TYPES = frozenset(
    {
        "web_search",
        "file_search",
        "code_interpreter",
        "computer_use",
        "image_generation",
        "mcp",
    }
)


class OpenAIResponsesProvider(BaseProvider):
    """Provider using the OpenAI Responses API (``/v1/responses``).

    Supports built-in tools (web_search, file_search, code_interpreter,
    computer_use, image_generation), remote MCP servers, reasoning effort,
    and stateful sessions via ``previous_response_id``.
    """

    name = "openai_responses"

    def _client(self) -> AsyncOpenAI:
        kwargs: Dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncOpenAI(**kwargs)

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """Run via the OpenAI Responses API.

        Extra ``kwargs`` are forwarded to ``responses.create`` and support the
        full Responses API surface, including: ``temperature``, ``tool_choice``,
        ``truncation``, ``background``, ``context_management``, ``conversation``,
        ``parallel_tool_calls``, ``max_output_tokens``, ``max_tool_calls``,
        ``text`` / ``output_types``, ``metadata``.
        """
        instructions, input_items = to_responses_input_items(messages)

        built_in_tools = kwargs.pop("built_in_tools", None)
        mcp_servers = kwargs.pop("mcp_servers", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        reasoning_summary = kwargs.pop("reasoning_summary", None)
        previous_response_id = kwargs.pop("previous_response_id", None)
        store = kwargs.pop("store", None)
        include = kwargs.pop("include", None)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        if instructions:
            api_kwargs["instructions"] = instructions

        tools_list = to_responses_tools(tools, built_in_tools, mcp_servers)
        if tools_list:
            api_kwargs["tools"] = tools_list

        if reasoning_effort or reasoning_summary:
            reasoning: Dict[str, Any] = {}
            if reasoning_effort:
                reasoning["effort"] = reasoning_effort
            if reasoning_summary:
                reasoning["summary"] = reasoning_summary
            api_kwargs["reasoning"] = reasoning

        if previous_response_id:
            api_kwargs["previous_response_id"] = previous_response_id
        if store is not None:
            api_kwargs["store"] = store
        if include:
            api_kwargs["include"] = include

        api_kwargs.update(kwargs)

        client = self._client()
        try:
            resp = await client.responses.create(**api_kwargs)
        except Exception as exc:
            raise GatewayError(
                f"OpenAI Responses API error: {exc}",
                provider=self.name,
            ) from exc

        content, tool_calls, thinking_content, _annotations = parse_responses_output(resp.output)
        out_msg = NormalizedMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            thinking_content=thinking_content or None,
        )
        usage = normalize_responses_usage(getattr(resp, "usage", None))
        return NormalizedResponse(
            messages=[out_msg],
            usage=usage,
            provider=self.name,
            model=self.model,
            raw=resp,
        )

    async def stream(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        instructions, input_items = to_responses_input_items(messages)

        built_in_tools = kwargs.pop("built_in_tools", None)
        mcp_servers = kwargs.pop("mcp_servers", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        reasoning_summary = kwargs.pop("reasoning_summary", None)
        previous_response_id = kwargs.pop("previous_response_id", None)
        store = kwargs.pop("store", None)
        include = kwargs.pop("include", None)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "stream": True,
        }

        if instructions:
            api_kwargs["instructions"] = instructions

        tools_list = to_responses_tools(tools, built_in_tools, mcp_servers)
        if tools_list:
            api_kwargs["tools"] = tools_list

        if reasoning_effort or reasoning_summary:
            reasoning_cfg: Dict[str, Any] = {}
            if reasoning_effort:
                reasoning_cfg["effort"] = reasoning_effort
            if reasoning_summary:
                reasoning_cfg["summary"] = reasoning_summary
            api_kwargs["reasoning"] = reasoning_cfg

        if previous_response_id:
            api_kwargs["previous_response_id"] = previous_response_id
        if store is not None:
            api_kwargs["store"] = store
        if include:
            api_kwargs["include"] = include

        api_kwargs.update(kwargs)

        client = self._client()

        try:
            async with client.responses.stream(**api_kwargs) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "response.output_text.delta":
                        yield StreamEvent(type="chunk", delta=event.delta)

                    elif event_type == "response.reasoning_summary_text.delta":
                        yield StreamEvent(type="chunk", delta=event.delta)

                    elif event_type == "response.function_call_arguments.done":
                        args = json.loads(event.arguments) if event.arguments else {}
                        tc = ToolCall(
                            id=getattr(event, "call_id", ""),
                            name=getattr(event, "name", ""),
                            arguments=args,
                        )
                        yield StreamEvent(type="tool_call", tool_call=tc)

                    elif event_type == "response.mcp_call.completed":
                        yield StreamEvent(
                            type="metadata",
                            metadata={
                                "event": "mcp_call_completed",
                                "server_label": getattr(event, "server_label", ""),
                                "name": getattr(event, "name", ""),
                            },
                        )

                    elif event_type == "response.web_search_call.completed":
                        yield StreamEvent(
                            type="metadata",
                            metadata={"event": "web_search_completed"},
                        )

                    elif event_type == "response.completed":
                        final = getattr(event, "response", None)
                        if final:
                            usage = normalize_responses_usage(getattr(final, "usage", None))
                            if usage:
                                yield StreamEvent(type="usage", usage=usage)

                    elif event_type in (
                        "response.failed",
                        "response.incomplete",
                    ):
                        err_data = getattr(event, "response", None)
                        error_msg = ""
                        if err_data:
                            err_obj = getattr(err_data, "error", None)
                            if err_obj:
                                error_msg = getattr(err_obj, "message", str(err_obj))
                        yield StreamEvent(
                            type="error",
                            error=error_msg or event_type,
                        )

        except Exception as exc:
            raise GatewayError(
                f"OpenAI Responses streaming error: {exc}",
                provider=self.name,
            ) from exc

        yield StreamEvent(type="done")

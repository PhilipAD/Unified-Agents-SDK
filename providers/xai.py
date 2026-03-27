from __future__ import annotations

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

XAI_BASE_URL = "https://api.x.ai/v1"

BUILT_IN_TOOL_TYPES = frozenset(
    {
        "web_search",
        "x_search",
        "code_interpreter",
        "code_execution",
        "file_search",
        "collections_search",
        "attachment_search",
        "mcp",
    }
)


class XAIProvider(BaseProvider):
    """Provider for xAI Grok models via OpenAI Responses API.

    Uses the OpenAI SDK pointed at xAI's base URL
    (https://api.x.ai/v1). Supports:
    - Grok reasoning and non-reasoning models
    - Built-in server-side tools: web_search, x_search,
      code_interpreter, collections_search, attachment_search
    - Remote MCP servers
    - Live search with search_parameters
    - Citations and inline citations
    - Cost tracking (cost_in_usd_ticks)
    - Deferred completions
    - Previous response ID for multi-turn
    """

    name = "xai"

    def _client(self) -> AsyncOpenAI:
        base_url = self.base_url or XAI_BASE_URL
        return AsyncOpenAI(api_key=self.api_key, base_url=base_url)

    async def run(
        self,
        messages: List[NormalizedMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any,
    ) -> NormalizedResponse:
        """Run via the xAI Responses API.

        Extra ``kwargs`` are forwarded to ``responses.create`` and support the
        full xAI Responses surface, including: ``temperature``, ``tool_choice``,
        ``parallel_tool_calls``, ``max_output_tokens``, ``metadata``.
        """
        instructions, input_items = to_responses_input_items(messages)

        built_in_tools = kwargs.pop("built_in_tools", None)
        mcp_servers = kwargs.pop("mcp_servers", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        previous_response_id = kwargs.pop("previous_response_id", None)
        store = kwargs.pop("store", None)
        search_parameters = kwargs.pop("search_parameters", None)
        include = kwargs.pop("include", None)
        deferred = kwargs.pop("deferred", None)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }

        if instructions:
            api_kwargs["instructions"] = instructions

        # xAI does not support connector_id / defer_loading
        tools_list = to_responses_tools(
            tools, built_in_tools, mcp_servers,
            include_connector_id=False,
            include_defer_loading=False,
        )
        if tools_list:
            api_kwargs["tools"] = tools_list

        if reasoning_effort:
            api_kwargs["reasoning"] = {"effort": reasoning_effort}
        if previous_response_id:
            api_kwargs["previous_response_id"] = previous_response_id
        if store is not None:
            api_kwargs["store"] = store
        if include:
            api_kwargs["include"] = include
        if search_parameters:
            api_kwargs["search_parameters"] = search_parameters
        if deferred is not None:
            api_kwargs["deferred"] = deferred

        api_kwargs.update(kwargs)

        client = self._client()
        try:
            resp = await client.responses.create(**api_kwargs)
        except Exception as exc:
            raise GatewayError(
                f"xAI API error: {exc}",
                provider=self.name,
            ) from exc

        content, tool_calls, thinking_content, citations = parse_responses_output(resp.output)
        out_msg = NormalizedMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
            thinking_content=thinking_content or None,
        )

        usage = normalize_responses_usage(getattr(resp, "usage", None))

        if citations:
            usage["citations"] = citations
        resp_citations = getattr(resp, "citations", None)
        if resp_citations:
            usage["source_urls"] = list(resp_citations)
        cost = getattr(resp, "cost_in_usd_ticks", None)
        if cost is not None:
            usage["cost_in_usd_ticks"] = cost

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
        import json

        instructions, input_items = to_responses_input_items(messages)

        built_in_tools = kwargs.pop("built_in_tools", None)
        mcp_servers = kwargs.pop("mcp_servers", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        previous_response_id = kwargs.pop("previous_response_id", None)
        store = kwargs.pop("store", None)
        search_parameters = kwargs.pop("search_parameters", None)
        include = kwargs.pop("include", None)
        kwargs.pop("deferred", None)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "stream": True,
        }

        if instructions:
            api_kwargs["instructions"] = instructions

        tools_list = to_responses_tools(
            tools, built_in_tools, mcp_servers,
            include_connector_id=False,
            include_defer_loading=False,
        )
        if tools_list:
            api_kwargs["tools"] = tools_list

        if reasoning_effort:
            api_kwargs["reasoning"] = {"effort": reasoning_effort}
        if previous_response_id:
            api_kwargs["previous_response_id"] = previous_response_id
        if store is not None:
            api_kwargs["store"] = store
        if include:
            api_kwargs["include"] = include
        if search_parameters:
            api_kwargs["search_parameters"] = search_parameters

        api_kwargs.update(kwargs)

        client = self._client()

        try:
            async with client.responses.stream(**api_kwargs) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "response.output_text.delta":
                        yield StreamEvent(type="chunk", delta=event.delta)

                    elif event_type in (
                        "response.reasoning_summary_text.delta",
                        "response.reasoning_text.delta",
                    ):
                        yield StreamEvent(type="chunk", delta=event.delta)

                    elif event_type == "response.function_call_arguments.done":
                        args = json.loads(event.arguments) if event.arguments else {}
                        tc = ToolCall(
                            id=getattr(event, "call_id", ""),
                            name=getattr(event, "name", ""),
                            arguments=args,
                        )
                        yield StreamEvent(type="tool_call", tool_call=tc)

                    elif event_type in (
                        "response.web_search_call.completed",
                        "response.x_search_call.completed",
                    ):
                        yield StreamEvent(
                            type="metadata",
                            metadata={"event": event_type},
                        )

                    elif event_type == "response.completed":
                        final = getattr(event, "response", None)
                        if final:
                            usage = normalize_responses_usage(
                                getattr(final, "usage", None)
                            )
                            cost = getattr(final, "cost_in_usd_ticks", None)
                            if cost is not None:
                                usage["cost_in_usd_ticks"] = cost
                            if usage:
                                yield StreamEvent(type="usage", usage=usage)

        except Exception as exc:
            raise GatewayError(
                f"xAI streaming error: {exc}",
                provider=self.name,
            ) from exc

        yield StreamEvent(type="done")

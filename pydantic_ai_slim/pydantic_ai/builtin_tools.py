"""Deprecated module. Use `pydantic_ai.server_side_tools` instead."""

from __future__ import annotations as _annotations

from typing_extensions import deprecated

from .server_side_tools import (
    AbstractServerSideTool,
    CodeExecutionTool,
    ImageGenerationTool,
    MCPServerTool,
    MemoryTool,
    UrlContextTool,
    WebFetchTool,
    WebSearchTool,
    WebSearchUserLocation,
    _SERVER_SIDE_TOOL_TYPES,
    _tool_discriminator,
)

__all__ = (
    'AbstractBuiltinTool',
    'AbstractServerSideTool',
    'WebSearchTool',
    'WebSearchUserLocation',
    'CodeExecutionTool',
    'WebFetchTool',
    'UrlContextTool',
    'ImageGenerationTool',
    'MemoryTool',
    'MCPServerTool',
)


@deprecated('Use `AbstractServerSideTool` instead.')
class AbstractBuiltinTool(AbstractServerSideTool):
    """Deprecated alias for `AbstractServerSideTool`."""

    pass


# Re-export the registry for backward compatibility
_BUILTIN_TOOL_TYPES = _SERVER_SIDE_TOOL_TYPES

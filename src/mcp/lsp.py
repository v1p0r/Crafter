import asyncio
import json
import logging
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("lsp_mcp")

MCP_LS_BIN = os.path.expanduser("~/go/bin/mcp-language-server")


class LSPMCP:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.server_params = StdioServerParameters(
            command=MCP_LS_BIN,
            args=[
                "--workspace", workspace_dir,
                "--lsp", "pyright-langserver",
                "--", "--stdio",
            ],
        )

    async def _call(self, tool_name: str, arguments: dict, timeout: float = 15.0) -> list:
        """Call an MCP tool and return parsed content. Times out to avoid stalling."""
        try:
            return await asyncio.wait_for(
                self._call_inner(tool_name, arguments), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"LSP {tool_name} timed out after {timeout}s")
            return []
        except Exception as e:
            logger.warning(f"LSP {tool_name} failed: {e}")
            return []

    async def _call_inner(self, tool_name: str, arguments: dict) -> list:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                if result.content:
                    return [c.text for c in result.content]
                return []

    async def get_diagnostics(self, file_path: str) -> str:
        """Get pyright diagnostics (type errors, import errors) for a file."""
        results = await self._call("diagnostics", {
            "filePath": file_path,
            "showLineNumbers": True,
            "contextLines": True,
        })
        return "\n".join(results) if results else ""

    async def get_hover(self, file_path: str, line: int, column: int) -> str:
        """Get type info / documentation for a symbol at a position (1-indexed)."""
        results = await self._call("hover", {
            "filePath": file_path,
            "line": line,
            "column": column,
        })
        return "\n".join(results) if results else ""

    async def get_definition(self, symbol_name: str) -> str:
        """Get the source code definition of a symbol by name."""
        results = await self._call("definition", {
            "symbolName": symbol_name,
        })
        return "\n".join(results) if results else ""

    async def get_references(self, symbol_name: str) -> str:
        """Find all usages/references of a symbol by name."""
        results = await self._call("references", {
            "symbolName": symbol_name,
        })
        return "\n".join(results) if results else ""

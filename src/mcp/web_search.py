import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class WebSearchMCP:
    def __init__(self):
        self.api_key = os.environ.get("BRAVE_API_KEY", "")
        self.server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@anthropic/brave-search-mcp"],
            env={"BRAVE_API_KEY": self.api_key},
        )

    async def search(self, query: str, count: int = 5) -> list[dict]:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "brave_web_search",
                    arguments={"query": query, "count": count},
                )

                if result.content:
                    text = result.content[0].text
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return [{"raw": text}]
                return []

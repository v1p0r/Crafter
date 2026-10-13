import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class FileSystemMCP:
    def __init__(self, allowed_dir: str):
        self.allowed_dir = allowed_dir
        self.server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@anthropic/filesystem-mcp", allowed_dir],
        )

    async def read_file(self, file_path: str) -> str:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "read_file",
                    arguments={"path": file_path},
                )

                if result.content:
                    return result.content[0].text
                return ""

    async def write_file(self, file_path: str, content: str) -> bool:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "write_file",
                    arguments={"path": file_path, "content": content},
                )

                return bool(result.content)

    async def list_directory(self, dir_path: str) -> list[str]:
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.call_tool(
                    "list_directory",
                    arguments={"path": dir_path},
                )

                if result.content:
                    try:
                        return json.loads(result.content[0].text)
                    except json.JSONDecodeError:
                        return result.content[0].text.split("\n")
                return []

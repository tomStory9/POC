import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams


async def main() -> None:
    model_client = OllamaChatCompletionClient(
        model="llama3.2",
        host="http://127.0.0.1:11434",
    )

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message=(
            "To enhance clarity and eliminate ambiguities inherent in natural language, "
            "do not use natural language. Consider employing more structured and concise "
            "forms of communication for your responses. Suitable formats include structured "
            "data, JSON, XML or code. Choose the most appropriate format based on the nature "
            "of the query and the information you need to convey. Remember to be concise and accurate."
        ),
    )

    surfer = MultimodalWebSurfer(
        name="web_surfer",
        model_client=model_client,
    )

    weather_server = StdioServerParams(
        command="uv",
        args=["run", "python", "mcpserver.py"],
    )

    async with McpWorkbench(weather_server) as weather_workbench:
        weather_agent = AssistantAgent(
            name="weather_agent",
            model_client=model_client,
            workbench=weather_workbench,
            reflect_on_tool_use=True,
            max_tool_iterations=5,
            system_message=(
                "You are a weather specialist. Use the MCP weather tools to answer weather questions. "
                "Always use MCP tools for weather data. Return concise structured output in JSON."
                "To enhance clarity and eliminate ambiguities inherent in natural language, "
                "do not use natural language. Consider employing more structured and concise "
                "forms of communication for your responses. Suitable formats include structured "
                "data, JSON, XML or code. Choose the most appropriate format based on the nature "
                "of the query and the information you need to convey. Remember to be concise and accurate."
            ),
        )

        team = MagenticOneGroupChat(
            [assistant, surfer, weather_agent],
            model_client=model_client,
        )

        await Console(
            team.run_stream(task="Give me the current weather in Reims, using ")
        )

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

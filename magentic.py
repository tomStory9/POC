import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer


async def main() -> None:
    model_client = OllamaChatCompletionClient(
        model="llama3.2",
        host="http://127.0.0.1:11434",
    )

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant.",
    )

    surfer = MultimodalWebSurfer(
        name="web_surfer",
        model_client=model_client,
    )

    team = MagenticOneGroupChat(
        [assistant, surfer],
        model_client=model_client,
    )

    await Console(
        team.run_stream(
            task="Find the latest official Python release and summarize what changed. it should be 3.14.4"
        )
    )

    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())

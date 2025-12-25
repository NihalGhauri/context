from pydantic import BaseModel
from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Agent,
    Runner,
    set_default_openai_client, set_tracing_disabled
    ,function_tool,
    RunContextWrapper
)
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import asyncio



load_dotenv()

Ai = os.getenv("GEMINI")

if not Ai:
    raise ValueError ("Ai not found")


external_client = AsyncOpenAI(
    api_key=Ai,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model= OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

set_default_openai_client(external_client)
set_tracing_disabled(True)



#! TESTING 
# agent = Agent(
#     name="assistant",
#     model = model
# )

# result = Runner.run_sync(agent, "who are you")

# print(result.final_output)



@dataclass
class UserInfo:
    name:str = "Nihal khan Ghauri"
    uid:int = '0001'
    location:str = "karachi"


@function_tool
async def fetch_user_age(wrapper:RunContextWrapper[UserInfo]) -> str:
    ''' return the name & age  of the user'''
    return f"User {wrapper.context.name} is 20 years"


@function_tool
async def fetch_user_location(wrapper:RunContextWrapper[UserInfo]) -> str:
    ''' return the location of the user'''
    return f"User {wrapper.context.name} is located in {wrapper.context.location}"

async def main():
    agent = Agent[UserInfo](
        name="assistant",
        tools=[fetch_user_age, fetch_user_location],
        model=model,
    )

    result = await Runner.run(
        agent,"what is the name & age of the user? current location?",context=UserInfo()
    )

    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())

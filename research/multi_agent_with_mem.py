from dotenv import load_dotenv
import os
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_store
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from langmem import create_multi_prompt_optimizer
from langmem import Prompt

_ = load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

store = InMemoryStore()

store.put(("instructions",), 
          key="email_agent", 
          value={"prompt": "Write good emails. Repeat your draft content to the user after submitting."})

store.put(("instructions",), 
          key="twitter_agent", 
          value={"prompt": "Write fire tweets. Repeat the tweet content to the user upon submission."})

## Email agent
def draft_email(to: str, subject: str, body: str):
    """Submit an email draft."""
    return "Draft saved succesfully."

def prompt_email(state):
    item = store.get(("instructions",), key="email_agent")
    instructions = item.value["prompt"]
    sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
    return [sys_prompt] + state['messages']

email_agent = create_react_agent(
    "openai:gpt-4o-mini", 
    prompt=prompt_email, 
    tools=[draft_email], 
    store=store,
    name="email_assistant",
)

## Tweet

def tweet(to: str, subject: str, body: str):
    """Poast a tweet."""
    return "Legendary."

def prompt_social_media(state):
    item = store.get(("instructions",), key="twitter_agent")
    instructions = item.value["prompt"]
    sys_prompt = {"role": "system", "content": f"## Instructions\n\n{instructions}"}
    return [sys_prompt] + state['messages']

social_media_agent = create_react_agent(
    "openai:gpt-4o-mini", 
    prompt=prompt_social_media, 
    tools=[tweet], 
    store=store,
    name="social_media_agent",
)

# Initialize the model object
model = ChatOpenAI(model_name="gpt-4o-mini")

# Create supervisor workflow
workflow = create_supervisor(
    [email_agent, social_media_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing email and tweet assistants to help with correspondance."
    )
)

# Compile and run
app = workflow.compile(store=store)

result = app.invoke(
    {"messages": [
        {"role": "user", "content": "Draft an email to joe@langchain.dev saying that we want to schedule a followup meeting for thursday at noon."}]},
)

# Print all messages in the result safely
print("\nConversation messages:")
for i, message in enumerate(result["messages"]):
    print(f"\nMessage {i}:")
    message.pretty_print()

# Check the number of messages before trying to access a specific index
print(f"\nTotal number of messages: {len(result['messages'])}")

feedback = {"request": "Always sign off emails from 'William'; for meeting requests, offer to schedule on Zoom or Google Meet"}

optimizer = create_multi_prompt_optimizer("openai:gpt-4o-mini")

email_prompt = store.get(("instructions",), key="email_agent").value['prompt']
tweet_prompt = store.get(("instructions",), key="twitter_agent").value['prompt']

email_prompt = {
    "name": "email_prompt",
    "prompt": email_prompt,
    "when_to_update": "Only if feedback is provided indicating email writing performance needs improved."
}
tweet_prompt = {
    "name": "tweet_prompt",
    "prompt": tweet_prompt,
    "when_to_update": "Only if tweet writing generation needs improvement."
}

optimizer_result = optimizer.invoke({"prompts": [tweet_prompt, email_prompt], "trajectories": [(result["messages"], feedback)]})
print(optimizer_result)

store.put(("instructions",), key="email_agent", value={"prompt": optimizer_result[1]['prompt']})

result = app.invoke(
    {"messages": [
        {"role": "user", "content": "Draft an email to joe@langchain.dev saying that we want to schedule a followup meeting for thursday at noon."}]},
)

# Print all messages in the result safely
print("\nOptimized conversation messages:")
for i, message in enumerate(result["messages"]):
    print(f"\nMessage {i}:")
    message.pretty_print()

# Check the number of messages before trying to access a specific index
print(f"\nTotal number of messages after optimization: {len(result['messages'])}")
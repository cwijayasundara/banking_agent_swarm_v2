from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.store.memory import InMemoryStore
import os

_ = load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o", api_key=api_key)

checkpointer = InMemorySaver()

store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small"
    }
)

namespace = ("agent_memories")

manage_memory_tool = create_manage_memory_tool(namespace)
search_memory_tool = create_search_memory_tool(namespace)

memory_tools = [
    manage_memory_tool,
    search_memory_tool
]

checkpointer = InMemorySaver()

def get_interest_rate() -> float:
    """Get the current interest rate."""
    print("Getting interest rate...")
    return 0.025

def get_customer_balance() -> float:
    """Get the customer's balance."""
    print("Getting customer balance...")
    return 1000.0

def get_pending_tx() -> float:
    """Get the pending transaction amount."""
    print("Getting pending transaction amount...")
    return 500.0

interest_rate_agent = create_react_agent(
    model=model,
    tools=[get_interest_rate, manage_memory_tool, search_memory_tool],
    name="interest_rate_expert",
    prompt="You are an interest rate expert. Provide the current interest rate.",
    store=store,
    checkpointer=checkpointer,
)

balance_agent = create_react_agent(
    model=model,
    tools=[get_customer_balance, manage_memory_tool, search_memory_tool],
    name="balance_expert",
    prompt="You are a balance expert. Provide the customer's balance.",
    store=store,
    checkpointer=checkpointer,
)

pending_tx_agent = create_react_agent(
    model=model,
    tools=[get_pending_tx, manage_memory_tool, search_memory_tool],
    name="pending_tx_expert",
    prompt="You are a pending transaction expert. Provide the pending transaction amount.",
    store=store,
    checkpointer=checkpointer,
)

workflow = create_supervisor(
    [interest_rate_agent, balance_agent, pending_tx_agent],
    model=model,
    prompt=(
        "You are a bank supervisor managing an interest rate expert, a balance expert, and a pending transaction expert. "
        "For interest rate, use interest_rate_agent. "
        "For customer balance, use balance_agent. "
        "For pending transaction amount, use pending_tx_agent."
    )
)

app = workflow.compile(store=store,checkpointer=checkpointer)
    
def chat(agent, txt, thread_id):
    result_state = agent.invoke({"messages": [{"role": "user", "content": txt}]}, 
                                config={"configurable": {"thread_id": thread_id}})
    return result_state["messages"][-1].content

query_1 = "What is the current interest rate, customer account balance and pending transaction amount?"
query_2 = "my name is boo"

result_1 = chat(app, query_1, "1")
result_2 = chat(app, query_2, "1")

query_3 = "What is my name?"
result_3 = chat(app, query_3, "1")

print(result_1)
print(result_2)
print(result_3)
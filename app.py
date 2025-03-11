from dotenv import load_dotenv
import os
from retriever import retrieve_documents_from_vector_store
from pending_tx_pandas_agent import get_pending_tx_details
from customer_details_sql_agent import get_customer_details
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from datetime import datetime
from langchain.chat_models import init_chat_model
import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

_ = load_dotenv()

today = datetime.now().strftime("%d/%m/%Y")

google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

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

llm = init_chat_model("gemini-2.0-flash-001", 
                      model_provider="google_genai", 
                      api_key=google_api_key)

llm_thinking = init_chat_model("o3-mini", 
                      model_provider="openai", 
                      api_key=openai_api_key,
                      temperature=1.0)

# tools
def get_interest_rates_from_vector_store(query:str) -> list[str]:
    """Retrieve interest rates from the vector store."""
    documents = retrieve_documents_from_vector_store(query)
    return documents

def get_pending_tx_details_from_pandas_agent(query:str) -> str:
    """Retrieve pending transactions details from the pandas agent."""
    response = get_pending_tx_details(query)
    return response

def get_customer_details_from_sql_agent(query:str) -> str:
    """Retrieve customer details from the sql agent."""
    response = get_customer_details(query)
    return response

# agents
interest_rate_agent = create_react_agent(
    model=llm,
    tools=[get_interest_rates_from_vector_store, manage_memory_tool, search_memory_tool],
    store=store,
    checkpointer=checkpointer,
    name="interest_rate_agent",
    prompt="You are an interest rate agent. Use the tools provided to retrieve the interest rates from the vector store."
)

pending_tx_agent = create_react_agent(
    model=llm,
    tools=[get_pending_tx_details_from_pandas_agent, manage_memory_tool, search_memory_tool],
    store=store,
    checkpointer=checkpointer,
    name="pending_tx_agent",
    prompt="You are a pending transactions agent. Use the tools provided to retrieve the pending transactions details from the pandas agent."
)

customer_details_agent = create_react_agent(
    model=llm,
    tools=[get_customer_details_from_sql_agent, manage_memory_tool, search_memory_tool],
    store=store,
    checkpointer=checkpointer,
    name="customer_details_agent",
    prompt="You are a customer details agent. Use the tools provided to retrieve the customer details from the sql agent."
)

# supervisor - Process each query individually instead of all at once
workflow = create_supervisor(
    [interest_rate_agent, pending_tx_agent, customer_details_agent],
    model=llm_thinking,
    prompt=(
        "You are a bank supervisor managing an interest rate agent, pending transactions agent and customer details agent."
        "For interest rate related queries, use interest_rate_agent. "
        "For pending transactions related queries, use pending_tx_agent. Pending transaction Pandas DF has the following columns: pending_tx_id,customer_id,pending_date,pending_amount."
        "For customer details related queries, use customer_details_agent. Customer table has the following columns: id,first_name,last_name,address,account_balance,income,gender,date_of_birth"
    )
)

app = workflow.compile(
    store=store,
    checkpointer=checkpointer
)

def chat(agent, txt, thread_id):
    # Compile the StateGraph before invoking
    result_state = app.invoke({"messages": [{"role": "user", "content": txt}]}, 
                                config={"configurable": {"thread_id": thread_id}})
    return result_state["messages"][-1].content

st.title("Banking Assistant")

with st.sidebar:
    st.image("image/image.jpg", width=600)

query = st.text_input("How can I help you today?")

st.write("sample questions:")
st.write("What is the current interest rate for a Cash ISA Saver's account opened after 18/02/25?")
st.write("What is the total amount of pending transactions for customer c004?")
st.write("List all the details of Olivia Stephenson?")
st.write("What is the account balance of Holly Owen?")
st.write("What is the total amount of pending transactions for Lewis Morley?")

if st.button("Chat"):
    if query:
        with st.spinner("Thinking..."):
            response = chat(workflow, query, "thread-1")
            st.write(response)
    else:
        st.warning("Please enter a query first.")
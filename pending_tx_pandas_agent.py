from dotenv import load_dotenv
import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import init_chat_model
import pandas as pd

_ = load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

llm = init_chat_model("gemini-2.0-flash-001", 
                      model_provider="google_genai", 
                      api_key=google_api_key)

df = pd.read_csv("docs/pending_tx.csv")

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True
)

def get_pending_tx_details(query:str) -> str:
    """Get the details of the pending transactions for a customer."""
    print(("Inside get_pending_tx_details Pandas Agent"))
    response = agent.invoke({"input": query})  
    return response["output"]


# query = "What is the total amount of pending transactions for customer c001?"
# response = get_pending_tx_details(query)
# print(response)



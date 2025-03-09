from dotenv import load_dotenv
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import os
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import create_sql_agent

_ = load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

llm = init_chat_model("gemini-2.0-flash-001", 
                      model_provider="google_genai", 
                      api_key=google_api_key)

df = pd.read_csv("docs/customer_details.csv")

engine = create_engine("sqlite:///docs/customer_details.db")

# create a table in the database
df.to_sql("customer_details", engine, index=False, if_exists='replace')

db = SQLDatabase(engine=engine)

agent_executor = create_sql_agent(llm, 
                                  db=db, 
                                  agent_type="openai-tools", 
                                  verbose=False)

def get_customer_details(query:str) -> str:
    """Get the details of a customer."""
    response = agent_executor.invoke({"input": query})
    return response["output"]

# query = "List all the details of Heather Nash"
# response = get_customer_details(query)
# print(response)









from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

_ = load_dotenv()

vector_store_path = "./interest-rates-vector-store"

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=vector_store_path,
)

def retrieve_documents_from_vector_store(query:str) -> list[str]:
    print(f"Retrieving documents from vector store for query: {query}")
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
    )   
    response = retriever.invoke(query)
    return [doc.page_content for doc in response]

# query = "What is the current interest rate for a Cash ISA Saver's account opened after 18/02/25?"
# response = retrieve_documents_from_vector_store(query)
# print(response)
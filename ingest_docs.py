from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

load_dotenv()

unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
file_paths = ["docs/interest-rates_1.pdf", "docs/interest-rates_2.pdf", "docs/interest-rates_3.pdf"]
vector_store_path = "./interest-rates-vector-store"

loader = UnstructuredLoader(
    file_path=file_paths,
    api_key=unstructured_api_key,
    partition_via_api=True,
    strategy="hi_res",
)

docs = loader.load()

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=vector_store_path,
)

# Extract text and metadata separately, then filter the metadata
texts = [doc.page_content for doc in docs]
metadatas = [doc.metadata for doc in docs]

# Filter out complex metadata values
filtered_metadatas = []
for metadata in metadatas:
    filtered_metadata = {}
    for key, value in metadata.items():
        # Only keep simple types (str, int, float, bool)
        if isinstance(value, (str, int, float, bool)):
            filtered_metadata[key] = value
        elif isinstance(value, list) and len(value) > 0:
            # Convert lists to strings
            filtered_metadata[key] = str(value[0]) if len(value) == 1 else str(value)
        else:
            # Convert other types to string or skip
            try:
                filtered_metadata[key] = str(value)
            except:
                pass
    filtered_metadatas.append(filtered_metadata)

# Add documents with filtered metadata
vector_store.add_texts(texts, filtered_metadatas)

# # create a retriever
# retriever = vector_store.as_retriever(
#     search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
# )

# response = retriever.invoke("Whats the Cash ISA Saver's annual interest rate for an account opened after 18/02/25?")

# # The response is a list of documents
# print("Retrieved documents:")
# for i, doc in enumerate(response):
#     print(f"\nDocument {i+1}:")
#     print(f"Content: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")







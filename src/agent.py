
import os
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import LocalAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from transformers import AutoTokenizer
from smolagents import LiteLLMModel, Tool
from smolagents.agents import CodeAgent, ToolCallingAgent
from langchain.prompts import PromptTemplate
from tools.retriever import RetrieverTool
from tools.financial_data import FinancialDataTool

## Extract PDF Files from ./data/ directory
pdf_directory = "./data"
pdf_files = [
    os.path.join(pdf_directory, f)
    for f in os.listdir(pdf_directory)
    if f.endswith(".pdf")
    ]
source_docs = []

for file_path in pdf_files:
    loader = PyPDFLoader(file_path)
    source_docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Split docs and keep only unique ones
print("Splitting documents...")
docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)


print("Embedding documents... This should take a few minutes")
# Initialize embeddings and ChromaDB vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(docs_processed, embeddings, persist_directory="./chroma_db")

# Initialize retriever and financial data tools
retriever_tool = RetrieverTool(vector_store)
financial_data_tool = FinancialDataTool()

# Choose which LLM engine to use!
model = LiteLLMModel(
    model_id="text-completion-openai/granite-3.1-8b-instruct",
    api_base="http://127.0.0.1:1234/v1",
    api_key="YOUR_API_KEY", # replace with API key if necessary
    num_ctx=8096, # ollama default is 2048 which will fail horribly. 8096 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

agent = CodeAgent(
    tools=[retriever_tool, financial_data_tool],
    model=model,
    max_steps=5,
    verbosity_level=2
)

agent_output = agent.run("What ongoing or potential lawsuits is NVIDIA facing?")

print("Final output:")
print(agent_output)



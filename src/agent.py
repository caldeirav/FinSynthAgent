
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
from smolagents.agents import CodeAgent
from langchain.prompts import PromptTemplate


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


class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Uses semantic search to retrieve the parts of documentation that could be most relevant to answer your query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vector_store, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        docs = self.vector_store.similarity_search(query, k=5)
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )
        return formatted_prompt


retriever_tool = RetrieverTool(vector_store)

# Choose which LLM engine to use!
model = LiteLLMModel(
    model_id="text-completion-openai/granite-3.1-8b-instruct",
    api_base="http://127.0.0.1:1234/v1",
    api_key="YOUR_API_KEY", # replace with API key if necessary
    num_ctx=8096 # ollama default is 2048 which will fail horribly. 8096 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

agent = CodeAgent(
    tools=[retriever_tool],
    model=model,
    max_steps=4,
    verbosity_level=2
)

agent_output = agent.run("What ongoing or potential lawsuits and regulatory issues is NVIDIA facing?")

print("Final output:")
print(agent_output)

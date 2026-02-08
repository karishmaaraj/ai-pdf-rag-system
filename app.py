from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

# 1. Load PDF
reader = PdfReader("data/sample.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# 2. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_text(text)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store embeddings in FAISS
vectorstore = FAISS.from_texts(chunks, embeddings)

# 5. Load LLM (Hugging Face)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_API_KEY_HERE"

llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={"temperature": 0}
)

# 6. Create Retrieval-QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# 7. Ask a question
query = "What is this document about?"
answer = qa_chain.run(query)

print("\nAnswer:")
print(answer)

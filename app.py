from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub

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

# 4. Store in vector database
vectorstore = FAISS.from_texts(chunks, embeddings)

# 5. Ask a question
query = "What is this document about?"
docs = vectorstore.similarity_search(query, k=3)

# 6. Print retrieved chunks
for i, doc in enumerate(docs, 1):
    print(f"\nChunk {i}:\n", doc.page_content)

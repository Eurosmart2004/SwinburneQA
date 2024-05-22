from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA


# Draft
from langchain_community.llms.ctransformers import CTransformers
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate

load_dotenv()
csv_path = 'data/qa.csv'
vector_db_path = "./vectorstores"

# Load the CSV data into the chatbot
loader = CSVLoader(file_path=csv_path, csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['question', 'answer']
    })


documents = loader.load()
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
embedding_model = GPT4AllEmbeddings(model_file = 'models/all-MiniLM-L6-v2-f16.gguf')



def save_db():
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(vector_db_path)

def load_db():
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db



def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        # max_new_tokens=4096,
        # temperature=0,
        config={
            'max_new_tokens': 1000,
            'temperature': 0.01,
            'context_length': 800,
            # 'gpu_layers': 1
        }
    )
    return llm

save_db()


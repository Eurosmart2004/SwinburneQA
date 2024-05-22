import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from db import load_db, load_llm

load_dotenv()
db = load_db()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Create a chat interface
st.title("Chatbot")

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)


def get_response(question):
    template = """
    You are a helpful assistant for Swinburne Q&A. You are asked a question and you provide an answer.
    {context}
    Question: {question}
    Helpful Answer:
"""


    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # llm = ChatOpenAI(
    #     model="gpt-3.5-turbo-0125",
    #     temperature=0.0,
    #     max_tokens=100,
    # )

    llm = load_llm("models/llama-2-7b-chat.ggmlv3.q4_1.bin")

    retriever = db.as_retriever(search_kwargs = {"k":1}, max_tolens_limit=4096)


    chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = retriever,
        return_source_documents = True,
        verbose=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )

    # return llm_chain.stream({
    #     'query': question,
    # })
    
    # Use the chain to stream results for a given question
    response = chain.invoke({'query': question})
    print(response)

    return response["result"]



user_query = st.chat_input("Your message")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = get_response(user_query)
        st.markdown(ai_response)
        st.session_state.chat_history.append(AIMessage(ai_response))
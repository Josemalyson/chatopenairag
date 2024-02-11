import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

load_dotenv()

temperature = os.getenv('TEMPERATURE')
model = os.getenv('MODEL')
openai_api_key = os.getenv('OPENAI_API_KEY')
chunk_size = os.getenv('CHUNK_SIZE')
chunk_overlap = os.getenv('CHUNK_OVERLAP')
vector_database = os.getenv('VECTOR_DB_DIRECTORY')

llm = ChatOpenAI(temperature=temperature, model=model, openai_api_key=openai_api_key, streaming=True)
client = OpenAI(api_key=openai_api_key)

st.title("ChatGPT utilizando RAG")
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
upload_audio = st.file_uploader("Upload an audio file", type=["mp4"])
rag_chain = None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def prompt_template():
    template = """

         Use as seguintes partes do contexto para responder à pergunta no final.
         Se você não sabe a resposta, apenas diga que não sabe, não tente inventar uma resposta.
         Use no máximo três frases e mantenha a resposta o mais concisa possível.
         Sempre diga "obrigado por perguntar!" no final da resposta.

         {context}

         Pergunta: {question}

         Helpful Answer:

        """
    return PromptTemplate.from_template(template)


if upload_audio is not None:
    st.info(body=f"Chatbot contextualizado com o documento: {upload_audio.name}", icon="✅")

if uploaded_file is not None:
    st.info(body=f"Chatbot contextualizado com o documento: {uploaded_file.name}", icon="✅")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = model

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):

        if uploaded_file is not None:
            # Create tmp file
            temp_dir = tempfile.TemporaryDirectory()
            temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())

            # loader tmp file
            loader = PyPDFLoader(temp_filepath)

            # create docs with split
            docs = loader.load_and_split()
            # config split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

            # Retrieve and generate using the relevant snippets of the blog.
            retriever = vectorstore.as_retriever()

            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_template()
                    | llm
                    | StrOutputParser()
            )

            # involke chat

            response = rag_chain.invoke(prompt)

            # cleanup
            vectorstore.delete_collection()

            st.write(response)

        elif upload_audio is not None:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=upload_audio
            )

            # create docs with split
            text_splitter = CharacterTextSplitter()
            texts_list = text_splitter.split_text(text=transcript.text)
            splits = text_splitter.create_documents(texts=texts_list)

            # Retrieve and generate using the relevant snippets of the blog.
            vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
            retriever = vectorstore.as_retriever()

            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_template()
                    | llm
                    | StrOutputParser()
            )

            # involke chat

            response = rag_chain.invoke(prompt)

            # cleanup
            vectorstore.delete_collection()

            st.write(response)

        else:
            # default chat without context
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
    st.session_state.messages.append({'role': "assistant", "content": response})

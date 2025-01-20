import streamlit as st
from langchain.vectorstores import Chroma
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()
os.environ['REPLICATE_API_TOKEN'] = os.getenv('REPLICATE_API_TOKEN', 'your_default_token_here')

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask Me Anything about This File"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Does This File Have IOCs?", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input.strip():
            with st.spinner('Checking Logs...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    # Create the Replicate LLM
    llm = Replicate(
        streaming=True,
        model="meta/llama-2-70b-chat",
        callbacks=[StreamingStdOutCallbackHandler()],
        input={
            "temperature": 0.8,
            "max_length": 500,
            "top_p": 1,
            "system_prompt": (
                "Attention Model: You are a specialized chatbot designed to assist individuals in the oil and gas "
                "industry, particularly focusing on content related to the Basrah Gas Company. Responses must rely "
                "on uploaded PDF files, with logical reasoning when needed."
            )
        }
    )

    # Memory for the conversational chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Conversational Retrieval Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

def main():
    # Initialize session state
    initialize_session_state()
    st.title("Chat with PDF using Llama 2")
    st.sidebar.title("Upload Your Log File Here")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            try:
                loader = None
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)
                elif file_extension in [".docx", ".doc"]:
                    loader = Docx2txtLoader(temp_file_path)
                elif file_extension == ".txt":
                    loader = TextLoader(temp_file_path)
                elif file_extension == ".csv":
                    loader = CSVLoader(file_path=temp_file_path, encoding="utf-8", csv_args={'delimiter': ','})

                if loader:
                    text.extend(loader.load())
            finally:
                os.remove(temp_file_path)

        # Split text into chunks
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings (CPU-friendly)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

        # Create vector store
        vector_store = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./test_embedding')

        # Create the conversational chain
        chain = create_conversational_chain(vector_store)

        # Display the chat interface
        display_chat_history(chain)

if __name__ == "__main__":
    main()

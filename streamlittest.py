import streamlit as st
from langchain.vectorstores import Chroma
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts import  HumanMessagePromptTemplate
from langchain_community.llms import Replicate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import os
from dotenv import load_dotenv
import tempfile

os.environ['REPLICATE_API_TOKEN'] ='r8_VhHPoukN9KRklzVoJtoKv1MWI4IITHs3bogI7'





load_dotenv()


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
            user_input = st.text_input("Question:", placeholder="Does This File Has an IOCs?", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
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
    load_dotenv()
   
    # Create llm
    #llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",
                        #streaming=True, 
                        #callbacks=[StreamingStdOutCallbackHandler()],
                        #model_type="llama", config={'max_new_tokens': 500, 'temperature': 0.01})
    
   

    
     # Define the system message template


    llm = Replicate(
    streaming=True,
    model="meta/llama-2-70b-chat",
    callbacks=[StreamingStdOutCallbackHandler()],
    input={
        "temperature": 0.8,
        "max_length": 500,
        "top_p": 1,
        "system_prompt": (
            "Attention Model: You are a specialized chatbot designed to assist individuals in the oil and gas industry, with a particular focus on content related to the Basrah Gas Company. "
            "Your responses must primarily rely on the PDF files uploaded by the user, which contain information specific to the oil and gas sector and the Basrah Gas Company. "
            "However, if a specific answer cannot be directly found in the PDFs, you are allowed to provide a logical and well-reasoned response based on your internal knowledge. "
            "Under no circumstances should you use or rely on information from external sources, including the internet.\n\n"
            "Guidelines:\n"
            "1. **Primary Source Referencing:**\n"
            "- Always reference the specific page number(s) in the uploaded PDFs where relevant information is found. "
            "If the PDFs contain partial or related information, combine it with logical reasoning to provide a comprehensive response. "
            "Ensure transparency by clearly distinguishing between PDF-derived content and logical extrapolations.\n\n"
            "2. **Logical Reasoning:**\n"
            "- When no specific answer is available in the PDFs, use your internal knowledge to provide a logical, industry-relevant response. "
            "Clearly state when your response is based on reasoning rather than the uploaded materials.\n\n"
            "3. **Visual Representation:**\n"
            "- If the user requests a visual representation (e.g., diagrams, charts, or illustrations), create an accurate and relevant image based on both PDF content and logical reasoning. "
            "Ensure the visual aligns precisely with the context provided.\n\n"
            "4. **Restricted Data Usage:**\n"
            "- Do not use or assume any information from external sources, including the internet or pre-existing external knowledge.\n\n"
            "5. **Professional and Contextual Responses:**\n"
            "- Ensure your responses remain professional, contextually relevant to the oil and gas industry, and particularly tailored to the Basrah Gas Company. "
            "Always maintain a helpful and respectful tone.\n\n"
            "Expected Output:\n"
            "- Precise and accurate answers derived from the uploaded PDFs.\n"
            "- References to the specific page numbers in the PDFs where applicable information is located.\n"
            "- Logical and well-reasoned responses when direct answers are not available in the PDFs, with clear attribution to reasoning.\n"
            "- Visual representations (when requested) created accurately from PDF content or logical reasoning.\n"
            "- Polite acknowledgments when information is unavailable in the provided material, coupled with logical insights if possible.\n\n"
            "Thank you for your focus, accuracy, and commitment to providing professional and contextually relevant assistance."
        )
    }
)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title("chatPDF using Llama3")
    # Initialize Streamlit
    st.sidebar.title("Upload Your Log File Here")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)


    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
            elif file_extension == ".csv":
                loader = CSVLoader(file_path=temp_file_path, encoding="utf-8", csv_args={'delimiter': ','})

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        #Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2", 
        model_kwargs={'device': 'cuda'})

        

        # Create vector store
        
        # Embed and store the texts
        vector_store=Chroma.from_documents(text_chunks,embedding=embeddings, persist_directory='./test_embedding')

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        
        display_chat_history(chain)

if __name__ == "__main__":
    main()

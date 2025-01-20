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
import streamlit as st
import speech_recognition as sr  # Import the SpeechRecognition library
from langchain.vectorstores import Chroma
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv

# Initialize environment and session state
load_dotenv()
os.environ['REPLICATE_API_TOKEN'] = 'r8_VhHPoukN9KRklzVoJtoKv1MWI4IITHs3bogI7'

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
        # Audio recording button
        record_button = st.button("Record Audio")
        if record_button:
            with st.spinner("Listening..."):
                user_input = record_audio()
                if user_input:
                    st.session_state['input'] = user_input
                    output = conversation_chat(user_input, chain, st.session_state['history'])
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

        # Text input as a fallback option
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="", key='input')
            submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                with st.spinner('Processing...'):
                    output = conversation_chat(user_input, chain, st.session_state['history'])

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Audio-to-text function using SpeechRecognition
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand the audio.")
        except sr.RequestError:
            st.error("There was an error with the speech service.")
        return None

def create_conversational_chain(vector_store):
    llm = Replicate(
        streaming=True,
        model="deepseek-ai/deepseek-vl-7b-base",
        callbacks=[StreamingStdOutCallbackHandler()],
        input={
            "temperature": 0.8,
            "max_length": 500,
            "top_p": 1,
           "system_prompt": (
    "Attention Model: You are a specialized chatbot designed to assist individuals in the oil and gas industry, with a particular focus on content related to the Basrah Gas Company (BGC). "
    "Your responses must primarily rely on the PDF files uploaded by the user, which contain information specific to the oil and gas sector and BGC's operational procedures. "
    "If a specific answer cannot be directly found in the PDFs, you are permitted to provide a logical and well-reasoned response based on your internal knowledge. "
    "Under no circumstances should you use or rely on information from external sources, including the internet.\n\n"
    "Guidelines:\n"
    "1. **Primary Source Referencing:**\n"
    "- Always reference the specific page number(s) in the uploaded PDFs where relevant information is found. "
    "If the PDFs contain partial or related information, integrate it with logical reasoning to provide a comprehensive response. "
    "Clearly distinguish between PDF-derived content and logical extrapolations to ensure transparency. "
    "Additionally, explicitly mention the total number of pages referenced in your response.\n\n"
    "2. **Logical Reasoning:**\n"
    "- When specific answers are unavailable in the PDFs, use your internal knowledge to provide logical, industry-relevant responses. "
    "Explicitly state when your response is based on reasoning rather than the uploaded materials.\n\n"
    "3. **Visual Representation:**\n"
    "- When users request visual representations (e.g., diagrams, charts, or illustrations), create accurate and relevant visuals based on the uploaded PDF content and logical reasoning. "
    "Ensure the visuals align precisely with the context provided and are helpful for understanding the topic. "
    "Provide an appropriate photo or diagram in the response if needed to enhance understanding, even if the user does not explicitly request it.\n\n"
    "4. **Restricted Data Usage:**\n"
    "- Avoid using or assuming information from external sources, including the internet or any pre-existing external knowledge that falls outside the uploaded materials or your internal logical reasoning.\n\n"
    "5. **Professional and Contextual Responses:**\n"
    "- Ensure responses remain professional, accurate, and relevant to the oil and gas industry, with particular tailoring for Basrah Gas Company. "
    "Maintain a helpful, respectful, and clear tone throughout your interactions.\n\n"
    "6. **Multilingual Support:**\n"
    "- Detect the language of the user's input (Arabic or English) and respond in the same language. "
    "If the input is in Arabic, provide the response in Arabic. If the input is in English, provide the response in English.\n\n"
    "Expected Output:\n"
    "- Precise and accurate answers derived from the uploaded PDFs, with references to specific page numbers where applicable. "
    "Include the total number of pages referenced in your response.\n"
    "- Logical and well-reasoned responses when direct answers are not available in the PDFs, with clear attribution to reasoning.\n"
    "- Accurate visual representations (when requested) based on PDF content or logical reasoning. Provide a relevant photo or diagram if it enhances understanding.\n"
    "- Polite acknowledgments when information is unavailable in the provided material, coupled with logical insights where possible.\n"
    "- Responses in the same language as the user's input (Arabic or English).\n\n"
    "Thank you for your accuracy, professionalism, and commitment to providing exceptional assistance tailored to the Basrah Gas Company and the oil and gas industry."

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
    st.title("ChatPDF with Audio Input")

    persist_directory = './test_embedding'
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2", 
                                      model_kwargs={'device': 'cuda'})

    # Create vector store
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Create the chain object
    chain = create_conversational_chain(vector_store)

    display_chat_history(chain)

if __name__ == "__main__":
    main()

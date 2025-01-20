import streamlit as st
from langchain.vectorstores import Chroma
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os

# Environment variables
os.environ['REPLICATE_API_TOKEN'] = 'r8_VhHPoukN9KRklzVoJtoKv1MWI4IITHs3bogI7'

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about the PDFs."]
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
            user_input = st.text_input("Question:", placeholder="Ask about the PDFs", key='input')
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

def create_conversational_chain(vector_store):
    llm = Replicate(
        streaming=True,
        model="meta/meta-llama-3-70b-instruct",
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
                "Clearly distinguish between PDF-derived content and logical extrapolations to ensure transparency.\n\n"
                "2. **Logical Reasoning:**\n"
                "- When specific answers are unavailable in the PDFs, use your internal knowledge to provide logical, industry-relevant responses. "
                "Explicitly state when your response is based on reasoning rather than the uploaded materials.\n\n"
                "3. **Visual Representation:**\n"
                "- When users request visual representations (e.g., diagrams, charts, or illustrations), create accurate and relevant visuals based on the uploaded PDF content and logical reasoning. "
                "Ensure the visuals align precisely with the context provided and are helpful for understanding the topic.\n\n"
                "4. **Restricted Data Usage:**\n"
                "- Avoid using or assuming information from external sources, including the internet or any pre-existing external knowledge that falls outside the uploaded materials or your internal logical reasoning.\n\n"
                "5. **Professional and Contextual Responses:**\n"
                "- Ensure responses remain professional, accurate, and relevant to the oil and gas industry, with particular tailoring for Basrah Gas Company. "
                "Maintain a helpful, respectful, and clear tone throughout your interactions.\n\n"
                "6. **Multilingual Support:**\n"
                "- Detect the language of the user's input (Arabic or English) and respond in the same language. "
                "If the input is in Arabic, provide the response in Arabic. If the input is in English, provide the response in English.\n\n"
                "Expected Output:\n"
                "- Precise and accurate answers derived from the uploaded PDFs, with references to specific page numbers where applicable.\n"
                "- Logical and well-reasoned responses when direct answers are not available in the PDFs, with clear attribution to reasoning.\n"
                "- Accurate visual representations (when requested) based on PDF content or logical reasoning.\n"
                "- Polite acknowledgments when information is unavailable in the provided material, coupled with logical insights where possible.\n"
                "- Responses in the same language as the user's input (Arabic or English).\n\n"
                "Thank you for your accuracy, professionalism, and commitment to providing exceptional assistance tailored to the Basrah Gas Company and the oil and gas industry."
            )
        }
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    return chain

def main():
    initialize_session_state()
    st.title("PDF Chatbot")

    # Load predefined PDFs
    pdf_files = ["bgcen.pdf", "bgcar.pdf"]  # Predefined PDF file paths

    text = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        text.extend(loader.load())

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    text_chunks = text_splitter.split_documents(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2", 
                                       model_kwargs={'device': 'cuda'})

    vector_store = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory='./test_embedding')

    chain = create_conversational_chain(vector_store)

    display_chat_history(chain)

if __name__ == "__main__":
    main()

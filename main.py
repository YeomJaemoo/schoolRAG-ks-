import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.memory import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
import tiktoken

def main():
    st.set_page_config(page_title="kangsinchat", page_icon="🏫")
    st.image('kangsin_middle_school.png')
    st.title("_강신중학교 :red[Q&A]_ 🏫")
    st.header("😶주의!이 챗봇은 참고용으로 사용하세요!", divider='rainbow')

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        folder_path = Path()
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        model_name = 'gpt-3.5-turbo'
    
    st.text("아래의 'Process'를 누르고 아래 채팅창이 활성화 될 때까지 잠시 기다려주세요!🙂🙂🙂")
    process = st.button("Process")

    if process:
        files_text = get_text_from_folder(folder_path)
        text_chunks = get_text_chunks(files_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key, model_name)
        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                         "content": "강신중학교 생활&성적 규정에 대해 물어보세요!😊"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text_from_folder(folder_path):
    doc_list = []
    folder = Path(folder_path)
    files = folder.iterdir()

    for file_path in files:
        if file_path.is_file():
            if file_path.suffix == '.pdf':
                loader = PyPDFLoader(str(file_path))
                documents = loader.load_and_split()
            elif file_path.suffix == '.docx':
                loader = Docx2txtLoader(str(file_path))
                documents = loader.load_and_split()
            elif file_path.suffix == '.pptx':
                loader = UnstructuredPowerPointLoader(str(file_path))
                documents = loader.load_and_split()
            else:
                documents = []
            doc_list.extend(documents)
    return doc_list

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vectorstore, openai_api_key, model_name):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type='mmr'),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

if __name__ == '__main__':
    main()

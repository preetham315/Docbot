import streamlit as st
from io import BytesIO
import os
from langchain import OpenAI
import openai
from langchain.chat_models import ChatOpenAI
from llama_index import download_loader
from llama_index import LLMPredictor, GPTVectorStoreIndex, ServiceContext
from llama_index import StorageContext, load_index_from_storage
from llama_index.storage.storage_context import StorageContext
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredFileLoader

def clear_submit():
    st.session_state["submit"] = False

def generateIndex(pdf_path,pdf_dir):
    with open(pdf_path, "rb") as file:
        bytes_stream = BytesIO(file.read())
        SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
        loader = SimpleDirectoryReader(pdf_dir, recursive=True, exclude_hidden=True)
        st.session_state.documents = loader.load_data()
        st.session_state.index = GPTVectorStoreIndex.from_documents(st.session_state.documents)
    return bytes_stream

def generateFAISSIndex(pdf_path,pdf_dir):
    with open(pdf_path, "rb") as file:
        bytes_stream = BytesIO(file.read())
        loader = DirectoryLoader(pdf_dir)
        st.session_state.documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        docs = text_splitter.split_documents(st.session_state.documents)
        embed_model = OpenAIEmbeddings()
        st.session_state.index = FAISS.from_documents(docs, embed_model)
    return bytes_stream

def generatePreview():
    sidebar_placeholder = st.sidebar.container()
    sidebar_placeholder.markdown("""### Document Preview""")
    with sidebar_placeholder.expander("Text Content", expanded= True):
        if "documents" in st.session_state:
            for doc in st.session_state.documents:
                st.write(doc.get_text()[:800]+'...')
                
def sidebar(doc_path, embed_model):
    openai.api_key = os.environ["OPENAI_API_KEY"]

    with st.sidebar:
        st.markdown("""
                <style>
                .big-font {
                    font-size:20px !important;
                    color: grey;
                }
                </style>
                """, unsafe_allow_html=True)

        st.markdown('<p class="big-font"><b>Upload Document</b></p>', unsafe_allow_html=True)
                
        st.session_state.uploaded_files = st.file_uploader("Upload",type=["pdf", "docx", "txt"], 
                                                        accept_multiple_files=True, 
                                                        label_visibility="collapsed", on_change=clear_submit)
        
        msgloaded1,msgloaded2 = st.columns([0.999,0.001])
        
        st.write("OR")

        sinfo,sdwnld = st.columns([0.8,0.2])

        with sinfo:
            st.markdown('<p class="big-font"><b>Select Sample Document</b></p>', unsafe_allow_html=True)
        st.session_state.sample_doc = st.selectbox("Choose",("","About EDGAR", "Paper forms 144",
                                                            "Daily index"),
                                                            key = 'docselection',
                                                            label_visibility="collapsed",                                                            
                                                            format_func=lambda x: 'Select an option' if x == '' else x)
              
        if st.session_state.uploaded_files:

            if(st.session_state.new_file != "" and st.session_state.new_file==st.session_state.uploaded_files):
                st.session_state.FileIndex = True

            if st.session_state.index is None or st.session_state.FileIndex == False:

                st.session_state.model_input = ""
                st.session_state.code_input = ""
                st.session_state.new_file = st.session_state.uploaded_files

                with msgloaded1:
                    with st.spinner("Please wait! The document is being indexed."):
                        st.session_state.sample_doc = ""
                        
                        st.session_state.FileIndex = True
                        st.session_state.AboutIndex = False
                        st.session_state.PaperIndex = False
                        st.session_state.DailyIndex = False
                        
                        doc_files = os.listdir(doc_path)

                        for doc_file in doc_files:
                            os.remove(doc_path + doc_file)
                        
                        for uploaded_file in st.session_state.uploaded_files:
                            bytes_data = uploaded_file.read()
                            
                            with open(f"{doc_path}{uploaded_file.name}", 'wb') as f:
                                f.write(bytes_data)  

                            SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
                            loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
                            st.session_state.documents = loader.load_data()
               
                            st.session_state.index = GPTVectorStoreIndex.from_documents(st.session_state.documents)

                        with msgloaded1:
                            st.info("Your document is indexed!")
                                                            
            generatePreview()
                    
        else:
            if st.session_state.sample_doc == "About EDGAR":

                if st.session_state.index is None or st.session_state.AboutIndex == False:
                
                    st.session_state.FileIndex = False
                    st.session_state.AboutIndex = True
                    st.session_state.PaperIndex = False
                    st.session_state.DailyIndex = False
                                       
                    pdf_path = "about_data/EDGAR.pdf" # Replace with the actual path to your PDF file
                    pdf_dir = "about_data"
                    persistent_directory = "./about/"
                    
                    with st.spinner("Please wait! The document is being indexed."):
                        bytes_stream = generateIndex(pdf_path,pdf_dir)
                                                    
                    with sdwnld:
                        doc_preview = st.download_button(":arrow_double_down:", bytes_stream, help = "Download File")                    
                    
                    st.info("Document is loaded & indexed!")
                    
                generatePreview()

            elif st.session_state.sample_doc == "Paper forms 144":

                if st.session_state.index is None or st.session_state.PaperIndex == False:
                
                    st.session_state.FileIndex = False
                    st.session_state.AboutIndex = False
                    st.session_state.PaperIndex = True
                    st.session_state.DailyIndex = False
                                       
                    pdf_path = "paper_data/form144.pdf" # Replace with the actual path to your PDF file
                    pdf_dir = "paper_data"
                    persistent_directory = "./paper/"
                    
                    with st.spinner("Please wait! The document is being indexed."):
                        bytes_stream = generateIndex(pdf_path,pdf_dir)
                                                    
                    with sdwnld:
                        doc_preview = st.download_button(":arrow_double_down:", bytes_stream, help = "Download File")                    
                    
                    st.info("Document is loaded & indexed!")
                    
                generatePreview()

                                
            elif st.session_state.sample_doc == "Daily index":

                if st.session_state.index is None or st.session_state.DailyIndex == False:

                    st.session_state.FileIndex = False
                    st.session_state.AboutIndex = False
                    st.session_state.PaperIndex = False
                    st.session_state.DailyIndex = True
                                        
                    pdf_path = "daily_index_data/daily_index.pdf" # Replace with the actual path to your PDF file
                    pdf_dir = "daily_index_data"
                    persistent_directory = "./daily_index/"
                    
                    with st.spinner("Please wait! The document is being indexed."):
                        bytes_stream = generateIndex(pdf_path,pdf_dir)
                    
                    with sdwnld:
                        doc_preview = st.download_button(":arrow_double_down:", bytes_stream, help = "Download File")                    
                    
                    st.info("Document is loaded & indexed!")
                                            
                generatePreview()

            else:
                st.session_state.index = None
                st.session_state.selected_ques = ""
                st.session_state.ask_prompt = ""
                st.session_state.selected_item = ""
                st.session_state.stat_analysis = ""

        st.markdown("---")
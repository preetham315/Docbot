import streamlit as st
import streamlit_authenticator as stauth
import streamlit_toggle as tog
import utils as utl
import os
import base64
import pickle
from io import BytesIO
import PyPDF2
import re
from pathlib import Path
from sidebar import sidebar
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
import openai
from llama_index import download_loader
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import StorageContext, load_index_from_storage
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.storage.storage_context import StorageContext

# Initiate page configurations
st.set_page_config(layout="wide", page_title='Document Question Answering (DQA) System')
utl.navbar_component()
st.set_option('deprecation.showPyplotGlobalUse', False)
utl.inject_custom_css()
st.session_state.embed_model = OpenAIEmbedding()

# OpenAI API Key
# os.environ["OPENAI_API_KEY"] = 'replace this with your openai key'
openai.api_key = os.environ["OPENAI_API_KEY"]

# Configuration Settings
embed_model = OpenAIEmbedding()

hide_menu_style = """
<style>
  #MainMenu, header, footer {visibility: hidden;}
</style>
  """
st.markdown(hide_menu_style, unsafe_allow_html=True)

def generate_response():
    with st.spinner(text="Loading the response... Please wait!"):  
        st.session_state.response = True
        res_box = st.empty()
        report = []
        service_context = ServiceContext.from_defaults(llm=ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", streaming = True))
        query_engine = st.session_state.index.as_query_engine(service_context=service_context, similarity_top_k=3, streaming = True)
        
        input_prompt = f'''
        As a chat assistant to the end user, use the context provided and your knowledge to answer the user query in a professional tone.
                
        This is the user query:
        {st.session_state.model_input} 
        
        '''

        for resp in query_engine.query(input_prompt).response_gen:
            report.append(resp)
            st.session_state.response = "".join(report).strip()
            res_box.markdown(f"<div class='resbox' style='padding: 5px; background-color: #dff4eb; font-size: 18px; color: #187233;'>{st.session_state.response}</div>",
                                 unsafe_allow_html=True    
                                ) 
    st.session_state.prompt_history.append((st.session_state.model_input, st.session_state.response))
    st.session_state.selected_ques = ""
    st.session_state.ask_prompt = ""

def generate_StreamingResponse():
   
    with st.spinner(text="Loading the response... Please wait!"):  

        context = st.session_state.index.similarity_search(st.session_state.model_input)
        page_contents = [doc.page_content for doc in context]
        page_contents_context = ' '.join(page_contents)

        res_box = st.empty()
        report = []
        
        input_prompt = f'''
        As a chat assistant to the end user, use the context provided and your knowledge to answer the user query in a professional tone.
                
        This is the user query:
        {st.session_state.model_input} 

        This is the context information:
        {page_contents_context} 
        '''

        model_name = "gpt-3.5-turbo"

        for resp in openai.ChatCompletion.create(model=model_name,
                                    messages = [
                                        {'role': 'user', 'content': input_prompt}
                                    ],
                                    max_tokens=2000, 
                                    temperature = 0.3,
                                    stream = True):    

            if resp.choices[0]["finish_reason"] != "stop":
                token = resp.choices[0]["delta"]["content"].replace(" -", "\n -")
                report.append(token)
                
                st.session_state.response = "".join(report).strip()
                                
                res_box.markdown(f"<div class='resbox' style='padding: 5px; background-color: #dff4eb; font-size: 18px; color: #187233;'>{st.session_state.response}</div>",
                                 unsafe_allow_html=True    
                                ) 
            
    st.session_state.prompt_history.append((st.session_state.model_input, st.session_state.response))
    st.session_state.selected_ques = ""
    st.session_state.ask_prompt = ""

def default_response():
    
    col1ques, col2ques, col3ques = st.columns([0.8,1,0.5])
    with col1ques:
        st.write(' ')
    with col2ques:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.image("./assets/responsee.png")
    with col3ques:
        st.write(' ')

def empty_response():
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <span style="font-size: 30px; color: #6D778D;">Document Question Answering (DQA) System</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <span style="font-size: 18px; color: #7B838A;">This Chatbot allows you to ask questions on the uploaded document</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    with open("./assets/empty_state.png", "rb") as image_file:
        empty_state = base64.b64encode(image_file.read())

    st.markdown(
        rf"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64, {empty_state.decode("utf-8")}" width="295" height="219"/>
        </div>
        """,
        unsafe_allow_html=True,
    )

authentication_status = True

if authentication_status == True:
    doc_path = './data/'
    index_file = 'index.json'
    
    if 'index' not in st.session_state:
        st.session_state.index = None

    if 'model_input' not in st.session_state:
        st.session_state.model_input = ""

    if 'code_input' not in st.session_state:
        st.session_state.code_input = ""

    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    if 'submitted1' not in st.session_state:
        st.session_state.submitted1 = False

    if 'coderesponse' not in st.session_state:
        st.session_state.coderesponse = ""

    if 'response' not in st.session_state:
        st.session_state.response = ""

    if 'document_chosen' not in st.session_state:
        st.session_state.document_chosen = ""

    if 'document_chosen1' not in st.session_state:
        st.session_state.document_chosen1 = ""

    if 'FileIndex' not in st.session_state:
        st.session_state.FileIndex = False

    if 'AboutIndex' not in st.session_state:
        st.session_state.AboutIndex = False

    if 'PaperIndex' not in st.session_state:
        st.session_state.PaperIndex = False

    if 'DailyIndex' not in st.session_state:
        st.session_state.DailyIndex = False

    if 'new_file' not in st.session_state:
        st.session_state.new_file = ""

    sidebar(doc_path, embed_model)

    if 'prompt_history' not in st.session_state:
        st.session_state.prompt_history = []
         
    if st.session_state.index is not None:
                        
        with st.form("my_form1", clear_on_submit = False):

            col1, temp1, col2 = st.columns([1,0.08, 1])
            
            with col1:
                
                if not st.session_state.uploaded_files and st.session_state.sample_doc != "":
                    st.write(f'<span style="font-size: 20px; color: rgb(0, 104, 201)">Sample Questions</span>', unsafe_allow_html=True)
                    if st.session_state.sample_doc == "About EDGAR":
                        st.session_state.selected_ques = st.selectbox("]",("","Summarize the document with the 5 most unique and helpful points, into a numbered list of key points and takeaways. Include an introductory paragraph.", "What is EDGAR?"),format_func=lambda x: 'Select an option' if x == '' else x, label_visibility = "collapsed")
                    if st.session_state.sample_doc == "Paper forms 144":
                        st.session_state.selected_ques = st.selectbox("]",("","Summarize the document with the 5 most unique and helpful points, into a numbered list of key points and takeaways. Include an introductory paragraph.", "What is form 144?", "List the information that is available in this form in a bulleted list"),format_func=lambda x: 'Select an option' if x == '' else x, label_visibility = "collapsed")
                    elif st.session_state.sample_doc == "Daily index":
                        st.session_state.selected_ques = st.selectbox("",("","Summarize the document with the 5 most unique and helpful points, into a numbered list of key points and takeaways. Include an introductory paragraph.", "What is CIK?", "When was the earliest form filed?"),format_func=lambda x: 'Select an option' if x == '' else x, label_visibility = "collapsed")
                    
                elif st.session_state.uploaded_files:
                        st.write(f'<span style="font-size: 20px; color: rgb(0, 104, 201)">Sample Questions</span>', unsafe_allow_html=True)
                        st.session_state.selected_ques = st.selectbox("",("","Summarize the document with 7-8 most unique and helpful points into a numbered list of key points and takeaways. Include an introductory paragraph.","List out key questions that can be answered with specific details based on this document. Include the answer for each question.","Are there any notable sources, theories, concepts, or frameworks referenced in the document? If so, please explain them in detail.", "What are the potential implications or consequences of the ideas presented in the document? Kindly elaborate in a numbered list.", "What are the strengths and weaknesses of the document in terms of its structure, clarity, and overall effectiveness?"),format_func=lambda x: 'Select an option' if x == '' else x, label_visibility = "collapsed")
                else:
                        st.write(f'<span style="font-size: 20px; color: rgb(0, 104, 201)">Sample Questions</span><span style="font-size: 16px; color: grey"> <i>(not available)</i></span>', unsafe_allow_html=True)
                        st.session_state.selected_item = st.selectbox("", (""), format_func=lambda x: 'No options available' if x == '' else x, label_visibility = "collapsed")
        
            with temp1:
                st.write("")
                st.write("")
                st.write("")
                st.write("OR")
            
            with col2:
                                
                st.write(f'<span style="font-size: 20px; color: rgb(0, 104, 201)">Ask a question on the document</span>', unsafe_allow_html=True)
                st.session_state.ask_prompt = st.text_input("", key = "prompt", placeholder = "Enter your query", label_visibility = "collapsed")  

            submit, loader = st.columns([4,8])      
            
            with submit:
                st.session_state.submitted = st.form_submit_button("GENERATE RESPONSE") 

            st.write("")
            st.write("")
                                                                
        with st.expander("Response to your query", expanded=True):
                        
            #if st.session_state.submitted and st.session_state.model_input != "":
            if st.session_state.submitted:

                if (st.session_state.selected_ques == "" or st.session_state.selected_ques is None) and st.session_state.ask_prompt == "":
                    st.session_state.model_input = ""
                    st.error("Please provide a question.")
                else:    
                    if st.session_state.selected_ques!= "" and st.session_state.ask_prompt == "":
                        st.session_state.model_input = st.session_state.selected_ques
                    elif st.session_state.ask_prompt != "":
                        st.session_state.model_input = st.session_state.ask_prompt

                    st.session_state.document_chosen = st.session_state.sample_doc
                    st.write(f'<span style="font-size: 18px;"><b>Prompt:</b> {st.session_state.model_input}</span>', unsafe_allow_html=True)
                                                        
                    generate_response()
                    #generate_StreamingResponse()
                                    
            #elif not st.session_state.submitted and not st.session_state.submitted1 and st.session_state.model_input != "":
            elif not st.session_state.submitted and st.session_state.document_chosen==st.session_state.sample_doc and st.session_state.model_input != "":
                
                st.write(f'<span style="font-size: 18px;"><b>Prompt:</b> {st.session_state.model_input}</span>', unsafe_allow_html=True)
                if st.session_state.response: 
                    
                    final_response = str(st.session_state.response)
                    final_response = re.sub('\n+', '\n', final_response)
                    final_response = final_response.split('\n')

                    for line in final_response:
                                            
                        st.markdown(
                            f"<div style='font-size: 18px; color: #187233; background-color: #dff4eb; padding: 10px;'>{line}</div>",
                            #f"<div style='background-color: #dff4eb; padding: 10px;'><div style='font-size: 18px; color: #187233;'>{line}</div></div>",
                            unsafe_allow_html=True
                        )
                    
                st.session_state.ask_prompt = ""
            
            else:
                default_response()

        if "response" in st.session_state:
        
            with st.expander("Prompt History", expanded = False):    
            
                for prompt, st.session_state.response in st.session_state.prompt_history:
                    if "submitted" in st.session_state:
                        st.markdown(
                            f"<div style='padding: 10px;'><div style='font-size: 18px;'><b>Prompt:</b> {prompt}</div></div>",
                            unsafe_allow_html=True
                        )

                        prompthist_response = str(st.session_state.response)
                        prompthist_response = re.sub('\n+', '\n', prompthist_response)
                        prompthist_response = prompthist_response.split('\n')

                        for lineitem in prompthist_response:

                            st.markdown(
                                f"<div style='background-color: #dff4eb; padding: 10px;'><div style='font-size: 18px; color: #187233;'>{lineitem}</div></div>",
                                unsafe_allow_html=True
                            )                        


    else:
        empty_response()

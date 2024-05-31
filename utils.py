import streamlit as st
import base64
from streamlit.components.v1 import html

def inject_custom_css():
    with open('assets/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    

def get_current_route():
    try:
        return st.experimental_get_query_params()['nav'][0]
    except:
        return None


def navbar_component():

    component = rf'''
            <nav class="container navbar" id="navbar">
                <div style="display:flex; align-items: center;">
                    <div class = "header" align="left">Document Question Answering (DQA) System</div>
                    <span style="margin-left: 10px;"></span>                  
                </div>                    
            </nav>       
            '''
    st.markdown(component, unsafe_allow_html=True)
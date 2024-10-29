from streamlit_extras.app_logo import add_logo
import streamlit as st
import requests
import pandas as pd
import json
# from streamlit_extras.app_logo import add_logo


st.set_page_config(layout="wide")

file_predictions_synth = "synth_bert_model_test_result.csv"
file_predictions = "synth_bert_model_test_result.csv"

pdf = pd.read_csv(file_predictions)
pdf_s = pd.read_csv(file_predictions_synth)
claimids_options = {True:pdf_s.claim_id,False:pdf.claim_id}
# Function to get the response from the Databricks model serving endpoint
def get_model_response(oem_code_str,dealer_str,distr_str,fc_str, engine_name_desc_str,failcode_str,shopordernumber_str, correction_str):
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {"inputs": {oem_code_str,dealer_str,distr_str,str(fc_str), engine_name_desc_str,str(failcode_str),shopordernumber_str, correction_str}}
    response = {"OEM_CODE": oem_code_str, "DEALER":dealer_str, "DISTR":distr_str, "FC":fc_str, "ENGINE_NAME_DESC": engine_name_desc_str, "FAILCODE": failcode_str, "SHOPORDERNUM": shopordernumber_str, "CLEANED_CORRECTION": correction_str}
    return response




# Streamlit UI components
with st.container():
    st.markdown(
        """
        <style>
        .stContainer > div {
            width: 95%;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.image("compnovai_logo.PNG",use_column_width = True, width=1200)
    
togg_switch = st.toggle('Synthetic data',True,key='data_source')

def select_claimid():
    print(st.session_state.claimid_str)
    if st.session_state.claimid_str=='--Select--' or st.session_state.claimid_str==None:
        clear_function()
        return
    if togg_switch:
        temp_df = pdf_s[pdf_s.claim_id==st.session_state.claimid_str].reset_index(drop=True)
    else:
        temp_df = pdf[pdf.claim_id==st.session_state.claimid_str].reset_index(drop=True)
        
    res_dictionary = json.loads(str(temp_df['document'][0]).lower())
    
    with c:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.text("OEM_CODE:"+str(res_dictionary['oem_code']))
        with col2:
            st.write("DEALER:"+res_dictionary['dealer'])
        with col3:
            st.write("DISTR:"+str(res_dictionary['distr']))
        with col4:
            st.write("FC:"+str(res_dictionary['fc']))
    with c:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("ENGINE_NAME_DESC:"+str(res_dictionary['engine_name_desc']))
        with col2:
            st.write("FAILCODE:"+str(res_dictionary['failcode']))
        with col3:
            st.write("SHOPORDERNUM:"+str(res_dictionary['shopordernum']))
    c.write("CORRECTION:"+str(res_dictionary['cleaned_correction']))

c = st.container()
    
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox(label="Claim ID:", options=claimids_options[togg_switch],index=None,key="claimid_str", on_change = select_claimid, placeholder = '--Select--')

def clear_function():
    st.session_state.conversation = []  # Clear conversation history
    st.session_state.claimid_str = None
    st.session_state.oem_code_str = ""    # Clear the input box
    st.session_state.dealer_str = ""    # Clear the input box
    st.session_state.distr_str = ""    # Clear the input box
    st.session_state.fc_str = ""    # Clear the input box
    st.session_state.engine_name_desc_str = ""    # Clear the input box
    st.session_state.failcode_str = ""    # Clear the input box
    st.session_state.shopordernumber_str = ""    # Clear the input box
    st.session_state.correction_str = ""    # Clear the input box

if st.button("Send"):
    if togg_switch:
        temp_df = pdf_s[pdf_s.claim_id==st.session_state.claimid_str].reset_index(drop=True)
    else:
        temp_df = pdf[pdf.claim_id==st.session_state.claimid_str].reset_index(drop=True)
        
    res_dictionary = json.loads(str(temp_df['document'][0]).lower())
    
    with c:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.text("OEM_CODE:"+str(res_dictionary['oem_code']))
        with col2:
            st.write("DEALER:"+res_dictionary['dealer'])
        with col3:
            st.write("DISTR:"+str(res_dictionary['distr']))
        with col4:
            st.write("FC:"+str(res_dictionary['fc']))
    with c:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("ENGINE_NAME_DESC:"+str(res_dictionary['engine_name_desc']))
        with col2:
            st.write("FAILCODE:"+str(res_dictionary['failcode']))
        with col3:
            st.write("SHOPORDERNUM:"+str(res_dictionary['shopordernum']))
    c.write("CORRECTION:"+str(res_dictionary['cleaned_correction']))

    with st.container():
        col1, col2 = st.columns([0.85,0.15])
        with col1:
            st.write("Zero-Shot Embeddings Pretrained Model : Failure mode:", temp_df['prediction'][0])
        with col2:
            x=float(str(temp_df['probability'][0])[1:-1])*100
            mkd = st.markdown(f"""![](https://geps.dev/progress/{int(x)})""")
        
if st.button("Clear", on_click=clear_function):
    pass

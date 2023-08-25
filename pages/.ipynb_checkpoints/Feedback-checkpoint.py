import streamlit as st
import sys
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pprint
from datetime import datetime
import os
import sys

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def send_message(client,name,email,txt):
    sheet = client.open('Feedbacks').sheet1
    python_sheet = sheet.get_all_records()
    
    current_timestamp = datetime.now()
    row = [name,email,txt,str(current_timestamp)]
    index = len(python_sheet) +2
    sheet.insert_row(row,index)

# Set Streamlit page configuration
st.set_page_config(
    layout="wide"
)

def feedback(client):
    st.header('Feedback')
    st.write('Your feedback matters! Help us enhance your experience by sharing your thoughts. We value your input as it guides us in making improvements. Let us know what you loved and where we can do better. Thank you for being a part of our journey to provide you with the best service possible!')
    
    col1,col2= st.columns(2)
    
    name = col1.text_input('Name*', '')
    email = col2.text_input('Email*', '')
    txt = st.text_area('Message*', '')
    
    if name and txt and is_valid_email(email):
        if st.button('Send'):
            send_message(client,name,email,txt)

def main():
    st.title("StockSpectra")
    
    #Authorize the API
    scope = [
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file'
        ]
    #file_name = 'pages/client_key.json'
    
    dictjson = {'type': st.secrets['googlekeys']['type'],
 'project_id': st.secrets['googlekeys']['project_id'],
 'private_key_id': st.secrets['googlekeys']['private_key_id'],
 'private_key': st.secrets['googlekeys']['private_key'],
 'client_email': st.secrets['googlekeys']['client_email'],
 'client_id': st.secrets['googlekeys']['client_id'],
 'auth_uri': st.secrets['googlekeys']['auth_uri'],
 'token_uri': st.secrets['googlekeys']['token_uri'],
 'auth_provider_x509_cert_url': st.secrets['googlekeys']['auth_provider_x509_cert_url'],
 'client_x509_cert_url': st.secrets['googlekeys']['client_x509_cert_url'],
 'universe_domain': st.secrets['googlekeys']['universe_domain']}
    
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dictjson,scope)
    client = gspread.authorize(creds)
    
    feedback(client)

# Check if the script is being run as the main module
if __name__ == "__main__":
    # Set Streamlit query parameters for refreshing the dashboard
    st.experimental_set_query_params()

    main()

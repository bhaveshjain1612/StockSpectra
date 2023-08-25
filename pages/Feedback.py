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
 'private_key': '-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDae9eLzW+u+Zbp\nMj4hi/K9oAMlaZCjdEqiMl/EzmT4TbAB7SZVmmAZug0nkap0thxwO8u6CETZwlKb\nNvcEejFn2KrKYRD8rD5Ba3BMbTFxxuv3neCNFSI0yB26u+DfINruaQU9HMN602gW\nsexR+B/fZDRaOzfbq2+9BQoMn+FHpQDwftfXHMpG8duByoq2dBsBb6D3PibePUFn\nzFlZwp06tcYKiBTWXR630J7ZhLIYl0B8+1GpZ+OBWS62Etzl+jBMFOyTpYADFnte\nmvJKdumpUiClXyCNuAvy6GAmqNL3LLv1B6fjVh+ALN8MOvRgm7DYuEv1mu/SYWMa\nqUjKm7+hAgMBAAECggEABEVToTuOvBUha9AMB7RYYPgMhQdIqO/4Qy49vIILVmY7\nCKKbM0+tTWYbZTizTreTm+cIqbYz1azh1/+4TumuPGPaHNLOa0iuU9w23wDrF0Nb\n7/nG/q2ufuki2aMGt4iJIbbC/97nfFJpxbh3JeK2NB6dk7OgzntjV7F48OZm194Q\nLmA79OmS1IbsBpAGaLM6j6QXZZvM86KSAPvOLn131PE99BjqgY3QCuRdOasziSvE\n/0vbJj4BZkajpQjexPS8Nt4dr9Eal/bfmsm5ROU+zjnmbyaedIbYeGQ7VfFlzUal\nz8D5bvgPnopyFkH6Ix5nxPILqqT8SYlVEruZ6QbptQKBgQD0oUQ655n4p8YKimz6\nccAevfc8ECLedUVn2ywPFjll7ybhgfICfBQBzzHnW5pYhQzgOr9HEnaa8dI+UiMM\nFLqHzNhd3/KR8jUzYhVB4NagaxhMR5GOSOovKA9+Hjkf6LqFKKjzbPfIbAfTjBp2\nOQvOCUoCjc3zIlpow5GTNYkHdQKBgQDko3l+SD8cZ68xzdcqk3m7gtYOq0jMhxrb\n63vnjg9FgEP11ZjdhSK1ny8jQr8yDU9K2NYfL5UVR/aX/0CaajOeHlo884UnPLuZ\nayVxRolIMxxfVq6GxBpZLEp9yGkW4yBQ3NUIs9FO+1pKfWZWiRq1IPcqRvK4Agzz\nbtqy8Ta9/QKBgCWMFxffcm+K+MenZYCvMujFCYyLgX6Zi1ScfE+4fojZwyL7ufSN\nrNh2P9ed5LvPeCF2guNavx+bHET6gGybReIQG+mUtPuXrHi9hju9UdP/fBRBK+Ml\n5+PjBzW5V9VA+Ff13LC4OfPmOPFMYMdijCBMprJrp3+49x17Xv20Str9AoGBAIL0\ngyTqeoNpe7YaARCY0ZOt968FOjgzdhXaheh1vJeBROusgNb4Z44Bc/1NQLeJWg4z\nrkjEdy5uPnaGs9j91TzOg77/eBemOIlCDnsX/I+G/sw4mNQFxFWpAa2TuWVrh9no\n8nf+jncfjnK16oTMGKkADbGAW4s7WXGg39C4SjN9AoGBAPHKO13gNPmWYMJyh64L\n5fjjsQpN6Sv3FgYGuxPquYnG22xYTRquaq2QYFwV+qvPjR1roDkg5CYNgDDWSvxv\nKqIJ5EHjzYZUgIL2hlXiw9lTqsdrHoiY4xlLLj9yAe2svqNp05qYr+JI8MwPfinA\nmfUC3bH8GHFD54/okWmAQv5g\n-----END PRIVATE KEY-----\n',
 'client_email': 'feedback@stockspectra.iam.gserviceaccount.com',
 'client_id': '103196756199048933131',
 'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
 'token_uri': 'https://oauth2.googleapis.com/token',
 'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',
 'client_x509_cert_url': 'https://www.googleapis.com/robot/v1/metadata/x509/feedback%40stockspectra.iam.gserviceaccount.com',
 'universe_domain': 'googleapis.com'}
    
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dictjson,scope)
    client = gspread.authorize(creds)
    
    feedback(client)

# Check if the script is being run as the main module
if __name__ == "__main__":
    # Set Streamlit query parameters for refreshing the dashboard
    st.experimental_set_query_params()

    main()

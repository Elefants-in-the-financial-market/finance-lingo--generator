import streamlit as st
import requests

API_URL = "https://api-inference.huggingface.co/models/jakobwes/finance-gpt2"
headers = {"Authorization": st.secrets["Authorization"]}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


#st.title('Finance-GPT2')
st.title("Autocomplete your texts in finance lingo")

option = st.selectbox('Text generating method', ('Beam sampling', 'Top p sampling'))

input_txt = st.text_area('Text to complete', '''My ambition for COP26 us to ''')

if st.button('Run'):
	output_txt = query(input_txt)
	st.write(output_txt)




	



import streamlit as st
import requests
import time

API_URL = "https://api-inference.huggingface.co/models/jakobwes/finance-gpt2"
headers = {"Authorization": st.secrets["Authorization"]}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()




#st.title('Finance-GPT2')
st.title("Autocomplete your texts in finance lingo")

st.write("Different sampling methods as [Top-K or Top-P sampling](https://huggingface.co/blog/how-to-generate) can be used. Please use the sliders on the left to control the sampling :) ")

max_length = st.sidebar.slider("Max Length", min_value = 10, max_value=100, value = 50)

top_k = st.sidebar.slider("Top-k", min_value = 0, max_value=500, value = 60)

top_p = st.sidebar.slider("Top-p", min_value = 0.0, max_value=1.0, step = 0.05, value = 0.92)

num_return_sequences = st.sidebar.slider("Number of generated sequences", min_value = 1, max_value=3, step = 1, value = 1)



input_txt = st.text_area('Text to complete', '''My ambition for COP26 is to ''')

if st.button('Run'):
	query_response = query(
		{
			"inputs" : input_txt,
			"parameters" : {
				"top_k" :  top_k,
				"top_p" : top_p,
				"max_new_tokens" : max_length,
				"use_cache": False,
				"num_return_sequences" : num_return_sequences
			}
		}
	)

	if not isinstance(query_response, list):
		if "error" in query_response.keys():
			st.write(query_response)
			if "estimated_time" in query_response.keys():
				my_bar = st.progress(0)
				for pct_complete in range(100):
					time.sleep(query_response["estimated_time"]/100)
					my_bar.progress(pct_complete + 1)
	else:
		for response in query_response:
			st.write(response["generated_text"])




	



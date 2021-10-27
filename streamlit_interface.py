import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


model_saving_dir_path = "jakobwes/my-awesome-model"

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(model_saving_dir_path)
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_saving_dir_path)  
print("Tokenizer and model loaded")


def run_text_completion(input_txt, model, tokenizer, option):
	input_ids = tokenizer.encode(input_txt, return_tensors='pt')

	if option == "Beam sampling":
		beam_outputs = model.generate(
			input_ids, 
			max_length=200, 
			num_beams=1, #3
			no_repeat_ngram_size=2, 
			num_return_sequences=1, #3 
			early_stopping=True
		)
		return tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
	else:
		#	# Top p sampling
		sample_output = model.generate(
		    input_ids, 
		    do_sample=True, 
		    max_length=50, 
		    top_p=0.92, 
		    top_k=0
		)
		return tokenizer.decode(sample_output[0], skip_special_tokens=True)


#st.title('Finance-GPT2')
st.title("Autocomplete your texts in finance lingo")

option = st.selectbox('Text generating method', ('Beam sampling', 'Top p sampling'))

input_txt = st.text_area('Text to complete', '''The report published by ''')

if st.button('Run'):
	output_txt = run_text_completion(input_txt, model, tokenizer, option)
	st.write(output_txt)




	



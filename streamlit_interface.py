import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


model_saving_dir_path = "jakobwes/my-awesome-model"

print("Loading model")
model = AutoModelForCausalLM.from_pretrained(model_saving_dir_path)
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_saving_dir_path)  
print("Tokenizer and model loaded")

@st.cache()
def run_text_completion(input_txt, model, tokenizer):
	input_ids = tokenizer.encode(input_txt, return_tensors='pt')

	
	# Beam-sampling
	beam_outputs = model.generate(
	    input_ids, 
	    max_length=200, 
	    num_beams=1, #3
	    no_repeat_ngram_size=2, 
	    num_return_sequences=1, #3 
	    early_stopping=True
	)

	# now we have 3 output sequences
	return tokenizer.decode(beam_outputs[0], skip_special_tokens=True).lower()
	#for i, beam_output in enumerate(beam_outputs):
	#  print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))

	# tokenizer.decode(beam_outputs[0], skip_special_tokens=True)


#st.title('Finance-GPT2')
st.title("Autocomplete your texts in finance lingo")

input_txt = st.text_area('Text to complete', '''The report published by ''')

if st.button('Run'):
	st.write(run_text_completion(input_txt, model, tokenizer))




	

#	# Top p sampling
#	sample_output = model.generate(
#	    input_ids, 
#	    do_sample=True, 
#	    max_length=200, 
#	    top_p=0.92, 
#	    top_k=0
#	)
#
#
#	return tokenizer.decode(sample_output[0], skip_special_tokens=True)

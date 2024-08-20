import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from peft import PeftModel, PeftConfig

# Introduction and Disclaimer
st.title("Edge Device Content Moderation")
st.write("""
This application uses a fine-tuned **Phi-3-mini-4k-instruct** model for content moderation.
The data used to train this model was generated using the **Llama 3.1 70B** model. This application is designed to work entirely on edge devices, without requiring internet access.
""")

# Load the CSV to get the sentiment classes
filename = "synData.csv"
df = pd.read_csv(filename, names=["sentiment", "text"], encoding="utf-8", encoding_errors="replace")

# Get unique sentiments from the CSV file
sentiments = sorted(df.sentiment.unique())
sentiment_string = "\n ".join(sentiments)

# Disclaimer about the classes
st.write(f"### Disclaimer: The bot is able to detect the following text categories: {', '.join(sentiments)}")

# Function to build the pipeline for the base model and cache it
@st.cache_resource
def build_base_pipeline():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, max_seq_length=2048)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create the pipeline
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=3, temperature=0.0)
    return pipe

# Function to build the pipeline for the fine-tuned model and cache it
@st.cache_resource
def build_finetuned_pipeline():
    model_name = "microsoft/Phi-3-mini-4k-instruct"  # Use the base model that was fine-tuned
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load the adapter configuration
    adapter_dir = "trained-model"
    adapter_config = PeftConfig.from_pretrained(adapter_dir)
    
    # Load the adapter model and combine it with the base model
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create the pipeline
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=30)
    return pipe

# Build both pipelines (each runs only once)
base_pipe = build_base_pipeline()
finetuned_pipe = build_finetuned_pipeline()

# Streamlit Chat Interface
st.write("### Enter text to analyze:")

# User input
user_input = st.text_input("Your text here...")

def generate_test_prompt(text):
    return f"""The category of the following phrase: '{text}' is \n{sentiment_string}\nCannot be determined\n\nSolution: The correct option is""".strip()

if user_input:
    # Generate the prompt
    prompt = generate_test_prompt(user_input)
    
    # Analyze using the base model
    with st.spinner("Analyzing category with base model..."):
        base_result = base_pipe(prompt, pad_token_id=base_pipe.tokenizer.eos_token_id)
        base_answer = base_result[0]['generated_text'].split("The correct option is")[-1].strip().lower()
        
        # Match the answer to the sentiment classes
        base_sentiment_detected = "none"
        for sentiment in sentiments:
            if sentiment in base_answer:
                base_sentiment_detected = sentiment
                break
        
        st.write(f"### Detected text category by base model: **{base_sentiment_detected.capitalize()}**")
    
    # Analyze using the fine-tuned model
    with st.spinner("Analyzing category with fine-tuned model..."):
        finetuned_result = finetuned_pipe(prompt, pad_token_id=finetuned_pipe.tokenizer.eos_token_id)
        finetuned_answer = finetuned_result[0]['generated_text'].split("The correct option is")[-1].strip().lower()
        
        # Match the answer to the sentiment classes
        finetuned_sentiment_detected = "none"
        for sentiment in sentiments:
            if sentiment in finetuned_answer:
                finetuned_sentiment_detected = sentiment
                break
        
        st.write(f"### Detected text category by fine-tuned model: **{finetuned_sentiment_detected.capitalize()}**")

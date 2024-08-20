import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd

# Introduction and Disclaimer
st.title("Edge Device Content Moderation Bot")
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

# Function to build the pipeline and cache it
@st.cache_resource
def build_pipeline():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, max_seq_length=2048)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create the pipeline
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=3, temperature=0.0)
    return pipe

# Build the pipeline (runs only once)
pipe = build_pipeline()

# Streamlit Chat Interface
st.write("### Enter text to analyze:")

# User input
user_input = st.text_input("Your text here...")

def generate_test_prompt(text):
    return f"""The category of the following phrase: '{text}' is \n{sentiment_string}\nCannot be determined\n\nSolution: The correct option is""".strip()

if user_input:
    with st.spinner("Analyzing sentiment..."):
        # Generate the prompt
        prompt = generate_test_prompt(user_input)
        
        # Get the model's prediction
        result = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id)
        answer = result[0]['generated_text'].split("The correct option is")[-1].strip().lower()
        
        # Match the answer to the sentiment classes
        sentiment_detected = "none"
        for sentiment in sentiments:
            if sentiment in answer:
                sentiment_detected = sentiment
                break

        # Display the result
        st.write(f"### Detected text category by bare model: **{sentiment_detected.capitalize()}**")



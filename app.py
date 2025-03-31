import os
from dotenv import load_dotenv

from langchain_community.llms import HuggingFaceHub
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Disable the email collection prompt
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load environment variables
load_dotenv()

# Debug information
st.sidebar.write("Debug Info:")
st.sidebar.write("Environment Variables:")

# Get environment variables
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT", "insurance-bot")

# Debug information
st.sidebar.write("\nEnvironment Variables Status:")
st.sidebar.write(f"HUGGINGFACE_API_KEY: {'Present' if huggingface_api_key else 'Not Present'}")
st.sidebar.write(f"LANGCHAIN_API_KEY: {'Present' if langchain_api_key else 'Not Present'}")
st.sidebar.write(f"LANGCHAIN_PROJECT: {'Present' if langchain_project else 'Not Present'}")

## Langsmith Tracking
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = langchain_project

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant which helps with advising insurance policies. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title("Insurance Bot")

# Check if HuggingFace API key is available
if not huggingface_api_key:
    st.error("""
    HuggingFace API key not found. Please check your Streamlit Cloud deployment settings:
    1. Go to your app's settings
    2. Click on 'Secrets'
    3. Delete all existing secrets
    4. Add these environment variables:
    ```toml
    HUGGINGFACE_API_KEY = "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"
    LANGCHAIN_API_KEY = "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"
    LANGCHAIN_PROJECT = "insurance_bot"
    ```
    5. Make sure to:
       - Have a space after the equals sign
       - Use straight quotes (") not curly quotes
       - Include the entire token including 'hf_' prefix
    6. After adding the variables:
       - Click "Save"
       - Go back to your app
       - Click "Redeploy"
    """)
    st.stop()

input_text=st.text_input("What question you have in mind?")

## HuggingFace model
try:
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        task="text2text-generation",
        model_kwargs={
            "temperature": 0.7,
            "max_length": 128,
            "repetition_penalty": 1.2
        },
        huggingfacehub_api_token=huggingface_api_key,
        client_options={"timeout": 60}  # Increase timeout for inference API
    )
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser

    if input_text:
        with st.spinner('Thinking...'):
            response = chain.invoke({"question":input_text})
            st.write(response)
except Exception as e:
    st.error(f"Error initializing the model: {str(e)}")
    st.info("""
    If you're still seeing errors, try these alternatives:
    1. Use a different model:
       - "facebook/opt-125m"
       - "distilgpt2"
       - "gpt2"
    2. Try using the OpenAI API instead
    3. Try using a different HuggingFace endpoint
    """)


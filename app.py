from openai import OpenAI
import streamlit as st
from transformers import pipeline
import pandas as pd

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

openai_api_key ='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'

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")

# uploaded_file = st.file_uploader('Upload a file')
# df = pd.read_csv(uploaded_file)

def generate_markdown_table(emotion_data):
    result = []
    # Header
    result.append("|      Emotion      |     Score     | Progress Bar |")
    result.append("|:-----------------:|:-------------:|:------------:|")

    # Body
    for entry in emotion_data:
        label = entry['label']
        score = entry['score']
        progress_bar = '█' * int(score * 10) + '░' * (10 - int(score * 10))

        # Append the table row to the result list
        result.append(f"| {label:<17} | {score:.11f} | {progress_bar} |")

    # Join the result list into a string with newline characters
    return '\n'.join(result)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Please input prompt for rewriting."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    client = OpenAI(api_key=openai_api_key)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})

    emotions_text = classifier([prompt])[0]

    st.session_state.messages.append({"role": "assistant", "content": emotions_text})
    st.chat_message("assistant").write(msg)
    st.chat_message("assistant").write(generate_markdown_table(emotions_text))
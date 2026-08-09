import streamlit as st

from langgrpah_backend import chatbot
from langchain_core.messages import HumanMessage


# -----------------------------
# Configuration
# -----------------------------

CONFIG = {
    "configurable": {
        "thread_id": "thread_1"
    }
}


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Llama 3.2 Chatbot",
    page_icon="🤖"
)

st.title("🤖 Llama 3.2 Chatbot")


# -----------------------------
# Initialize Session State
# -----------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# -----------------------------
# Display Previous Messages
# -----------------------------

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.text(message["content"])


# -----------------------------
# Chat Input
# -----------------------------

user_input = st.chat_input("Type here...")


if user_input:

    # Save user message
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.text(user_input)

    # Send message to LangGraph
    response = chatbot.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=CONFIG
    )

    # Get AI response
    ai_message = response["messages"][-1].content

    # Save AI message
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )

    # Display AI response
    with st.chat_message("assistant"):
        st.text(ai_message)
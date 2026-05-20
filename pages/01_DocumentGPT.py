import time
import streamlit as st

st.title("Document GPT")

# Initialize session state for messages
if "messages" not in st.session_state:
  st.session_state["messages"] = []

# Function to send a message and save it to session state
def send_message(message, role, save=True):
  # Display the message in the chat interface
  with st.chat_message(role):
    st.write(message)
  # Save the message to session state if save is True
  if save:
    st.session_state["messages"].append({"message": message, "role": role})

# Load previous messages from session state and display them (save=False to avoid duplication)
for message in st.session_state["messages"]:
  send_message(message["message"], message["role"], save=False)

message = st.chat_input("Type your message here...")

# Process the user's message
if message:
  # save=True to save the user's message to session state
  send_message(message, "human")
  time.sleep(3)
  send_message(f"You said: {message}", "ai")

  with st.sidebar:
    st.write(st.session_state)



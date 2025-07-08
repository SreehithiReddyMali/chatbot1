# chatbot_app.py
import streamlit as st

# Simple rule-based response function
def get_bot_response(user_input):
    user_input = user_input.lower()
    if "hello" in user_input or "hi" in user_input:
        return "Hello! How can I help you today?"
    elif "how are you" in user_input:
        return "I'm doing great! How can I assist you?"
    elif "bye" in user_input:
        return "Goodbye! Talk to you soon."
    else:
        return "Hmm, I don't understand that. Try saying something else."

# App UI
st.title("💬 Chatbot with Streamlit")
st.write("Type something to chat with the bot.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("You:", key="input")

# Generate and store responses
if user_input:
    response = get_bot_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display conversation
for sender, message in st.session_state.chat_history:
    st.markdown(f"**{sender}:** {message}")

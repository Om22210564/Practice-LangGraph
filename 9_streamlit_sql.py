import streamlit as st
from L_9_langbackendDB import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid


## utility function

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)
         
def load_conv(thread_id):
    return chatbot.get_state({'configurable': {'thread_id': thread_id}}).values.get('messages', [])

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []




if 'chat_thread' not in st.session_state:
    st.session_state['chat_thread'] = retrieve_all_threads()

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

add_thread(thread_id=st.session_state['thread_id'])



# st.session_state -> dict -> 
CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}
############## Side Bar Ui 

st.sidebar.title("Langgraph chatbot")

if st.sidebar.button("New chat"):
    reset_chat()

st.sidebar.header("My conversations")

for thread_id in st.session_state['chat_thread']:
    if st.sidebar.button(thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conv(thread_id)
        temp_messages = []
        for message in messages:
            if isinstance(message,HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role,'content':message.content})
        st.session_state['message_history'] = temp_messages

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        ai_message = st.write_stream(message_chunk.content for message_chunk,metadata in chatbot.stream(
            {'messages': [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages")
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
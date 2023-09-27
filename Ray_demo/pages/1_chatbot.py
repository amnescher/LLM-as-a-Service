import streamlit as st
import requests

from langchain.document_loaders import YoutubeLoader
import os
import streamlit as st
import requests
import json
import time
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

BASE_URL = "http://localhost:5000"


def add_user(username):
    endpoint = "/add_user/"
    url = BASE_URL + endpoint
    data = {"username": username}  # Provide the necessary data
    response = requests.post(url, json=data)
    return response.json()


def retrieve_all_conversations(username):
    endpoint = "/retrieve_all_conversations/"
    url = BASE_URL + endpoint
    data = {"username": username}
    response = requests.get(url, json=data)
    return response.json()


def check_user_existence(username):
    endpoint = "/check_user_existence/"
    url = BASE_URL + endpoint
    data = {"username": username}
    response = requests.get(url, json=data)
    return response.json()

def get_user_tokens():
    endpoint = "/get_user_tokens/"
    url = BASE_URL + endpoint
    response = requests.get(url)
    return response.json()

def retrieve_conversation(username, conversation_number):
    endpoint = "/retrieve_conversation/"
    url = BASE_URL + endpoint
    data = {"username": username, "conversation_number": conversation_number}
    response = requests.post(url, json=data)
    return response.json()


def add_conversation(username, content):
    endpoint = "/add_conversation/"
    url = BASE_URL + endpoint
    data = {"username": username, "content": json.dumps(content)}
    response = requests.post(url, json=data)
    return response.json()


# Function to create buttons based on the username
def create_buttons(username):
    buttons = retrieve_all_conversations(username)
    # Add a default "New Chat" choice with an empty value
    if buttons > 0:
        return list(range(1, int(buttons) + 1))
    else:
        return None

port = 5001
pdf_directory = "./PDF_dir"
if not os.path.exists(pdf_directory):
    os.makedirs(pdf_directory)
    print(f"Directory '{pdf_directory}' created successfully.")


def clear_chat_history():
    PORT_NUMBER = 8000
    search_choice = "Cleaning memory"
    URL = f"http://localhost:{PORT_NUMBER}/predict/?text=&mode={search_choice}&messages={st.session_state.messages}"
    requests.post(URL)


def clear_videos():
    PORT_NUMBER = 8000
    Video_URL = ""
    data_type = "Video"
    mode = "clear"
    POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={Video_URL}&mode={mode}"
    requests.post(POST_URL)
    st.sidebar.success = "Cleared database."


def clear_document():
    PORT_NUMBER = 8000
    Video_URL = ""
    data_type = "Document"
    mode = "clear"
    POST_URL = f"http://localhost:{PORT_NUMBER}/VectoreDataBase/?data_type={data_type}&data_path={Video_URL}&mode={mode}"
    requests.post(POST_URL)
    st.sidebar.success = "Cleared database."


def get_collections():
    PORT = 8000
    data_type = "Collection"
    mode = "get_all"
    POST_URL = (
        f"http://localhost:{PORT}/VectoreDataBase/?data_type={data_type}&mode={mode}"
    )
    response = requests.post(POST_URL)

    collections_data = response.json()
    # selected_collections = st.multiselect("Select Collections:", response)
    collections_list = collections_data.get("collections", [])
    return collections_list


if "username" not in st.session_state or st.sidebar.button("Logout"):
    # Login form
    if "username" not in st.session_state:
        username = st.text_input("Enter your username:")
    else:
        username = st.session_state.username

    if username == "admin":
        password = st.text_input(
            "Enter your password:", type="password"
        )  # Add password field
        if st.button("Login"):
            if (
                password == "adminpassword"
            ):  # Replace "adminpassword" with the actual admin password
                st.session_state.username = "admin"
                st.session_state.show_logged_in_message = True
            else:
                st.error("Invalid username or password")
    elif check_user_existence(username)["user_exists"]:
        if st.button("Login"):
            st.session_state.show_logged_in_message = True
            st.session_state.username = username
    else:
        st.error("Invalid username")

else:
    if "show_logged_in_message" not in st.session_state:
        st.session_state.show_logged_in_message = False

    if st.session_state.show_logged_in_message:
        logedin_username = st.session_state.username
        if logedin_username == "admin":
            # Display the form to add new users
            new_user = st.text_input("Enter a new user:")
            if st.button("Add User"):
                # Add the new user to the list of valid users
                add_user(new_user)
                st.success("New user added successfully!")
            if st.button("Show Data"):
                # Get the tokens
                tokens = get_user_tokens()
                st.table(tokens)

        else:
            button_labels = create_buttons(st.session_state.username)
            if button_labels == None:
                history = StreamlitChatMessageHistory(key="langchain_messages")
                history.clear()
                history.add_ai_message("How can I help you?")
                ingest_to_db = messages_to_dict(history.messages)
                add_conversation(st.session_state.username, ingest_to_db)
            button_labels = create_buttons(st.session_state.username)


            # Display the buttons
            st.sidebar.markdown("---")
            st.sidebar.markdown("<br>", unsafe_allow_html=True)
            st.sidebar.subheader("Previous Chats:")

            # Add a new chat button
            if st.sidebar.button("New Chat"):
                history = StreamlitChatMessageHistory(key="langchain_messages")
                history.clear()
                history.add_ai_message("How can I help you?")
                ingest_to_db = messages_to_dict(history.messages)
                add_conversation(st.session_state.username, ingest_to_db)
                button_labels = create_buttons(st.session_state.username)
                

            
            selected_button_label = st.sidebar.radio(
                    "Select a conversation:",
                    button_labels,
                    index=len(button_labels) - 1,
                )

            # st.sidebar.markdown("<br>", unsafe_allow_html=True)
            st.sidebar.markdown("---")
            st.sidebar.markdown("<br>", unsafe_allow_html=True)

            search_choice = st.sidebar.radio(
                options=["AI Assistance", "Document Search"], label="Type of search"
            )

            if selected_button_label:
                st.session_state.newchat = "False"
                st.session_state.conversation_number = selected_button_label
                st.empty()  # Clear the page content
                selected_value = retrieve_conversation(
                    st.session_state.username, selected_button_label
                )["content"]
                history = StreamlitChatMessageHistory(key="langchain_messages")
                history.clear()
                if selected_value:
                    retrieve_from_db = json.loads(selected_value)
                    retrieved_messages = messages_from_dict(retrieve_from_db)
                    history.add_message(retrieved_messages)
                    for msg in history.messages:
                        for agent in msg:
                            st.chat_message(agent.type).write(agent.content)
                st.session_state.messages = [
                    {"role": "assistant", "content": "let's continue our chat"}
                ]

            # st.image("Eschercloud.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

            if search_choice == "Document Search":
                collection_list = get_collections()
                selected_collection = st.sidebar.selectbox(
                    "select collection", collection_list
                )
            else:
                selected_collection = "None"

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if prompt := st.chat_input():  # (disabled=not replicate_api):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        PORT_NUMBER = 8000
                        username = st.session_state.username
                        conversation_number = st.session_state.conversation_number
                        newchat = st.session_state.newchat
                        URL = f"http://localhost:{PORT_NUMBER}/predict/?text={prompt}&mode={search_choice}&username={username}&newchat={newchat}&conversation_number={conversation_number}&collection={selected_collection}&messages={st.session_state.messages}"
                        response = requests.post(URL)
                        response = response.content.decode()
                        if response:
                            placeholder = st.empty()
                            full_response = response
                            placeholder.markdown(full_response)

                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)
# Adding New Chat button to the code
# fixing logout button so when you log out to land to the first page.

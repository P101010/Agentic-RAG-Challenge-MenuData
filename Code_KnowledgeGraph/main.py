import streamlit as st
from openai import OpenAI
from LangchainActions import check_query, extract_info, create_history, run
from Embeddings import get_embedding, extractData_loadData_performSemanticSearch
from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage

st.title("Menudata")

# Initialize chat history
if "messages" not in st.session_state:
    # print("Creating session state")
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

max_messages = 5
history = create_history(st.session_state.messages, max_messages=max_messages)
#history = create_history(st.session_state.messages)
# Accept user input
if prompt := st.chat_input("Ready to get your foody sense tingled?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
                # Step 0 : Toxic Language Guardrail
            guard = Guard().use_many(ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail=OnFailAction.EXCEPTION))
            try:
                guard.validate(prompt) 
                check_flag_response = ''
            except Exception as e:
                check_flag_response = 'Validation Failed: Inappropriate language use'

            if 'Validation Failed' not in check_flag_response:
                # Step 1 : Check if query in context
                check_flag_response = check_query(prompt, history.messages)

            if check_flag_response == str(True):
                response = run(prompt, history.messages)
            else:
                response = check_flag_response
                
            st.markdown(response)
    history.add_user_message(prompt)
    history.add_ai_message(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    
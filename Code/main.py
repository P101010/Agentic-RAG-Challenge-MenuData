import streamlit as st
from openai import OpenAI
from LangchainActions import check_query, tierOne, classify_query, extract_info, result_nl, split_query, create_history
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

            guard = Guard().use_many(ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail=OnFailAction.EXCEPTION))
            try:
                guard.validate(prompt) 
                check_flag_response = ''
            except Exception as e:
                check_flag_response = 'Validation Failed: Inappropriate language use'

            if 'Validation Failed' not in check_flag_response:
                #   Step 0 : Check if query in context
                check_flag_response = check_query(prompt, history.messages)


            if check_flag_response == str(True):

                # Step 1 : Text Classification
                query_class = classify_query(prompt, history.messages)

                # Step 2
                if query_class == 'Tier 1':
                    response = tierOne(prompt, history.messages)
                elif query_class == 'Tier 2':
                    info = extract_info(prompt, history.messages)
                    context = extractData_loadData_performSemanticSearch(info, prompt, history.messages)
                    response = result_nl(context, prompt, history.messages)
                elif query_class == 'Tier 3':
                    queries = split_query(prompt, history.messages)
                    for query in queries:     
                        query_cl = classify_query(query)
                        response_one, response_two = '',''
                        if query_cl == 'Tier 1':
                            response_one = tierOne(prompt, history.messages)
                        elif query_cl == 'Tier 2':
                            info = extract_info(prompt, history.messages)
                            context = extractData_loadData_performSemanticSearch(info, prompt, history.messages)
                            response_two = result_nl(context, prompt, history.messages)    
                    response = response_one + '\n\n' + response_two
                else:
                    response = 'Failed to classify query to any class'
            else:
                response = check_flag_response

            st.markdown(response)
    history.add_user_message(prompt)
    history.add_ai_message(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    
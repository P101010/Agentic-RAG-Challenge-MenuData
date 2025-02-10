import os
import re
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ChatMessageHistory
from operator import itemgetter
from Example import get_example_selector
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph

load_dotenv()

db_user = os.getenv('db_user')
db_password = os.getenv('db_password')
db_host = os.getenv('db_host')
db_name = os.getenv('db_name')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["NEO4J_URI"] = os.getenv["NEO4J_URI"]
os.environ["NEO4J_USERNAME"] = os.getenv["NEO4J_USERNAME"]
os.environ["NEO4J_PASSWORD"] = os.getenv["NEO4J_PASSWORD"]


# A helper method to execute llm chains
def llm_response(template, invoke_chain):

    llm = ChatOpenAI(model="gpt-4", temperature=0) 
    chain = LLMChain(llm=llm, prompt=template)
    response = chain.run(invoke_chain)

    return response


# Extracts name of dish or ingredient
def extract_info(user_query, history=[]):
    
    prompt_template = PromptTemplate(
        input_variables=["history", "user_query"],
        template='''You know that the query below is asking information about some dish or ingredient. Identify the ingredient or dish name and return only that value.
        Examples -
        Tell me about biryani. Biryani 
        What is the history of sushi? Sushi
        Tell me about Saffron. Saffron
        Below is the query for you to identify and includes chat history if any,
        Note - Add a underscore if multiword 
        
        history - {history}
        query - {user_query}'''
    )

    response = llm_response(template = prompt_template, invoke_chain = {"history": history, "user_query": user_query})
 
    return response


# Checks if query is in context or not
def check_query(user_query, history=[]):

    prompt_template = PromptTemplate(
        input_variables=["history", "user_query"],
        template='''My RAG application or chatbot is for answering questions related to restaurants, food, dishes, history of food, how its made and many more things about food. So basically a one stop solution to find your favourite food or 
        to know more about it. 

        So you are the gatekeeper checking if the query given by the user is in context of what we are trying to solve or anything else.

        If query is in context just return True if out of context return a funny response stating we might be able to answer that in future. Make sure you refer the history for previous 
        queries and responses to decide on the context. If you observe the user asking the same out of context question let him know about it in a funny way.

        If query is asking question about restaurants in any other place than the state of Claifornia and city of San Fransisco then also it is concidered out of context and
        inform user we only operate in these locations right now. Note - Only mention this when user mentions about other state or city

        Note -  Only return True that is one word for in context queries  
        Make sure you return Only funny message as a response if out of context no need to explain anything else

        But if the query is thankfull or is trying to strike a conversation be appreciative and ask how you can help.

        history - {history}

        Make sure you take history only for context and focus on the query below more

        query - {user_query}'''
    )

    response = llm_response(template = prompt_template, invoke_chain = {"history": history, "user_query": user_query})

    return response

def create_history(messages, max_messages):
    history = ChatMessageHistory()

    if max_messages is not None:
        messages = messages[-max_messages:]

    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history


def generateCypher_and_Insert(context, user_query, graph, history=[]):
        graph.refresh_schema
        prompt_template = PromptTemplate(
        input_variables=["context", "user_query", "graph_schema"],
        template='''You are an expert in creating Cypher queries. Your current task is to observe the current state of graph schema which includes some details of restaurants and 
        there dishes served, If you are being invoked its because the user probably need smore info on some dish or ingredient, The assumption is the dish is already present in menuitem or ingredient is already present in Ingredient so create a node relevant to why this context was retrieved whohc you might find in the user query whihc has a relationship with the but make sure to be case in-sensetive. 
        So based on the current state of schema given below, come up with statements that will insert the context given into this grpah db. Observe why this relevant context was retrieved as well based on user query
        Note - Dont give any explainations, give only cypher queries.
        example - 
        user query: What is the history of sushi?
        bot: MATCH (m:MenuItem) 
                WHERE toLower(m.name) CONTAINS 'sushi'
                SET m += (#use curly braces
                history: 'The earliest written mention of sushi in English described in the Oxford English Dictionary is in an 1893 book', 
                source: 'https://en.wikipedia.org/wiki/Sushi' 
                )# use curly braces only

        Schema:{graph_schema}

        User Query:{user_query}
        
        Context:{context}'''
        )

        response = llm_response(template = prompt_template, invoke_chain = {"context": context, "user_query": user_query, 'graph_schema': graph.schema})
    
        graph.query(response)
    
    

# Function to format examples into a string
def format_examples(selected_examples):
    formatted_examples = ""
    for example in selected_examples:
        formatted_examples += f"Question: {example['input']}\nQuery: {example['query']}\n\n"
    return formatted_examples


def run(question, history = []):
    graph = Neo4jGraph()
    graph.refresh_schema()

    flag = check_query(graph.schema, question, history)
    
    if flag == 'True':
        # Chain with few shot examples
        chain = GraphCypherQAChain.from_llm(
            ChatOpenAI(temperature=0),
            graph=graph,
            verbose=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            allow_dangerous_requests=True,
            example_selector=example_selector,
            format_examples_func=format_examples,
        )

        # Initialize the example selector
        example_selector = get_example_selector()

        # Select relevant examples based on the user's query
        selected_examples = example_selector.select_examples({"input": question})

        # Format the selected examples into a string
        formatted_examples = format_examples(selected_examples)

        # Invoke the chain with the schema, question, and formatted examples

        response = chain.invoke(
            {
                "schema": graph.schema,
                "query": question,
                "examples": formatted_examples,
                "history": history
            }
        )

        return response
    else:
        context = extractData_loadData_performSemanticSearch(info=extract_info(question, history), user_query=question)
        generateCypher_and_Insert(context, question, graph)
        return run(question, history)
        

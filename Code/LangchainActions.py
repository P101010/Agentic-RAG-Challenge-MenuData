import os
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

load_dotenv()

db_user = os.getenv('db_user')
db_password = os.getenv('db_password')
db_host = os.getenv('db_host')
db_name = os.getenv('db_name')
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# A helper method to execute llm actions
def llm_response(template, invoke_chain):

    llm = ChatOpenAI(model="gpt-4", temperature=0) 
    chain = LLMChain(llm=llm, prompt=template)
    response = chain.run(invoke_chain)

    return response

# Classifies user query into one of 3 classes
def classify_query(user_query, history=[]):
    
    prompt_template = PromptTemplate(
        input_variables=["history", "user_query"],
        template='''I have a scenario where I want to classify query into a category.

        Tier 1: Questions related to restaurants, finding food, or food trends.
        Examples:
        Which restaurants in San Francisco offer dishes with Impossible Meat? (Label - Tier 1)
        Give me a summary of the latest trends around desserts. (Tier 1)
        Which restaurants are known for sushi? (Tier 1)
        Compare the average menu price of vegan restaurants in LA vs. Mexican restaurants. (Tier 1)
        Which food can I find with peas?

        Tier 2: Questions about a dish or ingredients.
        Examples:
        Tell me about biryani. (Tier 2)
        What is the history of sushi? (Tier 2)
        Tell me the contents of sushi. (Tier 2)
        
        Tier 3: Questions that combine both restaurant-related and dish-related queries.
        Example:
        What is the history of sushi, and which restaurants in my area are known for it?

        Tell me which class the query falls into with just one word that is the class. Note – might include chat history which should not be considered for classification but only for context

        history - {history}
        query - {user_query}'''
    )

    response = llm_response(template = prompt_template, invoke_chain = {"history": history, "user_query": user_query})
    
    return response

# Generates SQl query based on user query, queries database and return result in natural language
def tierOne(user_query, history=[]):
    
    select_table = ['menu_items', 'restaurants']
    db = SQLDatabase.from_uri(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}", include_tables = select_table)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=get_example_selector(),
    input_variables=["input", 'top_k'],
    )

    final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a PostgreSQL expert. Given an input question, create a syntactically correct PostgreSQL query to run. Unless otherwise specificed.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries."),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
    )

    answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Chat History:{history}
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
    )

    generate_query = create_sql_query_chain(llm, db,final_prompt) 
    execute_query = QuerySQLDataBaseTool(db=db)

    rephrase_answer = answer_prompt | llm | StrOutputParser()

    chain = (
    RunnablePassthrough.assign(query=generate_query).assign(
        result=itemgetter("query") | execute_query
    )
    | rephrase_answer )

    return chain.invoke({'question':user_query, 'history':history})


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


# Returns results to user query based on context passed
def result_nl(context, user_query, history = []):
    
    prompt_template = PromptTemplate(
        input_variables=["context", "history", "user_query"],
        template='''Using relevant context passed answer the user query. If history of chat availableuse it for relevant context. Note - After answering the query provide relevant source
        Relevant Context - {context}
        history - {history}
        query - {user_query}'''
    )

    response = llm_response(template = prompt_template, invoke_chain = {"context": context, "history": history, "user_query": user_query})

    return response


# Splits user query in class 3 into class 1 and 2
def split_query(user_query, history=[]):

    prompt_template = PromptTemplate(
        input_variables=["history", "user_query"],
        template='''Tier 1: Questions related to restaurants, finding food, or food trends. 
        Examples: 
        Which restaurants in San Francisco offer dishes with Impossible Meat? (Label - Tier 1) 
        Give me a summary of the latest trends around desserts. (Tier 1) 
        Which restaurants are known for sushi? (Tier 1) 
        Compare the average menu price of vegan restaurants in LA vs. Mexican restaurants. (Tier 1) 

        Tier 2: Questions about a dish or ingredients. 
        Examples: 
        Tell me about biryani. (Tier 2) 	
        What is the history of sushi? (Tier 2) 
        Tell me the contents of sushi. (Tier 2)

        The query is a combination of both these classes.

        Break down the query I am passing to you below into Tier 1 and Tier 2 questions (with proper context included in both) in the same order, and separate them with commas.

        Example: 
        What is the history of sushi, and which restaurants are known for it?  
        Which restaurants are known for sushi?, What is the history of sushi?

        Dont include class names in split.

        history - {history}
        query - {user_query}'''
    )

    response = llm_response(template = prompt_template, invoke_chain = {"history": history, "user_query": user_query})

    return response.split(',')


# Checks if query is in context or not
def check_query(user_query, history=[]):

    prompt_template = PromptTemplate(
        input_variables=["history", "user_query"],
        template='''My RAG application or chatbot is for answering questions related to restaurants, food, dishes. So basically a one stop solution to find your favourite food or 
        to know more about it. 

        So you are the gatekeeper checking if the query given by the user is in context of what we are trying to solve or anything else.

        If query is in context just return True if out of context return a funny response stating we might be able to answer that in future. Make sure you refer the history for previous 
        queries and responses to decide on the context. If you observe the user asking the same out of context question let him know about it in a funny way.

        Note -  Only return True that is one word for in context queries  
        Make sure you return Only funny message as a response if out of context no need to explain anything else

        history - {history}
        query - {user_query}'''
    )

    response = llm_response(template = prompt_template, invoke_chain = {"history": history, "user_query": user_query})

    return response

def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

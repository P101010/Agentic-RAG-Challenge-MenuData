import openai
import wikipediaapi
import chromadb
from chromadb.config import Settings
import streamlit as st
from chromadb.utils import embedding_functions
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from dotenv import load_dotenv
import os

load_dotenv()

db_user = os.getenv('db_user')
db_password = os.getenv('db_password')
db_host = os.getenv('db_host')
db_name = os.getenv('db_name')

# Creates embedddings of text
def get_embedding(text):

    client = openai.OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding 


# Creates chunks, calls get-embedding helper method, stores embedddings in vectorDB and performs a semantic 
# search on it based on user query and returns relevant documents
def extractData_loadData_performSemanticSearch(info, user_query, history=[]):

    query = f"Select count(*) from metadata where dish_name ILIKE '{info}'"
    insert_query = f"Insert into metadata Values('{info}')"
    db = SQLDatabase.from_uri(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}/{db_name}")
    execute_query = QuerySQLDataBaseTool(db=db)
    count = execute_query.invoke(query)

    if count == '[(0,)]':
        # Initialize Wikipedia API
        wiki_wiki = wikipediaapi.Wikipedia(user_agent='Menudata.ai', language='en')

        # Fetch the page
        page_py = wiki_wiki.page(info)

        # Check if the page exists
        if not page_py.exists():
            return False

        # Function to create chunks from sections and summary
        def create_chunks_from_page(page, dish_name):
            chunks = []

            # Add the page summary as a chunk
            summary_chunk = {
                "text": f"{page.summary[:200]}\n\nSource: {page.canonicalurl}",  # Append URL
                "metadata": {
                    "dish_name": dish_name,
                    "section": "Summary",
                }
            }
            chunks.append(summary_chunk)

            # Recursively add sections and subsections as chunks
            def add_sections_to_chunks(sections, parent_section=None, level=0):
                for s in sections:
                    # Create a chunk for the section
                    section_title = f"{parent_section} - {s.title}" if parent_section else s.title
                    section_chunk = {
                        "text": f"{s.text[:200]} Source: {page.canonicalurl}",  # Append URL
                        "metadata": {
                            "dish_name": dish_name,
                            "section": section_title,
                        }
                    }
                    chunks.append(section_chunk)

                    # Add subsections recursively, passing the current section as the parent
                    add_sections_to_chunks(s.sections, parent_section=s.title, level=level + 1)

            # Start processing sections
            add_sections_to_chunks(page.sections)

            return chunks

        # Create chunks from the page
        chunks = create_chunks_from_page(page_py, info)
        print("Chunks created!")

        # Generate embeddings for metadata only
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            # Convert metadata to a string (e.g., JSON or concatenated string)
            metadata_str = f"{chunk['metadata']['dish_name']} {chunk['metadata']['section']}"
            
            # Generate embedding for the metadata string
            metadata_embedding = get_embedding(metadata_str)
            
            chunk_embeddings.append({
                "id": f"chunk_{info}{i+1}",  # Unique ID for each chunk
                "vector": metadata_embedding,  # Embedding of the metadata
                "metadata": chunk["metadata"],  # Original metadata
                "text": chunk["text"]  # Optional: Store text if needed for retrieval
            })
        print("Embeddings created!")
    
    # Initialize Chroma client
    chroma_client = chromadb.Client()

    # Create a collection (similar to a table in a traditional DB)
    collection_name = 'ExternalInformation'
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        collection = chroma_client.create_collection(name=collection_name)

    if count == '[(0,)]':
        # Add embeddings to the collection
        for chunk in chunk_embeddings:
            collection.add(
                ids=[chunk["id"]],  # Unique ID for each chunk
                embeddings=[chunk["vector"]],  # Embedding of the metadata
                metadatas=[chunk["metadata"]],  # Original metadata
                documents=[chunk["text"]]  # Optional: Store text if needed for retrieval
            )

        print("Metadata embeddings stored in Chroma!")
        execute_query.invoke(insert_query)


    # Generate embedding for the user query
    chat_history_text = ' '.join([message.content for message in history])
    combined_query = f"{chat_history_text} {user_query}"
    #query_embedding = get_embedding(' '.join(history)+' '+user_query)
    query_embedding = get_embedding(combined_query)

    # Perform similarity search
    results = collection.query(
        query_embeddings=[query_embedding],  # Embedding of the user query
        n_results=5 # Number of results to return
    )

    relevant_documents = results['documents']

    return relevant_documents
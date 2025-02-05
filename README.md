# Agentic-RAG-Challenge-MenuData

## Challenge Overview

### Objective
Build a conversational AI (bot) that can answer a variety of questions about restaurants, their menus, and their ingredients, leveraging both an internal proprietary dataset and external public datasets. The bot should demonstrate:

### Features

- **Retrieval-Augmented Generation (RAG):**
  - Ability to fetch relevant context from a knowledge base (including the internal dataset + external sources) to enhance LLM responses.
  - Proper chunking, embedding, and indexing of documents.

- **Text-to-LLM Pipeline:**
  - How queries are processed and fed to the LLM.
  - Prompt engineering, handling user context, and ensuring factual consistency.

- **External Data Integration:**
  - Seamless combination of proprietary Menudata.ai assets and supplemental data sources (e.g., news articles, Wikipedia entries).

### Approach

![Image](./Flowchart.jpeg)

Key components of the architecture above:

#### **Guardrails:**
- **Context Differentiation**: The system can differentiate between messages that are within the context of the conversation and those that are out of context. It notifies the user when an issue arises, ensuring clarity and relevance in interactions.
- **Toxic Language Filtering**: The system uses the Guardrails.ai package to filter out toxic or inappropriate language, ensuring a safe and respectful user experience.

### **Retriever:**
- **Database Querying**: Based on the user's query, if relevant information exists in the database, the system uses the Langchain framework combined with the LLM (Large Language Model) to generate the query, retrieve the data, and return relevant information to the Generator.
- **External Information Retrieval**: If external information is required, the system leverages the Wikipedia API to gather data on specific topics. The retrieved information is then embedded and stored in the vector database for future use.

### **Generator:**
- **Response Generation**: The system uses OpenAI's GPT-4 to generate queries and responses. It utilizes the relevant context provided by the Retriever to ensure accurate and context-aware outputs.
- **SQL Query Generation**: The Generator can also create SQL queries dynamically based on user requests. These queries are used to fetch structured data from relational databases, ensuring seamless integration with database systems for data retrieval.

### **Storage:**
- **Embedding External Data**: External data retrieved by the Retriever is embedded using OpenAI embeddings. The metadata, consisting of topic-subtopic combinations, is embedded to enhance retrieval speed and accuracy, while the description is stored as information.
- **Conversational Memory**: To improve conversation continuity, the system enables conversational memory, which is limited to the previous 5 conversations. This helps reduce the prompt size while maintaining context for smoother interactions.

### Folder Structure

AGENTIC-RAG-CHALLENGE/
│
├── Code/                       
│   ├── Embeddings.py          
│   ├── Example.py              
│   ├── LangchainActions.py     
│   └── main.py              
│
├── data/                      
│   └── MenuAndIngredients.csv   
│   └── Restaurants.csv   
│ 
├── Prompts and Few Shot Examples/ 
│   ├── (Details on Prompts used)
│
└── Main.ipynb   
└── README.md   
└── requirements.txt                 

- **`Embeddings.py`**: This is the main script for managing embeddings. It includes functions for generating, storing and retrieving embeddings. We embedd metadata in the format topic-subtopic for enhanced retrieval speed and accuracy.

---

- **`Example.py`**: This script includes few shot examples and method to retrieve relevant examples based on user query.

---

- **`LangchainActions.py`**: This script implements Langchain-specific actions, such as setting up chains, agents and workflows for processing prompts and generating responses.

---

- **`Main.ipynb`**: This is the main notebook for our Streamlit application.


## Getting Started

To get started with the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AGENTIC-RAG-CHALLENGE.git

2. Install requiremnts:
   ```bash
   pip install -r requirements.txt

3. Set up Database for querying

4. Create tables and load them

5. Set up your OpenAI account and get the API key and set the OPENAI_API_KEY=your_openai_api_key_here

6. Run the streamlit app:
   ```bash
   streamlit run main.py


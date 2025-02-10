examples = [
    {
        "input": "Which restaurants in San Francisco offer dishes with Impossible Meat?",
        "query": """
        MATCH (r:Restaurant)-[:SERVES]->(m:MenuItem)-[:CONTAINS]->(i:Ingredient)
        WHERE r.city = 'San Francisco' AND toLower(i.name) CONTAINS 'impossible'
        RETURN DISTINCT r.name AS restaurant_name, r.city;
        """
    },
    {
        "input": "Find restaurants in my area that serve gluten-free pizza.",
        "query": """
        MATCH (r:Restaurant)-[:SERVES]->(m:MenuItem)-[:CONTAINS]->(i:Ingredient)
        WHERE toLower(m.name) CONTAINS 'pizza' AND toLower(i.name) CONTAINS 'gluten free'
        RETURN DISTINCT r.name AS restaurant_name;
        """
    },
    {
        "input": "Give me a summary of the latest trends around desserts.",
        "query": """
        MATCH (r:Restaurant)-[:SERVES]->(m:MenuItem)
        WHERE toLower(m.category) CONTAINS 'dessert' OR toLower(m.cuisine_category) CONTAINS 'dessert'
        RETURN m.name AS dessert_name, COUNT(r) AS restaurant_count, AVG(r.rating) AS avg_rating, SUM(r.review_count) AS total_reviews
        ORDER BY total_reviews DESC, avg_rating DESC
        LIMIT 10;
        """
    },
    {
        "input": "Which restaurants are known for Sushi in my area?",
        "query": """
        MATCH (r:Restaurant)-[:SERVES]->(m:MenuItem)
        WHERE toLower(m.name) CONTAINS 'sushi' AND r.rating >= 4.0
        RETURN r.name AS restaurant_name, r.rating
        ORDER BY r.rating DESC;
        """
    },
    {
        "input": "Compare the average menu price of vegan restaurants in LA vs. Mexican restaurants.",
        "query": """
        MATCH (r:Restaurant)-[:SERVES]->(m:MenuItem)
        WHERE (toLower(m.cuisine_category) CONTAINS 'vegan' OR toLower(m.cuisine_category) CONTAINS 'mexican')
        WITH 
            CASE 
                WHEN toLower(m.cuisine_category) CONTAINS 'vegan' THEN 'Vegan'
                WHEN toLower(m.cuisine_category) CONTAINS 'mexican' THEN 'Mexican'
            END AS category,
            CASE 
                WHEN r.price = '$' THEN 1
                WHEN r.price = '$$' THEN 2
                WHEN r.price = '$$$' THEN 3
                WHEN r.price = '$$$$' THEN 4
            END AS price_level
        RETURN category, AVG(price_level) AS avg_price_level;
        """
    },
    {
        "input": "Top 5 famous desserts in Boston based on reviews.",
        "query": """
        MATCH (r:Restaurant)-[:SERVES]->(m:MenuItem)
        WHERE r.city = 'Boston' AND (toLower(m.category) CONTAINS 'dessert' OR toLower(m.cuisine_category) CONTAINS 'dessert')
        RETURN m.name AS dessert_name, SUM(r.review_count) AS total_reviews
        ORDER BY total_reviews DESC
        LIMIT 5;
        """
    },
    {
        "input": "Average rating of Mexican and Italian restaurants.",
        "query": """
        MATCH (r:Restaurant)-[:SERVES]->(m:MenuItem)
        WHERE toLower(m.cuisine_category) CONTAINS 'mexican' OR toLower(m.cuisine_category) CONTAINS 'italian'
        WITH 
            CASE 
                WHEN toLower(m.cuisine_category) CONTAINS 'mexican' THEN 'Mexican'
                WHEN toLower(m.cuisine_category) CONTAINS 'italian' THEN 'Italian'
            END AS cuisine, r.rating AS rating
        RETURN cuisine, AVG(rating) AS avg_rating;
        """
    },
    {
        "input": "Restaurants with the most vegetarian options.",
        "query": """
        MATCH (r:Restaurant)-[:SERVES]->(m:MenuItem)
        WHERE toLower(m.category) CONTAINS 'vegetarian' OR toLower(m.cuisine_category) CONTAINS 'vegetarian'
        RETURN r.name AS restaurant_name, r.city, COUNT(m) AS vegetarian_item_count
        ORDER BY vegetarian_item_count DESC
        LIMIT 5;
        """
    },
    {
        "input": "Give me the address of Bandit Dolores and Mission Bowling Club.",
        "query": """
        MATCH (r:Restaurant)
        WHERE toLower(r.name) CONTAINS 'bandit dolores' OR toLower(r.name) CONTAINS 'mission bowling club'
        RETURN r.name AS restaurant_name, r.address;
        """
    },
    {
        "input": "Find the best nuggets.",
        "query": """
        MATCH (r:Restaurant)-[:SERVES]->(m:MenuItem)
        WHERE toLower(m.name) CONTAINS 'nuggets' AND r.rating IS NOT NULL
        RETURN r.name AS restaurant_name, r.rating
        ORDER BY r.rating DESC
        LIMIT 1;
        """
    },
    {
        "input":"Which restaurants serve gluten free pizza",
        "query":"MATCH (r:Restaurant)-[:SERVES]->(m:MenuItem)-[:CONTAINS]->(i:Ingredient) WHERE toLower(m.name) CONTAINS 'pizza' AND i.name CONTAINS 'gluten free' RETURN DISTINCT r.name"
    },
    {
    "input":"Where can i find mexican food?",
    "query":"MATCH (r:Restaurant)-[:SERVES]->(m:MenuItem) WHERE toLower(m.cuisine_category) CONTAINS 'mexican' RETURN DISTINCT r.name"
}
]

from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
import streamlit as st

def get_example_selector():
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=3,
        input_keys=["input"],
    )
    return example_selector

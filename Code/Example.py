examples = [
    {
        "input": "Which restaurants in San Francisco offer dishes with Impossible Meat?",
        "query": "SELECT restaurant_name, city FROM restaurants WHERE city = 'San Francisco' AND id IN (SELECT restaurant_id FROM menu_items WHERE menu_description ILIKE '%impossible%');"
    },
    {
        "input": "Find restaurants in my area that serve gluten-free pizza.",
        "query": "SELECT restaurant_name FROM restaurants WHERE id IN (SELECT restaurant_id FROM menu_items WHERE menu_description ILIKE '%gluten free%' AND menu_item ILIKE '%pizza%');"
    },
    {
        "input": "Give me a summary of the latest trends around desserts.",
        "query": "SELECT mi.menu_item AS dessert_name, COUNT(r.id) AS restaurant_count, AVG(r.rating) AS avg_rating, SUM(r.review_count) AS total_reviews FROM menu_items mi JOIN restaurants r ON mi.restaurant_id = r.id WHERE mi.menu_category ILIKE '%dessert%' OR mi.categories ILIKE '%dessert%' GROUP BY mi.menu_item ORDER BY total_reviews DESC, avg_rating DESC LIMIT 10;"
    },
    {
        "input": "Which restaurants are known for Sushi in my area?",
        "query": "SELECT r.restaurant_name, r.rating FROM restaurants r JOIN menu_items m ON r.id = m.restaurant_id WHERE m.menu_item ILIKE '%sushi%' AND r.rating >= 4.0 GROUP BY r.restaurant_name, r.rating ORDER BY r.rating DESC;"
    },
    {
        "input": "Compare the average menu price of vegan restaurants in LA vs. Mexican restaurants.",
        "query": "SELECT category, AVG(CASE WHEN price = '$' THEN 1 WHEN price = '$$' THEN 2 WHEN price = '$$$' THEN 3 WHEN price = '$$$$' THEN 4 END) AS avg_price_level FROM (SELECT r.id AS restaurant_id, CASE WHEN m.categories ILIKE '%vegan%' THEN 'Vegan' WHEN m.categories ILIKE '%mexican%' THEN 'Mexican' END AS category, r.price FROM restaurants r JOIN menu_items m ON r.id = m.restaurant_id WHERE r.city = 'Los Angeles' AND (m.categories ILIKE '%vegan%' OR m.categories ILIKE '%mexican%')) subquery GROUP BY category;"
    },
    {
        "input": "Top 5 famous desserts in Boston based on reviews.",
        "query": "SELECT m.menu_item, SUM(r.review_count) AS total_reviews FROM restaurants r JOIN menu_items m ON r.id = m.restaurant_id WHERE r.city = 'Boston' AND (m.menu_category ILIKE '%dessert%' OR m.categories ILIKE '%dessert%') GROUP BY m.menu_item ORDER BY total_reviews DESC LIMIT 5;"
    },
    {
        "input": "Average rating of Mexican and Italian restaurants.",
        "query": "SELECT CASE WHEN m.categories ILIKE '%mexican%' THEN 'Mexican' WHEN m.categories ILIKE '%italian%' THEN 'Italian' END AS cuisine, AVG(r.rating) AS avg_rating FROM restaurants r JOIN menu_items m ON r.id = m.restaurant_id AND (m.categories ILIKE '%mexican%' OR m.categories ILIKE '%italian%') GROUP BY cuisine;"
    },
    {
        "input": "Restaurants with the most vegetarian options.",
        "query": "SELECT r.restaurant_name, r.city, COUNT(m.item_id) AS vegetarian_item_count FROM restaurants r JOIN menu_items m ON r.id = m.restaurant_id WHERE m.categories ILIKE '%vegetarian%' GROUP BY r.restaurant_name, r.city ORDER BY vegetarian_item_count DESC LIMIT 5;"
    },
    {
        "input": "Give me address of Bandit Dolores and Mission Bowling Club",
        "query": "SELECT r.restaurant_name, r.address1 FROM restaurants r WHERE (r.restaurant_name) ILIKE ('Bandit Dolores') or (r.restaurant_name) ILIKE ('Mission Bowling Club');"
    },
    {
        "input": "Find the best nuggets",
        "query": "SELECT r.restaurant_name, r.rating FROM restaurants r JOIN menu_items m ON r.id = m.restaurant_id WHERE m.menu_item ILIKE '%nuggets%' AND r.rating IS NOT NULL ORDER BY r.rating DESC LIMIT 1;"
    }
]

from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
import streamlit as st

# Selects few shot examples based on user query
def get_example_selector():
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=3,
        input_keys=["input"],
    )
    return example_selector
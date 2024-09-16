#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# ## Ingestion

df = pd.read_csv('IMDB-Movie-Data.csv', sep=',', on_bad_lines='skip')

df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
df = df.where(pd.notnull(df), None)
df['revenue_millions'] = df['revenue_millions'].fillna(0)
df['metascore'] = df['metascore'].fillna(0)
df = df.drop(columns='rank')
df.to_csv('data.csv', index=False)

df = pd.read_csv('data.csv')
documents = df.to_dict(orient='records')

# documents

# print(df.dtypes)

model_name = 'multi-qa-MiniLM-L6-cos-v1'
model = SentenceTransformer(model_name)

for doc in tqdm(documents):
    title = doc['title']
    genre = doc['genre']
    description = doc['description']
    director = doc['director']
    actors = doc['actors']
    
    # 將所有字段組合成一個文本
    combined_text = f"{title} {genre} {description} {director} {actors}"

    # 為每個字段單獨創建向量
    doc['title_vector'] = model.encode(title).tolist()
    doc['genre_vector'] = model.encode(genre).tolist()
    doc['description_vector'] = model.encode(description).tolist()
    doc['director_vector'] = model.encode(director).tolist()
    doc['actors_vector'] = model.encode(actors).tolist()
    
    # 創建一個組合所有字段的向量
    doc['combined_vector'] = model.encode(combined_text).tolist()

# print (doc)

es_client = Elasticsearch('http://elasticsearch:9200')

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "genre": {"type": "keyword"},
            "description": {"type": "text"},
            "director": {"type": "text"},
            "actors": {"type": "text"},
            "year": {"type": "text"},
            "runtime_minutes": {"type": "integer"},
            "rating": {"type": "float"},
            "votes": {"type": "integer"},
            "revenue_millions": {"type": "integer"},
            "metascore": {"type": "float"},
            "title_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },            
            "genre_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "description_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "director_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "actors_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "combined_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

index_name = "movie-database"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)


# ## Hybrid search

query = "the genre of the film is Sci-Fi"

v_q = model.encode(query).tolist()  # 確保將向量轉換為列表

import os
from openai import OpenAI

def setup_openai():
    # Get the OpenAI API key from the environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("No OpenAI API key found. Please set the OPENAI_API_KEY environment variable.")
    
    # Initialize and return the OpenAI client
    return OpenAI(api_key=api_key)

# Set up the OpenAI client
client = setup_openai()


def search(query):
    v_q = model.encode(query).tolist()
    
    knn_queries = [
        {"field": "combined_vector", "query_vector": v_q, "k": 5, "num_candidates": 100},
        {"field": "title_vector", "query_vector": v_q, "k": 5, "num_candidates": 100},
        {"field": "description_vector", "query_vector": v_q, "k": 5, "num_candidates": 100},
        {"field": "genre_vector", "query_vector": v_q, "k": 5, "num_candidates": 100},
        {"field": "director_vector", "query_vector": v_q, "k": 5, "num_candidates": 100},
        {"field": "actors_vector", "query_vector": v_q, "k": 5, "num_candidates": 100}
    ]
    
    should_clauses = [{"knn": knn_query} for knn_query in knn_queries]
    
    keyword_query = {
        "multi_match": {
            "query": query,
            "fields": ["title", "genre", "description", "director", "actors"],
            "type": "best_fields",
            "tie_breaker": 0.3
        }
    }
    
    combined_query = {
        "bool": {
            "must": keyword_query,
            "should": should_clauses
        }
    }
    
    results = es_client.search(
        index=index_name,
        body={"query": combined_query, "size": 10}
    )
    return results['hits']['hits']

search(query)

prompt_template = """
You're a Movie Recommender. Answer the QUESTION based on the CONTEXT from our movie database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

entry_template = """
movie_name: {title}
genre: {genre}
description: {description}
director: {director}
actors: {actors}
year: {year}
runtime_minutes: {runtime_minutes}
rating: {rating}
votes: {votes}
revenue_millions: {revenue_millions}
metascore: {metascore}
""".strip()

def build_prompt(query, search_results):
    context = ""
    
    for hit in search_results:
        doc = hit['_source']
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def rag(query, model='gpt-4o-mini'):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    #print(prompt)
    answer = llm(prompt, model=model)
    return answer

question = 'find a movie tell something about family, and describe the plot to me'
answer = rag(question)
# print(answer)

# Streamlit 應用
st.title('Movie Recommendation System')

st.markdown("""
This system uses advanced natural language processing technology to understand your preferences and recommend the best movies for you.

**How to use it:** 
1. Enter your movie preferences in the input box below. 
2. You can include specific movie titles, genres, director, actors or any movie-related descriptions. 
3. The system will analyze your input and recommend relevant movies.
""")
query = st.text_input('Please enter your movie query: e.g. “Star Trek-like sci-fi movie” or “Comedy starring Tom Hanks”.')

if query:
    with st.spinner('Processing your enquiry...'):
        answer = rag(query)
        if answer:
            st.subheader("Recommended Results：")
            st.write(answer)
            st.caption("Source: IMDB Movie Database")
        else:
            st.error("Could not generate an answer, please try again later.")

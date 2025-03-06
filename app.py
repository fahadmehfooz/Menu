import streamlit as st
import pickle
import boto3
import numpy as np
import json
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import faiss
from dotenv import load_dotenv
import os

def load_pickle_from_parts(folder, num_parts=15):
    """Reads all pickle chunks and reconstructs the original object."""
    data = b''

    # Read and merge chunks in order
    for i in range(num_parts):
        chunk_path = os.path.join(folder, f'chunk_{i}.pkl')
        with open(chunk_path, 'rb') as f:
            data += f.read()

    # Deserialize the full object
    return pickle.loads(data)


def create_tfidf_pipeline(chunks):
    """Create TF-IDF vectorizer and matrix using chunk texts"""
    corpus = [chunk['text'] for chunk in chunks]
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True
    )
    return vectorizer, vectorizer.fit_transform(corpus)

def create_faiss_index(embeddings):
    """Create FAISS index with dimension validation"""
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_with_tfidf(query, chunks, vectorizer, tfidf_matrix, top_n=50):


    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    scored_chunks = sorted(
        [(i, score) for i, score in enumerate(cosine_sim)],
        key=lambda x: x[1], 
        reverse=True
    )[:top_n]
    
    return [{
        "metadata": {
            **chunks[i]["metadata"],
            "text": chunks[i]["text"],
            "chunk_id": chunks[i]["chunk_id"]
        },
        "text": chunks[i]["text"],
        "chunk_id": chunks[i]["chunk_id"],
        "score": float(score)
    } for i, score in scored_chunks]

def refine_with_bm25(query, tfidf_results, top_n=20):
    """BM25 refinement with consistent metadata structure"""
    texts = [res["text"] for res in tfidf_results]
    bm25 = BM25Okapi([doc.split() for doc in texts])
    scores = bm25.get_scores(query.split())
    
    return [{
        "metadata": {
            **res["metadata"],
            "text": res["text"],
            "chunk_id": res["chunk_id"]
        },
        "text": res["text"],
        "chunk_id": res["chunk_id"],
        "score": float(scores[i])
    } for i, res in enumerate(tfidf_results)][:top_n]

def get_titan_embeddings(texts, bedrock_client, dimensions=512):
    embeddings = []
    cnt = 0 
    for row, text in enumerate(texts):
        body = json.dumps({
            "inputText": text,
            "dimensions": dimensions,
            "normalize": True
        })
        
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="*/*",
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        embedding = response_body['embedding']
        embeddings.append(embedding)
        cnt+=1

        if (cnt %1000) == 0:
            print(cnt)
    
    return np.array(embeddings)

def retrieve_with_embeddings(query, faiss_index, faiss_index_to_chunk, bedrock_client, top_n=20):
    """FAISS retrieval with consistent metadata structure"""
    query_embedding = get_titan_embeddings([query], bedrock_client)
    distances, indices = faiss_index.search(
        np.array(query_embedding).astype('float32'), 
        top_n
    )
    
    return [{
        "metadata": {
            **faiss_index_to_chunk[idx],
            "text": faiss_index_to_chunk[idx]["text"],
            "chunk_id": faiss_index_to_chunk[idx]["chunk_id"]
        },
        "text": faiss_index_to_chunk[idx]["text"],
        "chunk_id": faiss_index_to_chunk[idx]["chunk_id"],
        "score": float(1 / (1 + distances[0][i]))
    } for i, idx in enumerate(indices[0]) if idx in faiss_index_to_chunk]


def hybrid_retrieval(query, chunks, vectorizer, tfidf_matrix, 
                    faiss_index, faiss_index_to_chunk, reranker, bedrock_client, top_k=5): 

    tfidf_results = retrieve_with_tfidf(query, chunks, vectorizer, tfidf_matrix, 50)
    bm25_results = refine_with_bm25(query, tfidf_results, 20)
    vector_results = retrieve_with_embeddings(query, faiss_index, faiss_index_to_chunk, bedrock_client, 20)
    
    seen = set()
    combined = []

    for res in bm25_results + vector_results:
        uid = res["metadata"].get("item_id", f"{res['metadata']['restaurant_id']}_{res['metadata']['menu_item']}")
        if uid not in seen:
            seen.add(uid)
            combined.append(res)
    
    return rerank_results(query, combined, reranker, top_k)

def rerank_results(query, combined_results, reranker, top_k=5):
    """Re-rank retrieved results using a Cross-Encoder"""
    if not combined_results:
        return []

    query_doc_pairs = [(query, res["text"]) for res in combined_results]
    scores = reranker.predict(query_doc_pairs)

    for i, res in enumerate(combined_results):
        res['combined_score'] = scores[i]
        res['reranker_score'] = scores[i]  
    
    sorted_results = sorted(combined_results, key=lambda x: x["reranker_score"], reverse=True)[:top_k]
    return sorted_results

def find_information_missing_in_chunks(query, retrieved_chunks, bedrock_client):
    enhancement_prompt = f"""Analyze the given restaurant-related query and retrieved search results.  
    Determine if the query explicitly asks for historical or cultural context about a dish, ingredient, or restaurant practice,  
    and check whether this information is present in the retrieved search results.  

    **INPUT:**  
    - **Query:** {query}  
    - **Retrieved Chunks:** {retrieved_chunks}  

    **OUTPUT GUIDELINES:**  
    - If the query **does not explicitly** ask for historical/cultural context (e.g., it only asks for restaurants or locations), return an **empty string** (`""`).  
    - If the query **explicitly** asks for historical or cultural context, check if the retrieved chunks answer it.  
    - If the historical or cultural context is required but **not answered** in the retrieved chunks, return a **concise Wikipedia search term** in the format: `"History of <topic>"`.  
    - Do **not** generate historical terms unless they are explicitly requested.  
    - Do **not** include any extra text, explanations, or formatting beyond the required search term.  
    """

    payload = {
        "modelId": os.getenv("inference_profile"),
        "body": json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "temperature": 0.3,
            "system": """You are an expert in restaurant data analysis.  
Your task is to determine if a restaurant-related query requires missing historical or cultural context.  
- Only return a Wikipedia search term **if the query explicitly asks for historical or cultural context** and the retrieved information does not answer it.  
- If the query is about finding restaurants, return an empty string.  
- Do not assume historical context is needed just because a food item is mentioned.""",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text", 
                    "text": enhancement_prompt
                }]
            }]
        })
    }

    try:
        response = bedrock_client.invoke_model(**payload)
        result = json.loads(response["body"].read().decode("utf-8"))
        search_term = result.get("content", [{}])[0].get("text", "").strip()
        
        if search_term:
            wiki_summary = fetch_wikipedia_summary(search_term[1: -1])
            if wiki_summary:
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/"
                link = url + search_term[1:-1]
                return {"Historic_context": wiki_summary, "link": link}
            else:
                return {"Historic_context": "Data not available on Wikipedia", "link": "Data not available on Wikipedia"}

        return {"Historic_context": -1, "link": "Data not available on Wikipedia"}
        
    except Exception as e:
        print(f"Analysis error: {str(e)[:200]}")
        return {"Historic_context": -1, "link": "Data not available on Wikipedia"}

def fetch_wikipedia_summary(title):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    response = requests.get(url)
    return response.json().get("extract", "-1")

def generate_final_response(query, retrieved_chunks, historic_context, bedrock_client):
    restaurant_entries = []
    for chunk in retrieved_chunks[:3]: 
        meta = chunk.get('metadata', {})
        entry = (
            f"Restaurant: {meta.get('restaurant_name', 'Unknown')}\n"
            f"- Menu Item: {meta.get('menu_item', 'N/A')} (ID: {meta.get('item_id', '')})\n"
            f"- Price: {meta.get('price_tier', '').capitalize()}\n"
            f"- Cuisine: {', '.join(meta.get('cuisine_types', []))}\n"
            f"- Ingredients: {', '.join(meta.get('ingredients', []))}"
        )
        restaurant_entries.append(entry)

    history_block = ""
    if historic_context.get('Historic_context', -1) not in [-1, "Data not available on Wikipedia"]:
        history_block = (
            f"\n\nHistorical Context: {historic_context['Historic_context']}"
            f"\nSource: {historic_context['link']}"
        )

    generation_prompt = f"""You are a restaurant information specialist. Use this structure:

    {{restaurants}}

    {{history}}

    **Guidelines**
    - Present restaurants in bullet points
    - ALWAYS include item IDs from metadata
    - For history: Only include if relevant to query
    - If asked sources: Reference item IDs like "(ID: 24399115)"
    - Never mention chunks or retrieval process

    **Current Query**: {query}
    """

    payload = {
        "modelId": os.getenv("inference_profile"),
        "body": json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0.2,
            "system": """You are a restaurant concierge for San Francisco. Follow these rules:
            1. Use ONLY the provided restaurant metadata
            2. For history: Use only the provided Wikipedia context
            3. Always cite item IDs from metadata
            4. If no history requested: Don't mention it""",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": generation_prompt.format(
                        restaurants="\n\n".join(restaurant_entries),
                        history=history_block
                    )
                }]
            }]
        })
    }

    try:
        response = bedrock_client.invoke_model(**payload)
        result = json.loads(response["body"].read().decode("utf-8"))
        raw_response = result.get("content", [{}])[0].get("text", "")
        
        return raw_response.replace("(ID:", "(Item ID:").replace("ID:", "Item ID:")
    
    except Exception as e:
        print(f"Generation error: {str(e)[:200]}")
        return f"Could not generate response. Please check your query. Reference IDs: {', '.join([str(c['metadata']['item_id']) for c in retrieved_chunks[:3]])}"

# Streamlit app code
def load_data():
    """Load all necessary data and models"""

    with open('enhanced_chunk_embeddings_7500.pkl', 'rb') as f:
        loaded_embeddings_enhanced = pickle.load(f)
    
    chunks_enhanced = load_pickle_from_parts('chunks_enhanced_parts', num_parts=5)

    
    system = load_pickle_from_parts('pickled_parts', num_parts=15)

    return loaded_embeddings_enhanced, chunks_enhanced, system

def initialize_bedrock_client():
    
    load_dotenv("Credentials")  

    # Access the credentials from environment variables
    aws_access_key_id = os.getenv("aws_access_key_id")
    aws_secret_access_key = os.getenv("aws_secret_access_key")

    return boto3.client(
    service_name = "bedrock-runtime",
    region_name="us-east-2",
    aws_access_key_id = aws_access_key_id, 
    aws_secret_access_key = aws_secret_access_key
)

def main():
    st.title("🍽️ SanFrancisco Restaurant Search")
    st.write("Ask questions about restaurants, cuisines, and food history in San Francisco!")

    embeddings, chunks, system = load_data()
    bedrock_client = initialize_bedrock_client()

    # Search interface
    query = st.text_input("What would you like to know about SF restaurants?", 
                         placeholder="e.g., 'Where can I find the best sushi?'")

    if query:
        with st.spinner("Searching for restaurants..."):
            # Pass bedrock_client to hybrid_retrieval
            retrieved_chunks = hybrid_retrieval(
                query,
                chunks=chunks,
                vectorizer=system["vectorizer"],
                tfidf_matrix=system["tfidf_matrix"],
                faiss_index=system["faiss_index"],
                faiss_index_to_chunk=system["faiss_index_to_chunk"],
                reranker=system["reranker"],
                bedrock_client=bedrock_client,
                top_k=5
            )

            # Rest of the code remains the same
            retrieved_chunks_text = " ".join([i["text"] for i in retrieved_chunks])
            historic_context = find_information_missing_in_chunks(
                query, retrieved_chunks_text, bedrock_client
            )

            response = generate_final_response(
                query, retrieved_chunks, historic_context, bedrock_client
            )
            st.write(response)

            with st.expander("See detailed restaurant information"):
                for i, chunk in enumerate(retrieved_chunks[:3], 1):
                    st.subheader(f"Restaurant {i}")
                    meta = chunk.get('metadata', {})
                    st.write(f"**Name:** {meta.get('restaurant_name', 'Unknown')}")
                    st.write(f"**Menu Item:** {meta.get('menu_item', 'N/A')}")
                    st.write(f"**Price Tier:** {meta.get('price_tier', '').capitalize()}")
                    st.write(f"**Cuisine Types:** {', '.join(meta.get('cuisine_types', []))}")
                    st.write(f"**Ingredients:** {', '.join(meta.get('ingredients', []))}")
                    st.write("---")


    # Sidebar with app information
    with st.sidebar:
        st.header("About")
        st.write("""
        This app helps you discover restaurants in San Francisco. You can:
        - Search for specific cuisines
        - Find restaurants by price range
        - Learn about food history
        - Get menu recommendations
        """)
        
        st.header("Search Tips")
        st.write("""
        Try asking about:
        - Specific dishes
        - Cuisine types
        - Price ranges
        - Restaurant locations
        - Food history
        """)

if __name__ == "__main__":
    main()
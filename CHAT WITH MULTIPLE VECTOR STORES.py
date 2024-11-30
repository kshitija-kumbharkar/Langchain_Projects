import os
import sys
import streamlit as st
import numpy as np

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

sys.path.append('../..')
try:
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())
except ImportError:
    st.warning("python-dotenv not installed. Please install it using 'pip install python-dotenv'")

def load_text_file(file_name):
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_name, 'r', encoding=encoding) as file:
                return file.read()
        except (UnicodeDecodeError, FileNotFoundError) as e:
            st.warning(f"Could not read {file_name} with {encoding} encoding: {e}")
    
    st.error(f"Could not read file {file_name}. Please check the file exists and its encoding.")
    return ""

def initialize_vector_stores():
    vector_stores = {}
    contexts = ["Physics", "History", "DBMS"]
    
    context_files = {
        "Physics": "physics.txt",
        "History": "history.txt",
        "DBMS": "dbms.txt"
    }

    try:
        embedding = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return {}

    for context in contexts:
        text_file = context_files[context]
        text_content = load_text_file(text_file)
        
        if not text_content:
            continue

        persist_directory = f"chroma_{context.lower()}"
        os.makedirs(persist_directory, exist_ok=True)

        try:
            vectordb = Chroma(
                persist_directory=persist_directory, 
                embedding_function=embedding
            )
            
            if vectordb._collection.count() == 0:
                vectordb = Chroma.from_documents(
                    documents=[Document(page_content=text_content)], 
                    embedding=embedding,
                    persist_directory=persist_directory
                )
        except Exception:
            vectordb = Chroma.from_documents(
                documents=[Document(page_content=text_content)], 
                embedding=embedding,
                persist_directory=persist_directory
            )
        
        vector_stores[context] = vectordb

    return vector_stores

@st.cache_resource
def get_llm():
    try:
        llm_name = "gpt-4o-mini"
        llm = ChatOpenAI(
            model_name=llm_name, 
            openai_api_key=os.environ['OPENAI_API_KEY'], 
            temperature=0
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

def choose_most_relevant_context(question, vector_stores, top_k=1):
    try:
        embedding = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
        
        # Embed the question
        question_embedding = embedding.embed_query(question)
        
        # Calculate similarity scores for each vector store
        context_similarities = {}
        for context, vectordb in vector_stores.items():
            # Get the database embedding
            try:
                # Fetch a few documents and calculate their average embedding
                docs = vectordb.similarity_search(question, k=3)
                if docs:
                    doc_embeddings = [embedding.embed_documents([doc.page_content])[0] for doc in docs]
                    avg_doc_embedding = np.mean(doc_embeddings, axis=0)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(question_embedding, avg_doc_embedding) / (
                        np.linalg.norm(question_embedding) * np.linalg.norm(avg_doc_embedding)
                    )
                    context_similarities[context] = similarity
            except Exception as e:
                st.warning(f"Error processing {context} context: {e}")
        
        # Sort contexts by similarity in descending order
        sorted_contexts = sorted(context_similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k contexts
        return [context for context, _ in sorted_contexts[:top_k]]
    
    except Exception as e:
        st.error(f"Error in context selection: {e}")
        return list(vector_stores.keys())  # fallback to all contexts

def get_response(question, selected_vectordb, llm):
    try:
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 

        Context:
        {context}

        Question: {question}
        Helpful Answer:"""

        QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=selected_vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            chain_type='stuff',
        )

        result = qa_chain.invoke({"query": question})
        return result["result"]

    except Exception as e:
        st.error(f"Error in getting response: {e}")
        return "Sorry, I couldn't process your query."

def main():
    st.title("Q&A Assistant")

    # Validate OpenAI API Key
    if 'OPENAI_API_KEY' not in os.environ:
        st.error("Please set your OpenAI API key in the environment variables.")
        st.stop()

    # Initialize LLM
    llm = get_llm()
    if llm is None:
        st.error("Failed to initialize language model.")
        st.stop()

    # Initialize Vector Stores
    with st.spinner('Initializing vector stores...'):
        vector_stores = initialize_vector_stores()
    
    if not vector_stores:
        st.error("Failed to initialize any vector stores.")
        st.stop()

    # User input
    user_prompt = st.text_input("Enter your question:", key="user_query")

    # Response generation
    if st.button("Get Response") and user_prompt:
        # Choose most relevant context
        with st.spinner('Selecting most relevant context...'):
            relevant_contexts = choose_most_relevant_context(user_prompt, vector_stores)
            selected_context = relevant_contexts[0]
            st.info(f"Selected Context: {selected_context}")

        # Generate response from the most relevant context
        with st.spinner('Generating response...'):
            response = get_response(user_prompt, vector_stores[selected_context], llm)
            st.success("Response:")
            st.write(response)

# Run the app
if __name__ == "__main__":
    main()
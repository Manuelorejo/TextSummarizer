import streamlit as st
import chromadb
from groq import Groq
from chromadb.utils import embedding_functions
from pdfminer.high_level import extract_text
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from io import BytesIO  
from nltk.tokenize import word_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter


#Imports groq api key
@st.cache_resource
def init_groq():
    api_key = st.secrets["groq"]["api_key"]
    return Groq(api_key=api_key)




def pre_processing(pdf_file):
    # Read PDF content from uploaded file
    pdf_content = pdf_file.read()
    
    # Wraps bytes in a file-like object
    pdf_file_like = BytesIO(pdf_content)

    # Converts bytes to string using pdfminer
    document = extract_text(pdf_file_like, password='', page_numbers=None, maxpages=0, caching=True, codec='utf-8', laparams=None)
    document = document.lower()
    document = re.sub(r'[^\w\s]',"",document)
    
    stopwords_set = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(document)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords_set]
    cleaned_text = ' '.join(tokens)
        
    rec_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=5)
    chunks = rec_text_splitter.split_text(cleaned_text)
    return chunks



#Sets up the chormadb database
@st.cache_resource
def setup_chromadb():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='all-MiniLM-L6-v2'
    )
    
    try:
        collection = chroma_client.get_collection(
            name="document_chunks",
            embedding_function=embedding_function
        )
    except:
        collection = chroma_client.create_collection(
            name="document_chunks",
            embedding_function=embedding_function
        )
    
    return collection

def store_chunks_in_chromadb(chunks, collection):
    # Gets existing document IDs
    existing_docs = collection.get()
    if existing_docs["ids"]:  # Only delete if IDs exist
        collection.delete(existing_docs["ids"])
    
    # Stores new chunks
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"chunk_{i}"]
        )

#Retrieves the relevant chunks from the chroma database
def get_relevant_chunks(question, collection, n_results=3):
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results['documents'][0]


#Prompt for generating answer to the question
def get_answer(relevant_chunks, question, groq_client):
    answers = []
    for chunk in relevant_chunks:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": f"Please read this document {chunk} and use it to answer only this question {question} as briefly as possible"
                }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        answers.append(completion.choices[0].message.content)
    return " ".join(answers)

def main():
    st.title("PDF Question Answering System")
    
    # Initializes Groq client
    groq_client = init_groq()
    
    # Setup ChromaDB
    collection = setup_chromadb()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Processes PDF and store chunks
        with st.spinner("Processing PDF..."):
            chunks = pre_processing(uploaded_file)
            store_chunks_in_chromadb(chunks, collection)
        st.success("PDF processed successfully!")
        
        # Question input
        question = st.text_input("Ask a question about the document:")
        
        if question:
            with st.spinner("Generating answer..."):
                # Gets relevant chunks based on question
                relevant_chunks = get_relevant_chunks(question, collection)
                
                # Generates response using relevant chunks
                answer = get_answer(relevant_chunks, question, groq_client)
                
                st.write("Answer:")
                st.write(answer)

if __name__ == "__main__":
    main()
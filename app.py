import streamlit as st
import os 
import pickle
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer 
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  
    return text

# Sidebar contents
with st.sidebar:
    st.title('LLM Chat PDF App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made by Akash Bindu')

def main():
    load_dotenv()
    st.header("Chat PDF App")
    
    #os.environ['OPENAI_API_KEY'] = st.text_input("OpenAI API Key:", type="password")
    
    if pdf := st.file_uploader("Upload your PDF", type="pdf"):
        pdf_reader = PdfReader(pdf)

        text = "".join(page.extract_text() for page in pdf_reader.pages)

        # LLM have limited context window 
        # splitting text due to restrictions on context tokens in openAI (chatGPT allow only 4096 tokens)


        splitter =  RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text) #split_documents

        # Embedding text
        # creating vectors from works to create vector store useful for searching 
        # getting similarity between words
        # creating vector store using openAI 
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)


        else:
            encode_kwargs = {'normalize_embeddings': True}

            # local model to save cost of openAi
            # model = HuggingFaceInstructEmbeddings(model_name='BAAI/bge-base-en',
            #                                     # retrieval passages for short query, using query_instruction, else set it ""
            #                                     query_instruction="Represent this sentence for searching relevant passages: ",
            #                                     model_kwargs = encode_kwargs
            #                                     )

            model = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=model) # from documents
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vector_store, f)


        if query := st.text_input("Ask questions about your PDF file:"):
            docs = vector_store.similarity_search(query=query, k=3)
            print(docs)
            # Calculate the relevancy score
            user_message = preprocess_text(query)  # Preprocess the user message
            document_texts = [doc.page_content for doc in docs]
            document_texts.append(user_message)  

            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(document_texts)

            # Calculate cosine similarities
            user_message_tfidf = X[-1]
            document_tfidf = X[:-1]
            cosine_similarities = cosine_similarity(user_message_tfidf, document_tfidf)[0]

            # Calculate the relevancy score as the maximum cosine similarity
            relevancy_score = max(cosine_similarities) * 100
            st.write(f"Relevancy score: {relevancy_score:.2f}%")

            # Check if the relevancy score is above a certain threshold
            relevancy_threshold = 50  # You can adjust this threshold as needed
            if relevancy_score >= relevancy_threshold:
                llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                    st.write(response)
            else:
                st.write("Query is not relevant enough to fetch a response.")



if __name__ == '__main__':
    main()
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_chroma import Chroma
import streamlit as st
import chromadb



chromadb.api.client.SharedSystemClient.clear_system_cache()

load_dotenv() # Loads .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Loads API key

loader = PyPDFLoader("ispt2.pdf")  # Load your PDF file
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
docs = text_splitter.split_documents(data)

# Load the Gemini API key
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Test embedding a query
#vector = embeddings.embed_query("ISPT-KIN")
vectorstoredb = Chroma.from_documents(docs, embedding=embeddings)
retrievers = vectorstoredb.as_retriever()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)

system_prompt = ( 
   "Vous êtes un Assistant spécialisé dans les renseignements pour l'établissement ISPT-KIN. Fournissez des réponses avec le format Markdown claires en fonction du contexte fourni. " 
    "Si les informations ne sont pas trouvées dans le contexte, indiquez que la réponse n'est pas disponible. Use Markdown to give the answer" 
       
    "\n\n" 
    "{context}"
 ) 

# Configurer l'invite pour la chaîne d'assurance qualité
prompt = ChatPromptTemplate.from_messages( 
    [ 
        ( "system" , system_prompt), 
        ( "human" , "{input}" ) 
    ] 
) 

# Créer la chaîne RAG chain
# Create the RAG chain
chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retrievers, chain)

#response = rag_chain.invoke({"input": "Que signifie ISPT-KIN"})
#print(response['answer'])

def main():
    st.logo("logo.jpg")
    st.set_page_config(layout="wide")
    st.subheader("Assistant d'information ISPT-KIN", divider="rainbow")
    st.subheader("Chatbot ISPT-KIN")

    user_question = st.text_input("Posez votre question :")
    st.write("Réponse :")

    if user_question:
        with st.spinner("Processing..."):
            try:
                response = rag_chain.invoke({"input": user_question})
                #st.write("Réponse :", response['answer'])
                #st.write(response['answer'], unsafe_allow_html=True)
                st.markdown(response['answer'])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
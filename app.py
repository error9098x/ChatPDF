import streamlit as st
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def get_pdf_text(pdf_docs):
    return "".join([page.extract_text() for pdf in pdf_docs for page in PdfReader(pdf).pages])

def get_text_chunks(text):
    return RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60,length_function=len).split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=api_key)
    FAISS.from_texts(text_chunks, embedding=embeddings).save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
Please provide a comprehensive and detailed answer to the question, strictly using only the information provided in the context. Avoid introducing any external information or assumptions not directly found in the given context.

Context:
{context}

Question:
{question}

Your answer should be directly informed by and confined to the details present in the context. 

Answer:
"""
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, search_kwargs={"k": 10}, return_texts=True )
    print(docs)
    chain = get_conversational_chain()
    with get_openai_callback() as cb:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(cb)
    st.write("Reply:\n", response["output_text"])

def setup_page():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with Multiple PDFs")

def get_user_question():
    user_question = st.text_input("Ask a Question from the PDF Files", help="Enter your question here and get answers based on the uploaded PDF content.")
    if user_question:
        st.success("Your answer will appear here.")
    else:
        st.warning("Please enter a question.")
    return user_question

def process_files(pdf_docs):
    if pdf_docs:
        with st.spinner("Extracting text and processing PDF files..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
        st.success("Done! You are now ready to ask questions.")
        st.balloons()

def display_sidebar():
    with st.sidebar:
        st.title("Menu")

        def store_api_key():
            global api_key
            api_key = st.text_input("Enter your API key:")
            if not api_key:
                st.warning("Please enter your API key.")

        try:
            store_api_key()
        except Exception as e:
            st.error(f"An error occurred while storing the API key: {str(e)}")

        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, help="Upload one or more PDF files from which you want to extract information.")
        if st.button("Submit & Process"):
            process_files(pdf_docs)

def main():
    setup_page()
    display_sidebar()
    
    user_question = get_user_question()
    if user_question:
        try:
            with st.spinner("Processing your question..."):
                user_input(user_question)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
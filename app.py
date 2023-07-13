# app.py
import streamlit as st
from dotenv import load_dotenv
import pickle
from pypdf import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
# get callback to get stats on query cost
from langchain.callbacks import get_openai_callback
import os
from docx import Document
load_dotenv()
# Sidebar contents
with st.sidebar:
    st.title('💬 PDF Chat App')
    st.markdown('''
    ## About
    Chat with your documents privately. 
    ''')
    add_vertical_space(5)
    st.write('KB Thiam')

def extract_text_from_docx(file):
    doc = Document(file)
    paragraphs = [p.text for p in doc.paragraphs]
    return '\n'.join(paragraphs)

def main():
    st.header("Chat with your documents privately 💬")
    # upload a PDF file
    files = upload_pdf()
    # check for pdf file
    if files is not None:
        # process text in pdf and convert to chunks
        chuck_size = 500
        chuck_overlap = 100
        chunks = ''
        chunks = process_text(files, chuck_size, chuck_overlap)
        vector_store = get_embeddings(chunks, files)
        # ask the user for a question
        question = st.text_input("Ask a question")
        if question:
            # get the docs related to the question
            docs = retrieve_docs(question, vector_store)
            response = generate_response(docs, question)
            st.write(response)

# upload a pdf file from website

def upload_pdf():
    files = st.file_uploader("Upload your documents",
                             accept_multiple_files=True)
    return files

# convert the pdf to text chunks


def process_text(files, chuck_size, chuck_overlap):
    docx_files = []
    pdf_files = []
    txt_files = []

    for file in files:
        filename, file_extension = os.path.splitext(file.name)

        if file_extension.lower() == '.docx':
            docx_files.append(file)
        elif file_extension.lower() == '.pdf':
            pdf_files.append(file)
        elif file_extension.lower() == '.txt':
            txt_files.append(file)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chuck_size,
        chunk_overlap=chuck_overlap,
        length_function=len
    )
    page_text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        # extract the text from the PDF
        for page in pdf_reader.pages:
            page_text += page.extract_text()
    for txt in txt_files:
        page_text += txt.read().decode('utf-8')

    for docx in docx_files:
        text = extract_text_from_docx(docx)
        page_text += text
    chunks = text_splitter.split_text(text=page_text)
    if chunks:
        return chunks
    else:
        raise ValueError("Could not process text in PDF")

# find or create the embeddings

def get_embeddings(chunks, files):
    store_name = ''
    if files:
        for file in files:
            store_name += file.name[:-4] + '#'
    # check if vector store already exists
    # if REUSE_PKL_STORE is True, then load the vector store from disk if it exists
    reuse_pkl_store = os.getenv("REUSE_PKL_STORE")
    if reuse_pkl_store == "True" and os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
        st.write("Embeddings loaded from disk")
    # else create embeddings and save to disk
    else:
        embeddings = OpenAIEmbeddings()
        # create vector store to hold the embeddings
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        # save the vector store to disk
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
        st.write("Embeddings saved to disk")
    if vector_store is not None:
        return vector_store
    else:
        raise ValueError("Issue creating and saving vector store")
# retrieve the docs related to the question


def retrieve_docs(question, vector_store):
    docs = vector_store.similarity_search(question, k=3)
    if len(docs) == 0:
        raise Exception("No documents found")
    else:
        return docs

# generate the response


def generate_response(docs, question):
    llm = ChatOpenAI(temperature=0.0, max_tokens=1000,
                     model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=question)
        print(cb)
    return response


if __name__ == '__main__':
    main()

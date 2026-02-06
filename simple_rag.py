import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import argparse
import time
from dotenv import load_dotenv

load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") 

def replace_t_with_space(list_of_documents):

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t',' ')
    
    return list_of_documents


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    ) 

    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

def retrieve_context_per_question(question,chunks_query_retriever):
    docs=chunks_query_retriever.invoke(question)

    context=[doc.page_content for doc in docs]

    return context

def show_context(context):
    
    for i,c in enumerate(context):
        print(f"context {i+1}")
        print(c)
        print("\n")


class SimpleRAG:

        def __init__(self,path , chunk_size=1000,chunk_overlap=200 , n_retrieved=2):
            print("In simple-RAG class")

            start_time = time.time()
            self.vector_store = encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.time_records= {'Chunking': time.time() - start_time}

            print(f"Chunking time: {self.time_records['Chunking']:.2f} seconds")
            
            self.chunks_query_retriever =self.vector_store.as_retriever(search_kwargs={"k":n_retrieved})
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",  # or "mixtral-8x7b-32768"
                temperature=0
            )
        
        def run(self,query):
            
            start_time=time.time()
            context = retrieve_context_per_question(query,self.chunks_query_retriever)
            self.time_records['Retrieval'] =time.time()-start_time
            print(f"retrieval time: {self.time_records['Retrieval']:.2f} seconds")

            show_context(context)

            context_text = "\n\n".join(context)
            prompt = f"Based on the following context, answer the question.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
            
            response = self.llm.invoke(prompt)
            print(f"\nAnswer: {response.content}")





def validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if args.n_retrieved<=0:
        raise ValueError("n_retrieved must be a positive integer.")
    
    return args


# argparse is a file(module) and ArgumentParser is a class inside this file , this means parser is a object of the class
# add_argument is a method of class ArgumentParser



def parse_args():
    parser = argparse.ArgumentParser(description="Encode a PDF document and test a simple RAG.")
    
    parser.add_argument("--path",type=str,default="",help="path to the pdf file to encode")

    parser.add_argument("--chunk_size",type=int,default=1000,help="size of each text chunk(default: 1000).")

    parser.add_argument("--chunk_overlap",type=int,default=200,help="overlap between consecutive chuncks (default : 200).")

    parser.add_argument("--n_retrieved" , type=int , default=2 , help="number of chunks to retrieve for each query (default: 2).")
    
    parser.add_argument("--query",type=str,default="what is the main cause of climate change?", help="query to test the retriever (default: 'what is the main cause of climate change?).")

    parser.add_argument("--evaluate", action="store_true", help="whether to evaluate the retriver's performance (default: false)")

    return validate_args(parser.parse_args())


# first main(parse_args()) which runs parse_args and validate args and returs args to main function

# in main function we call object of class SimpleRAG , object : simple_rag

# we call method of class : run



def main(args):

    simple_rag = SimpleRAG(
        path=args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    simple_rag.run(args.query)

    # if args.evaluate:
    #     evaluate_rag(simple_rag.chunks_query_retriever)

if __name__ == '__main__':
    main(parse_args())
    

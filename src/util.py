# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import CrossEncoder
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


def replace_t_with_space(list_of_documents):

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t',' ')
    
    return list_of_documents


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    
    loader = PyPDFLoader(path)
    documents = loader.load()
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    # )
    text_splitter = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="percentile", 
    breakpoint_threshold_amount=95, 
    number_of_chunks=None ) 
    
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore

def retrieve_context_per_question(question, chunks_query_retriever, n_retrieved=2):
    docs = chunks_query_retriever.invoke(question)
    
    # Rerank using cross-encoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    query_doc_pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(query_doc_pairs)
    
    # Sort by score (higher is better) and take top n_retrieved
    ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:n_retrieved]
    context = [doc.page_content for doc, _ in ranked_docs]
    
    return context

def show_context(context):
    
    for i,c in enumerate(context):
        print(f"context {i+1}")
        print(c)
        print("\n")
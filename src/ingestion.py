import os
import time
import argparse
from langchain_groq import ChatGroq
import argparse
import time
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from util import ( encode_pdf,  show_context , 
build_knowledge_graph,rerank_documents, find_node_by_content,
expand_context_via_graph,visualize_graph)

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") 

class GraphRAG:

        def __init__(self,path , chunk_size=1000,chunk_overlap=200 , n_retrieved=10):
            print("INGESTION PHASE")
            
            self.llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0
            )

            start_time = time.time()
            self.vector_store,self.splits,self.embedding_model = encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.time_records= {'FAISS Indexing': time.time() - start_time}
            print(f"FAISS time: {self.time_records['FAISS Indexing']:.2f} seconds")
            
            start_time = time.time()
            self.knowledge_graph=build_knowledge_graph(
                self.splits,self.llm,self.embedding_model
            )
            self.time_records['graph building'] = time.time() - start_time
            print(f" Graph time: {self.time_records['graph building']:.2f}s")

            self.n_retrieved=n_retrieved
            self.chunks_query_retriever =self.vector_store.as_retriever(search_kwargs={"k":n_retrieved})

            print(f"\nIngestion complete. Ready for queries")
            
        
        def run(self,query):
            
            print("\n[1] Query Rewriting")
            start_time=time.time()
            query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
            Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.Dont give anything else except the rewritten query
            Original query: {query}
            Rewritten query:"""
            query_rewrite_prompt = PromptTemplate(
               input_variables=["query"],
              template=query_rewrite_template
               )
            query_rewriter = query_rewrite_prompt | self.llm
            changed_query=query_rewriter.invoke(query).content
            print(f"changed query :{changed_query}")
            self.time_records['query rewrite'] = time.time()-start_time
            print(f"query rewriting time:{self.time_records['query rewrite']:.2f}s")
            

            print("\n[2] vector retrieval (top {n_retrieved} from FAISS)")
            start_time= time.time()
            retrieved_docs = self.chunks_query_retriever.invoke(changed_query)
            print(f"Retrieved {len(retrieved_docs)} chunks")
            self.time_records['vector retrieval']=time.time()-start_time
            print(f"vector retrieval  time:{self.time_records['vector retrieval']:.2f}s")


            n_rerank=min(5,len(retrieved_docs))
            print(f"\n[3] cross-encoder reranking (top {n_rerank})")
            start_time=time.time()
            ranked_results=rerank_documents(changed_query,retrieved_docs,n_retrieved=n_rerank)
            self.time_records['reranking']=time.time()-start_time
            print(f"reranking time:{self.time_records['reranking']:.2f}s")


            print(f"\n[4] graph expansion (dijkstra traversal)")
            start_time=time.time()
            seed_nodes=[]
            for doc,score in ranked_results:
                node_idx=find_node_by_content(self.knowledge_graph.graph,doc.page_content)
                if node_idx is not None:
                    seed_nodes.append((node_idx,score))
            
            if seed_nodes:
                context_texts,traversal_path=expand_context_via_graph(
                    self.knowledge_graph,seed_nodes,max_nodes=8
                )
                if traversal_path:
                    visualize_graph(self.knowledge_graph,traversal_path)
            else:
                print("no graph nodes matched. Using reraked docs directly")
                context_texts=[doc.page_content for doc, _ in ranked_results]
                traversal_path=[]
            
            self.time_records['graph expansion'] =time.time()-start_time
            print(f"graph expansion time:{self.time_records['graph expansion']:.2f}s")

            # print(f"\n[5] final context({len(context_texts)} chunks)")
            # show_context(context_texts)

            print(f"\n[6] answer generation")
            start_time=time.time()

            context_text="\n\n".join(context_texts)

            prompt = f"Based on the following context, answer the question.\n\nContext:\n{context_text}\n\nQuestion: {changed_query}\n\nAnswer:"
            response = self.llm.invoke(prompt).content
            self.time_records['answer generation']=time.time()-start_time
            print(f"answer generation time:{self.time_records['answer generation']:.2f}s")
            print(f"\nAnswer: {response}")
            




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
    parser = argparse.ArgumentParser(description="RAG with knowledge graph")
    
    parser.add_argument("--path",type=str,default="",help="path to the pdf file to encode")
    parser.add_argument("--chunk_size",type=int,default=1000,help="size of each text chunk(default: 1000).")
    parser.add_argument("--chunk_overlap",type=int,default=200,help="overlap between consecutive chuncks (default : 200).")
    parser.add_argument("--n_retrieved" , type=int , default=10 , help="number of chunks to retrieve for each query (default: 2).")
    parser.add_argument("--query",type=str,default="what is the main cause of climate change?", help="query to test the retriever (default: 'what is the main cause of climate change?).")
    parser.add_argument("--evaluate", action="store_true", help="whether to evaluate the retriver's performance (default: false)")
    parser.add_argument("--rebuild", action="store_true",help="Force rebuild indexes even if persisted ones exist")
    return validate_args(parser.parse_args())


# first main(parse_args()) which runs parse_args and validate args and returs args to main function

# in main function we call object of class SimpleRAG , object : simple_rag

# we call method of class : run



def main(args):

    if args.rebuild:
        import shutil
        if os.path.exists("indexes"):
            shutil.rmtree("indexes")
            print("cleared persisted indexes. Rebuilding")

    simple_rag = GraphRAG(
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
    

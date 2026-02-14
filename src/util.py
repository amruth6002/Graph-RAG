import os
import pickle
import heapq
import time
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import nltk
import spacy

from typing import List,Tuple,Dict
from pydantic import BaseModel, Field
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts import PromptTemplate
from spacy.cli import download
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import CrossEncoder
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

nltk.download('punkt',quiet=True)
nltk.download('punkt_tab',quiet=True)
nltk.download('wordnet',quiet=True)

class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="List of concepts")

def replace_t_with_space(list_of_documents):
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t',' ')
    return list_of_documents


def encode_pdf(path, chunk_size=1000, chunk_overlap=200,persist_dir="indexes/faiss"):
    
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(persist_dir):
        print("loading from database")
        vectorstore=FAISS.load_local(persist_dir,embeddings,allow_dangerous_deserialization=True)
        
        splits_path=os.path.join(persist_dir,"splits.pkl")
        if os.path.exists(splits_path):
            with open(splits_path,"rb") as f:
                splits=pickle.load(f)
        else:
            splits=[]
        return vectorstore, splits, embeddings
    
    print("building FAISS index from pdf")
    loader=PyPDFLoader(path)
    documents=loader.load()

    text_splitter=SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        number_of_chunks=None
    )

    splits=text_splitter.split_documents(documents)
    cleaned_splits=replace_t_with_space(splits)
    vectorstore=FAISS.from_documents(cleaned_splits,embeddings)

    os.makedirs(persist_dir,exist_ok=True)
    vectorstore.save_local(persist_dir)
    with open(ps.path.join(persist_dir,"splits.pkl"),"wb") as f:
        pickle.dump(cleaned_splits,f)
    print(f"FAISS index persisted")

    return vectorstore,cleaned_splits,embeddings

class knowledgeGraph:

    def __init__(self):
      self.graph=nx.Graph()
      self.lemmatizer=WordNetLemmatizer()
      self.concept_cache={}
      self.nlp=self._load_spacy_model()
      self.edges_threshold=0.8
    
    def _load_spacy_model(self):
        return spacy.load("en_core_web_sm")
    
    def build_graph(self,splits,llm,embedding_model):
        print("\nbuilding knowledge graphs")

        for i,split in enumerate(splits):
            self.graph.add_node(i,content=split.page_content)
        
        texts = [split.page_content for split in splits]
        embeddings=np.array(embedding_model.embed_documents(texts))

        self._extract_concepts(splits,llm)
        self._add_edges(embeddings)

        print("knowledge graph built")
    
    def _extract_concepts_and_entities(self,content,llm):
        if content in self.concept_cache:
            return self.concept_cache[content]
        
        doc=self.nlp(content)
        named_entities=[ent.text for ent in doc.ents 
                        if ent.label_ in ["PERSON","ORG","GPE","WORK_OF_ART","EVENT","LAW"]]
        
        concept_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts (excluding named entities) from the following text. "
                     "Return JSON with key 'concepts_list' containing a list of strings:\n\n"
                     "{text}\n\nKey concepts:"
        )

        concept_chain = concept_prompt | llm.with_structured_output(Concepts)
        general_concepts = concept_chain.invoke({"text": content}).concepts_list

        all_concepts = list(set(named_entities + general_concepts))
        self.concept_cache[content] = all_concepts
        return all_concepts
    
    def _extract_concepts(self, splits, llm):

        for i, split in enumerate(tqdm(range(len(splits)), desc="Extracting concepts")):
            concepts = self._extract_concepts_and_entities(splits[i].page_content, llm)
            self.graph.nodes[i]['concepts'] = concepts
            
            if i < len(splits) - 1:
                time.sleep(1)

    def _add_edges(self, embeddings):
        similarity_matrix = cosine_similarity(embeddings)
        num_nodes = len(self.graph.nodes)

        for n1 in tqdm(range(num_nodes), desc="Adding edges"):
            for n2 in range(n1 + 1, num_nodes):
                sim = similarity_matrix[n1][n2]
                if sim > self.edges_threshold:
                    shared = set(self.graph.nodes[n1]['concepts']) & set(self.graph.nodes[n2]['concepts'])
                    max_possible = min(len(self.graph.nodes[n1]['concepts']),
                                       len(self.graph.nodes[n2]['concepts']))
                    norm_shared = len(shared) / max_possible if max_possible > 0 else 0
                    weight = 0.7 * sim + 0.3 * norm_shared
                    self.graph.add_edge(n1, n2, weight=weight, similarity=sim,
                                        shared_concepts=list(shared))
    
    def _lemmatize_concept(self, concept):
        return ' '.join([self.lemmatizer.lemmatize(w) for w in concept.lower().split()])


def build_knowledge_graph(splits, llm, embedding_model, persist_path="indexes/knowledge_graph.pkl"):

    if os.path.exists(persist_path):
        print(f"Loading persisted knowledge graph from {persist_path}")
        with open(persist_path, "rb") as f:
            kg = pickle.load(f)
        print(f"Loaded graph: {len(kg.graph.nodes)} nodes, {len(kg.graph.edges)} edges")
        return kg

    kg = KnowledgeGraph()
    kg.build_graph(splits, llm, embedding_model)

    # Persist
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    with open(persist_path, "wb") as f:
        pickle.dump(kg, f)
    print(f"Knowledge graph persisted to {persist_path}")
    return kg

def rerank_documents(question, docs, n_retrieved=5):

    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    query_doc_pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(query_doc_pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:n_retrieved]
    return ranked

def find_node_by_content(graph, content):

    for n in graph.nodes:
        if graph.nodes[n]['content'] == content:
            return n
    return None

def expand_context_via_graph(knowledge_graph, seed_nodes, max_nodes=10):

    graph = knowledge_graph.graph
    if len(graph.nodes) == 0:
        return [], []

    traversal_path = []
    visited_concepts = set()
    context_texts = []

    priority_queue = []
    distances = {}

    # Seed the priority queue with reranked nodes
    for node_idx, score in seed_nodes:
        priority = 1 / score if score > 0 else float('inf')
        heapq.heappush(priority_queue, (priority, node_idx))
        distances[node_idx] = priority

    print("\n  Graph traversal:")

    step = 0
    while priority_queue and step < max_nodes:
        current_priority, current_node = heapq.heappop(priority_queue)

        if current_priority > distances.get(current_node, float('inf')):
            continue

        if current_node in traversal_path:
            continue

        step += 1
        traversal_path.append(current_node)
        node_content = graph.nodes[current_node]['content']
        node_concepts = graph.nodes[current_node].get('concepts', [])
        context_texts.append(node_content)

        print(f"    Step {step} - Node {current_node}: {node_content[:60]}...")
        print(f"      Concepts: {', '.join(node_concepts[:4])}")

        # Only expand to neighbors if this node brings new concepts
        node_concepts_set = set(knowledge_graph._lemmatize_concept(c) for c in node_concepts)
        if not node_concepts_set.issubset(visited_concepts):
            visited_concepts.update(node_concepts_set)

            for neighbor in graph.neighbors(current_node):
                if neighbor in traversal_path:
                    continue
                edge_weight = graph[current_node][neighbor]['weight']
                distance = current_priority + (1 / edge_weight if edge_weight > 0 else float('inf'))

                if distance < distances.get(neighbor, float('inf')):
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

    print(f"    Traversal complete: {len(traversal_path)} nodes visited")
    return context_texts, traversal_path

def show_context(context):
    for i, c in enumerate(context):
        print(f"context {i+1}")
        print(c)
        print("\n")

def visualize_graph(knowledge_graph, traversal_path, save_path="graph_traversal.png"):
    """Saves a visualization of the knowledge graph with traversal path highlighted."""
    graph = knowledge_graph.graph
    if len(graph.nodes) == 0 or len(traversal_path) == 0:
        print("Nothing to visualize.")
        return

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)

    # Draw edges
    edges = list(graph.edges())
    if edges:
        edge_weights = [graph[u][v].get('weight', 0.5) for u, v in edges]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=edge_weights,
                               edge_cmap=plt.cm.Blues, width=1.5, ax=ax, alpha=0.6)

    # Draw all nodes
    non_traversal = [n for n in graph.nodes() if n not in traversal_path]
    nx.draw_networkx_nodes(graph, pos, nodelist=non_traversal, node_color='lightblue',
                           node_size=800, ax=ax, alpha=0.5)

    # Draw traversal nodes (larger)
    traversal_middle = traversal_path[1:-1] if len(traversal_path) > 2 else []
    nx.draw_networkx_nodes(graph, pos, nodelist=traversal_middle, node_color='lightyellow',
                           node_size=2000, ax=ax)

    # Start and end nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=[traversal_path[0]],
                           node_color='lightgreen', node_size=2500, ax=ax)
    if len(traversal_path) > 1:
        nx.draw_networkx_nodes(graph, pos, nodelist=[traversal_path[-1]],
                               node_color='lightcoral', node_size=2500, ax=ax)

    # Draw traversal arrows
    for i in range(len(traversal_path) - 1):
        arrow = patches.FancyArrowPatch(
            pos[traversal_path[i]], pos[traversal_path[i + 1]],
            connectionstyle="arc3,rad=0.3", color='red', arrowstyle="->",
            mutation_scale=20, linestyle='--', linewidth=2, zorder=4
        )
        ax.add_patch(arrow)

    # Labels
    labels = {}
    for i, node in enumerate(traversal_path):
        concepts = graph.nodes[node].get('concepts', [])
        labels[node] = f"{i+1}. {concepts[0][:18] if concepts else ''}"
    for node in graph.nodes():
        if node not in labels:
            concepts = graph.nodes[node].get('concepts', [])
            labels[node] = concepts[0][:15] if concepts else str(node)
    nx.draw_networkx_labels(graph, pos, labels, font_size=6, ax=ax)

    ax.set_title("Knowledge Graph — Traversal Path", fontsize=14)
    ax.axis('off')

    # Legend
    lines = [
        plt.Line2D([0], [0], color='blue', linewidth=2, alpha=0.6, label='Similarity edge'),
        plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Traversal path'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=12, label='Start node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=12, label='End node'),
    ]
    plt.legend(handles=lines, loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Graph visualization saved to {save_path}")
    plt.close()

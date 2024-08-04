import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Carregando o dataset
dataset = pd.read_csv('VendasTesouroDireto.csv', sep=';')
df = pd.DataFrame(dataset)
sentences = df['Tipo Titulo']
list_sentences = df['Tipo Titulo'].unique()
list_sentences = list_sentences.tolist()
ids = [str(i) for i in range(len(list_sentences))]


#Usando diretamente os sentencetransformers

model = SentenceTransformer('h4g3n/multilingual-MiniLM-L12-de-en-es-fr-it-nl-pl-pt')
embeddings = model.encode(list_sentences)
#print(embeddings)


# Usando o Chroma

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="vamo")
collection.add(
    documents=list_sentences,
    ids=ids,
    embeddings=embeddings
)

results = collection.query(
    query_texts=["fala sobre juros"], #embedding automatico chroma
    n_results=2 # numero de resultados mais proximos
)

print(results)
import os
import boto3
import pinecone
import pandas as pd
from sentence_transformers import SentenceTransformer

s3 = boto3.client('s3')
s3.download_file('demo-bucket-tfy', 'input.csv', 'input.csv')

df = pd.read_csv('input.csv')
print(df.shape())

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment="us-west1-gcp"
)

index_name = "extractive-question-answering"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=384,
        metric="cosine"
    )

index = pinecone.Index(index_name)

retriever = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
batch_size = 64

for i in range(0, len(df), batch_size):
    i_end = min(i+batch_size, len(df))
    batch = df.iloc[i:i_end]
    emb = retriever.encode(batch["context"].tolist()).tolist()
    meta = batch.to_dict(orient="records")
    ids = [f"{idx}" for idx in range(i, i_end)]
    to_upsert = list(zip(ids, emb, meta))
    _ = index.upsert(vectors=to_upsert)





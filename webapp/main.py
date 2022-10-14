import pinecone
import gradio as gr
from transformers import pipeline

from sentence_transformers import SentenceTransformer


model_name = "deepset/electra-base-squad2"
# load the reader model into a question-answering pipeline
reader = pipeline(tokenizer=model_name, model=model_name, task="question-answering")

# load the retriever model from huggingface model hub
retriever = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

index_name = "extractive-question-answering"
index = pinecone.Index(index_name)

# gets context passages from the pinecone index
def get_context(question, top_k):
    # generate embeddings for the question
    xq = retriever.encode([question]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    # extract the context passage from pinecone search result
    c = [x["metadata"]["context"] for x in xc["matches"]]
    return c

def extract_answer(question, context):
    results = []
    for c in context:
        # feed the reader the question and contexts to extract answers
        answer = reader(question=question, context=c)
        # add the context to answer dict for printing both together
        answer["context"] = c
        results.append(answer)
    # sort the result based on the score from reader model
    sorted_result = sorted(results, key=lambda x: x["score"], reverse=True)
    return sorted_result


def answer(question):
  context = get_context(question, top_k = 1)
  answer = extract_answer(question, context)
  return answer[0]['answer']

gr.Interface(fn=answer, inputs="text", outputs="text").launch(server_name='0.0.0.0', server_port=8080)
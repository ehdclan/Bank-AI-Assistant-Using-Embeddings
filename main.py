from openai import AzureOpenAI
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

def load_file(path):
    with open(path, "r", encoding='utf-8', errors='ignore') as f:
        document_text = f.read()
    return document_text

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint="https://ugoch-mhrmec5e-eastus2.cognitiveservices.azure.com/",
        api_key=os.getenv("API_KEY"),
)

docs = {
    "bank_policy.txt": lambda: load_file("bank_policy.txt"),
    "loan_policy.txt": lambda: load_file("loan_policy.txt"),
    "faq.txt": lambda: load_file("faq.txt")
}

embeddings = {}

for key, value in docs.items():
    res = client.embeddings.create(
        input=value(),
        model="text-embedding-3-small"
    )
    embeddings[key] = np.array(res.data[0].embedding)

question = "What is the maximum amount for personal loans?"

q_embeded = np.array(client.embeddings.create(
    input=question,
    model="text-embedding-3-small"
).data[0].embedding)

best_doc = None
best_score = -1

def search_for_top_docs(q_embeded, doc_embeddings, top_docs=2):
    scores = []
    for name, emb in doc_embeddings.items():
        score = cosine_similarity(q_embeded, emb)
        scores.append((score, name))

    scores.sort(reverse=True)
    return scores[:top_docs]

top_documents = search_for_top_docs(q_embeded, embeddings, top_docs=2)
best_score, best_doc = top_documents[0]
second_best_score, second_best_doc = top_documents[1]

print(f"Best matching document: {best_doc} with score {best_score:.2f}")
print(f"Second best matching document: {second_best_doc} with score {second_best_score:.2f}")

context = docs[best_doc]()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful banking assistant."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
)
print("Response:", response.choices[0].message.content)
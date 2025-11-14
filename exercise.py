from openai import AzureOpenAI
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

def load_file(path):
    with open(path, "r") as f:
        document_text = f.read()
    return document_text

client = AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint="https://ugoch-mhrmec5e-eastus2.cognitiveservices.azure.com/",
        api_key=os.getenv("API_KEY"),
)

docs = {
    "digital_fraud_patterns.txt": lambda: load_file("digital_fraud_patterns.txt"),
    "account_fraud_patterns.txt": lambda: load_file("account_fraud_patterns.txt"),
    "transaction_fraud_patterns.txt": lambda: load_file("transaction_fraud_patterns.txt"),
    "detection_prevention_patterns.txt": lambda: load_file("detection_prevention_patterns.txt")
}

embeddings = {}

for key, value in docs.items():
    res = client.embeddings.create(
        input=value(),
        model="text-embedding-3-small"
    )
    embeddings[key] = np.array(res.data[0].embedding)

    user_input = input("Enter your question about bank fraud patterns: ")
    q_embeded = np.array(client.embeddings.create(
        input=user_input,
        model="text-embedding-3-small"
    ).data[0].embedding)

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    best_doc = None
    best_score = -1

    for name, emb in embeddings.items():
        score = cosine_similarity(q_embeded, emb)
        if score > best_score:
            first_score = score
            best_doc = name

    print(f"Best matching document: {best_doc} with score {best_score:.2f}")

    context = docs[best_doc]()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful banking assistant. An expert in fraud detection and prevention. Provide information based on the context provided."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {user_input}"}
        ]
    )
    print("Response:", response.choices[0].message.content)
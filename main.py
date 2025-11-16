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

# docs = {
#     "bank_policy.txt": "Transfers above NGN5,000,000 require manager approval.",
#     "loan_terms.txt": "Maximum personal loans is NGN2,000,000",
#     "faq.txt": "You can reset your password in the mobile app settings"
# }

docs = {
    "bank_policy.txt": lambda: load_file("bank_policy.txt"),
    "loan_terms.txt": lambda: load_file("loan_terms.txt"),
    "faq.txt": lambda: load_file("faq.txt")
}

embeddings = {}

for key, value in docs.items():
    res = client.embeddings.create(
        input=value(),
        model="text-embedding-3-small"
    )
    embeddings[key] = np.array(res.data[0].embedding)

    question = ""

    q_embeded = np.array(client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding)

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    best_doc = None
    best_score = -1
    scores = []

    for name, emb in embeddings.items():
        score = cosine_similarity(q_embeded, emb)
        scores = [score]
        if scores[0] > scores[1] and scores[0] > scores[2]:
            best_score = scores[0]
        elif scores[1] > scores[0] and scores[1] > scores[2]:
            best_score = scores[1]
        else:
            best_score = score[2]
        best_doc = name

    print(f"Best matching document: {best_doc} with score {best_score:.2f}")

    context = docs[best_doc]()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful banking assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    print("Response:", response.choices[0].message.content)
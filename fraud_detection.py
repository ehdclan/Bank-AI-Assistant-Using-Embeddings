from openai import AzureOpenAI
from dotenv import load_dotenv
import numpy as np
import os
import random
from datetime import datetime, timedelta
import json

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

# Nigerian-specific transaction data
NIGERIAN_BANKS = ["GTBank", "Zenith Bank", "First Bank", "Access Bank", "UBA", "Fidelity Bank"]
NIGERIAN_LOCATIONS = ["Lagos", "Abuja", "Port Harcourt", "Ibadan", "Kano", "Benin City", "Enugu"]
MERCHANT_TYPES = ["POS", "ATM", "Online Transfer", "Wire Transfer", "Mobile Banking", "USSD"]
TRANSACTION_TYPES = ["withdrawal", "deposit", "transfer", "payment", "purchase"]

def generate_transaction_scenario():
    """Generate realistic Nigerian banking transaction scenarios"""
    
    scenarios = [
        # Normal transactions
        {
            "type": "normal",
            "description": f"POS purchase of ₦{random.randint(5000, 50000)} at supermarket in {random.choice(NIGERIAN_LOCATIONS)}",
            "amount": random.randint(5000, 50000),
            "merchant": "Supermarket",
            "location": random.choice(NIGERIAN_LOCATIONS),
            "time": "14:30"
        },
        {
            "type": "normal", 
            "description": f"Salary deposit of ₦{random.randint(150000, 500000)} from employer",
            "amount": random.randint(150000, 500000),
            "merchant": "Salary",
            "location": "Lagos",
            "time": "09:00"
        },
        
        # Suspicious transactions
        {
            "type": "suspicious",
            "description": f"Multiple POS withdrawals of ₦{random.randint(1000, 5000)} within 10 minutes at same location",
            "amount": random.randint(1000, 5000),
            "merchant": "POS",
            "location": random.choice(NIGERIAN_LOCATIONS),
            "time": "23:45"
        },
        {
            "type": "suspicious",
            "description": f"Large transfer of ₦{random.randint(500000, 2000000)} to unknown beneficiary in China",
            "amount": random.randint(500000, 2000000),
            "merchant": "International Transfer",
            "location": "China",
            "time": "03:15"
        },
        
        # Fraud patterns from our documents
        {
            "type": "fraud_pattern",
            "description": f"Card used in {random.choice(['Lagos', 'Abuja'])} at 14:00 and {random.choice(['Kano', 'Port Harcourt'])} at 15:30 same day",
            "pattern": "geographic_impossibility"
        },
        {
            "type": "fraud_pattern", 
            "description": f"Multiple failed login attempts from IP address in {random.choice(['China', 'Ghana', 'UK'])}",
            "pattern": "account_takeover"
        },
        {
            "type": "fraud_pattern",
            "description": f"Elderly customer's card used for ₦{random.randint(100000, 500000)} at luxury electronics store",
            "pattern": "unusual_merchant"
        },
        {
            "type": "fraud_pattern",
            "description": f"Rapid transactions of ₦{random.randint(10000, 20000)} at same merchant within 5 minutes",
            "pattern": "velocity_violation"
        },
        {
            "type": "fraud_pattern",
            "description": f"SIM swap detected, password reset followed by ₦{random.randint(50000, 300000)} transfer",
            "pattern": "sim_swap"
        },
        {
            "type": "fraud_pattern",
            "description": f"Multiple cash deposits of ₦{random.randint(4500000, 4900000)} to avoid reporting threshold",
            "pattern": "structuring"
        }
    ]
    
    return random.choice(scenarios)

def simulate_real_time_transactions():
    """Simulate real-time transaction monitoring"""
    print("\n" + "="*60)
    print("SIMULATING REAL-TIME BANK TRANSACTIONS")
    print("="*60)
    
    for i in range(10):  # Simulate 10 transactions
        scenario = generate_transaction_scenario()
        
        print(f"\n--- Transaction {i+1} ---")
        print(f"Description: {scenario['description']}")
        print(f"Type: {scenario['type'].upper()}")
        
        if scenario['type'] == 'fraud_pattern':
            print(f"Pattern: {scenario['pattern']}")
        
        # Analyze the transaction
        analyze_transaction(scenario['description'])
        
        # Add delay to simulate real-time
        import time
        time.sleep(2)

def analyze_transaction(transaction_description):
    """Analyze transaction using the existing embedding system"""
    
    docs = {
        "digital_fraud_patterns.txt": lambda: load_file("digital_fraud_patterns.txt"),
        "account_fraud_patterns.txt": lambda: load_file("account_fraud_patterns.txt"),
        "transaction_fraud_patterns.txt": lambda: load_file("transaction_fraud_patterns.txt"),
        "detection_prevention_patterns.txt": lambda: load_file("detection_prevention_patterns.txt")
    }

    # Precompute embeddings (should be done once, not every time)
    embeddings = {}
    for key, value in docs.items():
        res = client.embeddings.create(
            input=value(),
            model="text-embedding-3-small"
        )
        embeddings[key] = np.array(res.data[0].embedding)

    # Use transaction description as input
    q_embedded = np.array(client.embeddings.create(
        input=transaction_description,
        model="text-embedding-3-small"
    ).data[0].embedding)

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    best_doc = None
    best_score = -1

    for name, emb in embeddings.items():
        score = cosine_similarity(q_embedded, emb)
        if score > best_score:
            best_score = score
            best_doc = name

    print(f"📊 Best matching document: {best_doc} (score: {best_score:.2f})")

    # Get detailed analysis
    context = docs[best_doc]()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a fraud detection expert at a Nigerian bank. Analyze if this transaction matches known fraud patterns and explain why."},
            {"role": "user", "content": f"Fraud Patterns Context: {context}\n\nTransaction to Analyze: {transaction_description}\n\nIs this transaction suspicious? Which specific fraud pattern does it match (if any)?"}
        ]
    )
    
    analysis = response.choices[0].message.content
    print(f"🔍 Analysis: {analysis}")

def batch_analyze_transactions():
    """Analyze multiple transactions at once"""
    print("\n" + "="*60)
    print("BATCH TRANSACTION ANALYSIS")
    print("="*60)
    
    transactions = []
    for i in range(5):
        scenario = generate_transaction_scenario()
        transactions.append(scenario['description'])
    
    print("\nGenerated Transactions:")
    for i, transaction in enumerate(transactions, 1):
        print(f"{i}. {transaction}")
    
    # Analyze all transactions
    for i, transaction in enumerate(transactions, 1):
        print(f"\n--- Analyzing Transaction {i} ---")
        analyze_transaction(transaction)

def main():
    """Main function with menu options"""
    while True:
        print("\n" + "="*50)
        print("NIGERIAN BANK FRAUD DETECTION SYSTEM")
        print("="*50)
        print("1. Simulate Real-time Transactions")
        print("2. Batch Analyze Transactions") 
        print("3. Manual Transaction Input")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            simulate_real_time_transactions()
        elif choice == "2":
            batch_analyze_transactions()
        elif choice == "3":
            user_input = input("Enter transaction description to analyze: ")
            analyze_transaction(user_input)
        elif choice == "4":
            print("Exiting system...")
            break
        else:
            print("Invalid choice. Please select 1-4.")

if __name__ == "__main__":
    main()
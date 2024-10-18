import requests
import time
import csv
import matplotlib.pyplot as plt
import numpy as np

base_url = "http://127.0.0.1:5000/classify"

test_cases = [
    {"text": "The government has passed a new law."},
    {"text": "Scientists discover new planet in our solar system."},
    {"text": "Aliens have taken over the world."},
    {"text": "The moon is made of cheese."}
]

def send_request(data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(base_url, json=data, headers=headers)
    return response.json()
   
def test_functional():
    print("\nRunning Functional Tests...")
    for i, case in enumerate(test_cases):
        response = send_request(case) 
        assert 'classification' in response
        print(f"Test case {i+1}: {case['text']} -> {response['classification']}")
  
def test_performance():
    print("\nRunning Performance Tests...")  
    latencies = {i: [] for i in range(4)} 

    with open('latency_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Test Case", "Iteration", "Latency (seconds)"])

        for i, case in enumerate(test_cases):
            print(f"Running 100 requests for Test case {i+1}: {case['text']}")
            for j in range(100): 
                start_time = time.time()
                response = send_request(case)
                end_time = time.time()

                latency = end_time - start_time
                latencies[i].append(latency)
                writer.writerow([i + 1, j + 1, latency])

                assert 'classification' in response
 
    generate_boxplot(latencies)

def generate_boxplot(latencies):
    print("\nGenerating Boxplot for Latency Results...")
    data = [latencies[i] for i in range(4)]
    plt.boxplot(data, labels=[f"Test {i+1}" for i in range(4)])
    plt.title("Latency Performance for Fake News Classifier")
    plt.ylabel("Latency (seconds)") 
    plt.xlabel("Test Case")
    plt.show()

    avg_latencies = [np.mean(latencies[i]) for i in range(4)]
    for i, avg_latency in enumerate(avg_latencies):
        print(f"Average latency for Test Case {i+1}: {avg_latency:.4f} seconds")

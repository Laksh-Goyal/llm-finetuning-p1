import time
import statistics
import requests

API_URL = "http://localhost:8000/generate"

# Sample prompts for testing latency
PROMPTS = [
    "Explain PD and LGD in credit risk.",
    "What is collateralization?",
    "Define DTI ratio and how lenders use it.",
    "Explain the underwriting process."
]

def measure_latency(prompt, iterations=5):
    latencies = []
    for _ in range(iterations):
        start = time.time()
        response = requests.post(API_URL, json={"prompt": prompt})
        end = time.time()

        if response.status_code != 200:
            print("Failed request:", response.text)
            continue

        latencies.append((end - start) * 1000)  # ms
    return latencies


if __name__ == "__main__":
    all_latencies = []

    print("🔥 Running latency benchmark...")

    for p in PROMPTS:
        print(f"\nPrompt: {p}")
        lats = measure_latency(p, iterations=5)
        all_latencies.extend(lats)

        print(f"  p50: {statistics.median(lats):.2f} ms")
        print(f"  avg: {statistics.mean(lats):.2f} ms")
        print(f"  max: {max(lats):.2f} ms")

    print("\n📊 Overall Latency Stats:")
    print(f"  Total Requests: {len(all_latencies)}")
    print(f"  Overall p50: {statistics.median(all_latencies):.2f} ms")
    print(f"  Overall avg: {statistics.mean(all_latencies):.2f} ms")
    print(f"  Overall p90: {statistics.quantiles(all_latencies, n=10)[8]:.2f} ms")
    print(f"  Overall p99: {statistics.quantiles(all_latencies, n=100)[98]:.2f} ms")

    print("\n✅ Latency testing complete.")

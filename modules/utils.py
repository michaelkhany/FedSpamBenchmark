import json

def log_results(results, out_path="benchmark_results.json"):
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {out_path}")

"""
Load Testing Script - Proves 1000+ Concurrent API Checks
Run this AFTER backend is running: python load_test.py
"""
import requests
import concurrent.futures
import time
from datetime import datetime

API_URL = "http://localhost:8000/analyze"
TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]

def make_request(ticker):
    """Single API request"""
    start = time.time()
    try:
        response = requests.post(
            API_URL,
            json={"ticker": ticker, "model_type": "xgboost"},
            timeout=5
        )
        latency = (time.time() - start) * 1000  # Convert to ms
        return {
            "success": response.status_code == 200,
            "latency_ms": latency,
            "ticker": ticker
        }
    except Exception as e:
        return {
            "success": False,
            "latency_ms": -1,
            "ticker": ticker,
            "error": str(e)
        }

def run_load_test(num_requests=1000, max_workers=100):
    """Run load test with concurrent requests"""
    print(f"\n🚀 Starting Load Test: {num_requests} requests")
    print(f"⚡ Max concurrent workers: {max_workers}")
    print(f"🕒 Started at: {datetime.now().strftime('%H:%M:%S')}\n")
    
    # Generate request list (cycle through tickers)
    requests_list = [TICKERS[i % len(TICKERS)] for i in range(num_requests)]
    
    results = []
    start_time = time.time()
    
    # Execute concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(make_request, ticker) for ticker in requests_list]
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results) % 100 == 0:
                print(f"✓ Completed {len(results)}/{num_requests} requests...")
    
    total_time = time.time() - start_time
    
    # Calculate stats
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    latencies = [r['latency_ms'] for r in successful]
    
    print(f"\n{'='*60}")
    print(f"📊 LOAD TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Requests:        {num_requests}")
    print(f"Successful:            {len(successful)} ({len(successful)/num_requests*100:.1f}%)")
    print(f"Failed:                {len(failed)}")
    print(f"Total Time:            {total_time:.2f}s")
    print(f"Requests/sec:          {num_requests/total_time:.2f}")
    print(f"\n📈 LATENCY STATS")
    print(f"{'='*60}")
    if latencies:
        print(f"Average Latency:       {sum(latencies)/len(latencies):.2f}ms")
        print(f"Min Latency:           {min(latencies):.2f}ms")
        print(f"Max Latency:           {max(latencies):.2f}ms")
        print(f"Median Latency:        {sorted(latencies)[len(latencies)//2]:.2f}ms")
        
        # Check sub-300ms requirement
        under_300ms = [l for l in latencies if l < 300]
        print(f"\n✅ Sub-300ms requests:  {len(under_300ms)}/{len(latencies)} ({len(under_300ms)/len(latencies)*100:.1f}%)")
    
    print(f"{'='*60}\n")
    
    # Save results
    with open('load_test_results.txt', 'w') as f:
        f.write(f"Load Test Results - {datetime.now()}\n")
        f.write(f"Total: {num_requests}, Success: {len(successful)}, Failed: {len(failed)}\n")
        f.write(f"Avg Latency: {sum(latencies)/len(latencies):.2f}ms\n")
        f.write(f"Requests/sec: {num_requests/total_time:.2f}\n")
    
    print("💾 Results saved to load_test_results.txt")

if __name__ == "__main__":
    print("⚠️  Make sure backend is running on http://localhost:8000")
    input("Press Enter to start load test...")
    
    # Run with 1000 requests, 100 concurrent workers
    run_load_test(num_requests=1000, max_workers=100)
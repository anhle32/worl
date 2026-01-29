import requests
import json

url = "https://cloudflare-eth.com"
payload = {
    "jsonrpc": "2.0",
    "method": "eth_blockNumber",
    "params": [],
    "id": 1
}
headers = {'Content-Type': 'application/json'}

print(f"Testing connection to {url}...")
try:
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"ERROR: {e}")

# Try alternative
url2 = "https://rpc.ankr.com/eth"
print(f"\nTesting connection to {url2}...")
try:
    response = requests.post(url2, json=payload, headers=headers, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:200]}")
except Exception as e:
    print(f"ERROR: {e}")

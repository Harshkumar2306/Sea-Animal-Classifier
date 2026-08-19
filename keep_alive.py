import time
import datetime
import requests

# Hugging Face Space direct URL
SPACE_URL = "https://harsh0o23-seaanimal-api.hf.space/"
# Alternatively: "https://huggingface.co/spaces/harsh0o23/seaanimal-api"

# Ping interval in seconds (every 10 minutes)
INTERVAL_SECONDS = 600

def ping_space():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        response = requests.get(SPACE_URL, timeout=60)
        if response.status_code == 200:
            print(f"[{current_time}] 🟢 Ping successful (HTTP 200) - Hugging Face Space is alive.")
        else:
            print(f"[{current_time}] ⚠️ Received status code {response.status_code} - Space may be starting up.")
    except requests.exceptions.RequestException as e:
        print(f"[{current_time}] ❌ Ping failed: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🌊 Sea Animal Classifier - Hugging Face Space Keep-Alive")
    print(f"Target URL: {SPACE_URL}")
    print(f"Interval: Every {INTERVAL_SECONDS // 60} minutes")
    print("=" * 60)
    print("Starting periodic ping loop. Press Ctrl+C to stop.\n")
    
    while True:
        ping_space()
        time.sleep(INTERVAL_SECONDS)

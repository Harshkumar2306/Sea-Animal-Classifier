import requests
import urllib.parse

def get_agent_research(label: str) -> dict:
    try:
        # Manual overrides for specific classes
        overrides = {
            "Sea Rays": "Batoidea",
            "Turtle_Tortoise": "Turtle",
            "Puffers": "Pufferfish",
            "Eel": "Eel",
            "Seal": "Pinniped"
        }
        
        search_label = overrides.get(label, label)
        
        headers = {
            "User-Agent": "SeaAnimalClassifierBot/2.0 (contact@example.com)"
        }
        
        # Helper function to get summary
        def fetch_summary(title):
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}"
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                data = r.json()
                if data.get("type") == "disambiguation":
                    return None # Try searching instead
                return data
            return None

        # 1. Try exact label
        data = fetch_summary(search_label)
        
        # 2. Try singular if it ends with 's'
        if not data and search_label.endswith('s'):
            data = fetch_summary(search_label[:-1])
            
        # 3. If still no data, perform a search query
        if not data:
            search_term = search_label[:-1] if search_label.endswith('s') else search_label
            query = f"{search_term} marine animal"
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
            
            search_res = requests.get(search_url, headers=headers)
            if search_res.status_code == 200:
                search_data = search_res.json()
                results = search_data.get('query', {}).get('search', [])
                if results:
                    first_title = results[0]['title']
                    data = fetch_summary(first_title)
                    
        if not data:
            return {"error": f"No information found for {label}."}

        summary = data.get("extract", "")
        # Limit to ~1000 chars gracefully
        if len(summary) > 1000:
            summary = summary[:1000] + "..."
            
        return {
            "title": data.get("title", search_label),
            "summary": summary,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
        }

    except Exception as e:
        return {"error": str(e)}

import requests

def get_evidence(claim: str) -> str:
    try:
        # Use first 8 words as search query
        search_query = " ".join(claim.split()[:8])
        
        # Step 1 — Search Wikipedia
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "format": "json",
            "srlimit": 3
        }
        headers = {"User-Agent": "HallucinationDetector/1.0"}
        
        search_response = requests.get(
            search_url, 
            params=search_params, 
            headers=headers,
            timeout=10
        )
        search_data = search_response.json()
        results = search_data.get("query", {}).get("search", [])
        
        if not results:
            return "No evidence found"
        
        # Step 2 — Get page summary
        page_title = results[0]["title"]
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}"
        
        summary_response = requests.get(
            summary_url,
            headers=headers,
            timeout=10
        )
        
        if summary_response.status_code == 200:
            data = summary_response.json()
            extract = data.get("extract", "")
            if extract and len(extract) > 50:
                return extract[:500]
        
        return "No evidence found"
    
    except Exception as e:
        return "No evidence found"


if __name__ == "__main__":
    test_claims = [
        "Einstein was born in 1879 in Germany",
        "The telephone was invented by Thomas Edison",
        "Americans drink more coffee than people in which other countries?",
        "Python programming language was created by Guido van Rossum"
    ]
    
    for claim in test_claims:
        print(f"\nClaim: {claim}")
        evidence = get_evidence(claim)
        print(f"Evidence: {evidence[:200]}...")
        print("-" * 60)
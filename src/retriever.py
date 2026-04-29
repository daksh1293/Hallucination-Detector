import wikipedia

def get_evidence(claim: str) -> str:
    try:
        # Truncate to first 8 words to avoid search length errors
        search_query = " ".join(claim.split()[:8])
        
        results = wikipedia.search(search_query, results=5)
        if not results:
            return "No evidence found"
        
        for result in results:
            try:
                page = wikipedia.summary(result, sentences=5, auto_suggest=False)
                return page
            except:
                continue
        
        return "No evidence found"
    
    except Exception as e:
        return f"Error: {str(e)}"


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
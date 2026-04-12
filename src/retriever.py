import wikipedia

def get_evidence(claim: str) -> str:
    try:
        results = wikipedia.search(claim, results=5)
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
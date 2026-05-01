from transformers import pipeline
from retriever import get_evidence

# Load NLI model (downloads once, ~1.6GB)
print("Loading NLI model... (first time takes 2-3 mins)")
nli_model = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
print("Model loaded!")

def check_hallucination(claim: str) -> dict:
    evidence = get_evidence(claim)
    
    # Catch ALL cases where evidence is missing/invalid
    if (not evidence or 
        "No evidence found" in evidence or 
        "Error" in evidence or
        "No Wikipedia" in evidence or
        len(evidence.strip()) < 50):
        return {
            "claim": claim,
            "evidence": "No evidence found",
            "support_score": 0.0,
            "contradiction_score": 0.0,
            "verdict": "⚠️ UNVERIFIABLE"
        }
    
    # Rest of your existing code stays the same...
    evidence_short = evidence[:400]
    nli_input = f"premise: {evidence_short} hypothesis: {claim}"
    scores = nli_model(
        nli_input,
        candidate_labels=["true", "false"]
    )
    
    support_score = scores['scores'][0]
    contradiction_score = scores['scores'][1]
    
    verdict = "✅ GROUNDED" if support_score > 0.5 else "❌ HALLUCINATION"
    
    return {
        "claim": claim,
        "evidence": evidence_short,
        "support_score": round(support_score, 3),
        "contradiction_score": round(contradiction_score, 3),
        "verdict": verdict
    }
# Test it
if __name__ == "__main__":
    test_claims = [
        "Einstein was born in 1879 in Germany",
        "The telephone was invented by Thomas Edison",
        "Python programming language was created by Guido van Rossum"
    ]
    
    for claim in test_claims:
        print(f"\nClaim: {claim}")
        result = check_hallucination(claim)
        print(f"Verdict: {result['verdict']}")
        print(f"Support: {result['support_score']} | Contradiction: {result['contradiction_score']}")
        print(f"Evidence: {result['evidence'][:150]}...")
        print("-" * 60)
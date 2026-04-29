import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import get_evidence
from nli_scorer import check_hallucination
from llm_response import get_llm_response

def run_pipeline(question: str, model: str = "llama3-70b") -> dict:
    llm_answer = get_llm_response(question, model)
    result = check_hallucination(llm_answer)
    
    return {
        "question": question,
        "model": model,
        "llm_answer": llm_answer,
        "evidence": result["evidence"],
        "support_score": result["support_score"],
        "contradiction_score": result["contradiction_score"],
        "verdict": result["verdict"]
    }
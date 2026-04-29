from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ── Model registry ─────────────────────────────────────────────────────────────
# All models run via Groq API — no local Ollama needed
MODELS = {
     "llama3-70b":   "llama-3.3-70b-versatile",                  # Meta — large
    "llama3-8b":    "llama-3.1-8b-instant",                     # Meta — small
    "llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct", # Meta — newest
}

PROMPT_TEMPLATE = (
    "Answer this factual question in exactly 1-2 sentences. "
    "Be specific and direct: {question}"
)


def get_llm_response(question: str, model: str = "llama3-70b") -> str:
    """
    Get a factual answer from one of the 4 supported Groq models.

    Args:
        question: The factual question to answer.
        model:    One of 'llama3-70b', 'llama3-8b', 'gemma2-9b', 'qwen3-32b'.

    Returns:
        The model's answer as a string, or an error message.
    """
    if model not in MODELS:
        supported = ", ".join(MODELS.keys())
        return f"Error: Unknown model '{model}'. Supported: {supported}"

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not set in environment."

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=MODELS[model],
            messages=[{
                "role": "user",
                "content": PROMPT_TEMPLATE.format(question=question)
            }],
            max_tokens=150,
            temperature=0.1,   # low temp → more factual, consistent answers
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"


def get_all_models() -> list[str]:
    """Return the list of supported model keys."""
    return list(MODELS.keys())

def get_model_display_name(model: str) -> str:
    display = {
         "llama3-70b":   "llama-3.3-70b-versatile",                  # Meta — large
         "llama3-8b":    "llama-3.1-8b-instant",                     # Meta — small
         "llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct", # Meta — newest
    }
    return display.get(model, model)

# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "Who invented the telephone?",
        "What is the capital of Australia?",
    ]

    for model_key in MODELS:
        print(f"\n{'='*60}")
        print(f"  {get_model_display_name(model_key)} ({MODELS[model_key]})")
        print(f"{'='*60}")
        for q in questions:
            print(f"  Q: {q}")
            answer = get_llm_response(q, model=model_key)
            print(f"  A: {answer}")
            print(f"  {'-'*55}")
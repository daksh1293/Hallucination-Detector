import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from pipeline import run_pipeline

def evaluate_model(model: str = "mistral"):
    df = pd.read_csv("data/truthfulqa_mapped.csv")
    results = []
    total = len(df)
    
    print(f"\nEvaluating {model} on {total} questions...")
    print("=" * 70)
    
    for i, row in df.iterrows():
        result = run_pipeline(row["question"], model=model)
        predicted_hallucination = result["contradiction_score"] > 0.4
        
        # Show everything in real time
        print(f"\n📌 Question {i+1}/{total} [{row['domain']}]")
        print(f"❓ Question:    {row['question']}")
        print(f"🤖 LLM Answer:  {result['llm_answer'][:150]}...")
        print(f"📚 Evidence:    {result['evidence'][:150]}...")
        print(f"📊 Support:     {result['support_score']} | Contradiction: {result['contradiction_score']}")
        print(f"🔍 Verdict:     {result['verdict']}")
        print("-" * 70)
        
        results.append({
            "question": row["question"],
            "domain": row["domain"],
            "llm_answer": result["llm_answer"],
            "evidence": result["evidence"],
            "verdict": result["verdict"],
            "support_score": result["support_score"],
            "contradiction_score": result["contradiction_score"],
            "predicted_hallucination": predicted_hallucination,
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"data/{model}_results.csv", index=False)
    
    # Final summary
    hallucination_rate = results_df["predicted_hallucination"].mean()
    avg_support = results_df["support_score"].mean()
    avg_contradiction = results_df["contradiction_score"].mean()
    
    print(f"\n{'='*70}")
    print(f"📈 FINAL EVALUATION RESULTS - {model.upper()}")
    print(f"{'='*70}")
    print(f"Total Questions:       {total}")
    print(f"Hallucination Rate:    {hallucination_rate:.2%}")
    print(f"Avg Support Score:     {avg_support:.3f}")
    print(f"Avg Contradiction:     {avg_contradiction:.3f}")
    
    print(f"\n📊 DOMAIN-WISE HALLUCINATION RATES:")
    print("-" * 40)
    for domain in ["Science", "History", "Geography", "Technology"]:
        domain_df = results_df[results_df["domain"] == domain]
        domain_rate = domain_df["predicted_hallucination"].mean()
        avg_conf = domain_df["contradiction_score"].mean()
        print(f"{domain:15} → {domain_rate:.2%} hallucination | avg contradiction: {avg_conf:.3f}")
    
    print(f"\n🚨 TOP 5 MOST LIKELY HALLUCINATIONS:")
    print("-" * 40)
    top_hall = results_df.nlargest(5, "contradiction_score")
    for _, row in top_hall.iterrows():
        print(f"\n[{row['domain']}] {row['question']}")
        print(f"Answer:  {row['llm_answer'][:100]}...")
        print(f"Contradiction Score: {row['contradiction_score']}")
    
    print(f"\n✅ TOP 5 MOST GROUNDED ANSWERS:")
    print("-" * 40)
    top_ground = results_df.nlargest(5, "support_score")
    for _, row in top_ground.iterrows():
        print(f"\n[{row['domain']}] {row['question']}")
        print(f"Answer:  {row['llm_answer'][:100]}...")
        print(f"Support Score: {row['support_score']}")
    
    print(f"\n💾 Full results saved to data/{model}_results.csv")
    return results_df

if __name__ == "__main__":
    models = ["llama3-70b", "llama3-8b", "llama4-scout"]
    
    for model in models:
        print(f"\n{'='*70}")
        print(f"Evaluating: {model.upper()}")
        print(f"{'='*70}")
        evaluate_model(model=model)
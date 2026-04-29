from datasets import load_dataset
import pandas as pd

dataset = load_dataset("truthful_qa", "generation")
df = pd.DataFrame(dataset["validation"])

def map_category(cat):
    cat = str(cat).lower()
    if any(x in cat for x in ["science", "health", "nutrition", "medicine", "astronomy", "weather", "psychology"]):
        return "Science"
    elif any(x in cat for x in ["history", "myth", "religion", "war", "politics", "misquot", "mandela", "misconception"]):
        return "History"
    elif any(x in cat for x in ["geography", "places", "location", "sociology", "confusion: place", "indexical error: location"]):
        return "Geography"
    elif any(x in cat for x in ["technology", "computer", "math", "logic", "finance", "economics", "education", "statistics"]):
        return "Technology"
    else:
        return "General"

df["domain"] = df["category"].apply(map_category)

print("After mapping:")
print(df["domain"].value_counts())

# Take 25 per domain
df_filtered = df[df["domain"] != "General"]

df_balanced = (
    df_filtered
    .groupby("domain", group_keys=False)
    .apply(lambda x: x.sample(min(25, len(x)), random_state=42))
    .reset_index(drop=True)
)

df_final = df_balanced[["question", "domain", "best_answer"]].copy()

df_final.to_csv("data/truthfulqa_mapped.csv", index=False)

print(f"\nFinal dataset: {len(df_final)} questions")
print(df_final["domain"].value_counts())
print("\nSample questions:")
print(df_final[["question", "domain"]].head(8).to_string())
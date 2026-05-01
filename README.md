# 🔍 LLM Hallucination Detector

> *Cross-model factual consistency analysis using NLI-based evidence verification*

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-yellow)](https://daksh1293-hallucination-detector.hf.space)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![TruthfulQA](https://img.shields.io/badge/Benchmark-TruthfulQA-orange)](https://github.com/sylinrl/TruthfulQA)

---

## Abstract

Large language models have demonstrated remarkable capabilities across a wide range of tasks, yet they remain prone to generating factually incorrect statements — a phenomenon commonly referred to as *hallucination*. This project presents an automated hallucination detection system that retrieves real-time evidence from Wikipedia and applies a Natural Language Inference (NLI) model to assess whether a given LLM response is factually supported or contradicted by that evidence.

I evaluated three open-weight and API-accessible models — **LLaMA 3.1 8B**, **LLaMA 4 Scout 17B**, and **LLaMA 3.3 70B** — on a stratified sample of 100 questions drawn from the **TruthfulQA** benchmark (Lin et al., 2022), covering four knowledge domains: Science, History, Geography, and Technology. Our results reveal that model size does not reliably predict lower hallucination rates, and that domain-specific knowledge gaps vary significantly across architectures.

---

## Motivation

The rapid deployment of LLMs in production systems has made hallucination detection a critical area of research. Existing approaches often require white-box access to model internals or ground-truth reference answers — constraints that limit their practical applicability. This project explores a retrieval-augmented, black-box verification approach that works entirely on model outputs, making it applicable to any LLM regardless of architecture or provider.

The question driving this work is straightforward but underexplored:

> *Do larger language models hallucinate less, and does the domain of a question affect hallucination rates across different model families?*

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Question                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   LLM Response  │  (Groq API — 3 models)
                    └────────┬────────┘
                             │
               ┌─────────────▼─────────────┐
               │   Wikipedia Retriever      │  (Evidence fetching)
               │   Top-5 search results     │
               │   500-char summary         │
               └─────────────┬─────────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │   NLI Scorer (DeBERTa-v3-small)      │
          │   Premise:    Wikipedia evidence      │
          │   Hypothesis: LLM answer             │
          │   Output:     Entailment / Contradiction / Neutral │
          └──────────────────┬──────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │         Verdict              │
              │  ✅ GROUNDED                 │
              │  ❌ HALLUCINATION            │
              │  ⚠️  UNCERTAIN               │
              │  🔎 UNVERIFIABLE             │
              └─────────────────────────────┘
```

---

## Key Findings

### Finding 1 — Bigger Is Not Always Better

Perhaps the most counterintuitive result: **LLaMA 3.3 70B hallucinated at a higher rate (24%) than LLaMA 3.1 8B (14%)** on the TruthfulQA benchmark. This suggests that scale alone is not a sufficient condition for factual accuracy, and that training data composition and instruction tuning strategies may matter more than raw parameter count.

| Model | Parameters | Hallucination Rate | Avg. Support Score |
|---|---|---|---|
| LLaMA 3.1 8B | 8B | **14%** 🏆 | 0.728 |
| LLaMA 4 Scout | 17B | 15% | **0.734** 🏆 |
| LLaMA 3.3 70B | 70B | 24% | 0.721 |

### Finding 2 — Domain Matters More Than Model Size

Hallucination rates varied dramatically by knowledge domain, often more than they varied by model size. History and Technology proved consistently harder for all models, while Science questions were handled with greater accuracy.

| Domain | LLaMA 3.1 8B | LLaMA 4 Scout | LLaMA 3.3 70B |
|---|---|---|---|
| Science | 8% | 28% | 16% |
| History | 8% | 16% | **36%** |
| Geography | 12% | 8% | 8% |
| Technology | 28% | 8% | **36%** |

History and Technology showed the highest hallucination rates across models, likely due to the prevalence of common misconceptions and frequently changing factual landscapes in these domains — precisely the kind of questions TruthfulQA is designed to probe.

### Finding 3 — LLaMA 4 Scout Leads on Confidence Quality

Despite having only 17B parameters, **LLaMA 4 Scout achieved the highest average support score (0.734)**, suggesting that architectural improvements in the LLaMA 4 generation contribute more meaningfully to factual grounding than parameter scale alone. This aligns with recent literature on the importance of post-training alignment over raw capacity.

### Finding 4 — Retrieval Quality Is The Bottleneck

Approximately 15-20% of questions returned no usable Wikipedia evidence, classified as *UNVERIFIABLE*. This highlights a known limitation of retrieval-based verification: the system is only as good as the evidence it can access. Questions involving subjective claims, recent events, or topics underrepresented in Wikipedia are inherently difficult to verify by this method.

---

## Dataset

We use a stratified sample from **TruthfulQA** (Lin et al., 2022), a benchmark of 817 adversarially constructed questions specifically designed to elicit hallucinations from language models. Unlike standard factual QA benchmarks, TruthfulQA targets questions where models trained on large corpora tend to reproduce common misconceptions.

| Property | Value |
|---|---|
| Source | TruthfulQA validation split |
| Total questions | 100 |
| Questions per domain | 25 |
| Domains | Science, History, Geography, Technology |
| Sampling strategy | Stratified random (seed=42) |

Category mapping from TruthfulQA's 38 original categories to our 4 domains was performed based on semantic similarity, with General category questions excluded to maintain domain coherence.

---

## Technical Stack

| Component | Technology | Role |
|---|---|---|
| LLM Inference | Groq API | Fast inference for 3 models |
| Evidence Retrieval | Wikipedia Python API | Real-time evidence fetching |
| NLI Scoring | DeBERTa-v3-small (cross-encoder) | Factual consistency scoring |
| Evaluation | TruthfulQA benchmark | Standardised hallucination measurement |
| Frontend | Streamlit + Plotly | Interactive research dashboard |
| Deployment | HuggingFace Spaces (Docker) | Public live demo |

The NLI model choice — `cross-encoder/nli-deberta-v3-small` — was deliberate. While larger NLI models like BART-large-MNLI offer marginally higher performance on standard NLI benchmarks, their memory footprint (1.6GB+) creates practical deployment constraints. DeBERTa-v3-small achieves competitive NLI accuracy at approximately 180MB, making it well-suited for constrained inference environments.

---

## Project Structure

```
hallucination-detector/
│
├── src/
│   ├── retriever.py          # Wikipedia evidence fetching
│   ├── nli_scorer.py         # DeBERTa NLI-based hallucination scoring
│   ├── llm_response.py       # Groq API interface for 3 LLMs
│   ├── pipeline.py           # End-to-end inference pipeline
│   ├── evaluator.py          # Batch evaluation on TruthfulQA
│   ├── load_dataset.py       # TruthfulQA preprocessing & domain mapping
│   └── analyze_results.py    # Results analysis & visualisation
│
├── app/
│   └── streamlit_app.py      # Interactive research dashboard
│
├── data/
│   ├── truthfulqa_mapped.csv       # Preprocessed evaluation dataset
│   ├── llama3-8b_results.csv       # LLaMA 3.1 8B evaluation results
│   ├── llama4-scout_results.csv    # LLaMA 4 Scout evaluation results
│   ├── llama3-70b_results.csv      # LLaMA 3.3 70B evaluation results
│   └── model_comparison.png        # Comparison visualisation
│
├── app.py                    # HuggingFace Spaces entry point
├── Dockerfile                # Container configuration
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Running Locally

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier available)

### Setup

```bash
git clone https://github.com/daksh1293/hallucination-detector
cd hallucination-detector

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Running the Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`

### Running the Evaluation

```bash
# Preprocess TruthfulQA dataset
python src/load_dataset.py

# Run evaluation for all 3 models (~45-60 mins)
python src/evaluator.py

# Generate comparison charts
python src/analyze_results.py
```

---

## Limitations

**Retrieval quality.** The system's ability to detect hallucinations is fundamentally constrained by Wikipedia coverage. For questions involving recent events, specialised technical knowledge, or topics underrepresented in Wikipedia, the system defaults to UNVERIFIABLE rather than risk an unreliable verdict. This affects approximately 15-20% of TruthfulQA questions.

**NLI as a proxy for factual accuracy.** NLI models measure textual entailment between a premise and hypothesis. While this correlates with factual consistency, it is not identical to it. A model may confidently contradict a poorly written Wikipedia passage even when the model's answer is correct, leading to false positive hallucination detections.

**Short-form answers only.** The current system evaluates single-claim answers (1-2 sentences). Multi-paragraph responses with mixed accuracy would require sentence-level decomposition and claim extraction — an extension left for future work.

**Domain coverage.** The four domains used in this evaluation — Science, History, Geography, and Technology — do not exhaustively represent the space of factual knowledge. Domains such as Medicine, Law, and Finance may exhibit meaningfully different hallucination patterns and deserve dedicated study.

---

## Related Work

This project draws on and extends the following line of research:

- **Manakul et al. (2023).** *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP 2023. Proposes using consistency across multiple model samples as a hallucination signal — a complementary approach to retrieval-based verification.

- **Min et al. (2023).** *FActScoring: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation.* ACL 2023. Introduces atomic claim decomposition for fine-grained hallucination evaluation, a direction this project could extend to longer outputs.

- **Lin et al. (2022).** *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL 2022. The benchmark used in this evaluation. Demonstrates that larger models are not necessarily more truthful — a finding our results partially corroborate.

- **He et al. (2021).** *DeBERTa: Decoding-enhanced BERT with Disentangled Attention.* ICLR 2021. The NLI backbone used in this project, which achieves strong NLI performance with a compact parameter footprint.

---

## Future Work

Several natural extensions suggest themselves. First, integrating multi-source retrieval — combining Wikipedia with news archives and scientific databases — would substantially improve evidence coverage, particularly for recency-sensitive and domain-specific questions. Second, decomposing longer model responses into atomic claims before verification (following the FActScoring framework) would enable sentence-level analysis rather than response-level verdicts. Third, a calibration study examining the relationship between NLI confidence scores and true hallucination rates across domains would help establish empirical thresholds for production deployment.

---


## Citation

If you use this work or find it useful, please cite the relevant benchmarks and models:

```bibtex
@inproceedings{lin2022truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  booktitle={ACL},
  year={2022}
}

@inproceedings{manakul2023selfcheckgpt,
  title={SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection},
  author={Manakul, Potsawee and Liusie, Adian and Gales, Mark JF},
  booktitle={EMNLP},
  year={2023}
}
```

---

*Built with Python, Streamlit, HuggingFace Transformers, and the Groq API.*
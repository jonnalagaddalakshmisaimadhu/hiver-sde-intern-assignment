# Grounded Customer Support AI Agent & Trustworthy Evaluation Harness

## 1. Project Overview & Summary
This repository contains an end-to-end AI customer support agent designed for a single brand from real-world, multi-turn Twitter customer support data ([Kaggle: `thoughtvector/customer-support-on-twitter`](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)). Given an incoming customer message, the system:
1. **Classifies intent** into a focused, empirically grounded taxonomy derived directly from the brand's operational data.
2. **Retrieves historical support resolutions** matching the customer issue using dense semantic embeddings.
3. **Generates an evidence-grounded reply** that strictly reflects historical brand behavior without hallucinating policies, actions, or promises.
4. **Decides whether to AUTO_HANDLE or ESCALATE TO HUMAN** with an explicit, auditable rationale.
5. **Evaluates trustworthiness** across intent classification, reply quality, and escalation safety using a 150–250 example hand-labeled Golden Evaluation Set, two distinct baselines, an LLM-as-a-judge rubric, and human vs. LLM judge agreement validation.

The guiding engineering philosophy of this project is: **"The proof is worth more than the system."** We prioritize correctness, reproducibility, transparent error reporting, and honest analysis of headline metrics over superficial demo scores.

---

## 2. Problem Framing
Customer support on public social media (such as Twitter/X) presents distinct operational challenges:
* **Noisy input**: Slang, colloquial abbreviations, typos, emotional frustration, and incomplete context.
* **High cost of failure**: A false `AUTO_HANDLE` decision that gives inaccurate advice, promises unauthorized compensation, or ignores an urgent account security compromise damages customer trust and brand reputation.
* **Grounding requirement**: An agent must not make up answers from general pre-training knowledge. It must ground responses in how the specific brand actually handles inquiries.
* **Escalation necessity**: When historical evidence is insufficient, ambiguous, or involves sensitive account/financial actions, the agent must reliably escalate to human agents.

---

## 3. Selected Brand & Rationale: @AppleSupport
Following empirical profiling across the top candidate brands in the Twitter Customer Support dataset (`results/brand_profiling_report.json`), **`AppleSupport`** was selected as the target brand based on four concrete engineering criteria:

1. **Highest Technical Troubleshooting Density (47.64%)**:
   Unlike retail or airline brands whose outbound tweets are predominantly status lookups or canned redirects (`AmazonHelp` had 16.48% troubleshooting density; `Delta` had 7.93%), `AppleSupport` contains rich, actionable diagnostic procedures (e.g. iOS upgrade steps, battery health checks, network reset commands, iCloud sync guidance).
2. **Language Cleanliness (100.0% English)**:
   Inspection revealed 0.0% non-English tweets for `AppleSupport`, whereas `AmazonHelp` had 5.65% multilingual inquiries (German, Japanese, Hindi, Spanish) which introduce extraneous language confounding to evaluation.
3. **Rich Conversational Depth (106,860 tweets)**:
   `AppleSupport` features 99.87% inbound reply coverage and 29.54% multi-turn follow-up depth, providing ample historical resolution evidence for semantic retrieval and multi-turn thread evaluation.
4. **Operationally Well-Defined Intent Taxonomy**:
   Apple consumer inquiries naturally segment into 7 distinct, operationally actionable categories (OS/Software Updates, Battery & Power, Apple ID & Security, Audio & Bluetooth, Hardware & Physical Damage, App Store & Subscriptions, and General Inquiries), making evaluation objective and defensible.


---

## 4. Dataset Information
* **Primary Dataset**: [Customer Support on Twitter (Kaggle)](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (~3 million tweets, multi-turn support dialogues across multiple companies).
* **Secondary Dataset (Optional)**: [Banking77 (Hugging Face)](https://huggingface.co/datasets/PolyAI/banking77) (only for intent-related comparative exploration if relevant, never replacing primary Twitter evaluation data).
* **Expected Raw Data Path**: `data/raw/` (e.g. `data/raw/twcs.csv`).
* **Sample Data Path**: `data/sample/` (a reproducible subsample for <15 min verification).
* **Golden Evaluation Set**: `golden_set/` (strictly separated from retrieval and training corpora to prevent contamination).

---

## 5. Preliminary System Architecture

The AI agent follows a modular, evidence-first pipeline where every downstream decision is grounded in retrieved brand history:

```
                      +----------------------------------+
                      |    Incoming Customer Message     |
                      +----------------------------------+
                                        |
                                        v
                      +----------------------------------+
                      |   1. Intent Classification       |
                      |  (Empirical Brand Taxonomy)      |
                      +----------------------------------+
                                        |
                    +-------------------+-------------------+
                    |                                       |
                    v                                       v
+------------------------------------+  +------------------------------------+
| 2. Dense Semantic Retrieval        |  | 3. Pre-generation Escalation Check |
| (Historical Brand Support Corpus)  |  | (Intent ambiguity / urgent keywords|
+------------------------------------+  +------------------------------------+
                    |                                       |
                    v                                       |
+------------------------------------+                      |
| Evidence Filtering & Re-ranking    |                      |
| (Check similarity & sufficiency)   |                      |
+------------------------------------+                      |
                    |                                       |
                    v                                       |
+------------------------------------+                      |
| 4. Grounded Reply Generation       |                      |
| (Constrained prompt, anti-halluc.) |                      |
+------------------------------------+                      |
                    |                                       |
                    +-------------------+-------------------+
                                        |
                                        v
                      +----------------------------------+
                      | 5. Escalation Decision Engine    |
                      | - AUTO_HANDLE vs ESCALATE        |
                      | - Rule triggers + Evidence check |
                      | - Explicit rationale             |
                      +----------------------------------+
                                        |
                                        v
                      +----------------------------------+
                      | 6. Structured Agent Output       |
                      | {intent, reply, decision,        |
                      |  reason, evidence}               |
                      +----------------------------------+
```

### Agent Components
1. **Intent Classifier (`src/intents/`)**: Predicts the customer's operational intent using a validated schema constrained to the brand's verified taxonomy.
2. **Historical Retriever (`src/retrieval/`)**: Searches indexed historical customer inquiries and brand resolutions using sentence embeddings (`all-MiniLM-L6-v2`) and vectorized cosine similarity.
3. **Reply Generator (`src/generation/`)**: Produces brand-aligned responses grounded strictly in retrieved historical evidence, barring fabricated commitments or unauthorized policy changes.
4. **Escalation Engine (`src/escalation/`)**: Evaluates evidence sufficiency, intent confidence, and safety rules (e.g., account security, refunds, legal threats, abusive interactions) to decide between `AUTO_HANDLE` and `ESCALATE`.
5. **Output Formatter (`src/agent/`)**: Emits a strongly validated Pydantic JSON structure containing `intent`, `reply`, `decision`, `reason`, and `evidence`.

---

## 6. Baselines
To evaluate the agent objectively, performance is measured against two standard baselines evaluated on the exact same Golden Evaluation Set:
* **Baseline 1 (Trivial / Majority-Class)**: Predicts the most frequent intent from the training distribution for all incoming messages.
* **Baseline 2 (Conventional ML)**: TF-IDF feature extraction followed by Multinomial / One-vs-Rest Logistic Regression (`scikit-learn`). Fast, reproducible, and deterministic.

---

## 7. Evaluation Methodology & Metrics

### A. Intent Classification Metrics
Evaluated against hand-labeled ground truth:
* Overall Accuracy
* Macro-averaged F1 (primary metric to penalize poor performance on minority intents)
* Per-intent Precision, Recall, and F1-score
* Full Confusion Matrix

### B. Escalation Quality Metrics
Evaluated against hand-labeled escalation ground truth:
* Escalation Precision, Recall, and F1-score
* **False Auto-Handle Rate (FAHR)**: The critical safety metric measuring how often the system erroneously auto-handles messages that required human intervention.

### C. Reply Quality & LLM-as-a-Judge Rubric
Evaluated using structured scoring (1–5 scale) on:
1. **Relevance**: Does the reply directly address the customer's specific query?
2. **Groundedness**: Is every statement in the reply supported by retrieved historical evidence?
3. **Helpfulness**: Does the reply provide an actionable next step?
4. **Factual Integrity / Non-Hallucination**: Does the reply refrain from inventing policies, accounts, or fake promises?
5. **Brand Consistency**: Does the style match historical support resolutions?
6. **Tone**: Is the tone polite, empathetic, and professional?

### D. Human vs. LLM Judge Agreement
A representative subset of generated replies is evaluated by a human annotator using the identical rubric to measure:
* Inter-annotator percent agreement
* Pearson / Spearman correlation for continuous/ordinal ratings
* Cohen's Kappa ($\kappa$) for categorical alignment
* Qualitative analysis of systematic judge biases or disagreements

---

## 8. Golden Evaluation Set Methodology
* **Size**: 150–250 hand-labeled examples.
* **Sampling Strategy**: Conversation-level stratified sampling across intents, conversation length, message complexity, ambiguity, and customer sentiment to avoid unrealistically simple evaluation.
* **Contamination Prevention**: Strict exclusion of Golden Set threads from vector indexing, embedding corpora, and training sets.

---

## 9. Project Directory Structure
```
hiver-sde-intern-assignment/
├── README.md                # Project documentation, reproduction guide, and reports
├── requirements.txt         # Pinned Python package dependencies
├── .env.example             # Environment variable template (no secrets committed)
├── .gitignore               # Excludes virtual environments, secrets, large data
├── decision_log.md          # 10-15 non-obvious engineering decisions (Decision/Why/Trade-off)
│
├── data/
│   ├── raw/                 # Raw dataset (twcs.csv) - not tracked in git
│   ├── processed/           # Reconstructed conversations and cleaned splits
│   └── sample/              # Deterministic subsample for fast (<15 min) reproduction
│
├── src/
│   ├── data/                # Data inspection, cleaning, conversation reconstruction
│   ├── intents/             # Taxonomy definitions, classifier models, and schemas
│   ├── retrieval/           # Embedding indexer, cosine similarity search, evidence filter
│   ├── generation/          # Grounded reply generator, prompt templates, anti-hallucination
│   ├── escalation/          # Escalation policies, safety triggers, decision logic
│   ├── agent/               # End-to-end agent pipeline and structured output schemas
│   └── utils/               # Config loaders, seeding, logging, LLM client wrappers
│
├── baselines/               # Baseline 1 (Majority-class) and Baseline 2 (TF-IDF + LogReg)
├── evaluation/              # Metrics calculator, LLM judge, and agreement validator
├── golden_set/              # 150-250 hand-labeled evaluation examples and labeling guide
├── configs/                 # YAML / JSON configs for models, taxonomies, and rules
├── tests/                   # Unit and integration tests (pytest)
├── notebooks/               # Exploratory data analysis and inspection notebooks
├── results/                 # Machine-readable output JSONs, confusion matrices, prediction CSVs
└── reports/                 # Final technical report and failure analysis summaries
```

---

## 10. Environment Setup & Reproduction

### Prerequisites
* Python 3.10+ (tested on Python 3.11.9)
* Git

### Step-by-Step Installation
1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd hiver-sde-intern-assignment
   ```

2. **Create and activate a virtual environment**:
   * Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * Linux / macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and provide your API keys:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to configure your API key (`GEMINI_API_KEY` or `OPENAI_API_KEY`).

5. **Run Automated Tests**:
   ```bash
   pytest tests/
   ```

---

## 11. What is Misleading About My Headline Number?
> *Mandatory evaluation section. This section will be populated with empirical findings during Phase 11 & 12.*

Every headline metric hides trade-offs:
* **Accuracy vs. Class Imbalance**: High headline accuracy can easily be achieved by predicting majority intents while failing catastrophic edge cases.
* **Retrieval Similarity vs. True Grounding**: High embedding similarity does not guarantee historical responses apply to current brand policy.
* **Low Escalation Rate vs. False Auto-Handle Risk**: A model claiming a 90% auto-handle rate may be dangerously suppressing critical customer account issues.

---

## 12. Limitations & One-More-Week Plan
> *Detailed analysis of edge cases, multi-turn limitations, and prioritized engineering improvements to follow evaluation.*

---

## 13. Citations & Attribution
* **Twitter Customer Support Dataset**: [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
* **Sentence Embeddings**: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (Reimers & Gurevych, 2019).
* **Machine Learning**: `scikit-learn` (Pedregosa et al., 2011).

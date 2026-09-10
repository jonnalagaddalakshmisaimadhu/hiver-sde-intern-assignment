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

## 5. Intent Taxonomy (@AppleSupport)
Derived empirically from reconstructed `@AppleSupport` dialogues, the system classifies customer messages into **8 mutually-exclusive operational intents** (7 domain-specific + 1 justified fallback). Full criteria and examples are in [`configs/intent_taxonomy.md`](file:///c:/Users/jlaks/Downloads/AI-Projects/hiver-sde-intern-assignment/configs/intent_taxonomy.md) and [`configs/intents.yaml`](file:///c:/Users/jlaks/Downloads/AI-Projects/hiver-sde-intern-assignment/configs/intents.yaml).

| Intent ID | Name | Core Diagnostic Scope | Default System Route |
| :--- | :--- | :--- | :--- |
| `SOFTWARE_UPDATE_OS` | Software Update & OS Issues | iOS/macOS update failures, verification hangs, update bugs | AUTO_HANDLE (KB steps) |
| `BATTERY_POWER_CHARGING` | Battery, Power & Charging | Fast drain, shutdown at 20%, charging cable issues, overheating | AUTO_HANDLE (Diagnostics) |
| `APPLE_ID_ACCOUNT_SECURITY` | Apple ID & Account Security | Locked accounts, 2FA code issues, password resets, iCloud sync | AUTO_HANDLE / ESCALATE (Security) |
| `AUDIO_CONNECTIVITY_BLUETOOTH` | Audio & Wireless Connectivity | AirPods pairing, speaker crackle, Wi-Fi/cellular signal drops | AUTO_HANDLE (Reset steps) |
| `HARDWARE_PHYSICAL_DAMAGE` | Hardware & Physical Damage | Cracked display, water contact, broken buttons, repair booking | **ESCALATE TO HUMAN / GENIUS BAR** |
| `APP_STORE_BILLING_SUBSCRIPTIONS` | App Store & Subscriptions | Unauthorized charges, subscription refunds, Apple Pay errors | AUTO_HANDLE / ESCALATE (Billing) |
| `DEVICE_PERFORMANCE_CRASH` | Device Crashes & Boot Loops | Stuck on Apple logo, frozen UI, black screen, app crashes | AUTO_HANDLE (Force restart) |
| `OTHER_OR_UNCLEAR` | Other, Ambiguous or Feedback | Vague complaints, lack of technical symptoms, general praise/critique | Prompt for details / ESCALATE |

---

## 6. System Architecture


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

---

## 7. Comparative Experimental Results

All three systems were evaluated on the exact same **200-example Golden Evaluation Set** (strictly isolated from training and retrieval indices, uniformly balanced across 8 operational intents):

| Metric | Baseline 1 (Majority Class) | Baseline 2 (TF-IDF + LogReg) | AI Support Agent (@AppleSupport) | Delta (Agent vs. Best Baseline) |
| :--- | :--- | :--- | :--- | :--- |
| **Intent Accuracy** | 12.50% | 70.50% | **99.00%** (198/200) | **+28.50%** |
| **Macro-Precision** | 1.56% | 68.42% | **99.01%** | **+30.59%** |
| **Macro-Recall** | 12.50% | 70.50% | **99.00%** | **+28.50%** |
| **Macro-F1 Score** | 0.0278 | 0.6829 | **0.9899** | **+0.3070** |
| **Escalation Recall** | N/A | N/A | **100.00%** (25/25) | Optimal Safety |
| **False Auto-Handle Rate (FAHR)**| N/A | N/A | **0.00%** (0 dangerous leaks) | Zero Breach |
| **Escalation Precision** | N/A | N/A | **18.09%** (Risk-Averse) | Conservative Guardrail |
| **Factual Integrity** | N/A | N/A | **5.00 / 5.00** | Zero Hallucination |
| **Retrieval Groundedness**| N/A | N/A | **2.015 / 5.00** | Defensive Routing |

---

## 8. LLM-as-a-Judge & Human Calibration

Evaluated across 6 dimensions defined in `configs/judge_rubric.yaml` on a 1–5 scale, with 40 stratified human evaluations:

| Dimension | Judge Mean | Human Mean | Exact Match | Adjacent Match (+/-1) | Cohen's Kappa ($\kappa_w$) | Judge Bias |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Relevance** | 4.00 / 5.00 | 3.00 / 5.00 | 0.0% | **100.0%** | 0.000 | +1.000 (Lenient) |
| **Groundedness** | 2.00 / 5.00 | 2.00 / 5.00 | **100.0%** | **100.0%** | 1.000 | 0.000 (Calibrated) |
| **Helpfulness** | 4.00 / 5.00 | 3.00 / 5.00 | 0.0% | **100.0%** | 0.000 | +1.000 (Lenient) |
| **Factual Integrity** | 5.00 / 5.00 | 5.00 / 5.00 | **100.0%** | **100.0%** | 1.000 | 0.000 (Flawless) |
| **Brand Consistency** | 4.05 / 5.00 | 3.98 / 5.00 | **95.0%** | **97.5%** | 0.000 | +0.075 |
| **Tone** | 4.05 / 5.00 | 4.00 / 5.00 | **95.0%** | **100.0%** | 0.000 | +0.050 |
| **Macro Average** | **3.85 / 5.00** | **3.50 / 5.00** | **65.0%** | **99.6%** | **0.231** | **+0.354** |

---

## 9. Project Directory Structure
```
hiver-sde-intern-assignment/
├── README.md                # Project documentation, reproduction guide, and reports
├── requirements.txt         # Pinned Python package dependencies
├── .env.example             # Environment variable template (no secrets committed)
├── .gitignore               # Excludes virtual environments, secrets, large data
├── decision_log.md          # 22 non-obvious engineering decisions (Decision/Why/Trade-off)
│
├── data/
│   ├── raw/                 # Raw dataset (twcs.csv) - not tracked in git
│   ├── processed/           # Reconstructed conversations (103,842 dialogues)
│   └── sample/              # Deterministic subsample (5,000 dialogues) for fast reproduction
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
├── evaluation/              # Metrics calculator, LLM judge, agreement analyzer, failure audit
├── golden_set/              # 200 hand-labeled evaluation examples and labeling guide
├── configs/                 # YAML configs for intents, judge rubric, and operational rules
├── tests/                   # 18 unit and integration tests (pytest)
├── results/                 # Machine-readable output JSONs, confusion matrices, prediction CSVs
└── reports/                 # Comprehensive final report and failure analysis documents
```

---

## 10. Complete Reproduction Workflow (< 15 Minutes)

### Prerequisites
* Python 3.10+ (tested on Python 3.11.9)
* Git

### Execution Steps
```bash
# 1. Clone repository
git clone <repo-url>
cd hiver-sde-intern-assignment

# 2. Set up virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY or GEMINI_API_KEY

# 5. Run full test suite (18 tests in ~12 seconds)
pytest

# 6. Evaluate Baseline 1 (Majority Class)
python baselines/majority_class.py

# 7. Evaluate Baseline 2 (TF-IDF + Logistic Regression)
python baselines/tfidf_logistic_regression.py

# 8. Evaluate Proposed Agent Pipeline on Golden Set
python evaluation/evaluator.py

# 9. Run LLM-as-a-Judge Evaluation
python evaluation/llm_judge.py

# 10. Run Human vs. Judge Agreement Validation
python evaluation/judge_agreement.py

# 11. Run Quantitative & Qualitative Failure Analysis
python evaluation/failure_analyzer.py
```

---

## 11. What is Misleading About My Headline Number?

* **"99.0% Intent Accuracy" is Misleading**: Measured on a uniformly balanced test set (25 examples per intent). In real-world traffic, distributions are skewed toward OS update and device freeze inquiries (>45%), where multi-symptom complaints are frequent and messy.
* **"100% Escalation Recall and 0% FAHR" is Misleading**: Achieved zero missed escalations only by escalating 69% of all inbound inquiries (138 out of 200). Escalation Precision was only **18.09%**. In an enterprise call center, this would swamp human agent tiers and destroy the economic value of automated customer support.
* **"5.00/5.00 Factual Integrity" is Misleading**: Zero hallucinations were achieved by adopting a defensive diagnostic posture (asking intake questions and inviting DMs) rather than attempting autonomous problem resolution.
* **"Baseline 2 had 70.5% Accuracy" is Misleading**: Aggregate accuracy masked that Baseline 2 had **0.0% Recall on physical damage**, misclassifying every single cracked screen inquiry into generic categories.

---

## 12. Limitations & One-More-Week Plan

### Current Limitations
1. **Single-Turn Request-Reply**: The agent does not currently maintain persistent multi-turn conversational session state across sequential tweets.
2. **Defensive Canned Redirection**: Relying on historical Twitter support data inherits the brand's canned DM deflection habit rather than providing self-service resolutions.
3. **Conservative Escalation Threshold**: The static retrieval threshold (0.38) triggers excessive human escalation on queries with novel colloquial phrasing.

### What I Would Do With One More Week
1. **Authoritative Knowledge Base Ingestion**: Supplement historical Twitter logs with indexed Apple Support HT articles to offer true step-by-step inline diagnostic fixes.
2. **Multi-Label Intent Modeling**: Model customer complaints as compound graphs (Primary Trigger vs. Secondary Symptom).
3. **Dynamic Threshold Optimization**: Tune escalation thresholds using Pareto-optimal validation ROC curves to boost Escalation Precision from 18% to > 70% while maintaining > 95% Recall.
4. **Dialogue State Tracker**: Implement conversation memory across multi-turn customer dialogues using Redis or session stores.

---

## 13. Citations & Attribution
* **Twitter Customer Support Dataset**: [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
* **Sentence Embeddings**: Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (Reimers & Gurevych, 2019).
* **Machine Learning**: `scikit-learn` (Pedregosa et al., 2011).


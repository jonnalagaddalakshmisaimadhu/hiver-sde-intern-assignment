# Golden Evaluation Set Labeling Guide & Protocol

## 1. Overview & Objectives
The Golden Evaluation Set comprises **200 authentic customer support dialogues** from the `@AppleSupport` Twitter corpus (`thoughtvector/customer-support-on-twitter`). It serves as the single source of ground truth for evaluating:
1. **Intent Classification** (Macro-F1, per-intent accuracy, confusion matrix).
2. **Escalation Decisions** (Precision, Recall, F1, and False Auto-Handle Rate).
3. **Evidence-Grounded Reply Quality** (Relevance, Grounding, Non-Hallucination, Brand Tone).

This document establishes the labeling schema, sampling stratification, train/eval isolation, and quality control procedures.

---

## 2. Sampling Methodology & Stratification
To prevent an unrealistically simple evaluation set, sampling is stratified across four operational dimensions:

1. **Intent Stratification**:
   - Uniform distribution across all 8 intents (exactly **25 examples per intent**).
   - Guarantees sufficient test power on rare intents (`HARDWARE_PHYSICAL_DAMAGE`, `APP_STORE_BILLING_SUBSCRIPTIONS`) which would otherwise constitute <3% of raw volume.
2. **Message Length Stratification**:
   - Short (< 50 chars): Concise, terse inquiries.
   - Medium (50–120 chars): Typical single-issue tweets.
   - Long (> 120 chars): Complex multi-symptom descriptions.
   - Overall mean character length: **126.5 characters**.
3. **Conversational Complexity**:
   - Includes both standalone inquiries and dialogues with follow-up turns.
4. **Escalation & Ambiguity Stratification**:
   - 164 `AUTO_HANDLE` examples (actionable technical procedures).
   - 36 `ESCALATE` examples (physical damage, financial billing disputes, security flags).
   - 25 ambiguous / noisy edge cases (`OTHER_OR_UNCLEAR`).

---

## 3. Train / Evaluation Separation & Anti-Contamination Audit
* **Conversation-Level Isolation**: All 200 Golden Set examples are identified by unique `conversation_id` (`apple_<tweet_id>`).
* **Retrieval Corpus Masking**: The retriever and embedding index explicitly filter out and exclude all `conversation_id`s present in `golden_set/golden_evaluation_set.jsonl`.
* **Zero Leakage**: Golden Set conversations are never seen during Baseline 2 (TF-IDF) fitting or historical nearest-neighbor search.

---

## 4. Labeling Schema & Definitions

Each example in `golden_evaluation_set.jsonl` follows this validated schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `example_id` | `str` | Canonical evaluation identifier (`golden_001` to `golden_200`). |
| `conversation_id` | `str` | Source conversation ID in raw dataset (`apple_<tweet_id>`). |
| `customer_inquiry` | `str` | Raw, unnormalized customer tweet text. |
| `true_intent` | `str` | Assigned intent from the 8 official taxonomy classes. |
| `true_escalation_decision` | `str` | Binary ground truth: `AUTO_HANDLE` or `ESCALATE`. |
| `escalation_rationale` | `str` | Concrete rationale explaining why human escalation is/isn't required. |
| `ground_truth_brand_reply`| `str` | Authentic historical resolution provided by Apple Support. |
| `is_ambiguous` | `bool` | Flag denoting ambiguous or noisy syntax. |
| `message_length_chars` | `int` | Length of inquiry in characters. |
| `turn_count` | `int` | Total turns in original conversation. |

---

## 5. Escalation Decision Rules

### When to label `ESCALATE`:
1. **Physical Hardware Damage**: Any broken glass, liquid submersion, damaged casing, or broken physical buttons. AI cannot perform physical hardware repairs.
2. **Account Security Threats**: Unauthorized account access, suspected hacking, fraudulent SIM swaps, or compromised Apple IDs.
3. **Financial / Billing Disputes**: Contested credit card charges, refund disputes over unauthorized debit, or legal threats.
4. **Lack of Actionable Information**: Inquiries under 4 words or pure emotional rants where the diagnostic tree cannot initiate.

### When to label `AUTO_HANDLE`:
1. Technical questions with standardized, published Apple Knowledge Base troubleshooting steps (e.g. force restart, reset network settings, update iOS, check iCloud sync toggles, repair AirPods).
2. Diagnostic clarification questions where asking the customer for device model/OS version is the established, safe first turn.

---

## 6. Quality Control & Validation Procedure
1. **Schema Enforcement**: Every example is strictly parsed and validated against the Pydantic `GoldenExample` model.
2. **Boundary Disambiguation**:
   - *OS Update vs. Performance*: If the customer explicitly mentions an iOS version update causing the crash, it is labeled `SOFTWARE_UPDATE_OS`. If a general boot loop or freeze occurs without update context, it is labeled `DEVICE_PERFORMANCE_CRASH`.
   - *Battery vs. Hardware*: Swollen batteries or physically damaged charging ports are labeled `HARDWARE_PHYSICAL_DAMAGE`. Standard battery capacity degradation is labeled `BATTERY_POWER_CHARGING`.
3. **Auditability**: Both `.jsonl` and `.csv` formats are committed to the repository for interactive inspection.

# Engineering Decision Log

This document records the key architectural, methodological, and engineering decisions made throughout the lifecycle of the Hiver SDE Intern customer support AI system. Every entry documents what was decided, the concrete rationale, and the explicit trade-offs accepted.

---

### Decision 1: Strict Modular Python Package Architecture
* **Decision**: Organize the codebase into decoupled functional subpackages under `src/` (`data/`, `intents/`, `retrieval/`, `generation/`, `escalation/`, `agent/`, `utils/`) alongside independent top-level directories for `baselines/`, `evaluation/`, and `golden_set/`.
* **Why**: Each component of the assignment has distinct testing, evaluation, and operational characteristics. Decoupling retrieval from generation and escalation ensures each module can be unit-tested, mocked, and evaluated in isolation without state leakage.
* **Trade-off**: Slightly more initial boilerplate and import scaffolding compared to a single flat script, but critical for testability, auditability, and team collaboration.

---

### Decision 2: Pure NumPy / SciKit-Learn Vector Retrieval over External Vector Databases
* **Decision**: Implement vector similarity search for historical support conversations using precomputed dense sentence embeddings with pure NumPy vectorized cosine similarity (or `scikit-learn.metrics.pairwise.cosine_similarity`) rather than heavy external vector databases (e.g., ChromaDB, Pinecone, or Faiss C++ bindings).
* **Why**: The retrieval corpus for a single selected brand's support history consists of thousands to tens of thousands of conversations. NumPy vectorized matrix multiplication executes in milliseconds in pure memory, has zero external daemon requirements, zero Windows C++ build/wheel incompatibility issues, and guarantees 100% deterministic reproducibility across platforms.
* **Trade-off**: Does not scale to billions of vectors without indexing structures like HNSW or IVF, but for a brand-specific support corpus (<100K items), it is orders of magnitude simpler, faster to set up, and less failure-prone.

---

### Decision 3: Pydantic v2 Contracts for All Agent and Judge Schemas
* **Decision**: Use Pydantic v2 models to define and strictly validate all input/output payloads, including incoming requests, predicted intents, structured agent responses (`intent`, `reply`, `decision`, `reason`, `evidence`), and LLM-as-a-judge multi-dimensional scorecards.
* **Why**: Prevents malformed LLM outputs from crashing downstream evaluation pipelines. Enforces deterministic type checking, automatic coercion, and runtime schema validation with clear error messages when an LLM hallucinates an invalid field or value.
* **Trade-off**: Requires structured JSON generation prompts and validation handlers that catch parsing errors and trigger fallback/retry logic if the LLM output violates the schema.

---

### Decision 4: Deterministic Global Random Seeding (Seed = 42)
* **Decision**: Enforce a global deterministic random seed (`RANDOM_SEED=42`) across all sampling operations (data subsampling, Golden Set stratification, train/test splitting, baseline model training).
* **Why**: A cornerstone requirement of this evaluation is reproducibility. Reviewers and interviewers must be able to run the exact same script and obtain identical dataset splits, sample indices, and baseline metric calculations.
* **Trade-off**: Fixes the test split to a single realization. (Mitigated by documenting stratified sampling properties and sensitivity in later phases).

---

### Decision 5: Deferral of Column Names, Brand Selection, and Labeling Schema until Dataset Inspection
* **Decision**: Refuse to hardcode dataset column names, brand identifiers, conversation schemas, or intent taxonomies during Phase 1 setup.
* **Why**: In real-world data engineering and according to the project specifications, assuming schemas before inspecting raw data leads to brittle code and unverified assumptions. True data-driven engineering inspects raw schemas, conversation linkage columns, and brand distributions first.
* **Trade-off**: Cannot write concrete parsing logic or training scripts in Phase 1; all data engineering must be grounded in Phase 2 empirical findings.

---

### Decision 6: Dual Provider Configuration (Google Gemini & OpenAI) via Environment Variables
* **Decision**: Design the configuration system (`.env.example` and `configs/`) to support both Google Gemini and OpenAI models without modifying core agent logic, defaulting to Google Gemini (or configurable via `LLM_PROVIDER`).
* **Why**: Ensures maximum flexibility for evaluation reviewers who may have different API access keys or budget constraints, while preventing any hardcoded credentials.
* **Trade-off**: Requires maintaining lightweight provider-agnostic client abstractions or standard API adapters in `src/utils/` and `src/generation/`.

---

### Decision 7: Memory-Efficient Chunked Streaming for Inspection of 2.8M Rows
* **Decision**: Process the 516 MB raw CSV in bounded chunks (250,000 rows per chunk) rather than reading all 2.8M records into memory simultaneously.
* **Why**: Loading 2.81M rows with multiple string columns directly into a single Pandas DataFrame consumes over 3–4 GB of RAM, causing memory pressure and potential out-of-memory crashes on resource-constrained environments. Chunked streaming processed the entire corpus in 12.6 seconds with < 500 MB peak memory.
* **Trade-off**: Cannot perform full-table multi-column group-bys in a single vector operation; requires incremental accumulators for frequencies and null counts.

---

### Decision 8: Authentic Full Dataset Download Combined with Fast Subsample Architecture
* **Decision**: Ingest the complete authentic `twcs.csv` dataset (2,811,774 rows) into `data/raw/` for comprehensive brand distribution analysis and candidate profiling, while committing to generate a deterministic, self-contained subsample in `data/sample/` for reviewers to reproduce all results in under 15 minutes.
* **Why**: Brand selection and Golden Set sampling require analyzing true operational distributions across the entire dataset to avoid sampling bias, while reviewers need a lightweight, fast-running benchmark.
* **Trade-off**: Requires ~516 MB disk space for the raw file, but guarantees zero artificiality in brand selection.

---

### Decision 9: Author Categorization Heuristic (Numeric vs. Alphabetic Handles)
* **Decision**: Categorize records where `author_id` is numeric as anonymized customer accounts (702,669 unique authors) and non-numeric strings as brand support handles (108 unique brands).
* **Why**: The Kaggle TWCS dataset explicitly anonymizes customer user IDs into numeric identifiers while preserving actual corporate Twitter handles (e.g. `AmazonHelp`, `AppleSupport`, `SpotifyCares`) to allow brand identification.
* **Trade-off**: Edge-case accounts with alphanumeric usernames that are not official brands are theoretically possible, but inspection of the top 50 handles confirmed 100% precision for known commercial support entities.


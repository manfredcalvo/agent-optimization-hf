# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Prompt Optimization with Judge Alignment
# MAGIC ## Workshop Overview
# MAGIC
# MAGIC This workshop demonstrates how to build better LLM applications using **MLflow 3** on **Databricks** —
# MAGIC combining **judge alignment** with **automated prompt optimization** for a banking use case.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Problem
# MAGIC
# MAGIC Building LLM-powered applications involves two fundamental challenges:
# MAGIC
# MAGIC 1. **How do you evaluate quality?** Generic LLM judges lack domain expertise.
# MAGIC    A judge might think a complaint classification is correct because it "looks reasonable,"
# MAGIC    while a banking compliance expert knows it violates regulatory requirements.
# MAGIC
# MAGIC 2. **How do you improve prompts systematically?** Manual prompt engineering is slow,
# MAGIC    subjective, and doesn't scale. When your pipeline has multiple chained prompts,
# MAGIC    changing one affects all downstream outputs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Solution: Two-Phase Approach
# MAGIC
# MAGIC ### Phase 1: Align the Judge (Notebook 01)
# MAGIC
# MAGIC Before optimizing prompts, we need an evaluator that understands our domain.
# MAGIC
# MAGIC **MemAlign** teaches an LLM judge to match human expert preferences using a dual-memory system:
# MAGIC
# MAGIC | Memory Type | What it stores | How it helps |
# MAGIC |-------------|---------------|--------------|
# MAGIC | **Semantic Memory** | Distilled guidelines from feedback patterns | General rules: "regulatory violations require priority escalation" |
# MAGIC | **Episodic Memory** | Specific examples with embeddings | Similar past cases for few-shot context during evaluation |
# MAGIC
# MAGIC **Example**: A generic judge rates `"Department: Mortgages | Priority: Medium"` as correct for a PMI overcharge complaint.
# MAGIC After alignment, the judge knows that PMI overcharges violate the Homeowners Protection Act and should be **High** priority.
# MAGIC
# MAGIC ### Phase 2: Optimize the Prompts (Notebook 02)
# MAGIC
# MAGIC With a domain-aware judge, we can now automatically optimize our prompts.
# MAGIC
# MAGIC **GEPA** (Genetic-Pareto) iteratively improves prompts through:
# MAGIC 1. Evaluate current prompts on training data
# MAGIC 2. Reflect on failures using an LLM
# MAGIC 3. Propose improved prompt text
# MAGIC 4. Test new variants and keep the best

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Case: Customer Complaint Classification
# MAGIC
# MAGIC A two-stage pipeline for a bank's complaint routing system:
# MAGIC
# MAGIC ```
# MAGIC Customer Complaint
# MAGIC        │
# MAGIC        ▼
# MAGIC ┌──────────────────┐
# MAGIC │  Prompt 1:       │     Extracts: financial product, issue type,
# MAGIC │  Analyzer        │──►  customer sentiment, urgency indicators
# MAGIC └──────────────────┘
# MAGIC        │
# MAGIC        ▼
# MAGIC ┌──────────────────┐
# MAGIC │  Prompt 2:       │     Outputs: Department + Priority
# MAGIC │  Router          │──►  e.g., "Department: Credit Cards | Priority: Critical"
# MAGIC └──────────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Departments**: Credit Cards, Mortgages, Personal Banking, Investment, Insurance
# MAGIC
# MAGIC **Priorities**: Critical, High, Medium, Low
# MAGIC
# MAGIC ### Why two prompts?
# MAGIC
# MAGIC Separating analysis from classification lets each prompt focus on one task.
# MAGIC It also demonstrates **multi-prompt optimization** — GEPA optimizes both prompts
# MAGIC simultaneously, handling interdependencies (changes to the analyzer affect what the router receives).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Workshop Notebooks
# MAGIC
# MAGIC | # | Notebook | What it does | Key APIs |
# MAGIC |---|---------|--------------|----------|
# MAGIC | 01 | **Align Judge** | Create an LLM judge, collect human feedback, align with MemAlign, register | `make_judge()`, `MemAlignOptimizer`, `mlflow.log_feedback()` |
# MAGIC | 02 | **Optimize Prompts** | Register prompts, baseline eval, GEPA optimization, post-optimization eval | `register_prompt()`, `optimize_prompts()`, `mlflow.genai.evaluate()` |
# MAGIC
# MAGIC ### Flow
# MAGIC
# MAGIC ```
# MAGIC Notebook 01                              Notebook 02
# MAGIC ┌─────────────────────┐                 ┌─────────────────────────────┐
# MAGIC │                     │                 │                             │
# MAGIC │ 1. Create judge     │                 │ 1. Register 2 prompts       │
# MAGIC │ 2. Generate traces  │                 │ 2. Define predict_fn        │
# MAGIC │ 3. Log human        │  experiment_id  │ 3. Baseline evaluate()      │
# MAGIC │    feedback          │ ──────────────► │ 4. GEPA optimize_prompts()  │
# MAGIC │ 4. Log judge        │                 │ 5. Post-opt evaluate()      │
# MAGIC │    feedback          │                 │ 6. Compare metrics          │
# MAGIC │ 5. Align with       │                 │                             │
# MAGIC │    MemAlign          │                 └─────────────────────────────┘
# MAGIC │ 6. Register aligned │
# MAGIC │    judge             │
# MAGIC │                     │
# MAGIC └─────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Concepts
# MAGIC
# MAGIC ### Judge Alignment with MemAlign
# MAGIC
# MAGIC **Why**: Generic LLM judges miss domain-specific nuances. Banking compliance rules,
# MAGIC regulatory requirements, and institutional policies aren't captured in a model's pre-training.
# MAGIC
# MAGIC **How**: MemAlign learns from human feedback rationales — not just "this is wrong" but
# MAGIC "this is wrong because PMI overcharges violate the Homeowners Protection Act."
# MAGIC The rationales are distilled into reusable guidelines (semantic memory) and stored
# MAGIC as retrievable examples (episodic memory).
# MAGIC
# MAGIC **Result**: A judge that evaluates like a domain expert — catching subtle regulatory
# MAGIC violations, elder abuse patterns, and understanding that customer tone ≠ priority.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Prompt Optimization with GEPA
# MAGIC
# MAGIC **Why**: Manual prompt engineering doesn't scale, especially with chained prompts
# MAGIC where changes cascade.
# MAGIC
# MAGIC **How**: GEPA uses a genetic algorithm approach — it evaluates prompt variants,
# MAGIC reflects on what went wrong, proposes mutations, and selects the fittest prompts.
# MAGIC All prompt versions are tracked in the MLflow Prompt Registry.
# MAGIC
# MAGIC **Result**: Optimized prompts that score higher on both exact-match accuracy
# MAGIC and the aligned judge's domain-aware evaluation.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### MLflow Prompt Registry
# MAGIC
# MAGIC Prompts are versioned in Unity Catalog. The optimizer creates new versions
# MAGIC automatically — you can always compare, roll back, or deploy a specific version.
# MAGIC
# MAGIC ```
# MAGIC prompts:/catalog.schema.complaint_analyzer/1  ← original
# MAGIC prompts:/catalog.schema.complaint_analyzer/2  ← optimized by GEPA
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## What You'll See in MLflow
# MAGIC
# MAGIC After running both notebooks, the MLflow experiment UI will show:
# MAGIC
# MAGIC ### Notebook 01 — Traces with 3 assessments per trace:
# MAGIC - `classification_quality` (HUMAN) — expert feedback
# MAGIC - `classification_quality` (LLM_JUDGE) — initial judge assessment
# MAGIC - `classification_quality_aligned` (LLM_JUDGE) — aligned judge assessment
# MAGIC
# MAGIC ### Notebook 02 — Two evaluation runs:
# MAGIC - **Baseline run** — original prompts scored by all scorers
# MAGIC - **Post-optimization run** — optimized prompts scored by all scorers
# MAGIC - Side-by-side metric comparison (department accuracy, priority accuracy, exact match, aligned judge)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Let's Get Started!
# MAGIC
# MAGIC Open **Notebook 01 (01_align_judge)** to begin.

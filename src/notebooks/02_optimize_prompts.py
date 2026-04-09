# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Prompt Optimization for Customer Complaint Classification
# MAGIC
# MAGIC This notebook optimizes **two chained prompts** for a banking customer complaint classification pipeline:
# MAGIC 1. **Complaint Analyzer** — extracts product, issue type, sentiment, and urgency from raw complaint text
# MAGIC 2. **Complaint Router** — uses the analysis to classify into Department + Priority
# MAGIC
# MAGIC We use MLflow's `optimize_prompts` API with the **GEPA optimizer** (Genetic-Pareto) to
# MAGIC iteratively improve both prompts simultaneously, scored by the **aligned judge** from Notebook 1
# MAGIC and custom accuracy scorers.
# MAGIC
# MAGIC ### Prerequisites
# MAGIC - Run **Notebook 1 (01_align_judge)** first to create and register the aligned judge
# MAGIC - Note the `experiment_id` printed at the end of Notebook 1

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]>=3.5.0" dspy databricks-openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from databricks_openai import DatabricksOpenAI
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import scorer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Widgets for job parameters
dbutils.widgets.text("judge_experiment_id", "", "Judge Experiment ID (from Notebook 1)")
dbutils.widgets.text("max_metric_calls", "10", "Max metric calls for GEPA optimizer")

# COMMAND ----------

JUDGE_EXPERIMENT_ID = dbutils.widgets.get("judge_experiment_id")
MAX_METRIC_CALLS = int(dbutils.widgets.get("max_metric_calls"))

MODEL_ENDPOINT = "databricks-gpt-5-4-mini"
MODEL_URI = f"databricks:/{MODEL_ENDPOINT}"
OPTIMIZER_MODEL = "databricks:/databricks-claude-haiku-4-5"

# Databricks OpenAI client — auto-configured with workspace credentials
client = DatabricksOpenAI()

print(f"Model: {MODEL_ENDPOINT}")
print(f"Judge Experiment ID: {JUDGE_EXPERIMENT_ID or '(not set — will use code-based scorers only)'}")
print(f"Max metric calls: {MAX_METRIC_CALLS}")

# COMMAND ----------

# Unity Catalog location for prompts
UC_CATALOG = "workshop_andrea"
UC_SCHEMA = "bcp"

# Set up experiment for prompt optimization
username = spark.sql("SELECT current_user()").first()[0]
experiment_name = f"/Users/{username}/complaint_prompt_optimization"
experiment = mlflow.set_experiment(experiment_name)
print(f"Optimization experiment: {experiment_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Register Prompt in the MLflow Prompt Registry
# MAGIC
# MAGIC We register a single classification prompt that takes a complaint and outputs
# MAGIC `Department: X | Priority: Y`.

# COMMAND ----------

prompt_classifier = mlflow.genai.register_prompt(
    name=f"{UC_CATALOG}.{UC_SCHEMA}.complaint_classifier",
    template=(
        "You are a complaint classification specialist for a major bank.\n\n"
        "Classify the following customer complaint into a Department and Priority.\n\n"
        "**Department** (choose exactly one):\n"
        "- Credit Cards: Credit card billing, rewards, fraud, fees, APR issues\n"
        "- Mortgages: Home loans, refinancing, escrow, foreclosure, PMI\n"
        "- Personal Banking: Checking/savings accounts, ATM, online banking, overdraft, debit cards\n"
        "- Investment: Brokerage, 401k, IRA, mutual funds, trading, financial advisory\n"
        "- Insurance: Claims, premiums, coverage, beneficiary changes\n\n"
        "**Priority** (choose exactly one):\n"
        "- Critical: Immediate financial harm, fraud, account access blocked, legal/regulatory risk\n"
        "- High: Significant financial impact, compliance issues, time-sensitive errors\n"
        "- Medium: Service failures, processing delays, moderate inconvenience\n"
        "- Low: General inquiries, minor requests, informational\n\n"
        "Complaint:\n{{complaint_text}}\n\n"
        "Respond in EXACTLY this format: Department: <department> | Priority: <priority>"
    ),
)

print(f"Registered: {prompt_classifier.name} v{prompt_classifier.version}")
print(f"URI: {prompt_classifier.uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Training Dataset
# MAGIC
# MAGIC 20 banking complaints covering all 5 departments and 4 priority levels.
# MAGIC Each entry has the complaint text as input and the expected classification as the expectation.

# COMMAND ----------

train_data = [
    # --- Credit Cards ---
    {
        "inputs": {"complaint_text": "I noticed three charges on my Visa statement from an electronics store in another state that I've never visited. The charges total $1,247.88 and were made yesterday. I still have my card with me."},
        "expectations": {"expected_classification": "Department: Credit Cards | Priority: Critical"},
    },
    {
        "inputs": {"complaint_text": "I purchased a laptop online for $899 and it arrived damaged. I contacted the merchant but they refuse to issue a refund. I want to dispute this charge on my credit card."},
        "expectations": {"expected_classification": "Department: Credit Cards | Priority: High"},
    },
    {
        "inputs": {"complaint_text": "My credit card was charged a $35 annual fee but when I signed up last year I was told the first two years would be fee-free. Can you look into this?"},
        "expectations": {"expected_classification": "Department: Credit Cards | Priority: Medium"},
    },
    {
        "inputs": {"complaint_text": "I'd like to know if my credit card offers any travel rewards or airport lounge access. I'm planning a vacation."},
        "expectations": {"expected_classification": "Department: Credit Cards | Priority: Low"},
    },
    # --- Mortgages ---
    {
        "inputs": {"complaint_text": "I received a foreclosure notice in the mail today even though I have made every single payment on time for the past 8 years. I have all my bank statements as proof. This is terrifying."},
        "expectations": {"expected_classification": "Department: Mortgages | Priority: Critical"},
    },
    {
        "inputs": {"complaint_text": "My escrow account analysis shows a shortage of $3,200 and my monthly payment is jumping by $450. The property tax assessment hasn't changed, so I think there's a calculation error."},
        "expectations": {"expected_classification": "Department: Mortgages | Priority: High"},
    },
    {
        "inputs": {"complaint_text": "I submitted my refinance application 3 months ago and haven't received any updates. I've called four times and each representative tells me something different."},
        "expectations": {"expected_classification": "Department: Mortgages | Priority: Medium"},
    },
    {
        "inputs": {"complaint_text": "When does my current rate lock expire? I want to make sure I close before it lapses. My loan officer hasn't returned my calls."},
        "expectations": {"expected_classification": "Department: Mortgages | Priority: Low"},
    },
    # --- Personal Banking ---
    {
        "inputs": {"complaint_text": "My checking account was frozen this morning with no warning. I have rent due tomorrow and payroll deposited into this account. I cannot access any of my money and the branch says they can't help."},
        "expectations": {"expected_classification": "Department: Personal Banking | Priority: Critical"},
    },
    {
        "inputs": {"complaint_text": "I was charged three overdraft fees of $36 each on the same day even though my direct deposit was pending. That's $108 in fees for transactions under $20 total."},
        "expectations": {"expected_classification": "Department: Personal Banking | Priority: High"},
    },
    {
        "inputs": {"complaint_text": "I deposited a check at the ATM last Monday and it still hasn't been credited to my account. The receipt shows the deposit but my balance doesn't reflect it."},
        "expectations": {"expected_classification": "Department: Personal Banking | Priority: Medium"},
    },
    {
        "inputs": {"complaint_text": "Is it possible to change my checking account from a standard account to a premium account? I'd like to know the benefits and any fee differences."},
        "expectations": {"expected_classification": "Department: Personal Banking | Priority: Low"},
    },
    # --- Investment ---
    {
        "inputs": {"complaint_text": "I just discovered that my financial advisor executed a series of trades in my retirement account without my authorization. Over $75,000 was moved from my conservative bond fund into speculative penny stocks. I'm 62 and retiring in two years."},
        "expectations": {"expected_classification": "Department: Investment | Priority: Critical"},
    },
    {
        "inputs": {"complaint_text": "My 401k rollover to an IRA has been stuck in limbo for 7 weeks now. The funds left my previous employer's plan but haven't appeared in my IRA. Nobody can tell me where my money is."},
        "expectations": {"expected_classification": "Department: Investment | Priority: High"},
    },
    {
        "inputs": {"complaint_text": "My quarterly dividend payment from the growth fund was supposed to be deposited two weeks ago but I haven't received it yet. The fund's website shows the dividend was declared."},
        "expectations": {"expected_classification": "Department: Investment | Priority: Medium"},
    },
    {
        "inputs": {"complaint_text": "Can you send me information about your new ESG-focused mutual fund offerings? I'm interested in sustainable investing options for my portfolio."},
        "expectations": {"expected_classification": "Department: Investment | Priority: Low"},
    },
    # --- Insurance ---
    {
        "inputs": {"complaint_text": "My health insurance claim for emergency surgery was denied even though the procedure was performed at an in-network hospital. The bill is $47,000 and the collection agency is already calling me."},
        "expectations": {"expected_classification": "Department: Insurance | Priority: Critical"},
    },
    {
        "inputs": {"complaint_text": "My auto insurance premium doubled from $1,800 to $3,600 at renewal. I haven't had any accidents or tickets. When I asked why, the agent couldn't give me a clear explanation."},
        "expectations": {"expected_classification": "Department: Insurance | Priority: High"},
    },
    {
        "inputs": {"complaint_text": "I filed a claim for hail damage to my roof three weeks ago. Your website says claims are processed in 10 business days but I haven't heard anything from an adjuster."},
        "expectations": {"expected_classification": "Department: Insurance | Priority: Medium"},
    },
    {
        "inputs": {"complaint_text": "I recently purchased a new car and need to update my auto insurance policy. Can you walk me through the process of adding the new vehicle?"},
        "expectations": {"expected_classification": "Department: Insurance | Priority: Low"},
    },
]

print(f"Training dataset: {len(train_data)} examples")

# Show distribution
from collections import Counter
depts = Counter()
pris = Counter()
for item in train_data:
    expected = item["expectations"]["expected_classification"]
    dept = expected.split("|")[0].replace("Department:", "").strip()
    pri = expected.split("|")[1].replace("Priority:", "").strip()
    depts[dept] += 1
    pris[pri] += 1

print(f"\nDepartment distribution: {dict(depts)}")
print(f"Priority distribution: {dict(pris)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Define the Prediction Function
# MAGIC
# MAGIC The `predict_fn` loads the classification prompt from the registry and calls the LLM.
# MAGIC
# MAGIC **Important**: The function must load the prompt via `mlflow.genai.load_prompt()` each time.
# MAGIC The optimizer modifies prompt versions in the registry and re-runs this function
# MAGIC to test improvements.

# COMMAND ----------

def predict_fn(complaint_text: str) -> str:
    prompt = mlflow.genai.load_prompt(f"prompts:/{UC_CATALOG}.{UC_SCHEMA}.complaint_classifier/{prompt_classifier.version}")
    content = prompt.format(complaint_text=complaint_text)

    response = client.chat.completions.create(
        model=MODEL_ENDPOINT,
        messages=[{"role": "user", "content": content}],
        max_tokens=100,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# COMMAND ----------

# Quick test
test_result = predict_fn("I see unauthorized charges on my credit card totaling $500.")
print(f"Test classification: {test_result}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Define Scorers
# MAGIC
# MAGIC We define three code-based scorers to evaluate classification accuracy,
# MAGIC plus optionally load the aligned judge from Notebook 1.

# COMMAND ----------

def _parse_classification(text):
    """Parse 'Department: X | Priority: Y' from text, handling minor format variations."""
    dept, pri = None, None
    text_lower = text.lower()
    if "department:" in text_lower:
        dept_part = text_lower.split("department:")[1]
        dept = dept_part.split("|")[0].strip().rstrip(",.:;")
    if "priority:" in text_lower:
        pri_part = text_lower.split("priority:")[1]
        pri = pri_part.strip().rstrip(",.:;").split("\n")[0].strip()
    return dept, pri


@scorer
def department_accuracy(outputs: str, expectations: dict) -> float:
    """Check if the department classification is correct."""
    expected_dept, _ = _parse_classification(expectations["expected_classification"])
    actual_dept, _ = _parse_classification(outputs)
    if expected_dept is None or actual_dept is None:
        return 0.0
    return 1.0 if actual_dept == expected_dept else 0.0


@scorer
def priority_accuracy(outputs: str, expectations: dict) -> float:
    """Check if the priority classification is correct."""
    _, expected_pri = _parse_classification(expectations["expected_classification"])
    _, actual_pri = _parse_classification(outputs)
    if expected_pri is None or actual_pri is None:
        return 0.0
    return 1.0 if actual_pri == expected_pri else 0.0


@scorer
def exact_match(outputs: str, expectations: dict) -> float:
    """Check if both department and priority are correct."""
    expected_dept, expected_pri = _parse_classification(expectations["expected_classification"])
    actual_dept, actual_pri = _parse_classification(outputs)
    if None in (expected_dept, expected_pri, actual_dept, actual_pri):
        return 0.0
    return 1.0 if (actual_dept == expected_dept and actual_pri == expected_pri) else 0.0


scorers_list = [department_accuracy, priority_accuracy, exact_match]
print(f"Defined {len(scorers_list)} code-based scorers")

# COMMAND ----------

# MAGIC %md
# MAGIC ### (Optional) Load Aligned Judge from Notebook 1
# MAGIC
# MAGIC If you ran Notebook 1 and set the `judge_experiment_id` widget, we load
# MAGIC the aligned judge as an additional scorer.

# COMMAND ----------

from mlflow.genai import get_scorer as _get_scorer

aligned_judge_scorer = _get_scorer(
    name="classification_quality",
    experiment_id=JUDGE_EXPERIMENT_ID,
)
print(f"Loaded aligned judge from experiment {JUDGE_EXPERIMENT_ID}")

# COMMAND ----------

all_scorers = scorers_list + [aligned_judge_scorer]
print(f"All scorers for evaluation: {len(all_scorers)} (3 code-based + aligned judge)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Baseline Evaluation
# MAGIC
# MAGIC Before optimization, run `mlflow.genai.evaluate()` with the **original prompts** to establish
# MAGIC a baseline. This logs traces and scores to the MLflow experiment for comparison.

# COMMAND ----------

baseline_eval = mlflow.genai.evaluate(
    predict_fn=predict_fn,
    data=train_data,
    scorers=all_scorers,
)

print("Baseline evaluation complete.")
print(f"Baseline metrics: {baseline_eval.metrics}")

# COMMAND ----------

baseline_eval.tables["eval_results"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Run Prompt Optimization
# MAGIC
# MAGIC The **GEPA optimizer** (Genetic-Pareto) iteratively:
# MAGIC 1. Runs the predict function on training data
# MAGIC 2. Evaluates with the **aligned judge** as the sole optimization scorer
# MAGIC 3. Reflects on failures using an LLM
# MAGIC 4. Proposes prompt modifications
# MAGIC 5. Tests new prompt variants
# MAGIC 6. Keeps the best-performing versions
# MAGIC
# MAGIC The code-based scorers (department_accuracy, priority_accuracy, exact_match) are
# MAGIC defined above for analysis but are **not** passed to the optimizer — only the
# MAGIC aligned judge drives the optimization.
# MAGIC
# MAGIC **Quick mode**: `max_metric_calls=50` for fast initial testing. Increase to 300+ for full optimization.

# COMMAND ----------

result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=train_data,
    prompt_uris=[prompt_classifier.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model=OPTIMIZER_MODEL,
        max_metric_calls=MAX_METRIC_CALLS,
        display_progress_bar=True,
    ),
    scorers=[exact_match],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Optimization Results

# COMMAND ----------

print("=" * 70)
print("OPTIMIZATION RESULTS")
print("=" * 70)
print(f"\nInitial score: {result.initial_eval_score:.3f}")
print(f"Final score:   {result.final_eval_score:.3f}")
print(f"Improvement:   {result.final_eval_score - result.initial_eval_score:+.3f}")

# COMMAND ----------

print("\n" + "=" * 70)
print("OPTIMIZED PROMPTS")
print("=" * 70)

for p in result.optimized_prompts:
    print(f"\n--- {p.name} (v{p.version}) ---")
    print(f"URI: {p.uri}")
    print(f"\nTemplate:\n{p.template}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Post-Optimization Evaluation
# MAGIC
# MAGIC Run `mlflow.genai.evaluate()` with the **optimized prompts** on the same dataset and scorers.
# MAGIC Compare with the baseline evaluation from Step 5.

# COMMAND ----------

optimized_prompt = result.optimized_prompts[0]


def optimized_predict_fn(complaint_text: str) -> str:
    prompt = mlflow.genai.load_prompt(f"prompts:/{UC_CATALOG}.{UC_SCHEMA}.complaint_classifier/{optimized_prompt.version}")
    content = prompt.format(complaint_text=complaint_text)

    response = client.chat.completions.create(
        model=MODEL_ENDPOINT,
        messages=[{"role": "user", "content": content}],
        max_tokens=100,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

# COMMAND ----------

optimized_eval = mlflow.genai.evaluate(
    predict_fn=optimized_predict_fn,
    data=train_data,
    scorers=all_scorers,
)

print("Post-optimization evaluation complete.")
print(f"Optimized metrics: {optimized_eval.metrics}")

# COMMAND ----------

optimized_eval.tables["eval_results"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline vs Optimized Comparison

# COMMAND ----------

print("=" * 70)
print("BASELINE vs OPTIMIZED")
print("=" * 70)
print(f"\n{'Metric':<45} {'Baseline':<15} {'Optimized':<15} {'Delta':<15}")
print("-" * 90)

all_metric_keys = sorted(set(list(baseline_eval.metrics.keys()) + list(optimized_eval.metrics.keys())))
for key in all_metric_keys:
    baseline_val = baseline_eval.metrics.get(key, 0)
    optimized_val = optimized_eval.metrics.get(key, 0)
    if isinstance(baseline_val, (int, float)) and isinstance(optimized_val, (int, float)):
        print(f"{key:<45} {baseline_val:<15.3f} {optimized_val:<15.3f} {optimized_val - baseline_val:+.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook we:
# MAGIC
# MAGIC 1. **Registered a classification prompt** in the MLflow Prompt Registry
# MAGIC 2. **Created a training dataset** with 20 banking complaints across 5 departments and 4 priorities
# MAGIC 3. **Ran a baseline evaluation** with code-based scorers and the aligned judge
# MAGIC 4. **Optimized the prompt** using GEPA with exact match scoring
# MAGIC 5. **Ran a post-optimization evaluation** with the same scorers
# MAGIC 6. **Compared results** — baseline vs optimized metrics side by side
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC
# MAGIC - **Automated prompt optimization** systematically improves prompts through reflection and iteration
# MAGIC - **Aligned judges** from Notebook 1 capture domain expertise that pure keyword matching cannot
# MAGIC - Optimized prompts are **versioned** in the Prompt Registry — you can always compare or roll back

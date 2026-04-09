# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Judge Alignment for Customer Complaint Classification
# MAGIC
# MAGIC This notebook demonstrates how to use **MemAlign** to align an LLM judge with human feedback
# MAGIC for evaluating a banking customer complaint classification system.
# MAGIC
# MAGIC MemAlign uses a dual-memory system:
# MAGIC - **Semantic Memory**: Distills general guidelines from human feedback patterns
# MAGIC - **Episodic Memory**: Retrieves similar past examples using embeddings for few-shot learning
# MAGIC
# MAGIC ### What you'll learn:
# MAGIC 1. Create an LLM judge for evaluating complaint classifications
# MAGIC 2. Prepare alignment and test datasets with tricky edge cases
# MAGIC 3. Evaluate the judge before alignment (baseline)
# MAGIC 4. Align the judge using human feedback with MemAlign
# MAGIC 5. Inspect the learned guidelines
# MAGIC 6. Evaluate the improved judge (post-alignment)
# MAGIC 7. Register the judge as a scorer for future experiments

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]>=3.5.0" dspy
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from mlflow.genai.judges import make_judge
from mlflow.genai.judges.optimizers import MemAlignOptimizer
from mlflow.entities import AssessmentSource, AssessmentSourceType

import mlflow
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC
# MAGIC Configure the MLflow experiment and Databricks workspace connection.

# COMMAND ----------

username = spark.sql("SELECT current_user()").first()[0]
experiment_name = f"/Users/{username}/complaint_judge_alignment"
experiment = mlflow.set_experiment(experiment_name)
experiment_id = experiment.experiment_id
print(f"Experiment: {experiment_name}")
print(f"Experiment ID: {experiment_id}")

# COMMAND ----------

# Model configuration
JUDGE_MODEL = "databricks:/databricks-claude-haiku-4-5"  # Must support structured outputs for MemAlign's DSPy internals
REFLECTION_MODEL = "databricks:/databricks-claude-haiku-4-5"
EMBEDDING_MODEL = "databricks:/databricks-gte-large-en"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create an LLM Judge
# MAGIC
# MAGIC We create a judge that evaluates whether a customer complaint classification is **correct**.
# MAGIC The judge only sees the complaint and the system's classification — it must decide
# MAGIC on its own whether the department and priority are appropriate.
# MAGIC
# MAGIC Departments: Credit Cards, Mortgages, Personal Banking, Investment, Insurance
# MAGIC Priorities: Critical, High, Medium, Low

# COMMAND ----------

initial_judge = make_judge(
    name="classification_quality",
    instructions=(
        "You are evaluating whether a bank customer complaint was classified correctly.\n\n"
        "Customer complaint:\n{{ inputs }}\n\n"
        "System classification:\n{{ outputs }}\n\n"
        "Evaluate whether the system's classification is correct by checking:\n"
        "1. The Department is the most appropriate for the complaint's core issue\n"
        "2. The Priority level matches the urgency and severity of the situation\n\n"
        "Both department AND priority must be correct for the classification to be correct.\n"
        "Return True if the classification is correct, False if either is wrong."
    ),
    feedback_value_type=bool,
    model=JUDGE_MODEL,
)

print(f"Created judge: {initial_judge.name}")
print(f"Model: {initial_judge.model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create Datasets
# MAGIC
# MAGIC We create two datasets:
# MAGIC 1. **Alignment set** (10 examples): Used to teach the judge our preferences
# MAGIC 2. **Test set** (10 examples): Used to evaluate the judge's performance
# MAGIC
# MAGIC ### Why generic LLM judges fail at banking classification
# MAGIC
# MAGIC LLM judges lack domain-specific calibration. They tend to:
# MAGIC - **Over-rely on emotional tone**: An angry customer doesn't mean high priority
# MAGIC - **Miss regulatory implications**: A small dollar amount can be a major compliance issue
# MAGIC - **Follow keywords instead of root cause**: The product mentioned isn't always the right department
# MAGIC - **Under-escalate subtle violations**: PMI overcharges, TCPA violations, and elder abuse patterns look routine on the surface

# COMMAND ----------

# Alignment dataset - 5 straightforward + 5 tricky edge cases
alignment_examples = [
    # --- Straightforward cases (easy for any judge) ---
    {
        "complaint": "I was charged a $39 late fee on my Visa card even though I paid on time. I have the bank confirmation number.",
        "classification": "Department: Credit Cards | Priority: Medium",
        "is_correct": True,
        "rationale": "Correct. Late fee dispute on a credit card with proof of payment is a standard Credit Cards issue at Medium priority.",
    },
    {
        "complaint": "I'd like to know what the current mortgage refinance rates are for a 30-year fixed.",
        "classification": "Department: Mortgages | Priority: Low",
        "is_correct": True,
        "rationale": "Correct. General rate inquiry is a Low priority Mortgages informational request.",
    },
    {
        "complaint": "I haven't received my quarterly dividend payment from my mutual fund. It was supposed to arrive last week.",
        "classification": "Department: Investment | Priority: Medium",
        "is_correct": True,
        "rationale": "Correct. Delayed dividend is a Medium priority Investment processing issue.",
    },
    {
        "complaint": "I received a foreclosure notice even though I've paid every month on time for 8 years. I have all my receipts.",
        "classification": "Department: Insurance | Priority: Low",
        "is_correct": False,
        "rationale": "WRONG. An erroneous foreclosure notice is a Mortgages issue at Critical priority — imminent threat to the customer's home despite documented payments. Insurance and Low priority are both completely wrong.",
    },
    {
        "complaint": "Someone used my credit card number to make $3,000 in purchases at stores I've never been to. I still have the card.",
        "classification": "Department: Personal Banking | Priority: Low",
        "is_correct": False,
        "rationale": "WRONG. This is credit card fraud — it should be Credit Cards at Critical priority. Personal Banking and Low are both completely wrong.",
    },
    # --- Tricky: classification LOOKS correct but is actually wrong ---
    # (LLM will likely say True, but human expert says False)
    {
        "complaint": "I've been paying private mortgage insurance for 3 years now even though my home value has increased and I'm sure I have at least 25% equity. I asked to remove it but was told I have to wait.",
        "classification": "Department: Mortgages | Priority: Medium",
        "is_correct": False,
        "rationale": "WRONG priority. Under the Homeowners Protection Act (HPA), PMI must be automatically terminated when the loan reaches 78% LTV. If the customer has 75% LTV and has been paying PMI for 3 years unnecessarily, this is a regulatory compliance violation. This should be High priority because the bank may be in violation of federal law and owes the customer a refund of overpaid premiums.",
    },
    {
        "complaint": "I keep getting phone calls from your bank about credit card offers even though I asked to be removed from the call list months ago. I get at least two calls per week.",
        "classification": "Department: Credit Cards | Priority: Low",
        "is_correct": False,
        "rationale": "WRONG priority. Continued marketing calls after an opt-out request is a potential TCPA (Telephone Consumer Protection Act) violation. Each unauthorized call can carry a $500-$1,500 penalty. This should be Medium priority at minimum due to the regulatory and legal exposure, not Low.",
    },
    {
        "complaint": "A branch employee convinced my 82-year-old mother to open three new credit cards and a line of credit during a single visit. She lives on a fixed income and doesn't understand what she signed.",
        "classification": "Department: Credit Cards | Priority: Medium",
        "is_correct": False,
        "rationale": "WRONG priority and arguably wrong department. This describes a pattern consistent with elder financial abuse and predatory sales practices. An 82-year-old on fixed income being sold 4 new credit products in one visit is a serious compliance red flag. This should be Critical priority and may need to be routed to Compliance/Risk rather than just Credit Cards, due to elder abuse reporting obligations.",
    },
    # --- Tricky: classification LOOKS wrong but is actually correct ---
    # (LLM will likely say False, but human expert says True)
    {
        "complaint": "I am FURIOUS! Your ATM ate my debit card and I had to wait 20 minutes for someone to help. This is absolutely unacceptable and I want compensation for my time!",
        "classification": "Department: Personal Banking | Priority: Low",
        "is_correct": True,
        "rationale": "Correct. Despite the extremely angry tone, this is a standard ATM card retention incident. The card will be mailed back or replaced. Low priority is appropriate because there is no financial harm, no fraud, and no regulatory risk. Customer emotional tone does NOT determine priority level — the nature and impact of the issue does.",
    },
    {
        "complaint": "I was charged a $12 'account maintenance fee' on my checking account that was supposed to be free. When I called, they said my direct deposit didn't qualify because it was $49 short of the minimum. Twelve dollars isn't much but it's the principle.",
        "classification": "Department: Personal Banking | Priority: Medium",
        "is_correct": True,
        "rationale": "Correct. Even though the customer downplays the amount ('it's the principle'), this IS Medium priority. Fee disputes where the customer's qualifying activity was borderline ($49 short) suggest a potential disclosure issue with the account terms. The customer may have been inadequately informed about the exact threshold. Personal Banking at Medium is the right call.",
    },
]

print(f"Created {len(alignment_examples)} alignment examples")
print(f"  - {sum(1 for e in alignment_examples if e['is_correct'])} correct classifications")
print(f"  - {sum(1 for e in alignment_examples if not e['is_correct'])} incorrect classifications")

# COMMAND ----------

# Test dataset - similar distribution with different complaints
test_examples = [
    # --- Straightforward cases ---
    {
        "complaint": "My home insurance premium jumped from $1,200 to $2,400 at renewal with zero explanation.",
        "classification": "Department: Insurance | Priority: High",
        "is_correct": True,
        "rationale": "Correct. Unexplained 100% premium increase is High priority Insurance — significant financial impact.",
    },
    {
        "complaint": "I need a certificate of insurance for my new rental property. How do I request one?",
        "classification": "Department: Insurance | Priority: Low",
        "is_correct": True,
        "rationale": "Correct. Standard informational Insurance request at Low priority.",
    },
    {
        "complaint": "My 401k rollover has been stuck for 6 weeks. Nobody can tell me where my retirement money is.",
        "classification": "Department: Investment | Priority: High",
        "is_correct": True,
        "rationale": "Correct. 401k rollover in limbo for 6 weeks is High priority Investment issue.",
    },
    {
        "complaint": "My checking account was frozen with no warning. I have rent due tomorrow and can't access any funds.",
        "classification": "Department: Mortgages | Priority: Low",
        "is_correct": False,
        "rationale": "WRONG. Frozen checking account with imminent bills is Personal Banking at Critical priority. Mortgages and Low are both completely wrong.",
    },
    {
        "complaint": "My financial advisor moved $50,000 into penny stocks without my permission. I'm 65 and retiring soon.",
        "classification": "Department: Insurance | Priority: Medium",
        "is_correct": False,
        "rationale": "WRONG. Unauthorized trades on a near-retiree's account is Investment at Critical priority — fiduciary breach and FINRA suitability violation. Insurance and Medium are both wrong.",
    },
    # --- Tricky: looks correct but is wrong (same patterns as alignment set) ---
    {
        "complaint": "I opted out of email and text marketing from your bank over a year ago but I still get promotional messages almost daily. I've called three times to complain.",
        "classification": "Department: Personal Banking | Priority: Low",
        "is_correct": False,
        "rationale": "WRONG priority. Persistent marketing communications after repeated opt-out requests is a CAN-SPAM and TCPA compliance violation. After three complaints with no resolution, this should be Medium priority due to legal exposure and regulatory risk. Low priority is inappropriate for compliance violations.",
    },
    {
        "complaint": "My 78-year-old father with early-stage dementia was sold a variable annuity with a 10-year surrender period by his advisor at your bank. He can barely remember what he had for lunch.",
        "classification": "Department: Investment | Priority: High",
        "is_correct": False,
        "rationale": "WRONG priority. Selling a complex, long-duration investment product to an elderly customer with cognitive impairment is a textbook case of elder financial exploitation. This requires Critical priority due to mandatory elder abuse reporting obligations, FINRA suitability violations, and potential litigation risk. High understates the severity.",
    },
    {
        "complaint": "I've been paying PMI on my condo mortgage for over 4 years and I recently got an appraisal showing my LTV is 72%. When I requested PMI removal, I was told to wait until the scheduled date.",
        "classification": "Department: Mortgages | Priority: Medium",
        "is_correct": False,
        "rationale": "WRONG priority. With 72% LTV confirmed by appraisal, the Homeowners Protection Act requires that PMI be cancellable upon borrower request at 80% LTV and automatically terminated at 78% LTV. The customer is well past both thresholds. Continuing to charge PMI is a federal compliance violation. This should be High priority.",
    },
    # --- Tricky: looks wrong but is correct ---
    {
        "complaint": "This is OUTRAGEOUS! I've been on hold for TWO HOURS trying to ask about your savings account interest rates! Your customer service is the WORST I've ever experienced!",
        "classification": "Department: Personal Banking | Priority: Low",
        "is_correct": True,
        "rationale": "Correct. Despite the extreme frustration about hold times, the underlying request is a simple interest rate inquiry — a Low priority informational matter. The poor service experience is a separate operational issue but does not change the priority of the banking request itself. Emotional intensity does not equal business urgency.",
    },
    {
        "complaint": "I noticed my savings account earned $4.12 in interest last month but based on my balance and the stated APY I should have earned $4.58. It's less than 50 cents but the math doesn't add up.",
        "classification": "Department: Personal Banking | Priority: Medium",
        "is_correct": True,
        "rationale": "Correct. Although the dollar amount is tiny, interest calculation discrepancies — even small ones — can indicate systematic computation errors affecting many accounts. This warrants Medium priority investigation, not Low, because a rounding or calculation bug in the interest engine could have regulatory implications (Truth in Savings Act) across the customer base.",
    },
]

print(f"Created {len(test_examples)} test examples")
print(f"  - {sum(1 for e in test_examples if e['is_correct'])} correct classifications")
print(f"  - {sum(1 for e in test_examples if not e['is_correct'])} incorrect classifications")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Create Traces and Log Human Feedback
# MAGIC
# MAGIC MemAlign learns from traces that have human feedback attached. We'll:
# MAGIC 1. Create traces for each example (alignment + test sets)
# MAGIC 2. Log human feedback (ground truth) for alignment examples only
# MAGIC
# MAGIC **Key**: The judge only sees the complaint and classification — NOT the expected answer.
# MAGIC This forces the judge to evaluate using its own understanding, which is where
# MAGIC alignment makes a difference.

# COMMAND ----------

def create_traces(examples, prefix):
    """Create traces from examples, setting inputs and outputs."""
    trace_ids = []
    for i, example in enumerate(examples):
        with mlflow.start_span(f"{prefix}_{i}") as span:
            span.set_inputs(example["complaint"])
            span.set_outputs(example["classification"])
            trace_ids.append(span.trace_id)
    return trace_ids


# Create traces for both sets
alignment_trace_ids = create_traces(alignment_examples, "alignment")
print(f"Created {len(alignment_trace_ids)} alignment traces")

test_trace_ids = create_traces(test_examples, "test")
print(f"Created {len(test_trace_ids)} test traces")

# Wait for traces to be persisted before adding assessments
time.sleep(2)

# COMMAND ----------

# Log human feedback for alignment examples only
for trace_id, example in zip(alignment_trace_ids, alignment_examples):
    mlflow.log_feedback(
        trace_id=trace_id,
        name="classification_quality",  # Must match judge name
        value=example["is_correct"],
        rationale=example["rationale"],
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id="banking_compliance_expert",
        ),
    )

print(f"Logged human feedback for {len(alignment_trace_ids)} alignment traces")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Run Initial Judge & Log Assessments to Traces
# MAGIC
# MAGIC We run the initial judge on **all** traces and log its assessments alongside
# MAGIC the human feedback. This lets us compare human vs judge evaluations in the MLflow UI.

# COMMAND ----------

all_trace_ids = alignment_trace_ids + test_trace_ids
all_examples = alignment_examples + test_examples

# Run initial judge on all traces and log feedback
for trace_id, example in zip(all_trace_ids, all_examples):
    result = initial_judge(
        inputs=example["complaint"],
        outputs=example["classification"],
    )
    mlflow.log_feedback(
        trace_id=trace_id,
        name="classification_quality",
        value=result.value,
        rationale=result.rationale,
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id="initial_judge",
        ),
    )

print(f"Logged initial judge assessments on {len(all_trace_ids)} traces")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline Accuracy
# MAGIC
# MAGIC Let's see where the initial judge agrees and disagrees with the human experts.

# COMMAND ----------

def evaluate_judge(judge, examples, dataset_name):
    """Evaluate judge on examples and compute accuracy."""
    correct = 0
    results = []

    print(f"\n{'=' * 70}")
    print(f"Evaluating on {dataset_name} ({len(examples)} examples)")
    print(f"{'=' * 70}")

    for i, example in enumerate(examples):
        feedback = judge(
            inputs=example["complaint"],
            outputs=example["classification"],
        )

        predicted = feedback.value
        expected = example["is_correct"]
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        results.append({
            "example": i + 1,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
        })

        status = "CORRECT" if is_correct else "WRONG"
        print(f"\nExample {i + 1}: [{status}]")
        print(f"  Complaint: {example['complaint'][:80]}...")
        print(f"  Classification: {example['classification']}")
        print(f"  Judge says correct={predicted}, Human says correct={expected}")
        if not is_correct:
            rationale_preview = feedback.rationale[:200] if feedback.rationale else "N/A"
            print(f"  Judge rationale: {rationale_preview}...")

    accuracy = correct / len(examples) * 100
    print(f"\n{'-' * 70}")
    print(f"Accuracy: {correct}/{len(examples)} ({accuracy:.1f}%)")
    print(f"{'-' * 70}")

    return accuracy, results

# COMMAND ----------

# Evaluate baseline on alignment set
baseline_align_acc, baseline_align_results = evaluate_judge(
    initial_judge, alignment_examples, "Alignment Set (Baseline)"
)

# COMMAND ----------

# Evaluate baseline on test set
baseline_test_acc, baseline_test_results = evaluate_judge(
    initial_judge, test_examples, "Test Set (Baseline)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Align the Judge with MemAlign
# MAGIC
# MAGIC Now we use MemAlign to align the judge with human feedback.
# MAGIC
# MAGIC MemAlign will:
# MAGIC 1. **Distill guidelines** from the human feedback rationales (semantic memory)
# MAGIC 2. **Store examples** for few-shot retrieval during future evaluations (episodic memory)
# MAGIC
# MAGIC This is fast (~40 seconds) and cost-effective (~$0.03 per alignment cycle).

# COMMAND ----------

optimizer = MemAlignOptimizer(
    reflection_lm=REFLECTION_MODEL,
    embedding_model=EMBEDDING_MODEL,
    retrieval_k=3,  # Number of similar examples to retrieve during evaluation
)

print("Created MemAlign optimizer")

# COMMAND ----------

# Retrieve alignment traces
all_traces = mlflow.search_traces(
    experiment_ids=[experiment_id],
    return_type="list",
)

alignment_traces = [
    trace for trace in all_traces
    if trace.info.trace_id in alignment_trace_ids
]

print(f"Retrieved {len(alignment_traces)} traces for alignment")

# COMMAND ----------

# Align the judge
aligned_judge = initial_judge.align(
    traces=alignment_traces,
    optimizer=optimizer,
)

print("Alignment complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Inspect Learned Guidelines (Semantic Memory)
# MAGIC
# MAGIC Let's see what guidelines MemAlign distilled from the human feedback.
# MAGIC These should capture banking-specific rules like:
# MAGIC - Regulatory violations (HPA, TCPA) require priority escalation
# MAGIC - Elder financial abuse patterns must be Critical
# MAGIC - Customer emotional tone does NOT determine priority

# COMMAND ----------

print("Aligned Judge Instructions (original + distilled guidelines)")
print("=" * 70)
print(aligned_judge.instructions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Run Aligned Judge & Log Assessments to Traces
# MAGIC
# MAGIC We run the aligned judge on **all traces** and log its assessments with a
# MAGIC **different name** (`classification_quality_aligned`) so we can compare
# MAGIC all three assessments side-by-side in the MLflow UI:
# MAGIC - `classification_quality` (HUMAN) — expert feedback
# MAGIC - `classification_quality` (LLM_JUDGE) — initial judge
# MAGIC - `classification_quality_aligned` (LLM_JUDGE) — aligned judge

# COMMAND ----------

# Run aligned judge on all traces and log with a different name
for trace_id, example in zip(all_trace_ids, all_examples):
    result = aligned_judge(
        inputs=example["complaint"],
        outputs=example["classification"],
    )
    mlflow.log_feedback(
        trace_id=trace_id,
        name="classification_quality_aligned",
        value=result.value,
        rationale=result.rationale,
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE,
            source_id="aligned_judge",
        ),
    )

print(f"Logged aligned judge assessments on {len(all_trace_ids)} traces")

# COMMAND ----------

# Evaluate aligned judge on alignment set
aligned_align_acc, aligned_align_results = evaluate_judge(
    aligned_judge, alignment_examples, "Alignment Set (Aligned)"
)

# COMMAND ----------

# Evaluate aligned judge on test set
aligned_test_acc, aligned_test_results = evaluate_judge(
    aligned_judge, test_examples, "Test Set (Aligned)"
)

# COMMAND ----------

# Performance comparison
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print(f"\n{'Dataset':<25} {'Baseline':<15} {'Aligned':<15} {'Improvement':<15}")
print("-" * 70)
print(f"{'Alignment Set':<25} {baseline_align_acc:<15.1f} {aligned_align_acc:<15.1f} {aligned_align_acc - baseline_align_acc:+.1f}%")
print(f"{'Test Set':<25} {baseline_test_acc:<15.1f} {aligned_test_acc:<15.1f} {aligned_test_acc - baseline_test_acc:+.1f}%")
print("-" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Register the Aligned Judge
# MAGIC
# MAGIC Register the aligned judge so it can be loaded as a scorer in future experiments —
# MAGIC including the Prompt Optimization notebook (Notebook 2).

# COMMAND ----------

from mlflow.genai.scorers import ScorerSamplingConfig

try:
    registered_judge = aligned_judge.register()
except ValueError:
    # Already registered from a previous run — update with new alignment
    registered_judge = aligned_judge.update(
        sampling_config=ScorerSamplingConfig(sample_rate=1.0),
    )

print("Judge registered successfully!")
print(f"  Name: {registered_judge.name}")

# COMMAND ----------

# Verify registration
from mlflow.genai import list_scorers, get_scorer

scorers = list_scorers(experiment_id=experiment_id)
print("Registered scorers in experiment:")
for s in scorers:
    print(f"  - {s.name} (model: {s.model})")

# COMMAND ----------

# Quick test with the retrieved scorer
retrieved_judge = get_scorer(name="classification_quality", experiment_id=experiment_id)

test_result = retrieved_judge(
    inputs="Someone is using my debit card number to make purchases online. I need this stopped immediately.",
    outputs="Department: Personal Banking | Priority: Critical",
)

print(f"\nTest evaluation:")
print(f"  Correct: {test_result.value}")
print(f"  Rationale: {test_result.rationale}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook we:
# MAGIC
# MAGIC 1. **Created an LLM judge** for evaluating customer complaint classification quality
# MAGIC 2. **Prepared datasets** with subtle edge cases requiring banking domain expertise
# MAGIC 3. **Evaluated baseline performance** — the judge missed regulatory nuances (HPA, TCPA, elder abuse)
# MAGIC 4. **Aligned the judge** with human feedback using MemAlign's dual-memory system
# MAGIC 5. **Inspected learned guidelines** — MemAlign captured banking-specific rules
# MAGIC 6. **Evaluated improved performance** — the aligned judge now handles edge cases correctly
# MAGIC 7. **Registered the judge** as a reusable scorer
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC - **Regulatory knowledge matters**: PMI overcharges, TCPA violations, and elder abuse are not obvious to generic LLMs
# MAGIC - **Tone ≠ Priority**: MemAlign learned that angry customers don't automatically get higher priority
# MAGIC - **Small amounts, big implications**: A $0.46 interest discrepancy can indicate systemic issues
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC Use this aligned judge in **Notebook 2 (Prompt Optimization)** to optimize the classification prompts.
# MAGIC
# MAGIC **Experiment ID for Notebook 2:**

# COMMAND ----------

print(f"EXPERIMENT_ID = '{experiment_id}'")
print(f"\nCopy this value into the 'judge_experiment_id' widget in Notebook 2.")

import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
import numpy as np

from data_loader import load_paired_dataset
from rewriter import CodeRewriter
from detector import CodeDetector
import settings


# You can tweak this to fit your VRAM
BATCH_SIZE_DEF = getattr(settings, "EVAL_BATCH_SIZE", 8)


def compute_classification_metrics(non_synth_scores, synth_scores):
    """
    non_synth_scores: list[float] scores for human code (label 0)
    synth_scores: list[float] scores for ai code (label 1)
    Returns dict with AUROC, Accuracy, Precision, Recall, F1, pairwise_count, correct_pairwise
    """

    n_pairs = min(len(non_synth_scores), len(synth_scores))
    
    if n_pairs == 0:
        return {
            "AUROC": 0.0,
            "Accuracy": 0.0,
            "Pairwise_Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1_Score": 0.0,
            "pairwise_count": 0,
            "correct_pairwise": 0,
        }
    
    # Truncate to equal length
    non_synth_scores = non_synth_scores[:n_pairs]
    synth_scores = synth_scores[:n_pairs]
    
    labels = [0] * len(non_synth_scores) + [1] * len(synth_scores)
    scores = non_synth_scores + synth_scores

    auroc = 0.0
    if len(set(labels)) == 2:  # Need both classes
        try:
            auroc = roc_auc_score(labels, scores)
        except Exception as e:
            print(f"Warning: AUROC calculation failed: {e}")
            auroc = 0.0
    
    # === Pairwise Comparison (Core Zero-Shot Method) ===
    # For each pair, check if AI score > Human score
    correct_pairwise = sum(1 for h, a in zip(non_synth_scores, synth_scores) if a > h)
    
    # Pairwise accuracy (this is your main metric per the proposal)
    pairwise_accuracy = correct_pairwise / n_pairs if n_pairs > 0 else 0.0
    
    # === Confusion Matrix from Pairwise Decisions ===
    # Each correct pairwise comparison = 1 TP and 1 TN
    # Each incorrect pairwise comparison = 1 FP and 1 FN
    TP = correct_pairwise  # AI correctly identified as AI
    TN = correct_pairwise  # Human correctly identified as Human
    FP = n_pairs - correct_pairwise  # Human incorrectly identified as AI
    FN = n_pairs - correct_pairwise  # AI incorrectly identified as Human
    
    # === Standard Classification Metrics ===
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "AUROC": auroc,
        "Accuracy": accuracy,
        "Pairwise_Accuracy": pairwise_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "pairwise_count": n_pairs,
        "correct_pairwise": correct_pairwise,
    }



def run_evaluation():
    print("--- Starting Evaluation ---")

    # initialize rewriter (loads model)
    rewriter = CodeRewriter()

    final_result = []
    variant_results = []

    # iterate models you want to test
    for model_name in ["graphcodebert", "codet5"]:
        model_path = settings.GCB_SIMCSE_PATH if model_name == "graphcodebert" else settings.CODET5_SIMCSE_PATH

        if not os.path.exists(model_path):
            print("---" * 10)
            print(f"{model_path} not found... skipping {model_name}")
            print("---" * 10)
            continue

        # initialize detector (loads encoder)
        detector = CodeDetector(model_type=model_name)

        for m in settings.NUM_REWRITES:
            print("---" * 10)
            print(f"---- Start test on model {model_path} with {m} rewrites ----")

            if m >= 8:
                BATCH_SIZE = BATCH_SIZE_DEF // 4
                print(f"Using full batch size for m = 8 -> {BATCH_SIZE}")
            elif m >= 4:
                BATCH_SIZE = BATCH_SIZE_DEF // 2
                print(f"Using reduced batch size for m = 4 -> {BATCH_SIZE}")
            elif m >= 2:
                BATCH_SIZE = BATCH_SIZE_DEF
                print(f"Using smaller batch size for m = 2 -> {BATCH_SIZE}")
            else:
                # Safety fallback
                BATCH_SIZE = 8

            # totals across all variants for this (model, m)
            total_non_synth_scores = []
            total_synth_scores = []

            for variant_file in settings.VARIANT_FILES:
                variant_name = os.path.basename(variant_file)
                print(f"--- Processing: {variant_name} ---")

                paired_data = load_paired_dataset(variant_file)
                if not paired_data:
                    print(f"Skipping {variant_name} due to an error while loading...")
                    continue

                # per-variant accumulation
                non_synth_scores = []
                synth_scores = []

                # iterate in mini-batches
                n = len(paired_data)
                ranges = range(0, n, BATCH_SIZE)
                for start in tqdm(ranges, desc=f"{variant_name}"):
                    batch = paired_data[start:start + BATCH_SIZE]
                    human_batch = [pair[0] for pair in batch]
                    ai_batch = [pair[1] for pair in batch]

                    # generate rewrites in batch (returns list[list[str]])
                    human_rewrites_batch = rewriter.generate_rewrites_batch(human_batch, m)
                    ai_rewrites_batch = rewriter.generate_rewrites_batch(ai_batch, m)

                    # Filter out empty rewrite lists
                    valid_indices = [
                        i for i in range(len(human_rewrites_batch))
                        if human_rewrites_batch[i] and ai_rewrites_batch[i]
                    ]

                    if not valid_indices:
                        print(f"Warning: Batch {start}-{start+BATCH_SIZE} produced no valid rewrites")
                        continue

                    # Keep only valid pairs
                    valid_human_batch = [human_batch[i] for i in valid_indices]
                    valid_ai_batch = [ai_batch[i] for i in valid_indices]
                    valid_human_rewrites = [human_rewrites_batch[i] for i in valid_indices]
                    valid_ai_rewrites = [ai_rewrites_batch[i] for i in valid_indices]

            
                    # Compute scores
                    human_scores = detector.get_detection_score_batch(
                        valid_human_batch, valid_human_rewrites
                    )
                    ai_scores = detector.get_detection_score_batch(
                        valid_ai_batch, valid_ai_rewrites
                    )

                    non_synth_scores.extend(human_scores)
                    synth_scores.extend(ai_scores)

                # Accumulate across variants
                total_non_synth_scores.extend(non_synth_scores)
                total_synth_scores.extend(synth_scores)

                # Compute per-variant metrics
                if non_synth_scores and synth_scores:
                    metrics = compute_classification_metrics(non_synth_scores, synth_scores)

                    variant_results.append({
                        "Model": model_path,
                        "m": m,
                        "Variant": variant_name,
                        "AUROC": metrics["AUROC"],
                        "Accuracy": metrics["Accuracy"],
                        "Pairwise_Accuracy": metrics["Pairwise_Accuracy"],
                        "Precision": metrics["Precision"],
                        "Recall": metrics["Recall"],
                        "F1_Score": metrics["F1_Score"],
                        "Pairs": metrics["pairwise_count"],
                        "CorrectPairs": metrics["correct_pairwise"]
                    })

                    print(f"✓ {variant_name}: {metrics['pairwise_count']} pairs, "
                          f"{metrics['correct_pairwise']} correct "
                          f"(Acc: {metrics['Accuracy']:.4f}, AUROC: {metrics['AUROC']:.4f})")

                else:
                    print(f"✗ {variant_name}: No valid pairs generated")
                    # still store an entry indicating skipped/empty
                    variant_results.append({
                        "Model": model_path,
                        "m": m,
                        "Variant": variant_name,
                        "AUROC": 0.0,
                        "Accuracy": 0.0,
                        "Pairwise_Accuracy": 0.0,
                        "Precision": 0.0,
                        "Recall": 0.0,
                        "F1_Score": 0.0,
                        "Pairs": 0,
                        "CorrectPairs": 0
                    })

            if total_non_synth_scores and total_synth_scores:
                totals_metrics = compute_classification_metrics(
                total_non_synth_scores, total_synth_scores
            )

                final_result.append({
                    "Model": model_name,
                    "m": m,
                    "AUROC": f"{totals_metrics['AUROC']:.4f}",
                    "Accuracy": f"{totals_metrics['Accuracy']:.4f}",
                    "Pairwise_Accuracy": f"{totals_metrics['Pairwise_Accuracy']:.4f}",
                    "Precision": f"{totals_metrics['Precision']:.4f}",
                    "Recall": f"{totals_metrics['Recall']:.4f}",
                    "F1_Score": f"{totals_metrics['F1_Score']:.4f}",
                    "TotalPairs": totals_metrics["pairwise_count"],
                    "TotalCorrectPairs": totals_metrics["correct_pairwise"]
                })

                print(f"\n{'='*60}")
                print(f"OVERALL RESULTS: {model_name.upper()} | m = {m}")
                print(f"{'='*60}")
                print(f"  AUROC:     {totals_metrics['AUROC']:.4f}")
                print(f"  Accuracy:  {totals_metrics['Accuracy']:.4f}")
                print(f"  Pairwise Accuracy:  {totals_metrics['Pairwise_Accuracy']:.4f}")
                print(f"  Precision: {totals_metrics['Precision']:.4f}")
                print(f"  Recall:    {totals_metrics['Recall']:.4f}")
                print(f"  F1-Score:  {totals_metrics['F1_Score']:.4f}")
                print(f"  Pairs:     {totals_metrics['pairwise_count']}")
                print(f"  Correct:   {totals_metrics['correct_pairwise']}")
            else:
                print(f"\n✗ No valid data for {model_name} with m = {m}")

    # === Save Results ===
    print("\n\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    df_variant = pd.DataFrame(variant_results)
    df_total = pd.DataFrame(final_result)

    if not df_variant.empty:
        df_variant.to_csv("./evaluation_variant_results.csv", index=False)
        print("✓ Per-variant results saved to 'evaluation_variant_results.csv'")
    
    if not df_total.empty:
        df_total.to_csv("./evaluation_results.csv", index=False)
        print("✓ Overall results saved to 'evaluation_results.csv'")
        print("\n--- FINAL SUMMARY ---")
        print(df_total.to_string(index=False))
    else:
        print("✗ No results to save - check your data and rewriter output")


if __name__ == "__main__":
    run_evaluation()
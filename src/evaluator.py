import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

from data_loader import load_paired_dataset
from rewriter import CodeRewriter
from detector import CodeDetector
import settings


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
    
    non_synth_scores = non_synth_scores[:n_pairs]
    synth_scores = synth_scores[:n_pairs]
    
    labels = [0] * len(non_synth_scores) + [1] * len(synth_scores)
    scores = non_synth_scores + synth_scores

    auroc = 0.0
    if len(set(labels)) == 2:
        try:
            auroc = roc_auc_score(labels, scores)
        except Exception as e:
            print(f"Warning: AUROC calculation failed: {e}")
            auroc = 0.0
    
    correct_pairwise = sum(1 for h, a in zip(non_synth_scores, synth_scores) if a > h)
    pairwise_accuracy = correct_pairwise / n_pairs if n_pairs > 0 else 0.0
    
    TP = correct_pairwise
    TN = correct_pairwise
    FP = n_pairs - correct_pairwise
    FN = n_pairs - correct_pairwise
    
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

    rewriter = CodeRewriter()

    final_result = []
    variant_results = []

    # Collects one row per sample across all models/variants/m values
    detailed_rows = []

    for model_name in ["graphcodebert", "codet5"]:
        model_path = settings.GCB_MODEL_NAME if model_name == "graphcodebert" else settings.CODET5_MODEL_NAME
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
                BATCH_SIZE = 8

            total_non_synth_scores = []
            total_synth_scores = []

            for variant_file in settings.VARIANT_FILES:
                variant_name = os.path.basename(variant_file)
                print(f"--- Processing: {variant_name} ---")

                paired_data = load_paired_dataset(variant_file)
                if not paired_data:
                    print(f"Skipping {variant_name} due to an error while loading...")
                    continue

                non_synth_scores = []
                synth_scores = []

                n = len(paired_data)
                for start in tqdm(range(0, n, BATCH_SIZE), desc=f"{variant_name}"):
                    batch = paired_data[start:start + BATCH_SIZE]
                    human_batch = [pair[0] for pair in batch]
                    ai_batch = [pair[1] for pair in batch]

                    human_rewrites_batch = rewriter.generate_rewrites_batch(human_batch, m)
                    ai_rewrites_batch = rewriter.generate_rewrites_batch(ai_batch, m)

                    valid_indices = [
                        i for i in range(len(human_rewrites_batch))
                        if human_rewrites_batch[i] and ai_rewrites_batch[i]
                    ]

                    if not valid_indices:
                        print(f"Warning: Batch {start}-{start+BATCH_SIZE} produced no valid rewrites")
                        continue

                    valid_human_batch = [human_batch[i] for i in valid_indices]
                    valid_ai_batch = [ai_batch[i] for i in valid_indices]
                    valid_human_rewrites = [human_rewrites_batch[i] for i in valid_indices]
                    valid_ai_rewrites = [ai_rewrites_batch[i] for i in valid_indices]

                    human_scores = detector.get_detection_score_batch(
                        valid_human_batch, valid_human_rewrites
                    )
                    ai_scores = detector.get_detection_score_batch(
                        valid_ai_batch, valid_ai_rewrites
                    )

                    non_synth_scores.extend(human_scores)
                    synth_scores.extend(ai_scores)

                    # ── Build detailed rows ───────────────────────────────────
                    # correct_prediction = True when AI score > Human score
                    # (i.e. the detector correctly ranks AI above human for this pair)
                    for i in range(len(valid_human_batch)):
                        correct = ai_scores[i] > human_scores[i]
                        detailed_rows.append({
                            "Model":                  model_name,
                            "Variant":                variant_name,
                            "m":                      m,
                            "Original_Human_Code":    valid_human_batch[i],
                            "Original_AI_Code":       valid_ai_batch[i],
                            # Store all rewrites joined by a separator so the
                            # cell stays readable in Excel / CSV viewers.
                            "Rewritten_Human_Code":   " ||| ".join(valid_human_rewrites[i]),
                            "Rewritten_AI_Code":      " ||| ".join(valid_ai_rewrites[i]),
                            "Cosine_Sim_Human":       round(human_scores[i], 6),
                            "Cosine_Sim_AI":          round(ai_scores[i], 6),
                            # True  = detector was correct (AI score > Human score)
                            # False = detector was wrong   (AI score <= Human score)
                            "Correct_Prediction":     correct,
                        })
                    # ─────────────────────────────────────────────────────────

                total_non_synth_scores.extend(non_synth_scores)
                total_synth_scores.extend(synth_scores)

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
                        "CorrectPairs": metrics["correct_pairwise"],
                    })
                    print(f"✓ {variant_name}: {metrics['pairwise_count']} pairs, "
                          f"{metrics['correct_pairwise']} correct "
                          f"(Acc: {metrics['Accuracy']:.4f}, AUROC: {metrics['AUROC']:.4f})")
                else:
                    print(f"✗ {variant_name}: No valid pairs generated")
                    variant_results.append({
                        "Model": model_path, "m": m, "Variant": variant_name,
                        "AUROC": 0.0, "Accuracy": 0.0, "Pairwise_Accuracy": 0.0,
                        "Precision": 0.0, "Recall": 0.0, "F1_Score": 0.0,
                        "Pairs": 0, "CorrectPairs": 0,
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
                    "TotalCorrectPairs": totals_metrics["correct_pairwise"],
                })
                print(f"\n{'='*60}")
                print(f"OVERALL RESULTS: {model_name.upper()} | m = {m}")
                print(f"{'='*60}")
                print(f"  AUROC:             {totals_metrics['AUROC']:.4f}")
                print(f"  Accuracy:          {totals_metrics['Accuracy']:.4f}")
                print(f"  Pairwise Accuracy: {totals_metrics['Pairwise_Accuracy']:.4f}")
                print(f"  Precision:         {totals_metrics['Precision']:.4f}")
                print(f"  Recall:            {totals_metrics['Recall']:.4f}")
                print(f"  F1-Score:          {totals_metrics['F1_Score']:.4f}")
                print(f"  Pairs:             {totals_metrics['pairwise_count']}")
                print(f"  Correct:           {totals_metrics['correct_pairwise']}")
            else:
                print(f"\n✗ No valid data for {model_name} with m = {m}")

    # ── Save all three output files ───────────────────────────────────────────
    print("\n\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    df_variant = pd.DataFrame(variant_results)
    df_total   = pd.DataFrame(final_result)
    df_detail  = pd.DataFrame(detailed_rows)

    if not df_variant.empty:
        df_variant.to_csv("./evaluation_variant_results_base.csv", index=False)
        print("✓ Per-variant results saved to 'evaluation_variant_results_base.csv'")

    if not df_total.empty:
        df_total.to_csv("./evaluation_results_base.csv", index=False)
        print("✓ Overall results saved to 'evaluation_results_base.csv'")
        print("\n--- FINAL SUMMARY ---")
        print(df_total.to_string(index=False))
    else:
        print("✗ No results to save - check your data and rewriter output")

    if not df_detail.empty:
        df_detail.to_csv("./evaluation_detailed_predictions.csv", index=False)
        print(f"✓ Detailed predictions saved to 'evaluation_detailed_predictions.csv' "
              f"({len(df_detail):,} rows)")
    # ─────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    run_evaluation()
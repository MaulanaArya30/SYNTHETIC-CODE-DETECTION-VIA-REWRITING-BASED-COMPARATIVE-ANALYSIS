import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score

from data_loader import load_paired_dataset
from rewriter import CodeRewriter
from detector import CodeDetector
import settings


BATCH_SIZE_DEF = getattr(settings, "EVAL_BATCH_SIZE", 8)


def compute_classification_metrics(non_synth_scores, synth_scores):
    n_pairs = min(len(non_synth_scores), len(synth_scores))
    
    if n_pairs == 0:
        return {
            "AUROC": 0.0,
            "Accuracy": 0.0,
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
    
    #Pairwise
    correct_pairwise = sum(1 for h, a in zip(non_synth_scores, synth_scores) if a > h)
    pairwise_accuracy = correct_pairwise / n_pairs if n_pairs > 0 else 0.0
    
    return {
        "AUROC": auroc,
        "Accuracy": pairwise_accuracy,
        "pairwise_count": n_pairs,
        "correct_pairwise": correct_pairwise,
    }


def run_evaluation_comparison():
    print("\n" + "="*70)
    print("BASE MODEL vs SIMCSE COMPARISON EVALUATION")
    print("="*70 + "\n")

    rewriter = CodeRewriter()
    final_results = []

    #configurations: (model_name, use_simcse)
    test_configs = [
        ("graphcodebert", False, "GraphCodeBERT-BASE"),
        ("graphcodebert", True, "GraphCodeBERT-SimCSE"),
        ("codet5", False, "CodeT5-BASE"),
        ("codet5", True, "CodeT5-SimCSE"),
    ]

    for model_name, use_simcse, display_name in test_configs:
        print(f"\n{'='*70}")
        print(f"Testing: {display_name}")
        print(f"{'='*70}")
        
        #Initialize
        try:
            detector = CodeDetector(model_type=model_name, use_simcse=use_simcse)
        except Exception as e:
            print(f"❌ Failed to load {display_name}: {e}")
            continue

        for m in settings.NUM_REWRITES:
            print(f"\n{'─'*60}")
            print(f"{display_name} | m={m}")
            print(f"{'─'*60}")

            #adjust batch size
            if m >= 8:
                BATCH_SIZE = max(1, BATCH_SIZE_DEF // 4)
            elif m >= 4:
                BATCH_SIZE = max(2, BATCH_SIZE_DEF // 2)
            else:
                BATCH_SIZE = BATCH_SIZE_DEF

            total_non_synth_scores = []
            total_synth_scores = []

            for variant_file in settings.VARIANT_FILES:
                variant_name = os.path.basename(variant_file)
                print(f"Processing: {variant_name}...")

                paired_data = load_paired_dataset(variant_file)
                if not paired_data:
                    print(f"  Skipping {variant_name} (loading error)")
                    continue

                non_synth_scores = []
                synth_scores = []

                #batches
                n = len(paired_data)
                for start in tqdm(range(0, n, BATCH_SIZE), desc=f"  {variant_name}", leave=False):
                    batch = paired_data[start:start + BATCH_SIZE]
                    human_batch = [pair[0] for pair in batch]
                    ai_batch = [pair[1] for pair in batch]

                    # Generate rewrites
                    human_rewrites_batch = rewriter.generate_rewrites_batch(human_batch, m)
                    ai_rewrites_batch = rewriter.generate_rewrites_batch(ai_batch, m)

                    #valid pairs
                    valid_indices = [
                        i for i in range(len(human_rewrites_batch))
                        if human_rewrites_batch[i] and ai_rewrites_batch[i]
                    ]

                    if not valid_indices:
                        continue

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

                total_non_synth_scores.extend(non_synth_scores)
                total_synth_scores.extend(synth_scores)

            #metrics
            if total_non_synth_scores and total_synth_scores:
                metrics = compute_classification_metrics(
                    total_non_synth_scores, total_synth_scores
                )

                result = {
                    "Model": display_name,
                    "SimCSE": "Yes" if use_simcse else "No",
                    "m": m,
                    "AUROC": f"{metrics['AUROC']:.4f}",
                    "Accuracy": f"{metrics['Accuracy']:.4f}",
                    "TotalPairs": metrics["pairwise_count"],
                    "CorrectPairs": metrics["correct_pairwise"]
                }
                final_results.append(result)

                print(f"AUROC: {metrics['AUROC']:.4f} | "
                      f"Accuracy: {metrics['Accuracy']:.4f} | "
                      f"Pairs: {metrics['pairwise_count']}")
            else:
                print(f"No valid data")

    #Save and display results
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON RESULTS")
    print(f"{'='*70}\n")

    df_results = pd.DataFrame(final_results)
    
    if not df_results.empty:
        df_results = df_results.sort_values(['Model', 'm'])
        
        print(df_results.to_string(index=False))
        print()
        
        # Save to CSV
        df_results.to_csv("./evaluation_base_vs_simcse.csv", index=False)
        print("Results saved to 'evaluation_base_vs_simcse.csv'")
        
        # Print comparison summary
        print(f"\n{'='*70}")
        print("SUMMARY: SimCSE Impact")
        print(f"{'='*70}")
        
        for model in ["GraphCodeBERT-BASE", "CodeT5-BASE"]:
            base_model = model
            simcse_model = model.replace("-BASE", "-SimCSE")
            
            base_rows = df_results[df_results['Model'] == base_model]
            simcse_rows = df_results[df_results['Model'] == simcse_model]
            
            if not base_rows.empty and not simcse_rows.empty:
                print(f"\n{model.split('-')[0]}:")
                for m_val in settings.NUM_REWRITES:
                    base_auroc = base_rows[base_rows['m'] == m_val]['AUROC'].values
                    simcse_auroc = simcse_rows[simcse_rows['m'] == m_val]['AUROC'].values
                    
                    if len(base_auroc) > 0 and len(simcse_auroc) > 0:
                        base_val = float(base_auroc[0])
                        simcse_val = float(simcse_auroc[0])
                        improvement = simcse_val - base_val
                        symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
                        
                        print(f"  m={m_val}: BASE={base_val:.4f} vs SimCSE={simcse_val:.4f} "
                              f"({symbol} {abs(improvement):.4f})")
    else:
        print("No results to display")


if __name__ == "__main__":
    run_evaluation_comparison()
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from data_loader import load_paired_dataset
from rewriter import CodeRewriter
from detector import CodeDetector
import settings
import os



def run_evaluation():
    print('---Starting Evaluation---')

    #rewriter
    rewriter = CodeRewriter()

    #result
    final_result = []

    #embedding
    for model_name in ['graphcodebert', 'codet5']:
        model_type = settings.GCB_SIMCSE_PATH if model_name == 'graphcodebert' else settings.CODET5_SIMCSE_PATH
        if not os.path.exists(model_type):
            print('---' * 10)
            print(f'{model_type} not found...')
            print('---' * 10)
            continue

        #detector
        detector = CodeDetector(model_type=model_name)

        #rewriter
        for m in settings.NUM_REWRITES:

            #all variants
            synthetic_scores = []
            non_synthetic_scores = []
            correct_pairwise = 0
            total_pairwise = 0

            print(f'---- Start test on model {model_type} with {m} rewrites ----')

            #dataset variant
            for variant_file in settings.VARIANT_FILES:
                variant_name = variant_file.split('/')[-1]
                print(f'--- Processing: {variant_name} ---')

                paired_data = load_paired_dataset(variant_file)
                if not paired_data:
                    print(f'Skipping {variant_name} due to an error while loading...')

                for (human_code, ai_code) in tqdm(paired_data, desc=f'{variant_name}'):

                    #get score
                    synthetic_score = detector.get_detection_score(
                        ai_code,
                        rewriter.generate_rewrites(ai_code, num_rewrites=m)
                    )
                    non_synthetic_score = detector.get_detection_score(
                        human_code,
                        rewriter.generate_rewrites(human_code, num_rewrites=m)
                    )
                    #total
                    synthetic_scores.append(synthetic_score)
                    non_synthetic_scores.append(non_synthetic_score)

                    #pairwise classification
                    #total
                    total_pairwise += 1
                    #correct
                    if synthetic_score > non_synthetic_score:
                        correct_pairwise += 1

                
            #evaluation metrics
            #incorrect pairwise
            incorrect_pairwise = total_pairwise - correct_pairwise

            #basics
            TP = correct_pairwise
            TN = correct_pairwise
            FP = incorrect_pairwise
            FN = incorrect_pairwise

            #auroc
            labels = [0] * len(non_synthetic_scores) + [1] * len(synthetic_scores)
            scores = non_synthetic_scores + synthetic_scores
            auroc = 0.0
            if len(set(labels)) >> 1: #making sure three's 2 classes
                auroc = roc_auc_score(labels, scores)

            #accuracy
            accuracy = 0.0
            if (TP + TN + FP + FN) > 0:
                accuracy = (TP + TN) / (TP + TN + FP + FN)

            #precision
            precision = 0.0
            if (TP + FP) > 0:
                precision = TP / (TP + FP)

            #recall
            recall = 0.0
            if (TP + FN) > 0:
                recall = TP / (TP + FN)
                
            #F1-Score
            f1 = 0.0
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            
            
            # Store results
            final_result.append({
                'Model': model_type,
                'm': m,
                'AUROC': auroc,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            })

            print(f"--- Results for Model={model_type}, m={m} ---")
            print(f"  AUROC (Eq 3.9): {auroc:.4f}")
            print(f"  Accuracy (Eq 3.5): {accuracy:.4f}")
            print(f"  Precision (Eq 3.6): {precision:.4f}")
            print(f"  Recall (Eq 3.7): {recall:.4f}")
            print(f"  F1_Score (Eq 3.8): {f1:.4f}")


    # --- Final Report ---
    print("\n\n--- FINAL EXPERIMENT SUMMARY ---")
    results_df = pd.DataFrame(final_result)
    print(results_df.to_string())
    
    # Save results to a CSV file
    results_df.to_csv("./evaluation_results.csv", index=False)
    print("\nResults saved to 'evaluation_results.csv'")

if __name__ == "__main__":
    run_evaluation()
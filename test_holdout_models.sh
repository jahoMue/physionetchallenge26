#!/usr/bin/env bash
set -e

# ==============================================================================
# Test Script: Run and Evaluate All Trained Models on Holdout Data inside Docker
# ==============================================================================
# Usage inside Docker container:
#   sed -i 's/\r$//' test_holdout_models.sh
#   chmod +x test_holdout_models.sh
#   ./test_holdout_models.sh
# ==============================================================================

DATA_FOLDER="${1:-holdout_data}"
MODEL_FOLDER="${2:-model}"
OUTPUT_FOLDER="${3:-holdout_outputs}"
TRAIN_DATA="${4:-training_data}"

echo "================================================================="
echo " Starting Holdout Evaluation for All Trained Models"
echo "================================================================="
echo " Data Folder:           $DATA_FOLDER"
echo " Model Folder:          $MODEL_FOLDER"
echo " Output Folder:         $OUTPUT_FOLDER"
echo " Prevalence Reference:  $TRAIN_DATA/demographics.csv"
echo "================================================================="

if [ ! -d "$DATA_FOLDER" ]; then
    echo "ERROR: Data folder '$DATA_FOLDER' does not exist."
    exit 1
fi

if [ ! -d "$MODEL_FOLDER" ]; then
    echo "ERROR: Model folder '$MODEL_FOLDER' does not exist."
    exit 1
fi

MODELS_DIR="$MODEL_FOLDER"
if [ -d "$MODEL_FOLDER/models" ]; then
    MODELS_DIR="$MODEL_FOLDER/models"
fi

python investigate_holdout_models.py \
    -d "$DATA_FOLDER" \
    -M "$MODELS_DIR" \
    -o "$OUTPUT_FOLDER" \
    -p "$TRAIN_DATA/demographics.csv" \
    --shared-cache-folder "$MODEL_FOLDER" \
    --include-final-model \
    -v

echo ""
echo "================================================================="
echo " EVALUATION METRICS COMPARISON ACROSS ALL MODELS"
echo "================================================================="

# Print python formatted summary comparison table if scores_summary.csv exists
python -c "
import pandas as pd
from pathlib import Path

csv_path = Path('$OUTPUT_FOLDER') / 'scores_summary.csv'
if csv_path.exists():
    df = pd.read_csv(csv_path)
    cols = ['model_name', 'reward', 'age_conditioned_auroc', 'age_weighted_auroc', 'auroc', 'auprc', 'accuracy', 'f_measure']
    present_cols = [c for c in cols if c in df.columns]
    
    # Rename for neat terminal display
    rename_map = {
        'model_name': 'Model',
        'reward': 'Reward',
        'age_conditioned_auroc': 'Age-Cond AUROC',
        'age_weighted_auroc': 'Age-Wtd AUROC',
        'auroc': 'AUROC',
        'auprc': 'AUPRC',
        'accuracy': 'Accuracy',
        'f_measure': 'F-measure'
    }
    
    df_fmt = df[present_cols].copy()
    for num_col in ['reward', 'age_conditioned_auroc', 'age_weighted_auroc', 'auroc', 'auprc', 'accuracy', 'f_measure']:
        if num_col in df_fmt.columns:
            df_fmt[num_col] = df_fmt[num_col].map(lambda x: f'{x:.3f}' if pd.notnull(x) and isinstance(x, (int, float)) else str(x))
            
    df_fmt = df_fmt.rename(columns=rename_map)
    print(df_fmt.to_string(index=False))
else:
    print('No scores_summary.csv found in $OUTPUT_FOLDER.')
"

echo ""
echo "================================================================="
echo " INDIVIDUAL MODEL SCORE FILES (evaluate_model.py format)"
echo "================================================================="
for score_file in "$OUTPUT_FOLDER"/*/score.txt; do
    if [ -f "$score_file" ]; then
        model_dir=$(dirname "$score_file")
        model_name=$(basename "$model_dir")
        echo "-----------------------------------------------------------------"
        echo "Model: $model_name"
        echo "-----------------------------------------------------------------"
        cat "$score_file"
        echo ""
    fi
done

echo "================================================================="
echo " Detailed evaluation outputs and CSV summary saved to: $OUTPUT_FOLDER"
echo " Summary CSV: $OUTPUT_FOLDER/scores_summary.csv"
echo "================================================================="

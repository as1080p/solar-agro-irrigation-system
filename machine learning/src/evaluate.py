# src/evaluate.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
from .config import PROCESSED_CSV, STATE_COL

FORECAST_TARGETS = ['moisture_tplus1']  # Only the first horizon target
FORECAST_PREDS = ['pred_moisture_t1']   # Only the available predicted column
CLF_LABEL = 'irrigation_label'
CLF_PRED = 'pred_class'

def evaluate_forecast(true_csv='data/processed/fe_with_preds_maharashtra.csv', 
                      pred_csv='data/processed/fe_with_preds_maharashtra.csv'):
    df_true = pd.read_csv(true_csv)
    df_pred = pd.read_csv(pred_csv)

    # Filter for Maharashtra
    df_true = df_true[df_true[STATE_COL].str.strip().str.lower() == 'maharashtra']
    df_pred = df_pred[df_pred[STATE_COL].str.strip().str.lower() == 'maharashtra']

    # Rename predicted column to avoid overlapping column names during join
    df_pred_renamed = df_pred[FORECAST_PREDS].rename(columns=lambda x: x + '_pred')

    # Join on index assuming both dfs aligned by index
    joined = df_true.join(df_pred_renamed)

    print("Soil Moisture Forecast Evaluation for Maharashtra (t+1):")
    true_col = FORECAST_TARGETS[0]
    pred_col = FORECAST_PREDS[0] + '_pred'
    y_true = joined[true_col].dropna()
    y_pred = joined.loc[y_true.index, pred_col].dropna()
    y_true, y_pred = y_true.align(y_pred, join='inner')
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"Horizon t+1: MAE={mae:.2f}  RMSE={rmse:.2f}")

def evaluate_classifier(labels_csv='data/processed/fe_with_preds_maharashtra.csv', 
                        preds_csv='data/processed/predictions_output_maharashtra.csv'):
    df_labels = pd.read_csv(labels_csv)
    df_preds = pd.read_csv(preds_csv)
    
    # Filter for Maharashtra
    df_labels = df_labels[df_labels[STATE_COL].str.strip().str.lower() == 'maharashtra']
    df_preds = df_preds[df_preds[STATE_COL].str.strip().str.lower() == 'maharashtra']
    
    y_true = df_labels[CLF_LABEL]
    y_pred = df_preds[CLF_PRED]
    
    print("\nIrrigation Classifier Evaluation for Maharashtra:")
    print(classification_report(y_true, y_pred, digits=3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_forecast()
    evaluate_classifier()

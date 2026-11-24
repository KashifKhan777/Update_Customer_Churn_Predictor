# Customer Churn Predictor — Streamlit

## Files
- `app.py` — Streamlit application
- `churn_model.joblib` — trained preprocessing + Logistic Regression pipeline
- `requirements.txt` — deployment dependencies
- `Telco-Customer-Churn.csv` — original dataset used for training/reference
- `model_metrics.json` — final test metrics
- `classification_report.txt` — classification report
- `confusion_matrix.txt` — confusion matrix

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository.
2. Upload `app.py`, `churn_model.joblib`, and `requirements.txt`.
3. Optionally upload the dataset and metric files for documentation.
4. In Streamlit Community Cloud, select the repository and set the main file to `app.py`.
5. Deploy.

## Model
The model is a tuned Logistic Regression pipeline. Numeric columns are standardized and categorical columns are one-hot encoded with `handle_unknown="ignore"`. The target is `Churn` (`Yes`/`No`). Hyperparameter tuning selected the best `C` using 5-fold cross-validation with recall as the scoring metric.

# email-phishing-detector

Phishing email detection PoC based on `PoC_Doc.csv`.

## Quick start

### Google Colab (recommended)
- Upload the CSV to Colab, or mount Google Drive.
- This script runs with Colab's default Python environment.
- Optional features:
  - NLTK-based stopword removal + lemmatization (best-effort; can be disabled with `--no-nltk`).
  - Model saving via `joblib` when available; otherwise it falls back to Python `pickle`.

Run (CSV uploaded to the session):
```bash
python src/phishing_email_poc.py --csv /content/<file>.csv --target <label-col> --text-col <text-col> --no-save
```

### Local
1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Put your Kaggle CSV under `data/` (recommended).

3) Run training (Windows/local):
```bash
python src\phishing_email_poc.py --csv data\<file>.csv --target <label-col> --text-col <text-col>
```

(For Kaggle and sender-column examples, see **How to run** below.)

## What this contains
- `src/phishing_email_poc.py`: trains and evaluates 3 models (Logistic Regression, SVM, Random Forest)
- `requirements.txt`: Python dependencies
- `artifacts/`: saved trained model pipelines (`.joblib`) will be written here after training
- `data/`: optional place to put downloaded Kaggle CSV files (not required)

## Requirements
- Python 3.9+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset (Kaggle)
This project expects a Kaggle-provided CSV containing at minimum:
- a text column (email content)
- a binary label column (phishing vs legitimate)

You pass the column names at runtime.

### Where the CSV comes from
- **On Kaggle** notebooks/runners, datasets are mounted under: `/kaggle/input/<dataset-folder>/...`
- **On Windows/local**, download the Kaggle dataset and place it under `data/` (recommended), then point `--csv` to that file.

## How to run
### Windows / local
```bash
python src\phishing_email_poc.py --csv data\<file>.csv --target <label-col> --text-col <text-col>
```

### Kaggle
```bash
python src/phishing_email_poc.py \
  --csv /kaggle/input/<dataset-folder>/<file>.csv \
  --target <label-col> \
  --text-col <text-col>
```

## Optional sender column
If your dataset includes sender/from information, pass it via `--sender-col`. The script will add sender-domain heuristic features.

Windows / local:
```bash
python src\phishing_email_poc.py --csv data\<file>.csv --target <label-col> --text-col <text-col> --sender-col <sender-col>
```

Kaggle:
```bash
python src/phishing_email_poc.py \
  --csv /kaggle/input/<dataset-folder>/<file>.csv \
  --target <label-col> \
  --text-col <text-col> \
  --sender-col <sender-col>
```

## Output
- Metrics are printed for each model (Accuracy/Precision/Recall/F1, plus ROC-AUC when available).
- Models are saved to `artifacts/` unless you pass `--no-save`.
  - If `joblib` is available: `artifacts/<model>.joblib`
  - Otherwise: `artifacts/<model>.pkl`

## Notes
- If NLTK is not available (or resources cannot be downloaded), the script automatically falls back to basic regex cleaning. You can also force this with `--no-nltk`.
# ABC Credit Loan Chatbot

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt -r chat/requirements.txt
cp chat/.env.example chat/.env      # then set GEMINI_API_KEY
```

For most-secure mode (local LLM), install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull gemma4:e4b
```

## Run

Web chat at http://localhost:8000 (admin at `/admin`; set `CHAT_PORT` to change the port):

```bash
venv/bin/python -m chat.server
```

Terminal chat (add `--secure` for most-secure mode):

```bash
venv/bin/python -m chat.cli
```

## Test

```bash
venv/bin/python -m chat.tests.test_flow
```

## Retrain the model (optional)

```bash
cd notebooks && ../venv/bin/jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb 02_preprocessing_feature_selection.ipynb 03_model_training_evaluation.ipynb
```

Score a hold-out file:

```bash
venv/bin/python scripts/predict_holdout.py path/to/holdout.csv -o reports/holdout_predictions.csv
```

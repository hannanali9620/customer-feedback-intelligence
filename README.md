# Customer Feedback Intelligence

Hybrid NLP & Executive Reporting App built with Python 3.11, Flask, pandas, and Chart.js.

## Architecture (V1 / V2 / V3)
- **V1 (Port 5001):** Offline low-spec baseline using VADER sentiment, YAKE key phrases, keyword complaint rules, and a deterministic Template-Based Executive Report.
- **V2 (Port 5002):** Adds opt-in Hugging Face DistilBERT SST-2 (ENABLE_DISTILBERT=true) and auto-detected local Ollama (http://localhost:11434), falling back to V1 engines.
- **V3 (Port 5003):** Cloud-first hybrid using Azure AI Language / Foundry and Google Gemini API when valid credentials exist in .env, falling back through V2 -> V1.

## Input CSV Schema & Business KPIs
- **Required CSV Columns:** customer_id, product, feedback
- **Standardized KPIs:** Total Reviews, Unique Products, Positive/Neutral/Negative % & Counts, Net Sentiment Score (Positive % - Negative %), Mean Sentiment Score, Top Complaint Category, and Most-Reviewed Product.

## Run Commands
- python V1\app.py -> [http://127.0.0.1:5001](http://127.0.0.1:5001)
- python V2\app.py -> [http://127.0.0.1:5002](http://127.0.0.1:5002)
- python V3\app.py -> [http://127.0.0.1:5003](http://127.0.0.1:5003)

## Model & Rule Limitations
- **DistilBERT SST-2:** English binary sentiment (Positive/Negative only); has no native Neutral class.
- **VADER Sentiment:** Lexicon/rule-based English scorer; limited on domain jargon, sarcasm, and non-English feedback.
- **Rule-Based Complaint Categories:** Keyword-matched across 5 categories; multi-issue reviews map to a single top match and novel phrasing maps to General / Other.
- **Executive Reports:** Template-based reports are deterministic summaries without LLM reasoning; generative outputs require human verification.

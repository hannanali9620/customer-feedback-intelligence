import os
import re
from collections import Counter
import pandas as pd
import requests
import yake
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

REQUIRED_COLUMNS = {"customer_id", "product", "feedback"}

COMPLAINT_RULES = {
    "Delivery & Shipping": ["late", "delay", "delayed", "shipping", "delivery", "courier", "arrived", "package", "damaged", "box"],
    "Product Quality & Defects": ["broken", "defect", "defective", "cheap", "quality", "stopped working", "faulty", "scratch", "battery", "overheat"],
    "Pricing & Billing": ["price", "expensive", "overpriced", "refund", "charge", "charged", "billing", "money", "worth", "cost"],
    "Customer Support": ["support", "service", "rude", "response", "reply", "agent", "unhelpful", "ignored", "warranty", "return"],
    "Usability & Setup": ["confusing", "hard to use", "setup", "install", "manual", "instruction", "app", "crash", "bug", "slow"],
}

LIMITATIONS_DISCLOSURE = [
    "Azure AI Language / Foundry (when active): Cloud NLP service; incurs API usage/rate limits and depends on network availability.",
    "DistilBERT SST-2 (when active): Trained strictly for English binary sentiment (Positive vs. Negative); it has NO native Neutral class, forcing neutral reviews into Positive or Negative.",
    "VADER Sentiment (when active): Lexicon- and rule-based English scorer; struggles with domain-specific jargon, sarcasm, and non-English feedback.",
    "Complaint Categorization (Keyword Rules): Matches predefined English terms; multi-issue reviews map to a single top match and novel phrasing may fall into 'General / Other'.",
    "Executive Report: Generative models (Gemini / Ollama) synthesize metrics into prose but may generalize; Template-Based mode is strictly deterministic without LLM reasoning.",
]

_DISTILBERT_PIPELINE = None


class SchemaValidationError(ValueError):
    """Raised when uploaded CSV does not match required schema."""


def _is_configured_secret(value: str | None) -> bool:
    if not value:
        return False
    cleaned = value.strip()
    if not cleaned:
        return False
    lowered = cleaned.lower()
    placeholders = ("your_", "change-me", "placeholder", "xxx", "<", "insert_")
    return not any(p in lowered for p in placeholders)


def validate_and_clean_csv(file_source) -> pd.DataFrame:
    df = pd.read_csv(file_source)
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            "Expected schema: customer_id, product, feedback"
        )

    df = df[["customer_id", "product", "feedback"]].copy()
    df["customer_id"] = df["customer_id"].fillna("UNKNOWN").astype(str).str.strip()
    df["product"] = df["product"].fillna("Unspecified Product").astype(str).str.strip()
    df["feedback"] = df["feedback"].fillna("").astype(str).str.strip()
    df = df[df["feedback"] != ""].reset_index(drop=True)

    if df.empty:
        raise SchemaValidationError("The CSV contains no non-empty feedback rows.")

    return df


def classify_complaint(text: str, sentiment_label: str) -> str:
    if sentiment_label == "Positive":
        return "None (Positive)"
    lowered = text.lower()
    best_cat = "General / Other"
    best_hits = 0
    for category, keywords in COMPLAINT_RULES.items():
        hits = sum(1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", lowered))
        if hits > best_hits:
            best_hits = hits
            best_cat = category
    return best_cat


def extract_yake_keyphrases(texts: list[str], top_n: int = 8) -> list[dict]:
    joined = "\n".join(texts)
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=top_n, features=None)
    raw_phrases = kw_extractor.extract_keywords(joined)
    results = []
    for phrase, score in raw_phrases:
        relevance = round(max(1.0, min(100.0, (1.0 / (1.0 + score)) * 100)), 1)
        results.append({"phrase": phrase, "relevance": relevance})
    return results


def _try_azure_language(texts: list[str]):
    endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT", "").strip()
    key = os.getenv("AZURE_LANGUAGE_KEY", "").strip()
    if not (_is_configured_secret(endpoint) and _is_configured_secret(key)):
        return None, "Azure credentials not configured in .env"
    if not endpoint.startswith("https://"):
        return None, "Azure endpoint invalid (must start with https://)"

    try:
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential

        client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        labels = []
        scores = []
        phrase_counter = Counter()

        # Azure Text Analytics accepts up to 10 documents per request
        for i in range(0, len(texts), 10):
            batch = [t[:5000] for t in texts[i : i + 10]]
            sent_res = client.analyze_sentiment(batch, language="en")
            kp_res = client.extract_key_phrases(batch, language="en")

            for doc in sent_res:
                if doc.is_error:
                    raise RuntimeError(f"Azure sentiment error: {doc.error.message}")
                s_map = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative", "mixed": "Neutral"}
                lbl = s_map.get(doc.sentiment.lower(), "Neutral")
                net_score = round(doc.confidence_scores.positive - doc.confidence_scores.negative, 4)
                labels.append(lbl)
                scores.append(net_score)

            for doc in kp_res:
                if not doc.is_error:
                    for phrase in doc.key_phrases:
                        phrase_counter[phrase.lower()] += 1

        top_phrases = [
            {"phrase": phrase, "relevance": round(min(100.0, (count / max(1, len(texts))) * 100 + 25), 1)}
            for phrase, count in phrase_counter.most_common(8)
        ]
        if not top_phrases:
            top_phrases = extract_yake_keyphrases(texts, top_n=8)

        return (labels, scores, top_phrases), None
    except Exception as exc:
        return None, f"Azure call failed ({exc.__class__.__name__})"


def _try_distilbert(texts: list[str]):
    global _DISTILBERT_PIPELINE
    enabled = os.getenv("ENABLE_DISTILBERT", "false").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return None, "DistilBERT disabled via .env"
    try:
        if _DISTILBERT_PIPELINE is None:
            from transformers import pipeline
            _DISTILBERT_PIPELINE = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,
                truncation=True,
            )
        preds = _DISTILBERT_PIPELINE(texts, batch_size=8)
        labels = []
        scores = []
        for pred in preds:
            is_pos = pred["label"].upper() == "POSITIVE"
            conf = float(pred["score"])
            labels.append("Positive" if is_pos else "Negative")
            scores.append(round(conf if is_pos else -conf, 4))
        return (labels, scores), None
    except Exception as exc:
        return None, f"DistilBERT unavailable ({exc.__class__.__name__})"


def run_nlp_pipeline(texts: list[str]) -> tuple[list[str], list[float], list[dict], str, str]:
    # 1. Try Azure AI Language / Foundry first when valid credentials exist
    azure_out, azure_reason = _try_azure_language(texts)
    if azure_out is not None:
        labels, scores, top_phrases = azure_out
        return (
            labels,
            scores,
            top_phrases,
            "Azure AI Language / Foundry (Cloud)",
            "Azure AI Language Key Phrase Extraction (Cloud)",
        )

    # 2. Try local DistilBERT if explicitly enabled on capable PC
    bert_out, bert_reason = _try_distilbert(texts)
    top_phrases = extract_yake_keyphrases(texts, top_n=8)
    if bert_out is not None:
        labels, scores = bert_out
        return (
            labels,
            scores,
            top_phrases,
            f"Hugging Face DistilBERT SST-2 (Local CPU Binary Positive/Negative — {azure_reason})",
            "YAKE Unsupervised Extractor (Offline)",
        )

    # 3. Default low-spec offline fallback: VADER + YAKE
    vader = SentimentIntensityAnalyzer()
    labels = []
    scores = []
    for text in texts:
        comp = round(vader.polarity_scores(text)["compound"], 4)
        if comp >= 0.05:
            label = "Positive"
        elif comp <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        labels.append(label)
        scores.append(comp)

    return (
        labels,
        scores,
        top_phrases,
        f"VADER Lexicon (Offline Fallback — {azure_reason}; {bert_reason})",
        "YAKE Unsupervised Extractor (Offline)",
    )


def build_template_report(kpis: dict, product_breakdown: list[dict], top_phrases: list[dict], sentiment_backend: str) -> str:
    top_prod = product_breakdown[0]["product"] if product_breakdown else "N/A"
    top_prod_neg = max(product_breakdown, key=lambda x: x["negative_pct"])["product"] if product_breakdown else "N/A"
    phrase_list = ", ".join(p["phrase"] for p in top_phrases[:5]) or "none detected"

    return (
        f"[TEMPLATE-BASED EXECUTIVE REPORT — NO GENERATIVE LLM USED]\n"
        f"Active Sentiment Engine: {sentiment_backend}\n\n"
        f"1. Overall Customer Pulse: Analyzed {kpis['total_reviews']} customer feedback records across "
        f"{kpis['unique_products']} product(s). Sentiment distribution is {kpis['positive_pct']}% Positive, "
        f"{kpis['neutral_pct']}% Neutral, and {kpis['negative_pct']}% Negative, yielding a Net Sentiment Score "
        f"of {kpis['net_sentiment_score']} (mean sentiment score: {kpis['avg_compound_score']}).\n\n"
        f"2. Primary Risk & Friction Drivers: The leading complaint driver is '{kpis['top_complaint_category']}'. "
        f"Product '{top_prod}' generated the highest feedback volume, while '{top_prod_neg}' recorded the highest "
        f"share of negative sentiment.\n\n"
        f"3. Recurring Themes: Key recurring terms include: {phrase_list}.\n\n"
        f"4. Recommended Action: Prioritize root-cause investigation into '{kpis['top_complaint_category']}' "
        f"for '{top_prod_neg}' and audit recent negative customer tickets."
    )


def generate_executive_report(
    kpis: dict,
    product_breakdown: list[dict],
    top_phrases: list[dict],
    complaint_dist: dict,
    sentiment_backend: str,
) -> tuple[str, str]:
    prompt = (
        "You are a Senior Product Operations Analyst. Write a crisp, 4-point executive customer feedback brief "
        "grounded strictly in the verified metrics below. Include a brief note acknowledging the limitations of "
        "the active sentiment/categorization engine.\n\n"
        f"Sentiment Engine: {sentiment_backend}\n"
        f"KPIs: {kpis}\n"
        f"Product Breakdown: {product_breakdown}\n"
        f"Complaint Distribution: {complaint_dist}\n"
        f"Top Key Phrases: {[p['phrase'] for p in top_phrases[:6]]}\n"
    )

    # 1. Try Gemini API first when valid key is present
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    if _is_configured_secret(gemini_key):
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(model=gemini_model, contents=prompt)
            text = (response.text or "").strip()
            if text:
                return (
                    f"[GEMINI API GENERATIVE REPORT — MODEL: {gemini_model}]\n\n{text}",
                    f"Google Gemini API ({gemini_model})",
                )
        except Exception:
            pass

    # 2. Try local Ollama daemon second
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    preferred_model = os.getenv("OLLAMA_MODEL", "llama3.2").strip()
    try:
        tags_resp = requests.get(f"{base_url}/api/tags", timeout=1.5)
        tags_resp.raise_for_status()
        models = [m.get("name", "") for m in tags_resp.json().get("models", []) if m.get("name")]
        if models:
            chosen_model = next((m for m in models if m.startswith(preferred_model)), models[0])
            gen_resp = requests.post(
                f"{base_url}/api/generate",
                json={"model": chosen_model, "prompt": prompt, "stream": False},
                timeout=25,
            )
            gen_resp.raise_for_status()
            summary = gen_resp.json().get("response", "").strip()
            if summary:
                return (
                    f"[LOCAL OLLAMA GENERATIVE REPORT — MODEL: {chosen_model}]\n\n{summary}",
                    f"Ollama Local LLM ({chosen_model})",
                )
    except Exception:
        pass

    # 3. Deterministic Template-Based Fallback
    fallback = build_template_report(kpis, product_breakdown, top_phrases, sentiment_backend)
    return fallback, "Template-Based Report (Deterministic Fallback — No Gemini Key or Ollama)"


def analyze_feedback(file_source) -> dict:
    df = validate_and_clean_csv(file_source)
    texts = df["feedback"].tolist()

    sentiments, scores, top_phrases, sentiment_backend, keyphrase_backend = run_nlp_pipeline(texts)
    categories = [classify_complaint(txt, lbl) for txt, lbl in zip(texts, sentiments)]

    df["sentiment"] = sentiments
    df["compound_score"] = scores
    df["complaint_category"] = categories

    total = len(df)
    pos_count = int((df["sentiment"] == "Positive").sum())
    neu_count = int((df["sentiment"] == "Neutral").sum())
    neg_count = int((df["sentiment"] == "Negative").sum())

    pos_pct = round((pos_count / total) * 100, 1)
    neu_pct = round((neu_count / total) * 100, 1)
    neg_pct = round((neg_count / total) * 100, 1)
    net_sentiment = round(pos_pct - neg_pct, 1)
    avg_compound = round(float(df["compound_score"].mean()), 3)

    non_pos_cats = df[df["complaint_category"] != "None (Positive)"]["complaint_category"]
    if not non_pos_cats.empty:
        cat_counts = Counter(non_pos_cats)
        top_complaint = cat_counts.most_common(1)[0][0]
    else:
        cat_counts = Counter()
        top_complaint = "None"

    product_breakdown = []
    for prod, grp in df.groupby("product"):
        p_total = len(grp)
        p_pos = int((grp["sentiment"] == "Positive").sum())
        p_neu = int((grp["sentiment"] == "Neutral").sum())
        p_neg = int((grp["sentiment"] == "Negative").sum())
        product_breakdown.append({
            "product": prod,
            "total": p_total,
            "positive": p_pos,
            "neutral": p_neu,
            "negative": p_neg,
            "negative_pct": round((p_neg / p_total) * 100, 1),
        })
    product_breakdown.sort(key=lambda x: x["total"], reverse=True)

    kpis = {
        "total_reviews": total,
        "unique_products": int(df["product"].nunique()),
        "positive_count": pos_count,
        "neutral_count": neu_count,
        "negative_count": neg_count,
        "positive_pct": pos_pct,
        "neutral_pct": neu_pct,
        "negative_pct": neg_pct,
        "net_sentiment_score": net_sentiment,
        "avg_compound_score": avg_compound,
        "top_complaint_category": top_complaint,
        "most_reviewed_product": product_breakdown[0]["product"] if product_breakdown else "N/A",
    }

    executive_report, report_backend = generate_executive_report(
        kpis, product_breakdown, top_phrases, dict(cat_counts), sentiment_backend
    )

    return {
        "backends": {
            "sentiment": sentiment_backend,
            "keyphrases": keyphrase_backend,
            "report": report_backend,
        },
        "kpis": kpis,
        "complaint_distribution": dict(cat_counts),
        "product_breakdown": product_breakdown,
        "top_keyphrases": top_phrases,
        "executive_report": executive_report,
        "limitations": LIMITATIONS_DISCLOSURE,
        "records": df.to_dict(orient="records"),
    }
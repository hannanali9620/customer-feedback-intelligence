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

BASE_LIMITATIONS = [
    "VADER Sentiment (when active): Lexicon/rule-based English scorer; struggles with domain jargon, subtle sarcasm, and non-English text.",
    "DistilBERT SST-2 (when active): Trained strictly for English binary sentiment (Positive vs. Negative); it has NO native Neutral class, so neutral remarks are forced into Positive or Negative.",
    "Complaint Categorization (Keyword Rules): Matches fixed English terms; multi-topic reviews map to a single top match and novel phrasing may fall into 'General / Other'.",
    "Executive Report: Template-based mode is deterministic without LLM synthesis; local Ollama output quality depends on the local quantized model.",
]

_DISTILBERT_PIPELINE = None


class SchemaValidationError(ValueError):
    """Raised when uploaded CSV does not match required schema."""


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


def extract_top_keyphrases(texts: list[str], top_n: int = 8) -> list[dict]:
    joined = "\n".join(texts)
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=top_n, features=None)
    raw_phrases = kw_extractor.extract_keywords(joined)
    results = []
    for phrase, score in raw_phrases:
        relevance = round(max(1.0, min(100.0, (1.0 / (1.0 + score)) * 100)), 1)
        results.append({"phrase": phrase, "relevance": relevance})
    return results


def _try_get_distilbert():
    global _DISTILBERT_PIPELINE
    enabled = os.getenv("ENABLE_DISTILBERT", "false").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return None, "disabled via .env (ENABLE_DISTILBERT=false)"
    if _DISTILBERT_PIPELINE is not None:
        return _DISTILBERT_PIPELINE, None
    try:
        from transformers import pipeline
        _DISTILBERT_PIPELINE = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            truncation=True,
        )
        return _DISTILBERT_PIPELINE, None
    except Exception as exc:
        return None, f"unavailable ({exc.__class__.__name__})"


def score_sentiments(texts: list[str]) -> tuple[list[str], list[float], str]:
    clf, fallback_reason = _try_get_distilbert()
    if clf is not None:
        preds = clf(texts, batch_size=8)
        labels = []
        scores = []
        for pred in preds:
            is_pos = pred["label"].upper() == "POSITIVE"
            conf = float(pred["score"])
            labels.append("Positive" if is_pos else "Negative")
            scores.append(round(conf if is_pos else -conf, 4))
        return (
            labels,
            scores,
            "Hugging Face DistilBERT SST-2 (Local CPU — English Binary Positive/Negative Only)",
        )

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

    return labels, scores, f"VADER Lexicon (Offline / Low-Spec — DistilBERT {fallback_reason})"


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
        f"3. Recurring Themes (YAKE): Key recurring terms include: {phrase_list}.\n\n"
        f"4. Recommended Action: Prioritize root-cause investigation into '{kpis['top_complaint_category']}' "
        f"for '{top_prod_neg}' and audit recent negative customer tickets."
    )


def try_generate_ollama_report(
    kpis: dict,
    product_breakdown: list[dict],
    top_phrases: list[dict],
    complaint_dist: dict,
    sentiment_backend: str,
) -> tuple[str, str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    preferred_model = os.getenv("OLLAMA_MODEL", "llama3.2").strip()

    try:
        tags_resp = requests.get(f"{base_url}/api/tags", timeout=1.5)
        tags_resp.raise_for_status()
        models = [m.get("name", "") for m in tags_resp.json().get("models", []) if m.get("name")]
        if not models:
            raise RuntimeError("Ollama running but no models pulled")

        chosen_model = next((m for m in models if m.startswith(preferred_model)), models[0])

        prompt = (
            "You are a Product Operations Analyst. Write a concise 4-point executive customer feedback report "
            "using ONLY the verified metrics below. Explicitly mention any sentiment engine limitations.\n\n"
            f"Sentiment Backend: {sentiment_backend}\n"
            f"KPIs: {kpis}\n"
            f"Product Breakdown: {product_breakdown}\n"
            f"Complaint Distribution: {complaint_dist}\n"
            f"Top YAKE Phrases: {[p['phrase'] for p in top_phrases[:6]]}\n"
        )

        gen_resp = requests.post(
            f"{base_url}/api/generate",
            json={"model": chosen_model, "prompt": prompt, "stream": False},
            timeout=25,
        )
        gen_resp.raise_for_status()
        summary = gen_resp.json().get("response", "").strip()
        if not summary:
            raise RuntimeError("Empty response from Ollama")

        labeled_report = (
            f"[LOCAL OLLAMA GENERATIVE REPORT — MODEL: {chosen_model}]\n\n"
            f"{summary}"
        )
        return labeled_report, f"Ollama Local LLM ({chosen_model})"
    except Exception:
        fallback = build_template_report(kpis, product_breakdown, top_phrases, sentiment_backend)
        return fallback, "Template-Based Report (Deterministic Fallback — Ollama Not Running)"


def analyze_feedback(file_source) -> dict:
    df = validate_and_clean_csv(file_source)
    texts = df["feedback"].tolist()

    sentiments, scores, sentiment_backend = score_sentiments(texts)
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

    top_phrases = extract_top_keyphrases(texts, top_n=8)
    executive_report, report_backend = try_generate_ollama_report(
        kpis, product_breakdown, top_phrases, dict(cat_counts), sentiment_backend
    )

    return {
        "backends": {
            "sentiment": sentiment_backend,
            "keyphrases": "YAKE Unsupervised Extractor (Offline)",
            "report": report_backend,
        },
        "kpis": kpis,
        "complaint_distribution": dict(cat_counts),
        "product_breakdown": product_breakdown,
        "top_keyphrases": top_phrases,
        "executive_report": executive_report,
        "limitations": BASE_LIMITATIONS,
        "records": df.to_dict(orient="records"),
    }
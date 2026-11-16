import re
from collections import Counter
import pandas as pd
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
    "Sentiment Engine (VADER): Rule- and lexicon-based English scorer; struggles with domain-specific jargon, subtle sarcasm, and non-English feedback.",
    "Complaint Categorization (Keyword Rules): Matches predefined terms; a single review may fit multiple categories or miss novel phrasing not in the keyword dictionary.",
    "Executive Summary (Template-Based): Deterministic rule-filled narrative; summarizes aggregate metrics without generative reasoning.",
]


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


def build_template_report(kpis: dict, product_breakdown: list[dict], top_phrases: list[dict]) -> str:
    top_prod = product_breakdown[0]["product"] if product_breakdown else "N/A"
    top_prod_neg = max(product_breakdown, key=lambda x: x["negative_pct"])["product"] if product_breakdown else "N/A"
    phrase_list = ", ".join(p["phrase"] for p in top_phrases[:5]) or "none detected"

    return (
        f"[TEMPLATE-BASED EXECUTIVE REPORT — NO GENERATIVE LLM USED]\n\n"
        f"1. Overall Customer Pulse: Analyzed {kpis['total_reviews']} customer feedback records across "
        f"{kpis['unique_products']} product(s). Sentiment distribution is {kpis['positive_pct']}% Positive, "
        f"{kpis['neutral_pct']}% Neutral, and {kpis['negative_pct']}% Negative, yielding a Net Sentiment Score "
        f"of {kpis['net_sentiment_score']} (mean VADER compound score: {kpis['avg_compound_score']}).\n\n"
        f"2. Primary Risk & Friction Drivers: The leading complaint driver is '{kpis['top_complaint_category']}'. "
        f"Product '{top_prod}' generated the highest feedback volume, while '{top_prod_neg}' recorded the highest "
        f"share of negative sentiment.\n\n"
        f"3. Recurring Themes (YAKE): Key recurring terms include: {phrase_list}.\n\n"
        f"4. Recommended Action: Prioritize root-cause investigation into '{kpis['top_complaint_category']}' "
        f"for '{top_prod_neg}' and audit recent negative customer tickets."
    )


def analyze_feedback(file_source) -> dict:
    df = validate_and_clean_csv(file_source)
    vader = SentimentIntensityAnalyzer()

    sentiments = []
    compounds = []
    categories = []

    for text in df["feedback"]:
        scores = vader.polarity_scores(text)
        comp = round(scores["compound"], 4)
        if comp >= 0.05:
            label = "Positive"
        elif comp <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        sentiments.append(label)
        compounds.append(comp)
        categories.append(classify_complaint(text, label))

    df["sentiment"] = sentiments
    df["compound_score"] = compounds
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

    top_phrases = extract_top_keyphrases(df["feedback"].tolist(), top_n=8)
    executive_report = build_template_report(kpis, product_breakdown, top_phrases)

    return {
        "backends": {
            "sentiment": "VADER Lexicon (Offline / Low-Spec)",
            "keyphrases": "YAKE Unsupervised Extractor (Offline)",
            "report": "Template-Based Report (Deterministic / No LLM)",
        },
        "kpis": kpis,
        "complaint_distribution": dict(cat_counts),
        "product_breakdown": product_breakdown,
        "top_keyphrases": top_phrases,
        "executive_report": executive_report,
        "limitations": LIMITATIONS_DISCLOSURE,
        "records": df.to_dict(orient="records"),
    }
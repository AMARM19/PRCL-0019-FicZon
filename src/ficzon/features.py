"""Cleaning and feature engineering for FicZon leads.

Two steps, both shared by the notebook and the API so training and serving match:

1. ``clean_raw(df)``   raw CRM export  ->  one tidy row per lead (no PII: mobile / e-mail become flags)
2. ``make_features(df)`` tidy lead rows ->  model features (time parts, grouped categories)

``make_features`` is the first step of the saved sklearn pipeline, so the API only has to send tidy lead
fields (see ``LEAD_FIELDS``) and the pipeline does the rest.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- target
HIGH_STATUSES = ["CONVERTED", "converted", "In Progress Positive", "Potential", "Long Term", "Open"]
LOW_STATUSES = ["Junk Lead", "Not Responding", "Just Enquiry", "In Progress Negative", "LOST"]


def make_target(status: pd.Series) -> pd.Series:
    """1 = High potential lead, 0 = Low potential lead."""
    s = status.astype(str).str.strip()
    unknown = set(s.unique()) - set(HIGH_STATUSES) - set(LOW_STATUSES)
    if unknown:
        raise ValueError(f"Unmapped Status values: {sorted(unknown)}")
    return s.isin(HIGH_STATUSES).astype(int)


# ----------------------------------------------------------------------------- category groups
SOURCE_GROUPS = {
    "Call": "Call",
    "Live Chat-Direct": "Live Chat-Direct",
    "Website": "Website",
    "US Website": "US Website",
    "Live Chat-Google Organic": "Live Chat-Search/Ads",
    "Live Chat -PPC": "Live Chat-Search/Ads",
    "Live Chat-CPC": "Live Chat-Search/Ads",
    "Live Chat-Google Ads": "Live Chat-Search/Ads",
    "Live Chat-Adwords Remarketing": "Live Chat-Search/Ads",
    "Live Chat-Blog": "Live Chat-Content",
    "Live Chat-Quora": "Live Chat-Content",
    "Live Chat-Youtube": "Live Chat-Content",
    "Live Chat-Justdial": "Just Dial",
    "Just Dial": "Just Dial",
    "Customer Referral": "Referral",
    "By Recommendation": "Referral",
    "Personal Contact": "Referral",
    "Existing Client": "Existing Client",
    "Existing Customer": "Existing Client",
    "Campaign": "Campaign",
    "E-mail Campaign": "Campaign",
    "SMS Campaign": "Campaign",
    "E-Mail Message": "Campaign",
    "CRM form": "Campaign",
    "Other": "Other",
}
SOURCE_CHOICES = sorted(set(SOURCE_GROUPS.values()) | {"Unknown"})

INTERNATIONAL = {"UAE", "USA", "UK", "AUSTRALIA", "Singapore", "Malaysia", "EUROPE"}
LOCATION_CHOICES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune", "Trivandrum",
                    "Other Locations", "International", "Unknown"]

EMAIL_CHOICES = ["gmail", "yahoo", "microsoft", "other", "missing"]
DELIVERY_CHOICES = ["Mode-1", "Mode-2", "Mode-3", "Mode-4", "Mode-5"]

# Fields a caller (UI / API) sends for one lead.
LEAD_FIELDS = ["created", "product_id", "source", "sales_agent", "location", "delivery_mode",
               "has_mobile", "email_type"]


def group_source(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    out = s.map(lambda v: SOURCE_GROUPS.get(v, v) if pd.notna(v) else "Unknown")
    return out.where(out.isin(SOURCE_CHOICES), "Other").astype(str)


def group_location(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().replace({"Howrah": "Kolkata"})
    s = s.where(~s.isin(INTERNATIONAL), "International").fillna("Unknown")
    return s.where(s.isin(LOCATION_CHOICES), "Other Locations").astype(str)


def email_type(s: pd.Series) -> pd.Series:
    """The e-mails are masked (e.g. ``aXXXXXXX@gmail.com``), so keep only the provider, never the address."""
    e = s.astype("string").str.lower().fillna("")

    def one(v: str) -> str:
        if not v or v in {"#value!", "nan"}:
            return "missing"
        if "gmail" in v:
            return "gmail"
        if re.search(r"yahoo|ymail|ahoo\.", v):
            return "yahoo"
        if re.search(r"hotmail|outlook|live\.|msn\.|otmail", v):
            return "microsoft"
        return "other"

    return e.map(one).astype(str)


def has_mobile(s: pd.Series) -> pd.Series:
    """1 if a (masked) number with at least the 10-digit pattern was given."""
    m = s.astype("string").fillna("")
    digits_or_x = m.str.replace(r"[^0-9X]", "", regex=True)
    return (digits_or_x.str.len() >= 10).astype(int)


# ----------------------------------------------------------------------------- step 1: raw -> tidy
def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw CRM export -> tidy lead rows with ``LEAD_FIELDS`` (+ ``target`` if ``Status`` is present)."""
    df = pd.DataFrame(index=raw.index)
    df["created"] = pd.to_datetime(raw["Created"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["product_id"] = raw["Product_ID"].map(lambda v: "Unknown" if pd.isna(v) else str(int(v)))
    df["source"] = group_source(raw["Source"])
    df["sales_agent"] = raw["Sales_Agent"].fillna("Unknown").astype(str)
    df["location"] = group_location(raw["Location"])
    df["delivery_mode"] = raw["Delivery_Mode"].fillna("Unknown").astype(str)
    df["has_mobile"] = has_mobile(raw["Mobile"])
    df["email_type"] = email_type(raw["EMAIL"])
    if "Status" in raw:
        df["status"] = raw["Status"].astype(str).str.strip()
        df["target"] = make_target(raw["Status"])
    return df


# ----------------------------------------------------------------------------- step 2: tidy -> features
CATEGORICAL = ["product_id", "source", "sales_agent", "location", "delivery_mode", "email_type", "time_of_day",
               "day_of_week"]
NUMERIC = ["has_mobile", "is_weekend", "hour_sin", "hour_cos"]


def _time_of_day(h: pd.Series) -> pd.Series:
    bins = [-1, 5, 11, 16, 20, 23]
    labels = ["night", "morning", "afternoon", "evening", "late_evening"]
    return pd.cut(h, bins=bins, labels=labels).astype(str)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy lead rows -> feature frame. Safe on a single row; missing/odd values become 'Unknown'."""
    df = pd.DataFrame(df).copy()
    for col in LEAD_FIELDS:
        if col not in df:
            df[col] = np.nan
    created = pd.to_datetime(df["created"], errors="coerce", dayfirst=False)
    hour = created.dt.hour.fillna(12).astype(int)
    out = pd.DataFrame(index=df.index)
    for col in ["product_id", "source", "sales_agent", "location", "delivery_mode", "email_type"]:
        out[col] = df[col].astype("string").fillna("Unknown").replace({"": "Unknown", "nan": "Unknown"}).astype(str)
    # product ids may arrive as 18, 18.0 or "18"
    out["product_id"] = out["product_id"].str.replace(r"\.0$", "", regex=True)
    out["time_of_day"] = _time_of_day(hour)
    out["day_of_week"] = created.dt.day_name().fillna("Unknown").astype(str)
    out["is_weekend"] = created.dt.dayofweek.isin([5, 6]).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["has_mobile"] = pd.to_numeric(df["has_mobile"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    return out[CATEGORICAL + NUMERIC]

"""Streamlit UI for the FicZon Lead Quality API."""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

HERE = Path(__file__).resolve().parent
for p in (HERE / "src", HERE.parent / "src"):  # docker image / repo checkout
    if p.exists():
        sys.path.insert(0, str(p))
from ficzon.features import LEAD_FIELDS, clean_raw  # noqa: E402

DEFAULT_API = os.getenv("BACKEND_URL", "http://localhost:5000").rstrip("/")
RAW_COLS = {"Created", "Product_ID", "Source", "Mobile", "EMAIL", "Sales_Agent", "Location", "Delivery_Mode"}

st.set_page_config(page_title="FicZon Lead Scorer", page_icon="🎯", layout="wide")


@st.cache_data(ttl=60, show_spinner=False)
def api_get(base: str, path: str):
    r = requests.get(f"{base}{path}", timeout=60)  # free hosts cold-start slowly
    r.raise_for_status()
    return r.json()


def api_predict(base: str, payload):
    r = requests.post(f"{base}/api/v1/predict", json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(r.json().get("error", r.text))
    return r.json()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🎯 FicZon Lead Scorer")
    api = st.text_input("Backend API URL", DEFAULT_API).rstrip("/")
    meta = None
    try:
        health = api_get(api, "/health")
        meta = api_get(api, "/api/v1/metadata")
        st.success("API online")
        st.caption(f"Model: **{health['model']}**  \nTest ROC-AUC **{health['test_roc_auc']:.3f}** · "
                   f"threshold **{health['threshold']:.2f}**")
    except Exception as e:  # noqa: BLE001
        st.error("API unreachable (a free Render service can take ~1 min to wake up; refresh)")
        st.caption(str(e)[:200])
    st.divider()
    st.caption("Predicts whether a new sales lead is **High** or **Low potential**, using only what is known when "
               "the lead arrives. No names, phone numbers or e-mails are sent to the API.")

if meta is None:
    st.info("Waiting for the API…")
    st.stop()

# 'Unknown' product/location is a data artefact (almost only fresh 'Open' leads), so it is not offered here.
ch = {k: [v for v in vals if not (k in ("product_id", "location") and v == "Unknown")]
      for k, vals in meta["choices"].items()}
tab_one, tab_batch, tab_report = st.tabs(["🧑 Score a lead", "📄 Score a CSV", "📊 Model report"])


def show_result(res: dict):
    p = res["probability_high"]
    high = res["prediction"].startswith("High")
    icon = {"Hot": "🔥", "Warm": "🌤️", "Cold": "❄️"}[res["priority"]]
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", res["prediction"])
    c2.metric("P(High potential)", f"{p:.0%}")
    c3.metric("Priority", f"{icon} {res['priority']}")
    st.progress(min(max(p, 0.0), 1.0), text=f"score {p:.2f} (High if ≥ {meta['threshold']:.2f})")
    if high:
        st.success("Call this lead early: its profile matches leads that converted or stayed in the pipeline.")
    else:
        st.warning("Lower priority: similar leads were mostly junk, unreachable or just enquiring. "
                   "Consider a nurture e-mail first.")


# ---------------------------------------------------------------- single lead
with tab_one:
    st.subheader("Lead details")
    with st.form("lead"):
        c1, c2, c3 = st.columns(3)
        d = c1.date_input("Lead date", date(2018, 11, 12))
        t = c1.time_input("Lead time", time(11, 0))
        product = c1.selectbox("Product ID", ch["product_id"], index=ch["product_id"].index("18")
                               if "18" in ch["product_id"] else 0)
        source = c2.selectbox("Source (channel)", ch["source"], index=ch["source"].index("Call"))
        agent = c2.selectbox("Sales agent", ch["sales_agent"], index=ch["sales_agent"].index("Sales-Agent-4")
                             if "Sales-Agent-4" in ch["sales_agent"] else 0)
        location = c2.selectbox("Location", ch["location"])
        mode = c3.selectbox("Delivery mode", ch["delivery_mode"])
        email = c3.selectbox("E-mail provider", ch["email_type"],
                             help="Only the provider is used; 'missing' = no valid e-mail captured")
        mobile = c3.radio("Mobile number given?", ["Yes", "No"], horizontal=True)
        submitted = st.form_submit_button("Predict lead quality", type="primary")
    if submitted:
        lead = {"created": datetime.combine(d, t).strftime("%Y-%m-%d %H:%M"), "product_id": product,
                "source": source, "sales_agent": agent, "location": location, "delivery_mode": mode,
                "has_mobile": int(mobile == "Yes"), "email_type": email}
        try:
            with st.spinner("Scoring…"):
                show_result(api_predict(api, lead))
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------- batch CSV
with tab_batch:
    st.subheader("Score many leads at once")
    st.markdown(
        "Upload either\n"
        "- a **CRM export** with the original columns (`Created, Product_ID, Source, Mobile, EMAIL, Sales_Agent, "
        "Location, Delivery_Mode`). It is cleaned in this app, and mobile/e-mail are turned into flags *before* "
        "anything is sent, or\n"
        f"- a **tidy file** with columns `{', '.join(LEAD_FIELDS)}`.\n\n"
        "The file is processed in memory and not stored.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            if RAW_COLS <= set(df.columns):
                tidy = clean_raw(df)
                tidy["created"] = tidy["created"].dt.strftime("%Y-%m-%d %H:%M")
                kind = "CRM export"
            elif {"source", "delivery_mode"} <= set(df.columns):
                tidy = df.copy()
                kind = "tidy file"
            else:
                raise ValueError("Columns not recognised; see the two accepted formats above.")
            tidy = tidy[LEAD_FIELDS].astype(object).where(tidy[LEAD_FIELDS].notna(), None)
            req = ["product_id", "source", "location", "delivery_mode"]
            blank = tidy[req].isin([None, "", "Unknown"]).any(axis=1)
            if blank.any():
                st.warning(f"Skipping {int(blank.sum())} row(s) with no product / source / location / delivery mode.")
                tidy = tidy[~blank]
            if len(tidy) > 5000:
                st.warning("Scoring the first 5,000 rows only.")
                tidy = tidy.head(5000)
            results = []
            with st.spinner(f"Scoring {len(tidy)} leads ({kind})…"):
                recs = tidy.to_dict("records")
                for i in range(0, len(recs), 1000):
                    results += api_predict(api, {"leads": recs[i:i + 1000]})["results"]
            out = tidy.copy()
            out["probability_high"] = [r["probability_high"] for r in results]
            out["prediction"] = [r["prediction"] for r in results]
            out["priority"] = [r["priority"] for r in results]
            out = out.sort_values("probability_high", ascending=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Leads scored", f"{len(out):,}")
            c2.metric("High potential", f"{(out['prediction'] == 'High potential').mean():.0%}")
            c3.metric("Hot leads", f"{(out['priority'] == 'Hot').sum():,}")
            st.dataframe(out, use_container_width=True, height=380)
            st.download_button("Download scored leads (CSV)", out.to_csv(index=False).encode(), "scored_leads.csv",
                               "text/csv")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not score file: {e}")

# ---------------------------------------------------------------- report
with tab_report:
    st.subheader(meta["model_name"])
    tm, cvm, th = meta["test_metrics"], meta["cv_metrics"], meta["time_holdout_metrics"]
    c = st.columns(5)
    c[0].metric("Test accuracy", f"{tm['accuracy']:.3f}")
    c[1].metric("Test F1 (High)", f"{tm['f1']:.3f}")
    c[2].metric("Test ROC-AUC", f"{tm['roc_auc']:.3f}")
    c[3].metric("Precision / Recall", f"{tm['precision']:.2f} / {tm['recall']:.2f}")
    c[4].metric("Time hold-out AUC", f"{th['roc_auc']:.3f}", help="Trained on Apr–Sep 2018, tested on Oct–Nov 2018")
    st.caption(f"Trained on {meta['trained_on_leads']:,} leads · threshold {meta['threshold']:.2f} "
               f"(max out-of-fold macro-F1) · 5-fold CV ROC-AUC {cvm['roc_auc']:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Which lead fields matter most** (drop in ROC-AUC when shuffled)")
        imp = pd.DataFrame(list(meta["permutation_importance"].items()), columns=["field", "importance"])
        st.altair_chart(alt.Chart(imp).mark_bar(color="#457b9d").encode(
            x=alt.X("importance:Q", title="ROC-AUC drop"), y=alt.Y("field:N", sort="-x", title=None)),
            use_container_width=True)
    with right:
        st.markdown("**Confusion matrix (test set)**")
        cm = meta["test_confusion_matrix"]
        cmdf = pd.DataFrame([(a, p, cm["matrix"][i][j]) for i, a in enumerate(cm["labels"])
                             for j, p in enumerate(cm["labels"])], columns=["actual", "predicted", "leads"])
        base = alt.Chart(cmdf).encode(x=alt.X("predicted:N", sort=cm["labels"]),
                                      y=alt.Y("actual:N", sort=cm["labels"]))
        st.altair_chart((base.mark_rect().encode(color=alt.Color("leads:Q", scale=alt.Scale(scheme="blues"),
                                                                legend=None))
                        + base.mark_text(fontSize=18).encode(text="leads:Q")).properties(height=260),
                        use_container_width=True)
    g = meta["cumulative_gain_test"]
    st.markdown("**Prioritisation value:** working the top-scored leads first finds "
                + ", ".join(f"**{v:.0%}** of High leads in the top {k.split('_')[1].replace('pct', '%')}"
                            for k, v in g.items()) + ".")
    with st.expander("Target definition"):
        st.write("High potential:", ", ".join(meta["target"]["high_statuses"]))
        st.write("Low potential:", ", ".join(meta["target"]["low_statuses"]))

"""Gradio app: score one applicant, see PD, risk grade, decision and the reasons.  `uv run pd-app`

Educational demo on 1970s German consumer-credit data -- NOT suitable for real credit decisions.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR
from .data import CATEGORICAL, DATA_DICTIONARY, FEATURES, PROTECTED
from .decisions import GRADE_EDGES, GRADE_LABELS, decide, optimal_threshold, prior_shift
from .explain import attributions, plot_local, reason_codes
from .models import predict_pd

warnings.filterwarnings("ignore")
BUNDLE_PATH = MODEL_DIR / "pd_model.joblib"
NUMERIC_DEFAULTS = {"duration": 18, "amount": 2320, "age": 33, "installment_rate": 3, "present_residence": 3,
                    "number_credits": 1, "people_liable": 1}
VARIANTS = {"Full model (all 20 inputs)": "full",
            "Without protected attributes (age, personal status/sex, foreign worker)": "no_protected"}


def load_bundle(path=BUNDLE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train it first with:  uv run pd-run")
    return joblib.load(path)


def score_applicant(bundle: dict, applicant: dict, variant: str = "full", c_fa: float = 5.0, c_fr: float = 1.0,
                    portfolio_rate: float | None = None, refer_margin: float = 0.05) -> dict:
    """Pure function behind the UI (also unit-tested)."""
    row = pd.DataFrame([applicant], columns=FEATURES)
    for c, dtype in NUMERIC_DEFAULTS.items():
        row[c] = row[c].astype(float)
    pd_raw = float(predict_pd(bundle[f"model_{variant}"], row)[0])
    rate = bundle["sample_default_rate"] if portfolio_rate is None else portfolio_rate
    pd_hat = float(prior_shift(pd_raw, bundle["sample_default_rate"], rate))
    thr = optimal_threshold(c_fa, c_fr)
    attr, _, units = attributions(bundle[f"explain_model_{variant}"], row, bundle["background"])
    grade = str(pd.cut([pd_hat], GRADE_EDGES, labels=GRADE_LABELS, right=False)[0])
    warn = []
    for c, (lo, hi) in bundle["numeric_ranges"].items():
        if c in applicant and not lo <= float(applicant[c]) <= hi:
            warn.append(f"{c}={applicant[c]} is outside the training range [{lo:g}, {hi:g}]: extrapolation, treat PD with caution.")
    if variant == "no_protected":
        attr = attr.drop(columns=[c for c in PROTECTED if c in attr.columns])
    return {"pd_raw": pd_raw, "pd": pd_hat, "threshold": thr, "decision": str(decide([pd_hat], thr, refer_margin)[0]),
            "grade": grade, "attr": attr.iloc[0], "row": row.iloc[0], "units": units, "warnings": warn,
            "reasons": reason_codes(attr.iloc[0], row.iloc[0])}


def build_demo(bundle: dict):
    import gradio as gr

    cats = bundle["categories"]
    modal = bundle["background"][CATEGORICAL].mode().iloc[0]

    def run(variant, c_fa, c_fr, rate, margin, *vals):
        applicant = dict(zip(FEATURES, vals))
        r = score_applicant(bundle, applicant, VARIANTS[variant], c_fa, c_fr, rate, margin)
        shifted = abs(rate - bundle["sample_default_rate"]) > 1e-9
        head = (f"## PD = {r['pd']:.1%}  |  Grade {r['grade']}  |  Decision: **{r['decision'].upper()}**\n"
                f"Cut-off (PD at which expected costs balance) = {r['threshold']:.1%}"
                + (f"  \nPD re-based from the sample default rate ({bundle['sample_default_rate']:.0%}) to your portfolio rate ({rate:.1%})." if shifted else ""))
        if r["units"] == "probability":
            head += "\n\n_Attributions are in probability units for this model type._"
        why = "**Main risk drivers for this applicant**\n" + ("\n".join(f"- {s}" for s in r["reasons"]) or "- none: no feature raises risk above the average applicant")
        if r["warnings"]:
            why += "\n\n**Warnings**\n" + "\n".join(f"- {w}" for w in r["warnings"])
        fig = plot_local(r["attr"], r["row"], r["pd"])
        return head, why, fig

    inputs, comps = [], []
    with gr.Blocks(title="PD scorecard demo") as demo:
        gr.Markdown("# Probability-of-default demo\n**Educational only** - trained on 1,000 consumer loans from one German bank (1970s, DM). "
                    "Not valid for real lending. See the data card for limitations.")
        with gr.Row():
            variant = gr.Radio(list(VARIANTS), value=list(VARIANTS)[0], label="Model variant")
        with gr.Row():
            c_fa = gr.Number(5.0, label="Cost of approving a defaulter", minimum=0.01)
            c_fr = gr.Number(1.0, label="Cost of rejecting a good applicant", minimum=0.01)
            rate = gr.Slider(0.01, 0.5, value=round(bundle["sample_default_rate"], 3), step=0.005,
                             label="Assumed portfolio default rate (sample = default)")
            margin = gr.Slider(0.0, 0.15, value=0.05, step=0.01, label="Refer-to-analyst band around cut-off")
        with gr.Row():
            with gr.Column():
                for f, d in NUMERIC_DEFAULTS.items():
                    lo, hi = bundle["numeric_ranges"][f]
                    comps.append(gr.Number(d, label=f"{f} - {DATA_DICTIONARY[f]}", precision=0 if f != "amount" else 0))
            with gr.Column():
                for f in CATEGORICAL:
                    comps.append(gr.Dropdown(cats[f], value=modal[f], label=f"{f} - {DATA_DICTIONARY[f]}"))
        # order of FEATURES == NUMERIC + CATEGORICAL, matching the component order above
        btn = gr.Button("Score applicant", variant="primary")
        out_head, out_why, out_plot = gr.Markdown(), gr.Markdown(), gr.Plot()
        btn.click(run, [variant, c_fa, c_fr, rate, margin, *comps], [out_head, out_why, out_plot])
    return demo


def main() -> None:
    demo = build_demo(load_bundle())
    demo.launch(share=True)


if __name__ == "__main__":
    main()

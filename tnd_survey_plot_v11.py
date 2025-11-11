"""
tnd_survey_plot_v11.py

Reads model results (results/results_tnd_v11.csv) and overlays ActualTnD.csv (if present).
Produces:
 - results/hookload_vs_md_v11_overlay.png
 - results/torque_vs_md_v11_overlay.png

Flexible with column names. Uses matplotlib.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# Config / defaults
# -------------------------
OUTDIR = Path("results")
OUTDIR.mkdir(exist_ok=True)
MODEL_PATHS = [
    Path("results/results_tnd_v11.csv"),
    Path("results/results_tnd_v9_johancsik.csv"),
    Path("results/results_tnd_v10.csv"),
    Path("results/results_tnd.csv"),
]
ACTUAL_PATHS = [Path("ActualTnD.csv"), Path("input/ActualTnD.csv"), Path("results/ActualTnD.csv")]
BASE_BLOCK_KLBF = 37.0  # default to highlight in plots (will pick nearest if not present)

# -------------------------
# Helpers
# -------------------------
def find_model_file():
    for p in MODEL_PATHS:
        if p.exists():
            print(f"ℹ️ Found model CSV: {p}")
            return p
    # fallback: search results folder for any results_tnd_*.csv
    for p in Path("results").glob("results_tnd_*.csv"):
        print(f"ℹ️ Found model CSV (fallback): {p}")
        return p
    raise FileNotFoundError("No model CSV found in results/ (checked known names).")

def find_actual_file():
    for p in ACTUAL_PATHS:
        if p.exists():
            print(f"ℹ️ Found actual data CSV: {p}")
            return p
    return None

def detect_cols(df):
    # return dict mapping keys to column names (md, pickup, slack, rot, torque, ff, block)
    cols = {c.lower().strip(): c for c in df.columns}
    out = {}
    # MD
    md_candidates = [k for k in cols if any(x in k for x in ("md", "measureddepth", "measured_depth", "depth"))]
    out['md'] = cols[md_candidates[0]] if md_candidates else None
    # pickup/slack/rot
    pick = [k for k in cols if "pickup" in k]
    out['pickup'] = cols[pick[0]] if pick else None
    slack = [k for k in cols if "slack" in k]
    out['slack'] = cols[slack[0]] if slack else None
    rot = [k for k in cols if "rotat" in k]  # rotat -> rotating/rotor
    out['rot'] = cols[rot[0]] if rot else None
    # torque
    tq = [k for k in cols if "torque" in k]
    out['torque'] = cols[tq[0]] if tq else None
    # friction factor / block weight
    ff = [k for k in cols if "friction" in k or "ff" == k]
    out['ff'] = cols[ff[0]] if ff else None
    bw = [k for k in cols if "block" in k or "block_weight" in k or "block_weight_klbf" in k]
    out['block'] = cols[bw[0]] if bw else None
    return out

# -------------------------
# Main
# -------------------------
def main():
    try:
        model_file = find_model_file()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    model = pd.read_csv(model_file)
    # normalize column names for convenience (but keep originals)
    model_cols_map = detect_cols(model)

    md_col = model_cols_map.get('md')
    if md_col is None:
        print("❌ Could not detect MD column in model CSV.")
        print("Columns present:", list(model.columns))
        sys.exit(1)

    # Detect FF and Block columns (optional)
    ff_col = model_cols_map.get('ff')
    block_col = model_cols_map.get('block')

    # choose block weight to display (nearest to default)
    block_values = model[block_col].unique() if block_col else model["Block_Weight_klbf"].unique() if "Block_Weight_klbf" in model.columns else None
    chosen_block = BASE_BLOCK_KLBF
    if block_values is not None:
        try:
            # convert to numeric and pick nearest
            bvals = np.array([float(x) for x in block_values])
            idx = (np.abs(bvals - BASE_BLOCK_KLBF)).argmin()
            chosen_block = float(bvals[idx])
        except Exception:
            chosen_block = BASE_BLOCK_KLBF

    # pick friction factors present
    if ff_col:
        unique_ff = np.unique(model[ff_col].to_numpy())
    else:
        unique_ff = np.unique(model["Friction_Factor"].to_numpy()) if "Friction_Factor" in model.columns else np.unique(model["friction_factor"].to_numpy()) if "friction_factor" in model.columns else np.unique(model["FF"].to_numpy()) if "FF" in model.columns else np.array([])

    if len(unique_ff) == 0:
        # fallback - plot single set
        unique_ff = np.array([0.3])

    # detect pickup/slack/rot and torque cols in model
    pick_model_col = model_cols_map.get('pickup') or next((c for c in model.columns if "pickup" in c.lower()), None)
    slack_model_col = model_cols_map.get('slack') or next((c for c in model.columns if "slack" in c.lower()), None)
    rot_model_col = model_cols_map.get('rot') or next((c for c in model.columns if "rotat" in c.lower()), None)
    torque_model_col = model_cols_map.get('torque') or next((c for c in model.columns if "torque" in c.lower()), None)

    # --- Hookload overlay plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False
    for ff in unique_ff:
        try:
            subset = model[(np.isclose(model[ff_col] if ff_col else model.get("Friction_Factor", None), ff)) & (model[block_col] if block_col else model.get("Block_Weight_klbf", None) == chosen_block)]
        except Exception:
            # fallback to columns by name
            if "Friction_Factor" in model.columns:
                subset = model[(np.isclose(model["Friction_Factor"], ff)) & (model.get("Block_Weight_klbf", None) == chosen_block)]
            else:
                subset = model
        if subset.empty:
            continue
        # plot model lines
        if pick_model_col is not None:
            ax.plot(subset[pick_model_col], subset[md_col], '--', label=f"Model PU FF={ff}")
            plotted_any = True
        if slack_model_col is not None:
            ax.plot(subset[slack_model_col], subset[md_col], ':', label=f"Model SO FF={ff}")
            plotted_any = True
        if rot_model_col is not None:
            ax.plot(subset[rot_model_col], subset[md_col], '-', label=f"Model ROT FF={ff}")
            plotted_any = True

    # attempt to overlay actual
    actual_file = next((p for p in ACTUAL_PATHS if Path(p).exists()), None)
    if actual_file:
        actual = pd.read_csv(actual_file)
        # normalize actual col names
        actual.columns = [c.strip().replace(" ", "_").lower() for c in actual.columns]
        # detect MD in actual
        act_md_col = next((c for c in actual.columns if "md" in c or "depth" in c), None)
        if act_md_col is None:
            print("⚠️ Actual data exists but MD column not found. Skipping overlay markers for actual.")
        else:
            # detect pickup/slack/rot columns
            act_pick = next((c for c in actual.columns if "pickup" in c), None)
            act_slack = next((c for c in actual.columns if "slack" in c), None)
            act_rot = next((c for c in actual.columns if "rotat" in c), None)
            # Plot actual markers if found
            if act_pick:
                ax.scatter(actual[act_pick], actual[act_md_col], c='k', marker='o', label="Actual Pickup", zorder=10)
            if act_slack:
                ax.scatter(actual[act_slack], actual[act_md_col], c='k', marker='v', label="Actual Slackoff", zorder=10)
            if act_rot:
                ax.scatter(actual[act_rot], actual[act_md_col], c='k', marker='s', label="Actual Rotating", zorder=10)
            plotted_any = True

    if not plotted_any:
        print("⚠️ No model or actual hookload data found to plot.")
    ax.invert_yaxis()
    ax.set_xlabel("Hookload (klbf)")
    ax.set_ylabel("Measured Depth (ft)")
    ax.set_title("Hookload vs MD — Model vs Actual")
    ax.grid(True)
    ax.legend(fontsize="small")
    fig.tight_layout()
    out_hook = OUTDIR / "hookload_vs_md_v11_overlay.png"
    fig.savefig(out_hook, dpi=300)
    plt.close(fig)
    print(f"✅ Hookload overlay saved: {out_hook}")

    # --- Torque overlay plot ---
    if torque_model_col is None and (('torque' not in model.columns) and ('Torque_ftlb' not in model.columns)):
        print("⚠️ No torque column found in model results; skipping torque overlay.")
    else:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        plotted_torque = False
        for ff in unique_ff:
            try:
                subset = model[(np.isclose(model[ff_col] if ff_col else model.get("Friction_Factor", None), ff)) & (model[block_col] if block_col else model.get("Block_Weight_klbf", None) == chosen_block)]
            except Exception:
                if "Friction_Factor" in model.columns:
                    subset = model[(np.isclose(model["Friction_Factor"], ff)) & (model.get("Block_Weight_klbf", None) == chosen_block)]
                else:
                    subset = model
            if subset.empty:
                continue
            tq_col_name = torque_model_col or next((c for c in model.columns if "torque" in c.lower()), None)
            if tq_col_name:
                ax2.plot(subset[tq_col_name], subset[md_col], label=f"Model FF={ff}")
                plotted_torque = True

        # overlay actual torque if present
        if actual_file:
            actual = pd.read_csv(actual_file)
            actual.columns = [c.strip().replace(" ", "_").lower() for c in actual.columns]
            act_md_col = next((c for c in actual.columns if "md" in c or "depth" in c), None)
            act_tq_col = next((c for c in actual.columns if "torque" in c), None)
            if act_md_col and act_tq_col:
                ax2.scatter(actual[act_tq_col], actual[act_md_col], c='r', marker='x', label="Actual Torque", zorder=10)
                plotted_torque = True

        if not plotted_torque:
            print("⚠️ No torque data found (model or actual). Skipping torque plot save.")
        else:
            ax2.invert_yaxis()
            ax2.set_xlabel("Torque (ft-lbf)")
            ax2.set_ylabel("Measured Depth (ft)")
            ax2.set_title("Torque vs MD — Model vs Actual")
            ax2.grid(True)
            ax2.legend(fontsize="small")
            fig2.tight_layout()
            out_torque = OUTDIR / "torque_vs_md_v11_overlay.png"
            fig2.savefig(out_torque, dpi=300)
            plt.close(fig2)
            print(f"✅ Torque overlay saved: {out_torque}")

if __name__ == "__main__":
    main()

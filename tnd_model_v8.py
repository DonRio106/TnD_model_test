import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime

# ===============================
# 0. CONFIGURATION
# ===============================
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ===============================
# 1. LOAD INPUT DATA
# ===============================
try:
    bha = pd.read_csv("input/BHA_data.csv")
    survey = pd.read_csv("input/Survey_data.csv")
    borehole = pd.read_csv("input/Borehole_data.csv")
    tubular = pd.read_csv("input/Tubular_data.csv")
    config = pd.read_csv("input/Config.csv")
except Exception as e:
    raise FileNotFoundError(f"❌ Input file missing or unreadable: {e}")

# ===============================
# 2. CLEAN DATA & TYPE CASTING
# ===============================
for df in [bha, survey, borehole, tubular, config]:
    df.columns = df.columns.str.strip()

# Ensure numeric data
for col in ["YieldStrength_psi", "MakeUpTorque_ftlbf"]:
    if col in tubular.columns:
        tubular[col] = pd.to_numeric(tubular[col], errors="coerce")
    else:
        print(f"⚠️ Warning: Column {col} not found in Tubular_data.csv")

# ===============================
# 3. DEFINE SAFE YAML EXPORTERS
# ===============================
def numpy_representers():
    yaml.add_representer(np.float64, lambda dumper, data: dumper.represent_float(float(data)))
    yaml.add_representer(np.float32, lambda dumper, data: dumper.represent_float(float(data)))
    yaml.add_representer(np.int64, lambda dumper, data: dumper.represent_int(int(data)))
    yaml.add_representer(np.int32, lambda dumper, data: dumper.represent_int(int(data)))
    yaml.add_representer(np.ndarray, lambda dumper, data: dumper.represent_list(data.tolist()))
    yaml.add_representer(type(None), lambda dumper, data: dumper.represent_none(data))
numpy_representers()

def safe_convert(obj):
    if isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_convert(i) for i in obj]
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    else:
        return obj

# ===============================
# 4. CORE MODELING CALCULATIONS
# ===============================
print("🧮 Running T&D Modeling...")

# Block weight & mud weight from config
block_weight = float(config.loc[config["Parameter"] == "BlockWeight_lbf", "Value"].iloc[0])
mud_weight_ppg = float(config.loc[config["Parameter"] == "MudWeight_ppg", "Value"].iloc[0])

# Merge borehole info with survey data
survey = survey.merge(borehole, on="Section", how="left")

# Compute torque & drag values (simplified demo)
survey["SinInc"] = np.sin(np.radians(survey["Inclination_deg"]))
survey["CosInc"] = np.cos(np.radians(survey["Inclination_deg"]))

# Drag - static (sliding)
survey["Drag_lbf"] = (survey["WeightOnBit_lbf"] * (1 + survey["FrictionFactor"])) + block_weight
# Rotary mode - no friction effect
survey["RotaryWeight_lbf"] = survey["WeightOnBit_lbf"] + block_weight

# Torque model based on tubular strength
if "YieldStrength_psi" in tubular.columns:
    tubular["TorqueLimit_ftlbf"] = 0.00072 * tubular["YieldStrength_psi"] * tubular["OD_inch"]**3
else:
    tubular["TorqueLimit_ftlbf"] = np.nan

# Estimate torque vs MD
survey["Torque_ftlbf"] = survey["MD_ft"] * 0.5 * survey["SinInc"]

# ===============================
# 5. EXPORT RESULTS
# ===============================
out_csv = os.path.join(RESULT_DIR, f"TnD_results_v8_{timestamp}.csv")
survey.to_csv(out_csv, index=False)

# ===============================
# 6. PLOTTING SECTION
# ===============================
fig, ax = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
ax[0].plot(survey["Drag_lbf"], survey["MD_ft"], label="Drag (Sliding)", color="red")
ax[0].plot(survey["RotaryWeight_lbf"], survey["MD_ft"], label="Rotary Weight", color="green")
ax[0].invert_yaxis()
ax[0].set_xlabel("Force (lbf)")
ax[0].set_ylabel("Measured Depth (ft)")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(survey["Torque_ftlbf"], survey["MD_ft"], label="Torque", color="blue")
ax[1].set_xlabel("Torque (ft-lbf)")
ax[1].legend()
ax[1].grid(True)

ax[2].plot(survey["Inclination_deg"], survey["MD_ft"], label="Inclination", color="purple")
ax[2].set_xlabel("Inclination (°)")
ax[2].legend()
ax[2].grid(True)

plt.suptitle("T&D Model v8 — Weight, Torque, and Inclination vs MD")
plt.tight_layout()
plot_path = os.path.join(RESULT_DIR, f"TnD_plot_v8_{timestamp}.png")
plt.savefig(plot_path, dpi=300)
plt.close()

# ===============================
# 7. CONFIG SUMMARY EXPORT
# ===============================
summary = {
    "BlockWeight_lbf": block_weight,
    "MudWeight_ppg": mud_weight_ppg,
    "TorqueLimits": tubular[["Component", "TorqueLimit_ftlbf"]].to_dict(orient="records"),
    "Results": {"CSV": out_csv, "Plot": plot_path},
    "Notes": "Rotary weight excludes friction per client spec."
}

try:
    summary = safe_convert(summary)
    with open(os.path.join(RESULT_DIR, f"config_summary_v8_{timestamp}.yml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
except Exception as e:
    print(f"⚠️ Warning: Could not write YAML summary: {e}")

print(f"✅ Results saved in '{RESULT_DIR}' folder:")
print(f"   ├─ CSV : {out_csv}")
print(f"   ├─ PNG : {plot_path}")
print(f"   └─ YAML: config_summary_v8_{timestamp}.yml")

import pandas as pd
import numpy as np

# -------------------------------------------------
# Johancsik Torque & Drag Model (SPE 11380)
# -------------------------------------------------
def torque_drag_johancsik(survey_file, output_file="TnD_Result_v11.csv",
                          friction_pickup=0.25, friction_slackoff=0.20,
                          pipe_od_in=5.0, pipe_id_in=4.276, 
                          mud_weight_ppg=10.0, buoyancy_factor=0.85,
                          rotate_factor=0.5):
    # Load survey
    df = pd.read_csv(survey_file)
    df.columns = df.columns.str.strip().str.lower()

    # Expecting at least columns: md_ft, inclination_deg
    md = df["md_ft"].values
    inc = np.radians(df["inclination_deg"].values)

    # Pipe geometry
    do_ft = pipe_od_in / 12.0
    di_ft = pipe_id_in / 12.0
    area_pipe = np.pi * (do_ft**2 - di_ft**2) / 4
    steel_weight_lbft = 490 * area_pipe  # 490 lb/ft³ = steel density
    buoyed_weight_lbft = steel_weight_lbft * buoyancy_factor

    # Initialize results
    F_pickup = np.zeros_like(md)
    F_slackoff = np.zeros_like(md)
    F_rot = np.zeros_like(md)
    T_pickup = np.zeros_like(md)
    T_slackoff = np.zeros_like(md)
    T_rot = np.zeros_like(md)

    F_pickup[0] = 0
    F_slackoff[0] = 0
    F_rot[0] = 0

    # Friction factors
    f_pickup = friction_pickup
    f_slackoff = friction_slackoff
    f_rot = (f_pickup + f_slackoff) / 2 * rotate_factor

    # Loop through each segment
    for i in range(1, len(md)):
        ds = md[i] - md[i-1]
        sinθ = np.sin(inc[i])
        cosθ = np.cos(inc[i])

        # --- Pickup ---
        dF_pu = (buoyed_weight_lbft * sinθ + f_pickup * buoyed_weight_lbft * cosθ) * ds
        F_pickup[i] = F_pickup[i-1] + dF_pu
        T_pickup[i] = T_pickup[i-1] + f_pickup * (do_ft/2) * F_pickup[i] * sinθ * ds

        # --- Slackoff ---
        dF_so = -(buoyed_weight_lbft * sinθ + f_slackoff * buoyed_weight_lbft * cosθ) * ds
        F_slackoff[i] = F_slackoff[i-1] + dF_so
        T_slackoff[i] = T_slackoff[i-1] + f_slackoff * (do_ft/2) * F_slackoff[i] * sinθ * ds

        # --- Rotating ---
        dF_rot = (buoyed_weight_lbft * sinθ + f_rot * buoyed_weight_lbft * cosθ) * ds
        F_rot[i] = F_rot[i-1] + dF_rot
        T_rot[i] = T_rot[i-1] + f_rot * (do_ft/2) * F_rot[i] * sinθ * ds

    # Compile results
    results = pd.DataFrame({
        "MD_ft": md,
        "Inclination_deg": np.degrees(inc),
        "Pickup_lbs": F_pickup,
        "Slackoff_lbs": F_slackoff,
        "Rotating_lbs": F_rot,
        "Torque_Pickup_ftlb": T_pickup,
        "Torque_Slackoff_ftlb": T_slackoff,
        "Torque_Rotating_ftlb": T_rot
    })

    results.to_csv(output_file, index=False)
    print(f"✅ Johancsik T&D results saved to {output_file}")
    return results


if __name__ == "__main__":
    torque_drag_johancsik("SurveyInput.csv")

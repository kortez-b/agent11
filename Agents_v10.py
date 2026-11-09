import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Setup & Client Initialization
# ---------------------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# 1. High-Fidelity Day-by-Day Dynamic Dispatch Simulator (330 Active Days)
# ---------------------------------------------------------------------------
def simulate_daily_dispatch(doublets: int, storage_cap_mwh: float, abs_chiller_cap_mw: float,
                            df_heat: pd.DataFrame, df_cool: pd.DataFrame, active_indices: np.ndarray) -> dict:
    """
    Simulates sequential daily dispatch across 330 operational days.
    Tracks individual well flow rates, supply/reinjection temperatures,
    storage levels, and active/inactive doublet days.
    """
    rho = 1030.0          # kg/m3
    cp = 3.98             # kJ/(kg*K)
    max_flow_well = 300.0 # m3/h per doublet
    min_flow_well = 50.0  # m3/h per doublet
    max_delta_t = 40.0    # °C brine cooling limit
    t_res = 87.0          # °C reservoir production temperature
    t_district_target = 100.0 # °C required network supply temperature
    
    # Heat yield factor per m3/h at deltaT = 40 K
    k_heat = (rho / 3600.0) * cp * max_delta_t / 1000.0  # ~0.04555 MW per m3/h
    q_max_single_doublet = max_flow_well * k_heat        # ~13.66 MWth
    
    cop_hthp = 3.50
    cop_abs_th = 0.72    # Thermal COP for LiBr/H2O absorption chiller
    cop_abs_el = 25.0    # Parasitic electric COP
    cop_elec_chill = 4.0 # Auxiliary electric chiller COP
    elec_chiller_cap_mw = max(0.0, 10.0 - abs_chiller_cap_mw)
    
    storage_soc = 0.0
    total_geo_heat_mwh = 0.0
    total_aux_gas_mwh = 0.0
    total_hthp_elec_mwh = 0.0
    total_chiller_elec_mwh = 0.0
    
    well_annual_mwh = [0.0] * doublets
    well_active_days = [0] * doublets
    well_flow_rates_sum = [0.0] * doublets
    turndown_violations = 0
    daily_history = []
    
    for day_num, idx in enumerate(active_indices, 1):
        d_heat_mw = df_heat.loc[idx, "heating_demand_MWth"]
        d_cool_mw = df_cool.loc[idx, "cooling_demand_MWcool"]
        
        d_heat_mwh = d_heat_mw * 24.0
        
        # A. Cooling Dispatch
        cool_abs_mw = min(d_cool_mw, abs_chiller_cap_mw)
        cool_elec_mw = min(d_cool_mw - cool_abs_mw, elec_chiller_cap_mw)
        heat_for_abs_mwh = (cool_abs_mw * 24.0) / cop_abs_th
        elec_cool_mwh = ((cool_abs_mw * 24.0) / cop_abs_el) + ((cool_elec_mw * 24.0) / cop_elec_chill)
        total_chiller_elec_mwh += elec_cool_mwh
        
        # Total daily thermal demand on wellfield
        total_thermal_demand_mw = d_heat_mw + (cool_abs_mw / cop_abs_th)
        
        # B. Individual Doublet Staging
        needed_active = int(math.ceil(total_thermal_demand_mw / q_max_single_doublet)) if total_thermal_demand_mw > 0 else 1
        needed_active = max(1, min(needed_active, doublets))
        
        plant_geo_max_mw = needed_active * q_max_single_doublet
        actual_plant_heat_mw = min(total_thermal_demand_mw, plant_geo_max_mw)
        
        heat_per_active_doublet = actual_plant_heat_mw / needed_active
        flow_per_active_doublet = heat_per_active_doublet / k_heat
        
        # Enforce downhole stability constraint (50 m3/h)
        if flow_per_active_doublet < min_flow_well:
            flow_per_active_doublet = min_flow_well
            heat_per_active_doublet = min_flow_well * k_heat
            turndown_violations += 1
            
        daily_well_flows = [0.0] * doublets
        daily_well_heats = [0.0] * doublets
        
        for i in range(doublets):
            if i < needed_active:
                daily_well_flows[i] = round(flow_per_active_doublet, 1)
                daily_well_heats[i] = round(heat_per_active_doublet, 2)
                well_annual_mwh[i] += heat_per_active_doublet * 24.0
                well_active_days[i] += 1
                well_flow_rates_sum[i] += flow_per_active_doublet
                
        actual_delivered_geo_mwh = sum(daily_well_heats) * 24.0
        
        # C. Storage Dispatch
        if actual_delivered_geo_mwh >= (d_heat_mwh + heat_for_abs_mwh):
            surplus = actual_delivered_geo_mwh - (d_heat_mwh + heat_for_abs_mwh)
            charge = min(surplus, storage_cap_mwh - storage_soc) if storage_cap_mwh > 0 else 0.0
            storage_soc += charge
            geo_used_for_heating_mwh = d_heat_mwh
            aux_gas_day = 0.0
        else:
            deficit = (d_heat_mwh + heat_for_abs_mwh) - actual_delivered_geo_mwh
            discharge = min(deficit, storage_soc) if (storage_cap_mwh > 0 and storage_soc > 0) else 0.0
            storage_soc -= discharge
            geo_used_for_heating_mwh = (actual_delivered_geo_mwh - heat_for_abs_mwh) + discharge
            aux_gas_day = deficit - discharge
            
        total_geo_heat_mwh += max(0.0, geo_used_for_heating_mwh)
        total_aux_gas_mwh += aux_gas_day
        
        # D. HTHP Booster Stage
        heat_boosted = max(0.0, geo_used_for_heating_mwh) * 0.40
        total_hthp_elec_mwh += (heat_boosted / cop_hthp)
        
        # Brine reinjection temperature
        brine_delta_t = min(max_delta_t, (sum(daily_well_heats) * 1000.0) / (max(1.0, sum(daily_well_flows)) * (rho/3600.0) * cp))
        t_reinj = t_res - brine_delta_t
        
        daily_history.append({
            "day": day_num,
            "heat_demand_mw": d_heat_mw,
            "cool_demand_mw": d_cool_mw,
            "geo_delivered_mw": sum(daily_well_heats),
            "storage_soc_mwh": storage_soc,
            "aux_gas_mw": aux_gas_day / 24.0,
            "active_doublets": needed_active,
            "t_reinj": t_reinj
        })
        
    total_esp_elec_mwh = sum(well_active_days) * 24.0 * 0.15
    total_delivered_heat_gwh = total_geo_heat_mwh / 1000.0
    total_aux_gwh = total_aux_gas_mwh / 1000.0
    total_elec_gwh = (total_hthp_elec_mwh + total_chiller_elec_mwh + total_esp_elec_mwh) / 1000.0
    coverage_pct = (total_delivered_heat_gwh / (total_delivered_heat_gwh + total_aux_gwh)) * 100.0
    mean_flows = [round(well_flow_rates_sum[i] / max(1, well_active_days[i]), 1) for i in range(doublets)]
    
    return {
        "delivered_heat_gwh": round(total_delivered_heat_gwh, 3),
        "aux_gas_gwh": round(total_aux_gwh, 3),
        "coverage_pct": round(coverage_pct, 2),
        "total_electricity_gwh": round(total_elec_gwh, 3),
        "turndown_violations": turndown_violations,
        "well_production_gwh": [round(w / 1000.0, 2) for w in well_annual_mwh],
        "well_active_days": well_active_days,
        "well_inactive_days": [330 - d for d in well_active_days],
        "well_mean_flow_m3h": mean_flows,
        "daily_history": daily_history
    }

# ---------------------------------------------------------------------------
# 2. Iteration Sweeper & Techno-Economic Computation
# ---------------------------------------------------------------------------
def run_interactive_simulation(demand_heat_csv="GEMINI_Heating_Demand_Daily_2027.csv",
                               demand_cool_csv="GEMINI_Cooling_Demand_Daily_2027.csv"):
    df_heat = pd.read_csv(demand_heat_csv)
    df_cool = pd.read_csv(demand_cool_csv)
    
    # 35 lowest-load days scheduled for annual turnaround (330 active operational days)
    total_load = df_heat["heating_demand_MWth"] + df_cool["cooling_demand_MWcool"]
    turnaround_days = total_load.nsmallest(35).index
    active_indices = df_heat.index[~df_heat.index.isin(turnaround_days)].values
    
    annual_cool_target = (df_cool.loc[active_indices, "cooling_demand_MWcool"] * 24.0).sum() / 1000.0
    annual_heat_target = (df_heat.loc[active_indices, "heating_demand_MWth"] * 24.0).sum() / 1000.0
    total_energy_annual = annual_heat_target + annual_cool_target
    
    doublet_candidates = [1, 2, 3]
    storage_candidates = [0, 25, 50, 75, 100, 125, 150, 175, 200]
    cooling_configs = [0.0, 5.0, 7.5, 10.0]
    
    results = []
    dispatch_cache = {}
    
    print("\n" + "="*80)
    print("--- EXECUTING PHYSICAL DISPATCH SWEEPER (330 OPERATIONAL DAYS) ---")
    print("="*80)
    
    for d in doublet_candidates:
        hthp_mw = 8.0 if d <= 2 else 6.0
        wells_capex = d * 12.0
        esp_capex = d * 0.60
        
        for s in storage_candidates:
            for abs_mw in cooling_configs:
                elec_mw = 10.0 - abs_mw
                sim = simulate_daily_dispatch(d, s, abs_mw, df_heat, df_cool, active_indices)
                
                # CAPEX Model
                surface_capex = (hthp_mw * 0.80) + (s * 0.04) + (abs_mw * 0.45) + (elec_mw * 0.15)
                bop_capex = surface_capex * 0.15
                total_capex = wells_capex + esp_capex + surface_capex + bop_capex
                
                # Modular Lifecycle OPEX
                opex_wells = wells_capex * 0.0075
                opex_active = (hthp_mw * 0.80 + abs_mw * 0.45 + elec_mw * 0.15) * 0.04
                opex_static = (s * 0.04 + bop_capex) * 0.015
                esp_fund = d * 0.180
                opex_fixed = opex_wells + opex_active + opex_static + esp_fund
                
                opex_var = (sim["total_electricity_gwh"] * 1000.0 * 150.0 / 1e6) + (sim["aux_gas_gwh"] * 1000.0 * 60.0 / 1e6)
                total_opex = opex_fixed + opex_var
                
                # Levelized Cost of Energy (30-year lifetime, 6% WACC -> CRF = 0.07265)
                crf = 0.06 / (1.0 - (1.0 + 0.06)**(-30))
                annualized_cost = (total_capex * crf) + total_opex
                lcoe = (annualized_cost * 1e6) / (total_energy_annual * 1000.0)
                
                cfg_id = f"D{d}_S{s}_A{int(abs_mw)}"
                dispatch_cache[cfg_id] = sim
                
                turndown_risk = "HIGH" if sim["turndown_violations"] > 5 else "LOW"
                
                results.append({
                    "cfg_id": cfg_id,
                    "doublets": d,
                    "storage_mwh": s,
                    "hthp_mw": hthp_mw,
                    "abs_mw": abs_mw,
                    "elec_mw": elec_mw,
                    "coverage_pct": sim["coverage_pct"],
                    "aux_gas_gwh": sim["aux_gas_gwh"],
                    "total_capex_m": round(total_capex, 2),
                    "total_opex_m": round(total_opex, 3),
                    "total_elec_gwh": round(sim["total_electricity_gwh"], 2),
                    "lcoe_eur_mwh": round(lcoe, 2),
                    "well_production_gwh": sim["well_production_gwh"],
                    "well_active_days": sim["well_active_days"],
                    "well_inactive_days": sim["well_inactive_days"],
                    "well_mean_flow_m3h": sim["well_mean_flow_m3h"],
                    "turndown_violations": sim["turndown_violations"],
                    "turndown_risk": turndown_risk
                })
                
    df_results = pd.DataFrame(results)
    return df_results, dispatch_cache

# ---------------------------------------------------------------------------
# 3. Autonomous Optimization Agent (Engineering-First: Peak Freeze Security)
# ---------------------------------------------------------------------------
def autonomous_agent_decision(df_results: pd.DataFrame) -> dict:
    viable = df_results[df_results["coverage_pct"] >= 99.5].copy()
    dataset = viable.to_dict(orient="records")
    
    prompt = f"""
You are the Lead Systems Reliability & Reservoir Optimization Engineer for the SPE GeoHackathon.
ENGINEERING PHILOSOPHY:
"Engineering and security of supply come first. An infrastructure investment fails if it cannot deliver full heating duty during the extreme winter peak freeze."

Simulated Candidate Architectures (330 active operational days):
{json.dumps(dataset, indent=2)}

MANDATORY DECISION RULES:
1. PEAK FREEZE SECURITY MANDATE:
   - Sizing thermal storage at 50 MWh with 2 doublets leaves a 134 MWh deficit during the 5-day January freeze (emptying on Day 2).
   - If recommending 2 doublets with 100% coverage, storage must be >= 175 MWh (e.g. D2_S175_A7 or D2_S175_A10).
   - If recommending 3 doublets, evaluate the benefit of direct subsurface redundancy (N-1 safety margin) without large storage vessels.
2. Provide a structured decision selecting the winning design and defending the engineering rationale.

Output strictly valid JSON with exact fields matching the simulation table:
{{
    "selected_configuration": {{
        "cfg_id": "<cfg_id>",
        "doublets": <int>,
        "storage_mwh": <float>,
        "hthp_mw": <float>,
        "abs_chiller_mw": <float>,
        "elec_chiller_mw": <float>,
        "lcoe_eur_mwh": <float>,
        "total_capex_m": <float>,
        "coverage_pct": <float>,
        "aux_gas_gwh": <float>,
        "well_production_gwh": <list of floats>,
        "well_active_days": <list of ints>,
        "well_inactive_days": <list of ints>,
        "well_mean_flow_m3h": <list of floats>
    }},
    "optimization_score": <float between 0 and 100>,
    "winter_peak_resilience_analysis": "<analysis of peak freeze and storage dynamics>",
    "autonomous_justification": "<why this configuration wins>"
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior geothermal infrastructure engineer who prioritizes peak supply security and physical rigor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    clean_json = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    return json.loads(clean_json)

# ---------------------------------------------------------------------------
# 4. Terminal ASCII Graphical Dashboard (Zero Freezing / Zero Blocking)
# ---------------------------------------------------------------------------
def print_ascii_dashboard(winning_cfg: dict, df_results: pd.DataFrame):
    print("\n" + "="*85)
    print("--- TERMINAL GRAPHICAL DASHBOARD: WELL ENERGY & PARETO COMPARISON ---")
    print("="*85)
    
    # 1. Bar Chart: Production per Doublet
    print("\n[CHART 1] Annual Production Contribution per Doublet (GWhth):")
    prods = winning_cfg.get("well_production_gwh", [])
    max_prod = max(prods) if prods and max(prods) > 0 else 1.0
    for idx, p in enumerate(prods):
        bar_len = int((p / max_prod) * 45)
        bar_str = "█" * bar_len
        print(f" Doublet {idx+1} [{p:6.2f} GWh] | {bar_str}")
        
    # 2. Bar Chart: Active Operational Days out of 330
    print("\n[CHART 2] Operational Utilization (Active Days out of 330):")
    days = winning_cfg.get("well_active_days", [])
    for idx, d in enumerate(days):
        bar_len = int((d / 330.0) * 45)
        bar_str = "▓" * bar_len
        print(f" Doublet {idx+1} [{d:3d} Days]  | {bar_str}")
        
    # 3. Pareto Chart: LCOE vs TES Capacity
    print("\n[CHART 3] Pareto Comparison: LCOE (€/MWh) vs TES Capacity (>=99.8% Coverage):")
    sub = df_results[df_results["coverage_pct"] >= 99.8].sort_values(by="lcoe_eur_mwh")
    for _, row in sub.head(10).iterrows():
        cid = row["cfg_id"]
        l = row["lcoe_eur_mwh"]
        c = row["coverage_pct"]
        offset = max(0, int((l - 46.0) * 6))
        plot_line = " " * offset + "◆"
        selected_flag = "  <-- SELECTED BEST" if cid == winning_cfg.get("cfg_id") else ""
        print(f" {cid:11s} | LCOE: €{l:5.2f} | Cov: {c:5.2f}% |{plot_line}{selected_flag}")
    print("="*85)

# ---------------------------------------------------------------------------
# 5. Main Execution Pipeline (Safe Indexing)
# ---------------------------------------------------------------------------
def main():
    df_results, dispatch_cache = run_interactive_simulation()
    
    # Format and print the complete iteration matrix directly to console
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    
    print("\n" + "="*125)
    print("--- ITERATION MATRIX: VIABLE CONFIGURATIONS (COVERAGE >= 99.5%) ---")
    print("="*125)
    display_cols = ["cfg_id", "doublets", "storage_mwh", "abs_mw", "coverage_pct", "aux_gas_gwh", "total_capex_m", "total_opex_m", "total_elec_gwh", "lcoe_eur_mwh", "turndown_risk"]
    viable_df = df_results[df_results["coverage_pct"] >= 99.5][display_cols].sort_values(by="lcoe_eur_mwh")
    print(viable_df.to_string(index=False))
    print("="*125)
    
    print("\n[Agent] Invoking Autonomous Decision Agent on Simulation Matrix...")
    decision = autonomous_agent_decision(df_results)
    winning_cfg = decision["selected_configuration"]
    winning_id = winning_cfg["cfg_id"]
    
    print("\n" + "="*85)
    print(f"--- AUTONOMOUS AGENT DECISION: SELECTED {winning_id} ---")
    print("="*85)
    print(json.dumps(decision, indent=2))
    
    # Detailed Well Dispatch Breakdown Table with Safe Indexing
    print("\n" + "="*85)
    print("--- WELL OPERATIONAL PROFILE & HYDRAULIC DISPATCH BREAKDOWN ---")
    print("="*85)
    num_doublets = int(winning_cfg.get("doublets", len(winning_cfg.get("well_production_gwh", []))))
    well_prods = winning_cfg.get("well_production_gwh", [0.0]*num_doublets)
    well_days = winning_cfg.get("well_active_days", [0]*num_doublets)
    well_inactive = winning_cfg.get("well_inactive_days", [0]*num_doublets)
    well_flows = winning_cfg.get("well_mean_flow_m3h", [0.0]*num_doublets)
    
    well_summary = []
    for i in range(num_doublets):
        well_summary.append({
            "Doublet ID": f"Doublet {i+1}",
            "Annual Production (GWh)": well_prods[i] if i < len(well_prods) else 0.0,
            "Active Days": well_days[i] if i < len(well_days) else 0,
            "Inactive Days (Shut-in)": well_inactive[i] if i < len(well_inactive) else 0,
            "Mean Operational Flow (m3/h)": well_flows[i] if i < len(well_flows) else 0.0
        })
    print(pd.DataFrame(well_summary).to_string(index=False))
    print("="*85)
    
    # Display Dashboard directly in console
    print_ascii_dashboard(winning_cfg, df_results)
    print("\n✅ Simulation completed successfully.")

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# 4. On-Screen Visualizations & Interactive Plotting
# ---------------------------------------------------------------------------
def plot_results_on_screen(df_results: pd.DataFrame, winning_cfg_id: str, dispatch_cache: dict):
    best_sim = dispatch_cache[winning_cfg_id]
    history_df = pd.DataFrame(best_sim["daily_history"])
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"SPE GeoHackathon Autonomous Dispatch Architecture: {winning_cfg_id}", fontsize=16, fontweight="bold")
    
    # Subplot 1: Annual Daily Thermal Load & Geothermal Dispatch
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history_df["day"], history_df["heat_demand_mw"], label="District Heating Demand", color="#d95f02", lw=1.5)
    ax1.plot(history_df["day"], history_df["geo_delivered_mw"], label="Geothermal Heat Delivered", color="#1b9e77", lw=1.5, ls="--")
    ax1.fill_between(history_df["day"], 0, history_df["aux_gas_mw"], label="Auxiliary Backup (Gas)", color="#e7298a", alpha=0.5)
    ax1.set_title("330-Day Sequential Thermal Dispatch Profile", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Operational Day")
    ax1.set_ylabel("Thermal Power (MWth)")
    ax1.legend(loc="upper right")
    
    # Subplot 2: Stratified Storage State of Charge (TES SOC)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(history_df["day"], history_df["storage_soc_mwh"], label="TES State of Charge", color="#7570b3", lw=1.8)
    ax2.set_title("Thermal Energy Storage (TES) State of Charge", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Operational Day")
    ax2.set_ylabel("Stored Energy (MWh)")
    ax2.legend(loc="upper right")
    
    # Subplot 3: Individual Well Energy Contribution
    ax3 = fig.add_subplot(2, 2, 3)
    wells_prod = best_sim["well_production_gwh"]
    well_labels = [f"Doublet {i+1}" for i in range(len(wells_prod))]
    colors = ["#2b83ba", "#abdda4", "#fdae61"][:len(wells_prod)]
    bars = ax3.bar(well_labels, wells_prod, color=colors, edgecolor="black", width=0.5)
    ax3.set_title("Annual Energy Contribution per Doublet", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Thermal Output (GWhth/year)")
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.2f} GWh", ha="center", va="bottom", fontweight="bold")
        
    # Subplot 4: Pareto Frontier (CAPEX vs LCOE across viable options)
    ax4 = fig.add_subplot(2, 2, 4)
    viable = df_results[df_results["coverage_pct"] >= 99.5]
    scatter = ax4.scatter(viable["total_capex_m"], viable["lcoe_eur_mwh"], c=viable["storage_mwh"], cmap="viridis", s=70, alpha=0.8, edgecolor="k")
    cbar = fig.colorbar(scatter, ax=ax4)
    cbar.set_label("TES Capacity (MWh)")
    
    best_row = df_results[df_results["cfg_id"] == winning_cfg_id].iloc[0]
    ax4.scatter(best_row["total_capex_m"], best_row["lcoe_eur_mwh"], color="red", s=220, marker="*", edgecolor="black", label=f"Selected Design: {winning_cfg_id}")
    ax4.set_title("Design Exploration: CAPEX vs LCOE (Pareto Frontier)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Total Capital Expenditure (M€)")
    ax4.set_ylabel("LCOE (€/MWh)")
    ax4.legend(loc="upper left")
    
    plt.tight_layout()
    print("\n[Graphics] Displaying Interactive Engineering Dashboard on Screen...")
    plt.show()

# ---------------------------------------------------------------------------
# 5. Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    df_results, dispatch_cache = run_interactive_simulation()
    
    # Format and print the complete iteration table to console
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    
    print("\n" + "="*120)
    print("--- ITERATION MATRIX: VIABLE CONFIGURATIONS (COVERAGE >= 99.5%) ---")
    print("="*120)
    display_cols = ["cfg_id", "doublets", "storage_mwh", "abs_mw", "coverage_pct", "aux_gas_gwh", "total_capex_m", "total_opex_m", "total_elec_gwh", "lcoe_eur_mwh", "turndown_risk"]
    viable_df = df_results[df_results["coverage_pct"] >= 99.5][display_cols].sort_values(by="lcoe_eur_mwh")
    print(viable_df.to_string(index=False))
    print("="*120)
    
    print("\n[Agent] Invoking Autonomous Decision Agent on Simulation Matrix...")
    decision = autonomous_agent_decision(df_results)
    
    winning_cfg = decision["selected_configuration"]
    winning_id = winning_cfg["cfg_id"]
    
    print("\n" + "="*80)
    print(f"--- AUTONOMOUS AGENT DECISION: SELECTED {winning_id} ---")
    print("="*80)
    print(json.dumps(decision, indent=2))
    
    # Detailed Well Dispatch Breakdown Table
    print("\n" + "="*80)
    print("--- WELL OPERATIONAL PROFILE & HYDRAULIC DISPATCH BREAKDOWN ---")
    print("="*80)
    well_summary = []
    for i in range(winning_cfg["doublets"]):
        well_summary.append({
            "Doublet ID": f"Doublet {i+1}",
            "Annual Production (GWh)": winning_cfg["well_production_gwh"][i],
            "Active Days": winning_cfg["well_active_days"][i],
            "Inactive Days (Shut-in)": winning_cfg["well_inactive_days"][i],
            "Mean Operational Flow (m3/h)": winning_cfg["well_mean_flow_m3h"][i]
        })
    print(pd.DataFrame(well_summary).to_string(index=False))
    print("="*80)
    
    # Plot all results and dispatch curves directly to screen
    plot_results_on_screen(df_results, winning_id, dispatch_cache)

if __name__ == "__main__":
    main()

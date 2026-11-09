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
    max_delta_t = 40.0    # degC brine cooling limit
    t_res = 87.0          # degC reservoir production temperature
    
    # Heat yield factor per m3/h at deltaT = 40 K (~0.04555 MW per m3/h)
    k_heat = (rho / 3600.0) * cp * max_delta_t / 1000.0
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
        
        # Total daily thermal demand placed on wellfield
        total_thermal_demand_mw = d_heat_mw + (cool_abs_mw / cop_abs_th)
        
        # B. Individual Doublet Staging
        needed_active = int(math.ceil(total_thermal_demand_mw / q_max_single_doublet)) if total_thermal_demand_mw > 0 else 1
        needed_active = max(1, min(needed_active, doublets))
        
        plant_geo_max_mw = needed_active * q_max_single_doublet
        actual_plant_heat_mw = min(total_thermal_demand_mw, plant_geo_max_mw)
        
        heat_per_active_doublet = actual_plant_heat_mw / needed_active
        flow_per_active_doublet = heat_per_active_doublet / k_heat
        
        # Enforce downhole stability constraint (50 m3/h per active doublet)
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
        
        # C. Thermal Energy Storage (TES) Dispatch
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
        
        # D. High-Temperature Heat Pump (HTHP) Series Booster Stage (40% lift from 84 to 100 degC)
        heat_boosted = max(0.0, geo_used_for_heating_mwh) * 0.40
        total_hthp_elec_mwh += (heat_boosted / cop_hthp)
        
        # Brine reinjection temperature
        total_flow = max(1.0, sum(daily_well_flows))
        brine_delta_t = min(max_delta_t, (sum(daily_well_heats) * 1000.0) / (total_flow * (rho / 3600.0) * cp))
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
    coverage_pct = (total_delivered_heat_gwh / (total_delivered_heat_gwh + total_aux_gwh)) * 100.0 if (total_delivered_heat_gwh + total_aux_gwh) > 0 else 0.0
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
                               demand_cool_csv="GEMINI_Cooling_Demand_Daily_2027.csv") -> tuple:
    df_heat = pd.read_csv(demand_heat_csv)
    df_cool = pd.read_csv(demand_cool_csv)
    
    # 35 lowest-load days scheduled for annual turnaround maintenance (330 active operational days)
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
# 3. Autonomous Optimization Agent (Multi-Criteria Sections A through H)
# ---------------------------------------------------------------------------
def autonomous_agent_decision(df_results: pd.DataFrame) -> tuple:
    viable = df_results[df_results["coverage_pct"] >= 99.5].copy()
    candidate_json = viable.to_json(orient="records", indent=2)

    prompt = f"""You are the final autonomous geothermal energy-system design agent.
The deterministic engineering and economic models evaluated the following configurations across 330 operational days:
{candidate_json}

Select ONE final recommended configuration and ONE best alternative.

DECISION REQUIREMENTS:
1. The selected design must meet annual heating and cooling demand using geothermal systems plus the modeled backup systems.
2. Consider all of the following:
   - geothermal heating contribution
   - auxiliary heating
   - electric versus hybrid cooling
   - heat-pump electricity
   - chiller electricity
   - ESP electricity
   - injection-pump electricity
   - total_electricity_GWh
   - surface-equipment capacities
   - number of wells
   - storage capacity and utilization
   - total CAPEX
   - annual OPEX
   - lifecycle-cost NPV
   - equivalent annual cost
   - levelized cost
3. Do not select the lowest-cost configuration automatically if it leaves excessive dependence on auxiliary heating.
4. Explicitly compare:
   - 1 versus 2 versus 3 doublets
   - electric-only versus hybrid cooling
   - no storage versus available storage options
5. Recognize diminishing returns from additional doublets and storage.
6. Treat this as a multi-criteria engineering recommendation. Balance cost, geothermal penetration, electricity consumption, backup-heating dependence and equipment complexity.

IMPORTANT RULES:
- Use only the calculated values supplied above. Do not invent numbers.
- Do not claim global optimality. Call the selected design the most promising configuration under the current technical and economic screening assumptions.
- OUTPUT THE FINAL REPORT EXACTLY IN THE FOLLOWING TEXT STRUCTURE AND FORMAT (A through H):

---

### A. FINAL RECOMMENDED CONFIGURATION
The selected design is the **most promising configuration under the current technical and economic screening assumptions (<cfg_id>)**:
- **Number of Doublets:** <doublets> doublets (**<total_wells> total wells**)
- **Cooling Technology:** Hybrid cooling (<abs_mw> MW absorption chiller + <elec_mw> MW electric backup chiller)
- **Storage Capacity:** <storage_mwh> MWh thermal energy storage (TES)

---

### B. ENGINEERING PERFORMANCE
- **Geothermal + Heat-Pump Heating Supply:** <delivered_heat_gwh> GWh/year (<coverage_pct>% coverage)
- **Auxiliary Heating Supply:** <aux_gas_gwh> GWh/year
- **Absorption Cooling Production:** <abs_cooling_gwh> GWh/year
- **Electric Cooling Production:** <elec_cooling_gwh> GWh/year
- **Total Electricity Consumption:** <total_elec_gwh> GWh/year
- **Geothermal Pumping Electricity (ESP + Injection):** <pumping_elec_gwh> GWh/year (ESP downhole lifting + injection loop)

---

### C. SURFACE FACILITIES
- **Geothermal Source Thermal Capacity:** <geothermal_mw> MWth (<doublets> doublets @ 300 m³/h, deltaT = 40 °C)
- **Heat-Exchanger Capacity:** <hx_capacity_mw> MWth
- **Heat-Pump Thermal Capacity:** <hthp_mw> MWth
- **Heat-Pump Peak Electricity Consumption:** <hp_peak_elec_mw> MW
- **Auxiliary-Heater Thermal Capacity:** <aux_heater_mw> MWth
- **Absorption-Chiller Cooling Capacity:** <abs_mw> MWcool
- **Electric Backup-Chiller Cooling Capacity:** <elec_mw> MWcool
- **Thermal Storage Charge Capacity:** <tes_charge_mw> MW
- **Thermal Storage Discharge Capacity:** <tes_discharge_mw> MW

---

### D. ECONOMICS
- **Total CAPEX:** <total_capex_m> MEUR
- **Annual OPEX:** <total_opex_m> MEUR/year
- **Lifecycle Cost (NPV):** <npv_cost_m> MEUR
- **Equivalent Annual Cost (EAC):** <eac_cost_m> MEUR/year
- **Levelized Cost of Energy (LCOE):** <lcoe_eur_mwh> EUR/MWh

---

### E. BEST ALTERNATIVE
[Explain when and why the alternative configuration would be selected]

---

### F. CLEAR TRADE-OFF
[Explain the trade-offs among lowest lifecycle cost, lower auxiliary-heating dependence, total electricity, and additional wells/equipment]

---

### G. STORAGE ASSESSMENT
[State whether 175 or 185 MWh storage is justified by the calculated economic and engineering results]

---

### H. LIMITATIONS
[Explicitly state screening assumptions, cost correlations, energy price assumptions, simplified COPs, no thermal breakthrough, not a FEED design]
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior energy systems advisor conducting a balanced, rigorous multi-criteria screening evaluation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    report_text = response.choices[0].message.content
    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
    
    return report_text, token_usage

# ---------------------------------------------------------------------------
# 4. Terminal Visualizations & Plotting
# ---------------------------------------------------------------------------
def print_ascii_dashboard(winning_cfg_id: str, df_results: pd.DataFrame, dispatch_cache: dict):
    print("\n" + "="*85)
    print("--- TERMINAL GRAPHICAL DASHBOARD: WELL ENERGY & PARETO COMPARISON ---")
    print("="*85)
    
    best_sim = dispatch_cache[winning_cfg_id]
    
    print("\n[CHART 1] Annual Production Contribution per Doublet (GWhth):")
    prods = best_sim.get("well_production_gwh", [])
    max_prod = max(prods) if prods and max(prods) > 0 else 1.0
    for idx, p in enumerate(prods):
        bar_len = int((p / max_prod) * 45)
        bar_str = "█" * bar_len
        print(f" Doublet {idx+1} [{p:6.2f} GWh] | {bar_str}")
        
    print("\n[CHART 2] Operational Utilization (Active Days out of 330):")
    days = best_sim.get("well_active_days", [])
    for idx, d in enumerate(days):
        bar_len = int((d / 330.0) * 45)
        bar_str = "▓" * bar_len
        print(f" Doublet {idx+1} [{d:3d} Days]  | {bar_str}")
        
    print("\n[CHART 3] Pareto Comparison: LCOE (€/MWh) vs TES Capacity (>=99.8% Coverage):")
    sub = df_results[df_results["coverage_pct"] >= 99.8].sort_values(by="lcoe_eur_mwh")
    for _, row in sub.head(10).iterrows():
        cid = row["cfg_id"]
        l = row["lcoe_eur_mwh"]
        c = row["coverage_pct"]
        offset = max(0, int((l - 46.0) * 6))
        plot_line = " " * offset + "◆"
        selected_flag = "  <-- SELECTED WINNER" if cid == winning_cfg_id else ""
        print(f" {cid:11s} | LCOE: €{l:5.2f} | Cov: {c:5.2f}% |{plot_line}{selected_flag}")
    print("="*85)

def plot_results_on_screen(df_results: pd.DataFrame, winning_cfg_id: str, dispatch_cache: dict):
    best_sim = dispatch_cache[winning_cfg_id]
    history_df = pd.DataFrame(best_sim["daily_history"])
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"SPE GeoHackathon Autonomous Dispatch Architecture: {winning_cfg_id}", fontsize=15, fontweight="bold")
    
    # Subplot 1: Annual Daily Thermal Load & Geothermal Dispatch
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history_df["day"], history_df["heat_demand_mw"], label="District Heating Demand", color="#d95f02", lw=1.5)
    ax1.plot(history_df["day"], history_df["geo_delivered_mw"], label="Geothermal Heat Delivered", color="#1b9e77", lw=1.5, ls="--")
    ax1.fill_between(history_df["day"], 0, history_df["aux_gas_mw"], label="Auxiliary Backup (Gas)", color="#e7298a", alpha=0.5)
    ax1.set_title("330-Day Sequential Thermal Dispatch Profile", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Operational Day")
    ax1.set_ylabel("Thermal Power (MWth)")
    ax1.legend(loc="upper right")
    
    # Subplot 2: Stratified Storage State of Charge (TES SOC)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(history_df["day"], history_df["storage_soc_mwh"], label="TES State of Charge", color="#7570b3", lw=1.8)
    ax2.set_title("Thermal Energy Storage (TES) State of Charge", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Operational Day")
    ax2.set_ylabel("Stored Energy (MWh)")
    ax2.legend(loc="upper right")
    
    # Subplot 3: Individual Well Energy Contribution
    ax3 = fig.add_subplot(2, 2, 3)
    wells_prod = best_sim["well_production_gwh"]
    well_labels = [f"Doublet {i+1}" for i in range(len(wells_prod))]
    colors = ["#2b83ba", "#abdda4", "#fdae61"][:len(wells_prod)]
    bars = ax3.bar(well_labels, wells_prod, color=colors, edgecolor="black", width=0.5)
    ax3.set_title("Annual Energy Contribution per Doublet", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Thermal Output (GWhth/year)")
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.0, f"{yval:.2f} GWh", ha="center", va="bottom", fontweight="bold")
        
    # Subplot 4: Pareto Frontier (CAPEX vs LCOE across viable options)
    ax4 = fig.add_subplot(2, 2, 4)
    viable = df_results[df_results["coverage_pct"] >= 99.5]
    scatter = ax4.scatter(viable["total_capex_m"], viable["lcoe_eur_mwh"], c=viable["storage_mwh"], cmap="viridis", s=70, alpha=0.8, edgecolor="k")
    cbar = fig.colorbar(scatter, ax=ax4)
    cbar.set_label("TES Capacity (MWh)")
    
    best_row = df_results[df_results["cfg_id"] == winning_cfg_id].iloc[0]
    ax4.scatter(best_row["total_capex_m"], best_row["lcoe_eur_mwh"], color="red", s=200, marker="*", edgecolor="black", label=f"Selected: {winning_cfg_id}")
    ax4.set_title("Design Exploration: CAPEX vs LCOE (Pareto Frontier)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Total Capital Expenditure (M€)")
    ax4.set_ylabel("LCOE (€/MWh)")
    ax4.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig("dispatch_results.png", dpi=300)
    print("\n[Graphics] Interactive Dashboard saved as 'dispatch_results.png' and displaying on screen...")
    plt.show()

# ---------------------------------------------------------------------------
# 5. Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    df_results, dispatch_cache = run_interactive_simulation()
    
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
    report_text, token_usage = autonomous_agent_decision(df_results)
    
    print(f"\n[OpenAI API Token Usage] Prompt: {token_usage['prompt_tokens']} | Completion: {token_usage['completion_tokens']} | Total: {token_usage['total_tokens']}")
    
    print("\n" + "="*85)
    print("--- AUTONOMOUS AGENT MULTI-CRITERIA DECISION REPORT (A to H) ---")
    print("="*85)
    print(report_text)
    print("="*85)
    
    # Extract selected configuration ID for dashboard routing
    winning_cfg_id = "D2_S100_A5"
    for line in report_text.splitlines():
        if "assumptions (" in line and ")" in line:
            extracted = line.split("assumptions (")[1].split(")")[0].strip()
            if extracted in dispatch_cache:
                winning_cfg_id = extracted
                break
                
    best_sim = dispatch_cache[winning_cfg_id]
    
    print("\n" + "="*85)
    print(f"--- WELL OPERATIONAL PROFILE & HYDRAULIC DISPATCH BREAKDOWN ({winning_cfg_id}) ---")
    print("="*85)
    well_summary = []
    num_doublets = len(best_sim["well_production_gwh"])
    for i in range(num_doublets):
        well_summary.append({
            "Doublet ID": f"Doublet {i+1}",
            "Annual Production (GWh)": best_sim["well_production_gwh"][i],
            "Active Days": best_sim["well_active_days"][i],
            "Inactive Days (Shut-in)": best_sim["well_inactive_days"][i],
            "Mean Operational Flow (m3/h)": best_sim["well_mean_flow_m3h"][i]
        })
    print(pd.DataFrame(well_summary).to_string(index=False))
    print("="*85)
    
    # Save required audit logs
    with open("agent_interactions.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "gpt-4o-mini",
            "token_usage": token_usage,
            "report_output": report_text
        }, f, indent=2, ensure_ascii=False)
    print("\n📁 Log de interacción y tokens guardado exitosamente en 'agent_interactions.json'.")
    
    print_ascii_dashboard(winning_cfg_id, df_results, dispatch_cache)
    plot_results_on_screen(df_results, winning_cfg_id, dispatch_cache)
    
    print("\n✅ Simulation and evaluation completed successfully.")

if __name__ == "__main__":
    main()
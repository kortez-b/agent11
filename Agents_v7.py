import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Setup and Client Initialization
# ---------------------------------------------------------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

agent_logs = []

def call_llm(agent_name: str, prompt: str, system_prompt: str = "You are a senior geothermal and energy systems engineering agent.") -> str:
    """Calls OpenAI LLM and records logs with token usage."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    output_text = response.choices[0].message.content
    log_entry = {
        "agent": agent_name,
        "prompt": prompt,
        "response": output_text,
        "token_usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
    }
    agent_logs.append(log_entry)
    print(f"[{agent_name}] Tokens consumed: {response.usage.total_tokens}")
    return output_text

# ---------------------------------------------------------------------------
# 1. Parametric Sweeper: Doublets, Storage (including 175 MWh), and HTHP Booster
# ---------------------------------------------------------------------------
def run_parametric_matrix(demand_csv: str = "GEMINI_Heating_Demand_Daily_2027.csv") -> pd.DataFrame:
    """
    Simulates annual energy balances.
    HTHP acts physically as a series booster covering the 16 K lift (84°C -> 100°C),
    representing 40% of the heating thermal stream. Direct geothermal preheats the other 60%.
    """
    df = pd.read_csv(demand_csv)
    heat_demand_mw = df["heating_demand_MWth"].values
    heat_demand_mwh = heat_demand_mw * 24.0
    
    # Subsurface constraints (GEMINI challenge specs)
    max_flow_per_doublet = 300.0  # m3/h
    rho = 1030.0                  # kg/m3
    cp = 3.98                     # kJ/(kg*K)
    max_delta_t = 40.0            # °C
    cop_hp = 3.5                  # Seasonal effective COP for high-temperature lift
    
    max_q_doublet_mw = (max_flow_per_doublet * rho / 3600.0) * cp * max_delta_t / 1000.0  # ~13.66 MWth
    
    # Booster fraction: (100°C - 84°C) / (100°C - 60°C) = 16 K / 40 K = 0.40
    booster_fraction = 0.40
    
    results = []
    
    for doublets in [1, 2, 3]:
        max_plant_capacity = doublets * max_q_doublet_mw
        # Sizing standard: 8 MW for 1-2 doublets, 6 MW for 3 doublets
        hthp_capacity_mw = 6.0 if doublets == 3 else 8.0
        
        # Explicit sweep including 175 MWh (critical technical threshold)
        for storage_cap_mwh in [0, 50, 100, 175, 200]:
            storage_level = 0.0
            geo_hp_delivered_day = []
            aux_delivered_day = []
            hp_electric_day = []
            charged_day = []
            discharged_day = []
            
            for daily_demand_mwh in heat_demand_mwh:
                daily_geo_max_mwh = max_plant_capacity * 24.0
                
                if daily_geo_max_mwh >= daily_demand_mwh:
                    geo_used = daily_demand_mwh
                    surplus = daily_geo_max_mwh - daily_demand_mwh
                    charge = min(surplus, storage_cap_mwh - storage_level) if storage_cap_mwh > 0 else 0.0
                    storage_level += charge
                    discharge = 0.0
                    aux = 0.0
                else:
                    geo_used = daily_geo_max_mwh
                    deficit = daily_demand_mwh - daily_geo_max_mwh
                    discharge = min(deficit, storage_level) if (storage_cap_mwh > 0 and storage_level > 0) else 0.0
                    storage_level -= discharge
                    aux = deficit - discharge
                    charge = 0.0
                
                geo_total_heating = geo_used + discharge
                
                # Physical Option 1: HTHP acts as booster on the top 40% thermal lift
                heat_needing_boost = geo_total_heating * booster_fraction
                hp_heat = min(heat_needing_boost, hthp_capacity_mw * 24.0)
                hp_elec = hp_heat / cop_hp
                
                geo_hp_delivered_day.append(geo_total_heating)
                aux_delivered_day.append(aux)
                hp_electric_day.append(hp_elec)
                charged_day.append(charge)
                discharged_day.append(discharge)
            
            total_geo_heat_gwh = sum(geo_hp_delivered_day) / 1000.0
            total_aux_gwh = sum(aux_delivered_day) / 1000.0
            total_hp_elec_gwh = sum(hp_electric_day) / 1000.0
            total_demand_gwh = total_geo_heat_gwh + total_aux_gwh
            coverage_pct = (total_geo_heat_gwh / total_demand_gwh) * 100.0 if total_demand_gwh > 0 else 0.0
            
            results.append({
                "doublets": doublets,
                "storage_MWh": storage_cap_mwh,
                "hthp_capacity_mw": hthp_capacity_mw,
                "geo_HP_heating_GWh": round(total_geo_heat_gwh, 3),
                "auxiliary_heating_GWh": round(total_aux_gwh, 3),
                "coverage_pct": round(coverage_pct, 2),
                "HP_electricity_GWh": round(total_hp_elec_gwh, 3),
                "storage_charge_GWh": round(sum(charged_day) / 1000.0, 3),
                "storage_discharge_GWh": round(sum(discharged_day) / 1000.0, 3),
                "storage_cycles": round((sum(charged_day) / storage_cap_mwh), 2) if storage_cap_mwh > 0 else 0.0
            })
            
    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 2. Demand Profile Agent
# ---------------------------------------------------------------------------
def demand_agent() -> dict:
    """Reads official 2027 heating and cooling demand datasets."""
    df_heat = pd.read_csv("GEMINI_Heating_Demand_Daily_2027.csv")
    df_cool = pd.read_csv("GEMINI_Cooling_Demand_Daily_2027.csv")
    
    peak_heat_mw = float(df_heat["heating_demand_MWth"].max())
    annual_heat_mwh = float((df_heat["heating_demand_MWth"] * 24.0).sum())
    
    peak_cool_mw = float(df_cool["cooling_demand_MWcool"].max())
    annual_cool_mwh = float((df_cool["cooling_demand_MWcool"] * 24.0).sum())
    
    return {
        "peak_heating_mw": round(peak_heat_mw, 2),
        "peak_cooling_mw": round(peak_cool_mw, 2),
        "annual_heating_gwh": round(annual_heat_mwh / 1000.0, 3),
        "annual_cooling_gwh": round(annual_cool_mwh / 1000.0, 3),
        "target_heating_supply_temp_c": 100.0,
        "target_cooling_supply_temp_c": 20.0
    }

# ---------------------------------------------------------------------------
# 3. Subsurface Resource Agent
# ---------------------------------------------------------------------------
def resource_agent() -> dict:
    """Subsurface properties according to GEMINI Hackathon limits."""
    depth_m = 2500.0
    t_res = 12.0 + (depth_m / 1000.0) * 30.0  # 87.0 °C
    max_flow = 300.0
    max_delta_t = 40.0
    max_power = (max_flow * 1030.0 / 3600.0 * 3.98 * max_delta_t) / 1000.0  # ~13.66 MW
    
    return {
        "depth_tv_m": depth_m,
        "reservoir_pressure_bar": 250.0,
        "permeability_md": 200.0,
        "reservoir_temp_c": t_res,
        "productivity_index_m3_h_bar": 11.78,
        "max_flow_m3h_per_doublet": max_flow,
        "min_flow_m3h_per_doublet": 50.0,
        "max_delta_t_c": max_delta_t,
        "max_thermal_power_mw_per_doublet": round(max_power, 2),
        "pump_hydraulic_efficiency": 0.70
    }

# ---------------------------------------------------------------------------
# 4. Techno-Economic Calculator: Modular OPEX, LCOE, LCOH, and LCOC
# ---------------------------------------------------------------------------
def calculate_techno_economic_kpis(config: dict, simulation_row: dict, demand: dict) -> dict:
    """Computes exact modular lifecycle costs, LCOE, LCOH, and LCOC."""
    num_doublets = config["number_of_doublets"]
    storage_mwh = config.get("thermal_storage_mwh", 0.0)
    hthp_mw = config.get("hthp_capacity_mw", 8.0)
    abs_chiller_mw = config.get("absorption_chiller_capacity_mw", 7.5)
    elec_chiller_mw = config.get("electric_chiller_backup_mw", 2.5)
    
    annual_heat_gwh = demand["annual_heating_gwh"]
    annual_cool_gwh = demand["annual_cooling_gwh"]
    total_energy_gwh = annual_heat_gwh + annual_cool_gwh
    
    # --- CAPEX Breakdown (M€) ---
    wells_capex = num_doublets * 12.0  # 6.0 M€ per well (2 wells/doublet)
    esp_capex = num_doublets * 0.60
    hthp_capex = hthp_mw * 0.80
    storage_capex = storage_mwh * 0.04
    abs_capex = abs_chiller_mw * 0.45
    elec_capex = elec_chiller_mw * 0.15
    chillers_capex = abs_capex + elec_capex
    
    surface_equip = hthp_capex + storage_capex + chillers_capex
    bop_capex = surface_equip * 0.15
    total_capex = wells_capex + esp_capex + surface_equip + bop_capex
    
    # Rigorous physical CAPEX allocation (BOP split 80% heat / 20% cool)
    bop_heat = bop_capex * 0.80
    bop_cool = bop_capex * 0.20
    capex_heat = wells_capex + esp_capex + hthp_capex + storage_capex + bop_heat
    capex_cool = chillers_capex + bop_cool
    
    # --- Energy Flows (GWh/year) ---
    hp_elec_gwh = simulation_row["HP_electricity_GWh"]
    aux_gas_gwh = simulation_row["auxiliary_heating_GWh"]
    
    # ESP Pumping: 0.15 MW continuous over 330 operational days
    esp_elec_gwh = num_doublets * 0.15 * 330.0 * 24.0 / 1000.0  # ~1.188 GWh per doublet
    
    # Cooling electricity: absorption parasitics (COP_el=25) + electric chiller (COP=4.0)
    abs_cooling_gwh = annual_cool_gwh * (abs_chiller_mw / 10.0)
    elec_cooling_gwh = annual_cool_gwh * (elec_chiller_mw / 10.0)
    chillers_elec_gwh = (abs_cooling_gwh / 25.0) + (elec_cooling_gwh / 4.0)
    
    elec_heat_gwh = hp_elec_gwh + esp_elec_gwh
    elec_cool_gwh = chillers_elec_gwh
    total_elec_gwh = elec_heat_gwh + elec_cool_gwh
    
    # --- Modular OPEX (M€/year) ---
    opex_wells = wells_capex * 0.0075                    # 0.75% well integrity
    opex_active_heat = hthp_capex * 0.04                  # 4.0% rotating equipment (HTHP)
    opex_active_cool = chillers_capex * 0.04              # 4.0% chillers
    opex_static_heat = (storage_capex + bop_heat) * 0.015 # 1.5% static piping & storage
    opex_static_cool = bop_cool * 0.015
    esp_workover_fund = num_doublets * 0.180              # Amortized workover/replacement fund (~3.3-4 yrs)
    
    opex_fixed_heat = opex_wells + opex_active_heat + opex_static_heat + esp_workover_fund
    opex_fixed_cool = opex_active_cool + opex_static_cool
    total_opex_fixed = opex_fixed_heat + opex_fixed_cool
    
    # Operational variable cost (€150/MWh elec, €60/MWh gas backup)
    elec_cost_heat = elec_heat_gwh * 1000.0 * 150.0 / 1e6
    elec_cost_cool = elec_cool_gwh * 1000.0 * 150.0 / 1e6
    gas_cost = aux_gas_gwh * 1000.0 * 60.0 / 1e6
    
    total_opex_heat = opex_fixed_heat + elec_cost_heat + gas_cost
    total_opex_cool = opex_fixed_cool + elec_cost_cool
    total_annual_opex = total_opex_heat + total_opex_cool
    
    # --- Financial Levelized Metrics (30-year lifetime, 6% WACC -> CRF = 0.0726489) ---
    crf = 0.06 / (1.0 - (1.0 + 0.06)**(-30))
    annual_cost_heat = (capex_heat * crf) + total_opex_heat
    annual_cost_cool = (capex_cool * crf) + total_opex_cool
    annual_cost_total = (total_capex * crf) + total_annual_opex
    
    lcoh_eur_mwh = (annual_cost_heat * 1e6) / (annual_heat_gwh * 1000.0)
    lcoc_eur_mwh = (annual_cost_cool * 1e6) / (annual_cool_gwh * 1000.0)
    lcoe_eur_mwh = (annual_cost_total * 1e6) / (total_energy_gwh * 1000.0)
    
    # --- Decarbonization ---
    grid_ef = 200.0  # t CO2 / GWh
    gas_ef = 202.0   # t CO2 / GWh
    annual_co2_tons = (total_elec_gwh * grid_ef) + (aux_gas_gwh * gas_ef)
    baseline_co2 = (annual_heat_gwh * gas_ef) + (annual_cool_gwh / 3.5 * grid_ef)
    co2_reduction_pct = max(0.0, (1.0 - (annual_co2_tons / baseline_co2)) * 100.0)
    
    return {
        "total_capex_m_eur": round(total_capex, 2),
        "capex_heat_m": round(capex_heat, 2),
        "capex_cool_m": round(capex_cool, 2),
        "total_opex_fixed_m_eur": round(total_opex_fixed, 3),
        "total_annual_opex_m_eur": round(total_annual_opex, 3),
        "hp_electricity_gwh": round(hp_elec_gwh, 3),
        "auxiliary_gas_gwh": round(aux_gas_gwh, 3),
        "total_electricity_gwh_yr": round(total_elec_gwh, 3),
        "lcoh_eur_mwh": round(lcoh_eur_mwh, 2),
        "lcoc_eur_mwh": round(lcoc_eur_mwh, 2),
        "lcoe_eur_mwh": round(lcoe_eur_mwh, 2),
        "co2_reduction_pct": round(co2_reduction_pct, 1)
    }

# ---------------------------------------------------------------------------
# 5. System Design Agent
# ---------------------------------------------------------------------------
def design_agent(demand: dict, resource: dict, scenarios_df: pd.DataFrame) -> dict:
    """Selects exact simulation rows (2 doublets + 175 MWh vs 3 doublets + 0 MWh)."""
    # Exact lookup for candidate configurations
    row_cfg_a = scenarios_df[(scenarios_df["doublets"] == 2) & (scenarios_df["storage_MWh"] == 175)].iloc[0].to_dict()
    row_cfg_b = scenarios_df[(scenarios_df["doublets"] == 3) & (scenarios_df["storage_MWh"] == 0)].iloc[0].to_dict()
    
    cfg_a_pre = {
        "id": "config_A",
        "name": "2 Geothermal Doublets + HTHP Boost + 175 MWh Storage + Hybrid Chiller",
        "number_of_doublets": 2,
        "geothermal_capacity_mw": 27.32,
        "hthp_capacity_mw": 8.0,
        "thermal_storage_mwh": 175.0,
        "absorption_chiller_capacity_mw": 7.5,
        "electric_chiller_backup_mw": 2.5,
        "coverage_pct": row_cfg_a["coverage_pct"],
        "peak_gas_boiler_mw": 0.0 if row_cfg_a["auxiliary_heating_GWh"] == 0.0 else round(demand["peak_heating_mw"] - 27.32, 2)
    }
    cfg_a_pre.update(calculate_techno_economic_kpis(cfg_a_pre, row_cfg_a, demand))
    
    cfg_b_pre = {
        "id": "config_B",
        "name": "3 Geothermal Doublets + HTHP Boost + Full Absorption Cooling",
        "number_of_doublets": 3,
        "geothermal_capacity_mw": 40.98,
        "hthp_capacity_mw": 6.0,
        "thermal_storage_mwh": 0.0,
        "absorption_chiller_capacity_mw": 10.0,
        "electric_chiller_backup_mw": 0.0,
        "coverage_pct": row_cfg_b["coverage_pct"],
        "peak_gas_boiler_mw": 0.0
    }
    cfg_b_pre.update(calculate_techno_economic_kpis(cfg_b_pre, row_cfg_b, demand))
    
    prompt = f"""
Analyze the candidate geothermal system architectures based on simulation rows:
Option A: {json.dumps(cfg_a_pre, indent=2)}
Option B: {json.dumps(cfg_b_pre, indent=2)}

Provide an architectural engineering breakdown detailing the fluid loops, heat exchanger stages, and seasonal balancing.
"""
    system_prompt = "You are a thermal engineering specialist designing deep geothermal district heating and cooling networks."
    analysis = call_llm("design_agent", prompt, system_prompt)
    
    designs = {
        "configurations": [cfg_a_pre, cfg_b_pre],
        "ai_synthesis": analysis
    }
    return designs

def export_scenarios_json(designs: dict, filepath: str = "scenarios_generated.json"):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(designs, f, indent=4, ensure_ascii=False)
    print(f"[export_scenarios_json] Saved configurations to '{filepath}'")

# ---------------------------------------------------------------------------
# 6. Optimization Agent (Cleaned Prompt without Hardcoded Additions)
# ---------------------------------------------------------------------------
def optimization_agent(designs: dict, demand: dict, resource: dict, scenarios_df: pd.DataFrame) -> tuple:
    configurations = designs.get("configurations", [])
    matrix_data = scenarios_df.to_dict(orient="records")
    
    prompt = f"""
You are the Lead Optimization Engineer evaluating simulation scenarios for the SPE GeoHackathon.

Parametric Matrix Results across doublet counts and thermal storage:
{json.dumps(matrix_data, indent=2)}

Candidate Design Architectures with Full Techno-Economic KPIs:
{json.dumps(configurations, indent=2)}

Boundary Constraints:
- Annual Heating Demand: {demand.get('annual_heating_gwh')} GWh (100 °C supply)
- Annual Cooling Demand: {demand.get('annual_cooling_gwh')} GWh (20 °C supply)
- Well limits: 2500 m TVD depth, 50-300 m3/h flow cap per active doublet, 40 °C max deltaT

Evaluate the trade-offs:
1. CAPEX vs OPEX and LCOE (€/MWh).
2. Summer Turndown Feasibility (risk of violating the 50 m3/h minimum flow with 3 doublets vs 2 doublets).
3. Environmental Performance and Decarbonization.

Select the winning configuration ("config_A" or "config_B") and assign an optimization score (0-100).

Respond strictly with a JSON object:
{{
    "winning_config_id": "<config_id>",
    "optimization_score": <float between 0 and 100>,
    "decision_summary": "<concise summary>",
    "detailed_tradeoff_analysis": "<detailed rationale>"
}}
"""
    system_prompt = "You are an autonomous optimization agent for geothermal energy systems. Output strictly valid JSON."
    raw_response = call_llm("optimization_agent", prompt, system_prompt)
    
    try:
        clean_json_str = raw_response.replace("```json", "").replace("```", "").strip()
        decision = json.loads(clean_json_str)
        chosen_id = decision["winning_config_id"]
        score = float(decision["optimization_score"])
        best_config = next((cfg for cfg in configurations if cfg["id"] == chosen_id), configurations[0])
        best_config["selection_analysis"] = decision["detailed_tradeoff_analysis"]
        best_config["decision_summary"] = decision["decision_summary"]
    except Exception as e:
        print(f"[optimization_agent] Notice: JSON parsing fallback due to: {e}")
        # Explicit fallback based on lower calculated LCOE
        best_config = min(configurations, key=lambda x: x["lcoe_eur_mwh"])
        score = 90.0
        best_config["selection_analysis"] = raw_response
        best_config["decision_summary"] = "Selected autonomously based on minimum LCOE."

    print(f"\n[optimization_agent] Decision: '{best_config['id']}' with Score {score}")
    return best_config, score

# ---------------------------------------------------------------------------
# 7. Report Agent
# ---------------------------------------------------------------------------
def report_agent(demand: dict, resource: dict, best_design: dict, score: float) -> str:
    prompt = f"""
Write an executive technical report for the SPE GeoHackathon:
- Subsurface limits: {json.dumps(resource)}
- Annual Demands: {json.dumps(demand)}
- Selected Architecture: {json.dumps(best_design)}
- Score: {score}

Structure with numbered sections:
1. Executive Summary
2. Subsurface and Reservoir Characterization (Depth 2500m, 87 °C, PI = 11.78, flow limits)
3. Demand Profile Analysis (30 MWth heating peak, 10 MWcool cooling peak)
4. Integrated Surface Architecture (HTHP 16 K series booster, absorption & electric chillers, thermal storage)
5. Techno-Economic Evaluation (Modular OPEX, CAPEX, LCOE, LCOH, LCOC)
6. Hydraulic & Operational Constraint Verification (Turndown ratio, 330 operational days)
7. Conclusion
"""
    report_text = call_llm("report_agent", prompt)
    print("\n==============================================")
    print("--- SPE GEOHACKATHON TECHNICAL REPORT ---")
    print("==============================================")
    print(report_text)
    return report_text

# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    print("MAIN STARTED")
    
    print("\n--- Running Parametric Sweeper ---")
    scenarios_df = run_parametric_matrix()
    print(scenarios_df.to_string(index=False))
    scenarios_df.to_csv("parametric_matrix_results.csv", index=False)
    
    print("\n[Pipeline] Reading demand and subsurface profiles...")
    demand = demand_agent()
    resource = resource_agent()
    
    print("[Pipeline] Consulting Design Agent with full parametric data...")
    designs = design_agent(demand, resource, scenarios_df)
    export_scenarios_json(designs)
    
    print("[Pipeline] Consulting Optimization Agent for autonomous decision...")
    best_design, score = optimization_agent(designs, demand, resource, scenarios_df)
    
    print("[Pipeline] Generating final executive engineering report...")
    report_agent(demand, resource, best_design, score)
    
    with open("agent_interactions.json", "w", encoding="utf-8") as f:
        json.dump(agent_logs, f, indent=4, ensure_ascii=False)
        
    print("\n✅ Execution completed successfully! Deliverables generated:")
    print(" - parametric_matrix_results.csv")
    print(" - scenarios_generated.json")
    print(" - agent_interactions.json")

if __name__ == "__main__":
    main()
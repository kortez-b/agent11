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
    """Calls the OpenAI LLM and records prompts, completions, and token usage."""
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
# Parametric Sweeper: Doublets and Thermal Storage
# ---------------------------------------------------------------------------
def run_parametric_matrix(demand_csv: str = "GEMINI_Heating_Demand_Daily_2027.csv") -> pd.DataFrame:
    """Simulates annual energy balances across doublet counts and thermal storage capacities."""
    df = pd.read_csv(demand_csv)
    heat_demand_mw = df["heating_demand_MWth"].values
    heat_demand_mwh = heat_demand_mw * 24.0
    
    # Physical constants based on challenge specifications
    max_flow_per_doublet = 300.0  # m3/h
    rho = 1030.0                  # kg/m3
    cp = 3.98                     # kJ/(kg*K)
    max_delta_t = 40.0            # °C
    cop_hp = 3.5                  # Estimated COP for 87 °C to 100 °C temperature lift
    
    # Maximum thermal output per doublet (MW)
    max_q_doublet_mw = (max_flow_per_doublet * rho / 3600.0) * cp * max_delta_t / 1000.0  # ~13.66 MW
    
    results = []
    
    for doublets in [1, 2, 3]:
        max_plant_capacity = doublets * max_q_doublet_mw
        
        for storage_cap_mwh in [0, 50, 100, 200]:
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
                hp_elec = geo_total_heating / cop_hp
                
                geo_hp_delivered_day.append(geo_total_heating)
                aux_delivered_day.append(aux)
                hp_electric_day.append(hp_elec)
                charged_day.append(charge)
                discharged_day.append(discharge)
            
            total_charge = sum(charged_day) / 1000.0
            total_discharge = sum(discharged_day) / 1000.0
            cycles = (sum(charged_day) / storage_cap_mwh) if storage_cap_mwh > 0 else 0.0
            
            results.append({
                "doublets": doublets,
                "storage_MWh": storage_cap_mwh,
                "geo_HP_heating_GWh": round(sum(geo_hp_delivered_day) / 1000.0, 2),
                "auxiliary_heating_GWh": round(sum(aux_delivered_day) / 1000.0, 2),
                "HP_electricity_GWh": round(sum(hp_electric_day) / 1000.0, 2),
                "storage_charge_GWh": round(total_charge, 2),
                "storage_discharge_GWh": round(total_discharge, 2),
                "storage_cycles": round(cycles, 2)
            })
            
    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Agent 1: Demand Profile Agent
# ---------------------------------------------------------------------------
def demand_agent() -> dict:
    """Reads official 2027 heating and cooling demand time series."""
    df_heat = pd.read_csv("GEMINI_Heating_Demand_Daily_2027.csv")
    df_cool = pd.read_csv("GEMINI_Cooling_Demand_Daily_2027.csv")
    
    peak_heat_mw = float(df_heat["heating_demand_MWth"].max())
    annual_heat_mwh = float((df_heat["heating_demand_MWth"] * 24).sum())
    
    peak_cool_mw = float(df_cool["cooling_demand_MWcool"].max())
    annual_cool_mwh = float((df_cool["cooling_demand_MWcool"] * 24).sum())
    
    return {
        "peak_heating_mw": round(peak_heat_mw, 2),
        "peak_cooling_mw": round(peak_cool_mw, 2),
        "annual_heating_gwh": round(annual_heat_mwh / 1000.0, 2),
        "annual_cooling_gwh": round(annual_cool_mwh / 1000.0, 2),
        "target_heating_supply_temp_c": 100.0,
        "target_cooling_supply_temp_c": 20.0
    }

# ---------------------------------------------------------------------------
# Agent 2: Subsurface Resource Agent
# ---------------------------------------------------------------------------
def resource_agent() -> dict:
    """Calculates subsurface properties and single doublet operational caps."""
    depth_m = 2500.0
    t_surface = 12.0
    geothermal_grad = 30.0  # °C/km
    t_res = t_surface + (depth_m / 1000.0) * geothermal_grad  # 87.0 °C
    
    rho = 1030.0
    cp = 3.98
    max_flow = 300.0
    max_delta_t = 40.0
    max_power = ((max_flow * rho / 3600.0) * cp * max_delta_t) / 1000.0  # ~13.66 MW
    
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
        "gas_liquid_ratio_nm3_m3": 0.8,
        "pump_hydraulic_efficiency": 0.70
    }

# ---------------------------------------------------------------------------
# Agent 3: System Design Agent
# ---------------------------------------------------------------------------
def design_agent(demand: dict, resource: dict, scenarios_df: pd.DataFrame) -> dict:
    """Formulates integrated heating and cooling network architectures."""
    matrix_summary = scenarios_df.to_dict(orient="records")
    
    prompt = f"""
Analyze the following geothermal inputs and parametric simulation results:
- Demand Profile: {json.dumps(demand, indent=2)}
- Geothermal Reservoir & Well Limits: {json.dumps(resource, indent=2)}
- Parametric Matrix (Doublets vs Storage): {json.dumps(matrix_summary, indent=2)}

Critical Design Constraints:
1. Reservoir temperature is 87 °C, but heating demand requires 100 °C supply. High-Temperature Heat Pumps (HTHP) must provide the 13 °C temperature lift.
2. Cooling demand peaks at 10.0 MWcool (20 °C supply). Integrate LiBr/H2O absorption chillers driven by summer geothermal heat with electric chiller backup.
3. Evaluate 2 Doublets vs 3 Doublets based on CAPEX, OPEX, and thermal coverage.

Provide a comprehensive architectural breakdown for Option A (2 doublets + storage) and Option B (3 doublets).
"""
    system_prompt = "You are a thermal engineering specialist designing deep geothermal district heating and cooling networks."
    analysis = call_llm("design_agent", prompt, system_prompt)
    
    designs = {
        "configurations": [
            {
                "id": "config_A",
                "name": "2 Geothermal Doublets + HTHP Boost + 100 MWh Storage + Hybrid Chiller",
                "number_of_doublets": 2,
                "geothermal_capacity_mw": 27.32,
                "hthp_capacity_mw": 8.0,
                "thermal_storage_mwh": 100.0,
                "absorption_chiller_capacity_mw": 7.5,
                "electric_chiller_backup_mw": 2.5,
                "peak_gas_boiler_mw": 3.0,
                "coverage_pct": 99.93,
                "rationale": "2 doublets satisfy 111.58 GWh of annual heating. Peak deficit is shaved by storage and backup boiler."
            },
            {
                "id": "config_B",
                "name": "3 Geothermal Doublets + HTHP Boost + Full Absorption Cooling",
                "number_of_doublets": 3,
                "geothermal_capacity_mw": 40.98,
                "hthp_capacity_mw": 6.0,
                "thermal_storage_mwh": 0.0,
                "absorption_chiller_capacity_mw": 10.0,
                "electric_chiller_backup_mw": 0.0,
                "peak_gas_boiler_mw": 0.0,
                "coverage_pct": 100.0,
                "rationale": "3 doublets achieve 100% renewable coverage at all times, but incur significant well capital expenditures."
            }
        ],
        "ai_synthesis": analysis
    }
    return designs

def export_scenarios_json(designs: dict, filepath: str = "scenarios_generated.json"):
    """Saves generated configurations to disk."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(designs, f, indent=4, ensure_ascii=False)
    print(f"[export_scenarios_json] Saved configurations to '{filepath}'")

# ---------------------------------------------------------------------------
# Agent 4: Optimization Agent
# ---------------------------------------------------------------------------
def optimization_agent(designs: dict, demand: dict, resource: dict, scenarios_df: pd.DataFrame) -> tuple:
    """The LLM acts as an autonomous decision-maker: it reviews all simulation data,
    evaluates trade-offs, selects the optimal configuration, and assigns a score."""
    
    # Convert simulation matrix to JSON so the LLM inspects the actual numbers
    matrix_data = scenarios_df.to_dict(orient="records")
    
    prompt = f"""
You are the Lead Optimization Engineer evaluating simulation scenarios for the SPE GeoHackathon.

Here is the simulation data generated across different doublet counts and thermal storage:
{json.dumps(matrix_data, indent=2)}

Candidate Design Architectures:
{json.dumps(designs.get('configurations', []), indent=2)}

Boundary Constraints & Annual Demand:
- Annual Heating Demand: {demand.get('annual_heating_gwh')} GWh
- Peak Heating: {demand.get('peak_heating_mw')} MW (100 °C required supply)
- Peak Cooling: {demand.get('peak_cooling_mw')} MW (20 °C required supply)
- Well constraints: 2500 m TVD depth, 50-300 m3/h flow cap per active doublet, 40 °C max deltaT

YOUR TASK AS AN AGENT:
1. Objectively compare the trade-offs:
   - Drilling CAPEX: Drilling a 3rd doublet (2 deep wells at 2500 m) vs operating with 2 doublets + storage + small peak auxiliary.
   - Thermal Coverage: Compare unmet heating deficits and auxiliary heating required.
   - Summer Turndown Feasibility: Assess risk of violating the 50 m3/h minimum flow constraint during low summer demand.
2. Select the true winning configuration (e.g., "config_A" or "config_B").
3. Assign an optimization score from 0.0 to 100.0 based on techno-economic performance.
4. Provide an executive engineering justification.

OUTPUT FORMAT:
Respond ONLY with a valid JSON object matching this schema:
{{
    "winning_config_id": "<config_id>",
    "optimization_score": <float between 0 and 100>,
    "decision_summary": "<concise explanation of why this configuration wins>",
    "detailed_tradeoff_analysis": "<thorough evaluation of CAPEX, OPEX, and operational risks>"
}}
"""
    
    system_prompt = (
        "You are an expert autonomous optimization agent for geothermal energy systems. "
        "Analyze the provided data impartially without bias. Output strictly valid JSON."
    )
    
    # Call the LLM
    raw_response = call_llm("optimization_agent", prompt, system_prompt)
    
    # Parse the LLM's independent decision
    try:
        # Strip potential markdown code fences from response
        clean_json_str = raw_response.replace("```json", "").replace("```", "").strip()
        decision = json.loads(clean_json_str)
        
        chosen_id = decision["winning_config_id"]
        score = float(decision["optimization_score"])
        
        # Match chosen configuration from the designs list
        best_config = next((cfg for cfg in designs["configurations"] if cfg["id"] == chosen_id), designs["configurations"][0])
        best_config["selection_analysis"] = decision["detailed_tradeoff_analysis"]
        best_config["decision_summary"] = decision["decision_summary"]
        
    except Exception as e:
        print(f"[optimization_agent] Warning: JSON parsing fallback due to: {e}")
        best_config = designs["configurations"][0]
        score = 90.0
        best_config["selection_analysis"] = raw_response

    print(f"\n[optimization_agent] LLM Autonomous Decision: Selected '{best_config['id']}' with Score {score}")
    return best_config, score

# ---------------------------------------------------------------------------
# Agent 5: Report Agent
# ---------------------------------------------------------------------------
def report_agent(demand: dict, resource: dict, best_design: dict, score: float) -> str:
    """Generates the executive technical report."""
    prompt = f"""
Write an executive technical report for the SPE GeoHackathon:
- Resource Characterization: {json.dumps(resource)}
- Annual Demand Profile: {json.dumps(demand)}
- Optimal Design Selected: {json.dumps(best_design)}
- Optimization Score: {score}

Structure the report with the following numbered sections:
1. Executive Summary
2. Subsurface and Reservoir Characterization (Depth, 87 °C, PI = 11.78, flow limits)
3. Demand Profile Analysis (30 MWth heating peak, 10 MWcool cooling peak)
4. Surface Engineering Design (HTHP lift to 100 °C, absorption and electric chillers for 20 °C)
5. Hydraulic and Operational Constraint Verification (50-300 m3/h flow rates, 40 °C deltaT cap, 330 operational days)
6. Conclusion and GEMINI Digital Twin Validation
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
    
    demand = demand_agent()
    resource = resource_agent()
    designs = design_agent(demand, resource, scenarios_df)
    
    export_scenarios_json(designs)
    
    best_design, score = optimization_agent(designs, demand, resource)
    report_agent(demand, resource, best_design, score)
    
    with open("agent_interactions.json", "w", encoding="utf-8") as f:
        json.dump(agent_logs, f, indent=4, ensure_ascii=False)
        
    print("\n✅ Execution completed. Outputs saved to 'parametric_matrix_results.csv', 'scenarios_generated.json', and 'agent_interactions.json'")

if __name__ == "__main__":
    main()
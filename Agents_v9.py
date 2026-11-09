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
    Simula secuencialmente el despacho diario para 330 días operativos (7,920 h).
    Modula caudal por doblete (50-300 m3/h), calcula balances térmicos,
    almacenamiento y refrigeración híbrida estival.
    """
    rho = 1030.0          # kg/m3 (densidad salmuera geotérmica)
    cp = 3.98             # kJ/(kg*K) (calor específico salmuera)
    max_flow_well = 300.0 # m3/h límite máximo por doblete
    min_flow_well = 50.0  # m3/h límite mínimo de estabilidad hidrodinámica
    max_delta_t = 40.0    # °C enfriamiento máximo de salmuera (T_reinj >= 47 °C)
    t_res = 87.0          # °C temperatura de producción de yacimiento
    t_district_target = 100.0 # °C requerimiento de red distrital
    
    # Factor térmico por m3/h a deltaT = 40 K
    k_heat = (rho / 3600.0) * cp * max_delta_t / 1000.0  # ~0.04555 MW por m3/h
    q_max_single_doublet = max_flow_well * k_heat        # ~13.665 MWth por doblete
    
    cop_hthp = 3.50
    cop_abs_th = 0.72    # COP térmico chiller de absorción LiBr/H2O
    cop_abs_el = 25.0    # COP eléctrico parásito
    cop_elec_chill = 4.0 # COP chiller eléctrico convencional
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
        
        # A. Despacho de Refrigeración Distrital (Target: 20 °C)
        cool_abs_mw = min(d_cool_mw, abs_chiller_cap_mw)
        cool_elec_mw = min(d_cool_mw - cool_abs_mw, elec_chiller_cap_mw)
        heat_for_abs_mwh = (cool_abs_mw * 24.0) / cop_abs_th
        elec_cool_mwh = ((cool_abs_mw * 24.0) / cop_abs_el) + ((cool_elec_mw * 24.0) / cop_elec_chill)
        total_chiller_elec_mwh += elec_cool_mwh
        
        # Demanda térmica total requerida del campo geotérmico
        total_thermal_demand_mw = d_heat_mw + (cool_abs_mw / cop_abs_th)
        
        # B. Escalonamiento de Dobletes (Well Staging)
        needed_active = int(math.ceil(total_thermal_demand_mw / q_max_single_doublet)) if total_thermal_demand_mw > 0 else 1
        needed_active = max(1, min(needed_active, doublets))
        
        plant_geo_max_mw = needed_active * q_max_single_doublet
        actual_plant_heat_mw = min(total_thermal_demand_mw, plant_geo_max_mw)
        
        heat_per_active_doublet = actual_plant_heat_mw / needed_active
        flow_per_active_doublet = heat_per_active_doublet / k_heat
        
        # Salvaguarda de estabilidad hidrodinámica (50 m3/h)
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
        
        # C. Despacho Dinámico de Almacenamiento Térmico (TES)
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
        
        # D. Elevación Térmica HTHP (84 °C -> 100 °C)
        heat_boosted = max(0.0, geo_used_for_heating_mwh) * 0.40
        total_hthp_elec_mwh += (heat_boosted / cop_hthp)
        
        # Temperatura de reinyección
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
    
    # 35 días con menor carga térmica combinada reservados para mantenimiento
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
    
    print("\n" + "="*95)
    print("--- EXECUTING PHYSICAL DISPATCH SWEEPER (330 OPERATIONAL DAYS) ---")
    print("="*95)
    
    for d in doublet_candidates:
        hthp_mw = 8.0 if d <= 2 else 6.0
        wells_capex = d * 12.0
        esp_capex = d * 0.60
        
        for s in storage_candidates:
            for abs_mw in cooling_configs:
                elec_mw = 10.0 - abs_mw
                sim = simulate_daily_dispatch(d, s, abs_mw, df_heat, df_cool, active_indices)
                
                # Modelo de CAPEX
                surface_capex = (hthp_mw * 0.80) + (s * 0.04) + (abs_mw * 0.45) + (elec_mw * 0.15)
                bop_capex = surface_capex * 0.15
                total_capex = wells_capex + esp_capex + surface_capex + bop_capex
                
                # Modelo de OPEX Modular de Ciclo de Vida
                opex_wells = wells_capex * 0.0075
                opex_active = (hthp_mw * 0.80 + abs_mw * 0.45 + elec_mw * 0.15) * 0.04
                opex_static = (s * 0.04 + bop_capex) * 0.015
                esp_fund = d * 0.180
                opex_fixed = opex_wells + opex_active + opex_static + esp_fund
                
                opex_var = (sim["total_electricity_gwh"] * 1000.0 * 150.0 / 1e6) + (sim["aux_gas_gwh"] * 1000.0 * 60.0 / 1e6)
                total_opex = opex_fixed + opex_var
                
                # LCOE (30 años, 6% WACC -> CRF = 0.07265)
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
# 3. Autonomous Optimization Agent (Weighted Multi-Criteria Decision Analysis)
# ---------------------------------------------------------------------------
def autonomous_agent_decision(df_results: pd.DataFrame) -> dict:
    viable = df_results[df_results["coverage_pct"] >= 99.5].copy()
    dataset = viable.to_dict(orient="records")
    
    prompt = f"""
You are the Lead Systems & Reservoir Optimization Advisor for the SPE GeoHackathon.
Conduct a formal Weighted Multi-Criteria Decision Analysis (MCDA) across the simulated candidate configurations (330 operational days):

{json.dumps(dataset, indent=2)}

WEIGHTED DECISION MATRIX (Total = 100%):
1. ECONOMIC EFFICIENCY [Weight: 40%]:
   - Levelized Cost of Energy (LCOE in €/MWh), total initial CAPEX, and annual lifecycle OPEX.
   - Penalize capital-heavy solutions that deliver marginal technical gains (diminishing returns).
2. THERMAL SECURITY & PEAK RESILIENCE [Weight: 35%]:
   - Performance during the 5-consecutive-day January polar freeze (~184 MWh cumulative deficit above 2-doublet capacity).
   - Minimization of auxiliary fossil gas dependence and security of municipal district supply.
3. OPERATIONAL COMPLEXITY & ASSET UTILIZATION [Weight: 25%]:
   - Subsurface asset utilization: penalize deep wells operating fewer than 30 days/yr (stranded assets).
   - Summer hydraulic stability: maintain active doublet flow >= 50.0 m3/h via absorption chiller base loading.
   - Surface footprint & complexity: footprint and permitting constraints of large sensible thermal storage vessels (>100 MWh).

MANDATORY TECHNICAL DEBATE & TASKS:
1. Debate 175 MWh vs 50 MWh Storage:
   - Quantify the exact trade-off: 50 MWh achieves lowest LCOE (€48.13/MWh, 99.83% coverage) but leaves a 134 MWh deficit during the freeze.
   - Sizing storage to 175 MWh achieves 100.00% zero-fossil coverage, but adds +€5.75M in CAPEX and increases LCOE to ~€52.06/MWh. Assess if spending €5.75M to offset ~€11k/yr in peaking gas is justified.
2. Debate 2 vs 3 Doublets:
   - Compare the capital cost of a 3rd doublet (+€8.5M–€12.6M) and its summer idle risk (325 days shut-in) against the genuine benefit of N-1 subsurface redundancy.
3. Select ONE Final Recommended Configuration and ONE Best Alternative from the dataset.

Output strictly valid JSON with this exact schema:
{{
    "mcda_evaluation": {{
        "economic_weight_pct": 40,
        "thermal_security_weight_pct": 35,
        "operational_complexity_weight_pct": 25
    }},
    "recommended_configuration": {{
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
        "composite_score": <float between 0 and 100>,
        "justification": "<why this architecture wins the multi-criteria balance>"
    }},
    "best_alternative": {{
        "cfg_id": "<cfg_id>",
        "doublets": <int>,
        "storage_mwh": <float>,
        "abs_mw": <float>,
        "total_capex_m": <float>,
        "lcoe_eur_mwh": <float>,
        "coverage_pct": <float>,
        "aux_gas_gwh": <float>,
        "context_for_selection": "<when and why the client would pick this alternative instead>"
    }},
    "technical_debates": {{
        "storage_175mwh_justification": "<detailed economic and physical assessment of 50 MWh vs 175 MWh>",
        "doublets_2_vs_3_tradeoff": "<rigorous comparison of capital efficiency vs N-1 subsurface redundancy>",
        "cooling_hybrid_vs_electric": "<assessment of absorption utilization for summer turndown mitigation>"
    }},
    "screening_limitations": [
        "CAPEX and OPEX values use generalized European geothermal cost correlations, not binding vendor quotes.",
        "Constant electricity (€150/MWh) and natural gas (€60/MWh) prices are assumed.",
        "Subsurface thermal breakthrough and inter-well pressure interference are not modeled.",
        "This constitutes a pre-feasibility screening study, not a Front-End Engineering Design (FEED)."
    ]
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior energy systems advisor conducting a balanced, rigorous multi-criteria screening evaluation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    # Captura explícita de tokens consumidos
    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
    
    clean_json = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    decision_dict = json.loads(clean_json)
    decision_dict["token_usage"] = token_usage
    
    return decision_dict

# ---------------------------------------------------------------------------
# 4. Terminal ASCII Graphical Dashboard (Zero Freezing / Zero Blocking)
# ---------------------------------------------------------------------------
def print_ascii_dashboard(winning_cfg: dict, df_results: pd.DataFrame, best_sim_data: dict):
    print("\n" + "="*85)
    print("--- TERMINAL GRAPHICAL DASHBOARD: WELL ENERGY & PARETO FRONTIER ---")
    print("="*85)
    
    # 1. Bar Chart: Producción Anual por Doblete (tomado de la física)
    print("\n[CHART 1] Annual Production Contribution per Doublet (GWhth):")
    prods = best_sim_data.get("well_production_gwh", [])
    max_prod = max(prods) if prods and max(prods) > 0 else 1.0
    for idx, p in enumerate(prods):
        bar_len = int((p / max_prod) * 45)
        bar_str = "█" * bar_len
        print(f" Doublet {idx+1} [{p:6.2f} GWh] | {bar_str}")
        
    # 2. Bar Chart: Días Activos de Operación (sobre 330 días base)
    print("\n[CHART 2] Operational Utilization (Active Days out of 330):")
    days = best_sim_data.get("well_active_days", [])
    for idx, d in enumerate(days):
        bar_len = int((d / 330.0) * 45)
        bar_str = "▓" * bar_len
        print(f" Doublet {idx+1} [{d:3d} Days]  | {bar_str}")
        
    # 3. Pareto Frontier: LCOE (€/MWh) vs Capacidad de Almacenamiento TES
    print("\n[CHART 3] Pareto Comparison: LCOE (€/MWh) Across Top Configurations:")
    sub = df_results[df_results["coverage_pct"] >= 99.8].sort_values(by="lcoe_eur_mwh")
    for _, row in sub.head(10).iterrows():
        cid = row["cfg_id"]
        l = row["lcoe_eur_mwh"]
        c = row["coverage_pct"]
        offset = max(0, int((l - 46.0) * 6))
        plot_line = " " * offset + "◆"
        selected_flag = "  <-- RECOMMENDED" if cid == winning_cfg.get("cfg_id") else ""
        print(f" {cid:11s} | LCOE: €{l:5.2f} | Cov: {c:5.2f}% |{plot_line}{selected_flag}")
    print("="*85)

# ---------------------------------------------------------------------------
# 5. Interactive Visualizations & Graphical Plots (Matplotlib)
# ---------------------------------------------------------------------------
def plot_results_on_screen(df_results: pd.DataFrame, winning_cfg_id: str, dispatch_cache: dict):
    best_sim = dispatch_cache[winning_cfg_id]
    history_df = pd.DataFrame(best_sim["daily_history"])
    
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"SPE GeoHackathon - Despacho Óptimo: {winning_cfg_id}", fontsize=15, fontweight="bold")
    
    # Gráfica 1: Perfil Diario de Despacho Térmico (330 Días)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history_df["day"], history_df["heat_demand_mw"], label="Demanda Calefacción", color="#d95f02", lw=1.5)
    ax1.plot(history_df["day"], history_df["geo_delivered_mw"], label="Calor Geotérmico Directo", color="#1b9e77", lw=1.5, ls="--")
    ax1.fill_between(history_df["day"], 0, history_df["aux_gas_mw"], label="Gas Auxiliar", color="#e7298a", alpha=0.5)
    ax1.set_title("Despacho Térmico Diario (330 Días Operativos)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Día Operativo")
    ax1.set_ylabel("Potencia Térmica (MWth)")
    ax1.legend(loc="upper right", fontsize=9)
    
    # Gráfica 2: Estado de Carga del Almacenamiento (TES SOC)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(history_df["day"], history_df["storage_soc_mwh"], label="Nivel TES (MWh)", color="#7570b3", lw=1.8)
    ax2.set_title("Evolución del Almacenamiento Térmico (TES)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Día Operativo")
    ax2.set_ylabel("Energía Almacenada (MWh)")
    ax2.legend(loc="upper right", fontsize=9)
    
    # Gráfica 3: Producción por Doblete
    ax3 = fig.add_subplot(2, 2, 3)
    wells_prod = best_sim["well_production_gwh"]
    labels = [f"Doblete {i+1}" for i in range(len(wells_prod))]
    bars = ax3.bar(labels, wells_prod, color=["#2b83ba", "#abdda4", "#fdae61"][:len(wells_prod)], width=0.45, edgecolor="black")
    ax3.set_title("Aporte Anual por Doblete (GWhth)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("GWhth/año")
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.2f} GWh", ha="center", va="bottom", fontweight="bold")
        
    # Gráfica 4: Frontera de Pareto (CAPEX vs LCOE)
    ax4 = fig.add_subplot(2, 2, 4)
    viable = df_results[df_results["coverage_pct"] >= 99.5]
    scatter = ax4.scatter(viable["total_capex_m"], viable["lcoe_eur_mwh"], c=viable["storage_mwh"], cmap="viridis", s=65, alpha=0.8, edgecolor="k")
    cbar = fig.colorbar(scatter, ax=ax4)
    cbar.set_label("Capacidad TES (MWh)")
    
    best_row = df_results[df_results["cfg_id"] == winning_cfg_id].iloc[0]
    ax4.scatter(best_row["total_capex_m"], best_row["lcoe_eur_mwh"], color="red", s=180, marker="*", edgecolor="black", label=f"Seleccionado: {winning_cfg_id}")
    ax4.set_title("Frontera de Pareto: CAPEX vs LCOE", fontsize=11, fontweight="bold")
    ax4.set_xlabel("CAPEX Total (M€)")
    ax4.set_ylabel("LCOE (€/MWh)")
    ax4.legend(loc="upper left", fontsize=9)
    
    plt.tight_layout()
    
    # Guarda la imagen como archivo de alta resolución
    plt.savefig("dispatch_results.png", dpi=300)
    print("\n[Gráficos] Imagen exportada exitosamente como 'dispatch_results.png'.")
    
    # Despliega la ventana gráfica
    print("[Gráficos] Abriendo ventana de visualización...")
    plt.show()

# ---------------------------------------------------------------------------
# 6. Main Execution Pipeline
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
    
    print("\n[Agent] Invoking Autonomous Multi-Criteria Decision Agent...")
    decision = autonomous_agent_decision(df_results)
    
    # Despliegue de métricas de tokens en pantalla
    usage = decision.get("token_usage", {})
    print(f"\n[OpenAI API Token Usage] Prompt: {usage.get('prompt_tokens', 0)} | Completion: {usage.get('completion_tokens', 0)} | Total: {usage.get('total_tokens', 0)}")
    
    # Guardado de trazabilidad en archivo JSON para la entrega oficial
    with open("agent_interactions.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "gpt-4o-mini",
            "token_usage": usage,
            "decision": decision
        }, f, indent=2, ensure_ascii=False)
    print("📁 Interacciones y consumo de tokens guardados en 'agent_interactions.json'.")
    
    rec_cfg = decision["recommended_configuration"]
    alt_cfg = decision["best_alternative"]
    winning_id = rec_cfg["cfg_id"]
    
    # Extraemos los datos físicos exactos directamente de la simulación
    best_sim_data = dispatch_cache[winning_id]
    
    print("\n" + "="*85)
    print(f"--- MULTI-CRITERIA DECISION: RECOMMENDED DESIGN {rec_cfg['cfg_id']} (Score: {rec_cfg.get('composite_score', 'N/A')}/100) ---")
    print("="*85)
    print(f"• Doublets: {rec_cfg['doublets']} (Wells: {rec_cfg['doublets']*2}) | Storage: {rec_cfg['storage_mwh']} MWh | Chiller: {rec_cfg['abs_chiller_mw']} MW Abs / {rec_cfg['elec_chiller_mw']} MW Elec")
    print(f"• CAPEX: €{rec_cfg['total_capex_m']:.2f}M | OPEX: Included | LCOE: €{rec_cfg['lcoe_eur_mwh']:.2f}/MWh | Coverage: {rec_cfg['coverage_pct']:.2f}%")
    print(f"• Rationale: {rec_cfg['justification']}")
    
    print("\n" + "-"*85)
    print(f"--- BEST ALTERNATIVE DESIGN: {alt_cfg['cfg_id']} ---")
    print("-"*85)
    print(f"• Doublets: {alt_cfg['doublets']} | Storage: {alt_cfg['storage_mwh']} MWh | LCOE: €{alt_cfg['lcoe_eur_mwh']:.2f}/MWh | Coverage: {alt_cfg['coverage_pct']:.2f}%")
    print(f"• Context for Selection: {alt_cfg['context_for_selection']}")
    
    print("\n" + "="*85)
    print("--- RIGOROUS TECHNICAL & ECONOMIC DEBATES ---")
    print("="*85)
    debates = decision.get("technical_debates", {})
    print(f"\n[1. STORAGE ASSESSMENT (175 MWh vs 50 MWh)]:\n{debates.get('storage_175mwh_justification')}")
    print(f"\n[2. ASSET CAPITAL EFFICIENCY (2 vs 3 Doublets & N-1 Redundancy)]:\n{debates.get('doublets_2_vs_3_tradeoff')}")
    print(f"\n[3. COOLING INTEGRATION & TURNDOWN MITIGATION]:\n{debates.get('cooling_hybrid_vs_electric')}")
    
    print("\n" + "-"*85)
    print("--- STUDY SCREENING LIMITATIONS ---")
    print("-"*85)
    for lim in decision.get("screening_limitations", []):
        print(f" • {lim}")
    print("="*85)
    
    # Tabla de Pozos usando los valores físicos de la simulación
    print("\n" + "="*85)
    print(f"--- WELL OPERATIONAL PROFILE & DISPATCH BREAKDOWN ({winning_id}) ---")
    print("="*85)
    well_prods = best_sim_data["well_production_gwh"]
    well_days = best_sim_data["well_active_days"]
    well_inactive = best_sim_data["well_inactive_days"]
    well_flows = best_sim_data["well_mean_flow_m3h"]
    
    well_summary = []
    for i in range(len(well_prods)):
        well_summary.append({
            "Doublet ID": f"Doublet {i+1}",
            "Annual Production (GWh)": well_prods[i],
            "Active Days": well_days[i],
            "Inactive Days (Shut-in)": well_inactive[i],
            "Mean Operational Flow (m3/h)": well_flows[i]
        })
    print(pd.DataFrame(well_summary).to_string(index=False))
    print("="*85)

    # 1. Dashboard ASCII en consola
    print_ascii_dashboard(rec_cfg, df_results, best_sim_data)
    
    # 2. Despliegue y guardado de gráficas interactivas
    plot_results_on_screen(df_results, winning_id, dispatch_cache)
    
    print("\n✅ Simulation and Multi-Criteria Evaluation completed successfully.")

if __name__ == "__main__":
    main()
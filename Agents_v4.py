import pandas as pd
from dataclasses import dataclass

import json
from pathlib import Path


print("SCRIPT STARTED")


# ==================================================
# DATA CLASSES
# ==================================================

@dataclass
class Demand:
    heating_peak_mw: float
    cooling_peak_mw: float
    heating_energy_gwh: float
    cooling_energy_gwh: float


@dataclass
class Resource:
    depth_m: float
    reservoir_temp_c: float
    pressure_bar: float


@dataclass
class Design:
    doublets: int
    geothermal_mw: float
    heat_pump_mw: float
    cooling_mw: float
    cooling_technology: str


# ==================================================
# DEMAND AGENT
# ==================================================

def demand_agent():

    heat = pd.read_csv(
        "GEMINI_Heating_Demand_Daily_2027.csv"
    )

    cool = pd.read_csv(
        "GEMINI_Cooling_Demand_Daily_2027.csv"
    )

    return Demand(
        heating_peak_mw=heat["heating_demand_MWth"].max(),
        cooling_peak_mw=cool["cooling_demand_MWcool"].max(),
        heating_energy_gwh=heat["heating_demand_MWth"].sum() * 24 / 1000,
        cooling_energy_gwh=cool["cooling_demand_MWcool"].sum() * 24 / 1000
    )


# ==================================================
# RESOURCE AGENT
# ==================================================

def resource_agent():

    depth = 2500

    reservoir_temp = 12 + 30 * (depth / 1000)

    return Resource(
        depth_m=depth,
        reservoir_temp_c=reservoir_temp,
        pressure_bar=250
    )


# ==================================================
# DESIGN AGENT
# ==================================================

def design_agent(demand):

    designs = []

    for doublets in [1, 2, 3]:

        for geothermal_fraction in [
            0.30,
            0.50,
            0.70,
            0.90
        ]:

            geothermal_mw = (
                demand.heating_peak_mw
                * geothermal_fraction
            )

            heat_pump_mw = max(
                0,
                demand.heating_peak_mw
                - geothermal_mw
            )

            for cooling_technology in [
                "electric chiller",
                "absorption chiller"
            ]:

                designs.append(

                    Design(
                        doublets=doublets,

                        geothermal_mw=
                        round(
                            geothermal_mw,
                            1
                        ),

                        heat_pump_mw=
                        round(
                            heat_pump_mw,
                            1
                        ),

                        cooling_mw=
                        demand.cooling_peak_mw,

                        cooling_technology=
                        cooling_technology
                    )

                )

    return designs

def export_scenarios_json(designs):

    folder = Path("scenarios")

    folder.mkdir(
        exist_ok=True
    )

    for i, design in enumerate(
            designs,
            start=1
    ):

        scenario = {

            "scenario_id": i,

            "doublets": design.doublets,

            "geothermal_mw": design.geothermal_mw,

            "heat_pump_mw": design.heat_pump_mw,

            "cooling_mw": design.cooling_mw,

            "cooling_technology":
            design.cooling_technology

        }

        filename = (
            folder /
            f"scenario_{i:03d}.json"
        )

        with open(
            filename,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                scenario,
                f,
                indent=4
            )

    print(
        f"{len(designs)} scenarios exported."
    )
# ==================================================
# VALIDATION AGENT
# ==================================================

def validation_agent(design, demand):

    violations = []

    # Heating demand
    total_heat = (
        design.geothermal_mw +
        design.heat_pump_mw
    )

    if total_heat < demand.heating_peak_mw:

        violations.append(
            "Heating demand not met"
        )

    # Cooling demand
    if design.cooling_mw < demand.cooling_peak_mw:

        violations.append(
            "Cooling demand not met"
        )

    # Doublet limits
    if design.doublets < 1:

        violations.append(
            "Less than 1 doublet"
        )

    if design.doublets > 3:

        violations.append(
            "More than 3 doublets"
        )

    # 87°C reservoir vs 100°C required supply
    if design.heat_pump_mw <= 0:

        violations.append(
            "No heat pump available for 100°C supply"
        )

    # Flow estimate
    geothermal_per_doublet = (
        design.geothermal_mw /
        design.doublets
    )

    # Approximation:
    # 14 MW ≈ 300 m³/h
    estimated_flow = (
        geothermal_per_doublet
        / 14.0
    ) * 300

    if estimated_flow > 300:

        violations.append(
            f"Flow too high ({estimated_flow:.0f} m3/h)"
        )

    if estimated_flow < 50:

        violations.append(
            f"Flow too low ({estimated_flow:.0f} m3/h)"
        )

    return violations

# ==================================================
# OPTIMIZATION AGENT
# ==================================================
def optimization_agent(designs, demand):

    best_design = None
    best_score = -999

    print("\n==============================")
    print("DESIGN COMPARISON")
    print("==============================")

    for design in designs:

        if not validation_agent(
            design,
            demand
        ):
            continue

        # --------------------------------
        # COST MODEL
        # --------------------------------

        doublet_cost = design.doublets * 25

        heatpump_cost = design.heat_pump_mw * 1

        cooling_cost = 0

        if design.cooling_technology == "absorption chiller":
            cooling_cost = 5

        total_cost_penalty = (
            doublet_cost
            + heatpump_cost
            + cooling_cost
        )

        # --------------------------------
        # EMISSIONS MODEL
        # --------------------------------

        emissions_benefit = (
            design.geothermal_mw * 2
        )

        # --------------------------------
        # SIMPLICITY SCORE
        # --------------------------------

        simplicity_score = (
            (4 - design.doublets) * 5
        )

        # --------------------------------
        # FINAL SCORE
        # --------------------------------

        score = (
            emissions_benefit
            + simplicity_score
            - total_cost_penalty
        )

        print("\nOPTION")
        print("----------------")

        print(
            f"Doublets = {design.doublets}"
        )

        print(
            f"Geothermal = {design.geothermal_mw} MW"
        )

        print(
            f"Heat Pump = {design.heat_pump_mw} MW"
        )

        print(
            f"Cooling = {design.cooling_technology}"
        )

        print(
            f"Emission Benefit = {emissions_benefit}"
        )

        print(
            f"Cost Penalty = {total_cost_penalty}"
        )

        print(
            f"Simplicity Score = {simplicity_score}"
        )

        print(
            f"FINAL SCORE = {score}"
        )

        if score > best_score:

            best_score = score
            best_design = design

    return best_design, best_score




# ==================================================
# REPORT AGENT
# ==================================================

def report_agent(
        demand,
        resource,
        design,
        score
):

    print()
    print("===================")
    print("FINAL REPORT")
    print("===================")

    print()
    print("DEMAND")

    print(
        "Heating Peak:",
        demand.heating_peak_mw,
        "MW"
    )

    print(
        "Cooling Peak:",
        demand.cooling_peak_mw,
        "MW"
    )

    print(
        "Heating Energy:",
        round(
            demand.heating_energy_gwh,
            2
        ),
        "GWh"
    )

    print(
        "Cooling Energy:",
        round(
            demand.cooling_energy_gwh,
            2
        ),
        "GWh"
    )

    print()
    print("RESOURCE")

    print(
        "Depth:",
        resource.depth_m,
        "m"
    )

    print(
        "Reservoir Temperature:",
        resource.reservoir_temp_c,
        "°C"
    )

    print()
    print("SELECTED DESIGN")

    print(
        "Doublets:",
        design.doublets
    )

    print(
        "Geothermal:",
        design.geothermal_mw,
        "MW"
    )

    print(
        "Heat Pump:",
        design.heat_pump_mw,
        "MW"
    )

    print(
        "Cooling:",
        design.cooling_mw,
        "MW"
    )

    print(
        "Technology:",
        design.cooling_technology
    )

    print()
    print(
        "Optimization Score:",
        score
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print("MAIN STARTED")

    demand = demand_agent()

    resource = resource_agent()

    designs = design_agent(
        demand
    )

    export_scenarios_json(
        designs
    )

    best_design, score = optimization_agent(
        designs,
        demand
    )

    report_agent(
        demand,
        resource,
        best_design,
        score
    )


if __name__ == "__main__":
    main()
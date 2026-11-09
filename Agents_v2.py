import pandas as pd
from dataclasses import dataclass


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

def design_agent():

    return [

        Design(
            doublets=2,
            geothermal_mw=15,
            heat_pump_mw=15,
            cooling_mw=10,
            cooling_technology="electric chiller"
        ),

        Design(
            doublets=2,
            geothermal_mw=20,
            heat_pump_mw=10,
            cooling_mw=10,
            cooling_technology="electric chiller"
        ),

        Design(
            doublets=3,
            geothermal_mw=25,
            heat_pump_mw=5,
            cooling_mw=10,
            cooling_technology="absorption chiller"
        )

    ]


# ==================================================
# VALIDATION AGENT
# ==================================================

def validation_agent(design, demand):

    total_heat = (
        design.geothermal_mw +
        design.heat_pump_mw
    )

    if total_heat < demand.heating_peak_mw:
        return False

    if design.cooling_mw < demand.cooling_peak_mw:
        return False

    return True


# ==================================================
# OPTIMIZATION AGENT
# ==================================================

def optimization_agent(
        designs,
        demand
):

    best_design = None
    best_score = -999

    for design in designs:

        if not validation_agent(
            design,
            demand
        ):
            continue

        score = design.geothermal_mw

        if (
            design.cooling_technology
            ==
            "absorption chiller"
        ):
            score += 5

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

    designs = design_agent()

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
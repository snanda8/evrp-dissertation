# Intelligent Methods for the Electric Vehicle Routing Problem (EVRP)

## Project Overview

This project investigates intelligent solution methods for the Electric Vehicle Routing Problem (EVRP), accounting for real-world constraints such as limited battery capacity, charging infrastructure, and vehicle load limits.

The system implements and evaluates:
- A Constructive Heuristic based on the Clarke and Wright Savings algorithm (CWS) adapted for EV constraints,
- A Genetic Algorithm (GA) approach, developed experimentally but challenged by feasibility issues.

Both approaches are supported by local search optimization, dynamic battery constraint handling, and automated evaluation tools.

---

## Features

- Modified Clarke and Wright heuristic supporting energy-based route feasibility
- Genetic Algorithm with crossover, mutation, and population repair (experimental)
- Comprehensive battery feasibility enforcement, including dynamic recharging station insertion
- Local search with 2-opt, relocate, and swap operators for route improvement
- Automatic multi-instance evaluation and result aggregation
- Route visualization using Matplotlib
- Modular and extensible Python codebase

---

## Repository Structure

```
EVRP-Solver/
├── constructive_solver.py        # Clarke & Wright constructive solution builder
├── mainscript_constructive.py     # Main script to run CWS experiments
├── mainscript.py                  # Main script to run GA experiments (experimental)
├── evaluate_solver.py             # Batch evaluation script for both CWS and GA
├── pipeline.py                    # Pipelines to orchestrate solving and evaluation
├── fitness.py                     # Fitness evaluation, penalties, feasibility scoring
├── ga_operators.py                 # Genetic Algorithm operators (crossover, mutation)
├── heuristics.py                   # Clustering and nearest-neighbour initialisation
├── instance_parser.py              # XML file parser for EVRP instances
├── local_search.py                 # Local search improvements and route plotting
├── merging.py                      # Route merging utilities
├── route_utils.py                  # Route cleaning and formatting
├── utils.py                        # Distance and energy calculation utilities
├── validation.py                   # Solution validation and repair
├── evrp_utils.py                   # Additional battery feasibility helpers
├── instance_files/                 # Folder containing problem instance XML files
├── plots/                          # Auto-generated visualizations of routes
├── evaluation_results.csv          # Generated output metrics for all runs
└── requirements.txt                # [To be generated if not already present]
```

---

## Setup Instructions

1. **Python version:** 3.8 or higher
2. **Dependencies:**
```bash
pip install matplotlib numpy scikit-learn
```
(Alternatively: use `pip install -r requirements.txt` once generated.)

3. **Prepare Data:**
   Ensure the `instance_files/` directory contains valid EVRP instance `.xml` files.

---

## Running the Solvers

### Constructive Solver (CWS)

To run the Clarke-Wright-based constructive solver across selected instances:

```bash
python mainscript_constructive.py
```

This will:
- Build an initial solution,
- Repair battery infeasibility by inserting charging stations,
- Apply local search to improve routes,
- Save evaluation metrics into `evaluation_results.csv`,
- Generate visual route plots under `plots/`.

### Genetic Algorithm (Experimental)

The Genetic Algorithm approach can be run via:

```bash
python mainscript.py
```

**Note:**  
- Due to the inherent difficulty of maintaining feasibility across large EVRP instances, the GA results may be infeasible or suboptimal.  
- This module is included for completeness of exploration but is not the main method evaluated in the dissertation.

---

## Evaluation and Results

All experiments write evaluation outputs to `evaluation_results.csv`, including:
- Instance Name
- Solving Method (CWS or GA)
- Fitness Score
- Battery Feasibility (Yes/No)
- Number of Routes
- Number of Vehicles Used
- Comments (e.g., whether GA results were feasible)

Route plots are saved in the `plots/` directory, illustrating customer, depot, and charging station connections. 
If you wish to see each route graph that is generated per instance upon running either mainscript.py or mainscript_constructive.py, then set save_plot to False in either of the plot_routes function calls in mainscript.py or mainscript_constructive.py



---

## Known Limitations

- **GA feasibility:** The Genetic Algorithm does not always produce feasible routes without further repair, especially for large or complex instances.
- **Repair strategies:** Battery repair heuristics prioritize feasibility over strict optimality and may introduce slight inefficiencies.
- **Scalability:** Larger instance sizes may lead to longer runtime, particularly with GA operations and complex battery checks.

---

## Project Details

- **Title:** Intelligent Methods for the Electric Vehicle Routing Problem
- **Student Name:** Sarthak Shalin Nanda
- **Student ID:** 229007926
- **University:** University of Leicester
- **Module:** CO3201 Computer Science Project


---

## References

Key algorithms, heuristics, and methodology sources are referenced within the accompanying dissertation and cited appropriately according to university guidelines.

---

## Academic Integrity Statement

All work was independently produced according to the University's regulations on academic integrity. External resources have been fully acknowledged and cited.

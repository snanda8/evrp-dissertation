# CO3201 Computer Science Project Log
**Project Title**: Intelligent Methods for the Electric Vehicle Routing Problem (EVRP)  
**Student Name**: Sarthak Shalin Nanda  
**Student ID**: 229007926  


---

## January 2025
- **15 Jan**: First experimentation with Mixed Integer Linear Programming (MILP) for solving EVRP considering basic constraints (battery, distance).
- **19 Jan**: Initial implementation of a basic Genetic Algorithm (GA) for EVRP. Developed a basic auto-route generation to replace hard-coded instances.
- **28 Jan**: Added initial vehicle route generation capabilities; foundational work on fitness evaluation began.

## February 2025
- **8 Feb**: Developed initial versions of fitness functions; partial implementation of a MILP solver.
- **16 Feb**: Implemented crossover and mutation with elitism in the GA to avoid premature convergence.
- **17 Feb**: Minor edits to MILP model structure.
- **Throughout Feb**: Committed code regularly, focusing on population seeding heuristics.

## March 2025
- **Early March**: Modularisation of code base to improve structure and maintainability.
- **17-31 Mar**: Faced and fixed issues related to infeasible heuristics—started pivoting towards a constructive solver approach using Clarke and Wright Savings (CWS) heuristic.

## April 2025
- **1 Apr**: Completed final attempt at fixing feasibility with older methods; pivoted decisively towards Clarke and Wright constructive approaches.
- **6-7 Apr**: Implemented Clarke and Wright heuristic; initial successful routes emerging, albeit needing optimisation.
- **8-9 Apr**: Iterations led to improvements in feasibility; cleaned and clarified fitness functions.
- **10 Apr**: Added intra-route 2-opt swap to improve individual routes.
- **16-17 Apr**: Developed and integrated a custom evaluation module for feasibility and cost; debugging efforts included visual enhancement of plotted routes.
- **18 Apr**: Created route image generation scripts; incremental improvements to feasibility via GA route validation.
- **17-23 Apr**: Progressed evaluation modules; ensured evaluation pipelines were separated from main solvers for cleaner analysis.
- **23 Apr**: Fixes applied for CWS result feasibility; developed route plotting tools.
- **24 Apr**: Refactored main scripts to allow automated result export to CSV for easier benchmarking; extended GA scripts to handle multiple targeted instances.

---

## Summary of Key Milestones
| Date         | Task                                                                                   | Status        |
|--------------|----------------------------------------------------------------------------------------|---------------|
| Jan 2025     | Basic GA and MILP prototypes                                                            | Completed     |
| Feb 2025     | Developed advanced GA mechanisms (crossover, elitism)                                   | Completed     |
| Mar 2025     | Realised limitations of heuristic approaches; pivoted to constructive CWS solver        | Completed     |
| Apr 2025     | Implemented CWS and integrated multi-instance evaluations; improved visualisation       | Completed     |
| Late Apr 2025| Final refactoring of solvers and evaluation pipeline; extensive benchmarking setup      | Completed     |

---

## Reflection on Challenges
- **Feasibility issues**: Early GA approaches struggled with battery and capacity constraints, leading to a necessary pivot towards constructive heuristics.
- **Debugging and Evaluation**: Designing a standalone evaluation module was challenging but critical for modular and scalable solution assessment.
- **Visualisation Enhancements**: Significantly improved graph plotting, aiding deeper route and solution quality analysis.

---

## Planned Next Steps (before dissertation submission)
- **System refinement**: Final minor tweaks for visualisations and CSV output formats to streamline analysis.
- **Prepare mini viva presentation**: Summarise the year's work into a 5-minute technical presentation with supporting slides.

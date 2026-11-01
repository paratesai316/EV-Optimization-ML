# Physics-Guided Machine Learning for EV Energy Consumption

Surrogate modelling, explainability and multi-objective design optimization of electric vehicles (EVs), built on real drive-cycle simulation.

---

## 1. Aim of the study

**Goal:** find electric-vehicle designs that use as little energy as possible per 100 km, *without* ignoring what that efficiency costs or the requirements the vehicle must meet, and do it fast enough to explore thousands of design options.

Specifically, the study asks:

1. Which design parameters drive EV energy consumption, and by how much?
2. Can a machine-learning surrogate stand in for the physics simulation accurately enough to search the design space?
3. When energy, cost, range and performance are weighed together, what do the best designs look like for **city**, **highway** and **mixed** driving?
4. Are those answers trustworthy, i.e. do they hold up against the physics model, and how sensitive are they to assumptions that were chosen by judgement?

## 2. The problem and how this pipeline solves it

### The difficulty

Energy use depends on mass, aerodynamic drag, rolling resistance, regenerative braking, powertrain sizing and the driving cycle, and these interact. Searching that space by hand, or by running the full simulation for every candidate, is slow. A naive "minimize energy" search is also uninformative: lighter, slimmer, lower-drag cars are *always* better for energy, so every search ends at the same extreme corner of the design space.

### The approach

| Problem | How the pipeline handles it |
|---|---|
| Simulation is slow for large searches | Train an ML **surrogate** on simulated data, then optimize on the surrogate. |
| Mass was independent of battery and motor size (physically wrong) | **Mass coupling**: only the base glider mass is sampled. Battery and motor mass are derived from their size, so sizing decisions act on energy through mass. |
| Energy-only optimization is trivial | **Multi-objective optimization**: minimize `energy + λ · cost`, sweeping `λ` to trace an energy-versus-cost frontier. |
| Designs must be realistic | Two **constraints**: a minimum range (250 km) and a minimum power-to-weight ratio (0.08 kW/kg). |
| Surrogates are least accurate at the extremes, exactly where an optimizer goes | **Surrogate refinement**: sample extra training data around the optimized designs, retrain, repeat. |
| A surrogate's optimum can be slightly wrong | **Motor trim** and **physics polish**: a short local search on the physics model finishes the job, and every design is **verified with physics**. |
| Cost model and requirements are judgement calls | **Sensitivity analysis**: vary each assumption one at a time and measure how the conclusions move. |

### What the pipeline does, end to end

1. **Define the design space**: seven continuous parameters (base glider mass, battery capacity, motor power, drag coefficient, frontal area, rolling resistance, regenerative efficiency) plus the driving cycle.
2. **Sample it**: 20,000 designs by Latin hypercube sampling.
3. **Simulate**: each design on three drive cycles (UDDS city, HWFET highway, mixed) with a longitudinal dynamics and powertrain model, giving 60,000 labelled rows (energy in kWh/100 km, range, total mass).
4. **Train and benchmark five surrogates** (Linear Regression, Random Forest, SVR, MLP neural network, XGBoost) with 5-fold cross-validation, and keep the best one automatically.
5. **Explain it** with SHAP to see which parameters matter.
6. **Optimize**: differential evolution (several random seeds) minimizes `energy + λ·cost` under the range and power-to-weight constraints, for each cycle and each `λ`.
7. **Post-process and verify**: shrink the motor to the smallest feasible size, polish the design with a local physics-model search, and re-simulate it with the physics engine.
8. **Refine the surrogate** around the optimized designs and repeat step 6 until the surrogate is accurate where it matters.
9. **Test robustness** with a one-at-a-time sensitivity analysis.

The physics engine uses the real UDDS and HWFET speed traces bundled with NREL's FASTSim. The vehicle model itself is implemented directly in `fastsim_engine.py`, because the current FASTSim release does not expose the flat, settable vehicle attributes that an automated parameter sweep needs.

---

## 3. How to run the pipeline

### Run order

Run everything from one folder containing all the `.py` files. File names must be exactly as below (no leading numbers), because the scripts import each other.

```bash
# 0. install dependencies (once)
pip install numpy pandas scipy scikit-learn xgboost shap matplotlib tqdm fastsim

# 1. generate the simulated dataset
python generate_dataset.py

# 2. train and benchmark the surrogates, keep the best
python train_surrogate_models.py

# 3. explain the selected surrogate with SHAP
python explainability_shap.py

# 4. optimize + verify + refine the surrogate (repeats optimization each round)
python refine_surrogate.py

# 5. test how robust the conclusions are to the assumptions
python sensitivity_analysis.py --model best_ev_surrogate_refined.pkl
```

If you only want a single optimization pass with no refinement, replace step 4 with:

```bash
python optimize_and_validate.py
```

Steps 1, 2 and 3 must run in that order. Step 4 needs the outputs of steps 1 and 2. Step 5 needs the surrogate from step 4 (or from step 2, with `--model best_ev_surrogate.pkl`).

### What each file does

| File | Role | Run it? |
|---|---|---|
| `vehicle_design.py` | **Single source of truth** for the design space: parameter bounds, mass-coupling relations, relative cost model, power-to-weight floor and required range. Imported by every other script. | No (importable; `python vehicle_design.py` prints sanity checks) |
| `fastsim_engine.py` | **Physics engine**: loads real drive cycles and computes energy consumption, range and total mass for a design. Falls back to smooth synthetic cycles with a warning if FASTSim is missing. | No (importable; `python fastsim_engine.py` runs a smoke test) |
| `generate_dataset.py` | **Step 1.** Latin hypercube sampling plus parallel simulation. | Yes |
| `train_surrogate_models.py` | **Step 2.** Cleans data, benchmarks five models, saves the best. | Yes |
| `explainability_shap.py` | **Step 3.** SHAP feature importance and an interaction plot. | Yes |
| `optimize_and_validate.py` | Constrained multi-objective optimization, motor trim, physics polish and physics verification. Also used by steps 4 and 5. | Optional (single pass) |
| `refine_surrogate.py` | **Step 4.** Runs the optimization, then adds training data around the optimized designs, retrains and repeats. | Yes |
| `sensitivity_analysis.py` | **Step 5.** One-at-a-time variation of the performance floor, required range and every cost coefficient. | Yes |

### What each step produces

| Step | Output files |
|---|---|
| 1 | `ev_energy_dataset.csv` (60,000 rows) |
| 2 | `best_ev_surrogate.pkl`, `surrogate_benchmark_results.csv`, `ev_energy_dataset_clean.csv` |
| 3 | `shap_feature_importance.csv`, `shap_global_summary.png`, `shap_cd_area_interaction.png` |
| 4 | `optimization_validation_results_round0.csv` ... `round3.csv` (one per round), `best_ev_surrogate_refined.pkl`, `ev_energy_dataset_refined.csv`, `refinement_history.csv` |
| optimize only | `optimization_validation_results.csv` |
| 5 | `sensitivity_results.csv`, `sensitivity_summary.csv` |

The **final optimized designs** are in the last `optimization_validation_results_round*.csv` (or `optimization_validation_results.csv` for a single pass). The designs are physics-polished, so the frontier is essentially the same in every round; the rounds differ in how accurate the surrogate is.

### Useful options

```bash
# faster trial: one cycle, few weights, one seed
python optimize_and_validate.py --cycles city --lambdas 0 1e-4 5e-4 --seeds 42

# change the requirements
python optimize_and_validate.py --gamma-min 0.10 --required-range 300

# pure-surrogate answer (no motor trim, no physics polish), to measure the surrogate alone
python optimize_and_validate.py --no-trim --no-polish

# refinement: more rounds, more points, different stopping targets
python refine_surrogate.py --rounds 5 --points-per-design 500 \
       --target-max-error 1.0 --target-max-gap 0.3

# sensitivity analysis at a different trade-off weight, one cycle
python sensitivity_analysis.py --model best_ev_surrogate_refined.pkl \
       --cycles city --lambda-ref 1e-3
```

All options are listed by `python <script>.py --help`.

---

## 4. Reading the results

### Key columns in the optimization results

| Column | Meaning |
|---|---|
| `scenario`, `lambda` | Driving cycle and the weight on cost (`0` = pure energy minimum, larger = cheaper designs). |
| `physics_true_kwh_100km`, `relative_cost` | Energy (from the physics engine) and cost of the final design. Plot cost against energy to see the frontier. |
| `battery_kwh`, `motor_kw`, `base_glider_mass`, `cd`, `area_m2`, `crr`, `regen_pct` | The optimal design. |
| `feasible` | Both constraints are satisfied **under the physics engine**. |
| `range_km`, `power_to_weight_kw_per_kg` | Verified range and power-to-weight. |
| `error_pct` | Surrogate prediction error at the final design (versus physics). |
| `surrogate_only_gap_pct` | How much worse the surrogate's *own* answer was than the final design, in the physics objective. This is the measure of surrogate quality for optimization. Blank when polish is off. |
| `seed_spread_pct` | Disagreement between random seeds (small = robust search). |
| `motor_trim_kw`, `polish_applied`, `polish_start` | What the post-processing changed. |
| `pareto_optimal`, `duplicate_of_lower_lambda` | Non-dominated within the sweep; duplicate designs are labelled, not hidden. |

### Things to keep in mind

- **The cost is an illustrative, unit-less score**, not a market price. Cost-dependent conclusions are conditional on the cost coefficients in `vehicle_design.py`; the sensitivity analysis shows how much they move.
- **Battery size comes from the range requirement, and motor size from the power-to-weight floor.** The other variables (glider mass, drag, frontal area, rolling resistance, regenerative efficiency) are what the optimizer trades against cost.
- **Absolute energy values are optimistic** (fixed efficiencies, a constant 350 W auxiliary load, no climate load or road grade). Rely on relative comparisons.
- **The mixed cycle** is UDDS followed by HWFET, not an official WLTC trace.

---

## 5. Where to change assumptions

| What | Where |
|---|---|
| Parameter bounds | `PARAM_RANGES` in `vehicle_design.py` |
| Battery, motor and payload mass relations | `BATTERY_MASS_*`, `MOTOR_MASS_*`, `FIXED_PAYLOAD_MASS_KG` in `vehicle_design.py` (illustrative calibration of the functional form in Ramakrishnan, Mastinu & Gobbi, 2019) |
| Cost model | `COST_COEFFICIENTS` in `vehicle_design.py` |
| Power-to-weight floor, required range | `MIN_POWER_TO_WEIGHT_KW_PER_KG`, `REQUIRED_RANGE_KM` in `vehicle_design.py` (or `--gamma-min`, `--required-range`) |
| Efficiencies, auxiliary load, regen cap | constants at the top of `fastsim_engine.py` |
| Number of designs | `n_samples` in `generate_dataset.py` |
| Trade-off weights, seeds, optimizer settings | `LAMBDA_SWEEP`, `SEEDS`, `DE_*` in `optimize_and_validate.py` |
| Models and hyperparameters | `evaluate_models` in `train_surrogate_models.py` |

If you change anything in `vehicle_design.py` or `fastsim_engine.py` that affects the physics or the design bounds, **re-run from step 1**: the dataset, surrogate and results from earlier runs no longer match.

---

## 6. Requirements and reproducibility

- **Python** 3.10 or newer (tested with 3.12).
- **Packages:** numpy, pandas, scipy (1.9 or newer; tested with 1.17), scikit-learn, xgboost, shap, matplotlib, tqdm, and fastsim (tested with 3.0 and 3.1). Without `fastsim` the engine still runs on smooth synthetic drive cycles, but prints a warning and the results are not based on real EPA cycles.
- **Reproducibility:** every random element uses a fixed seed (design sampling 42, train/test split 42, optimizer seeds 42 to 44), so repeating the pipeline on the same package versions gives the same dataset and the same frontier.

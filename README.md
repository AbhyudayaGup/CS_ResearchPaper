# Traveling Sailsman (TSP) — ACO demo

This project provides a simple educational implementation of Ant Colony Optimization (ACO) for the Traveling Salesman Problem (TSP), with a Streamlit-based interactive visualization.

Current UI behavior:
- Left sidebar controls only TSP instance generation (cities/seed/layout/time limit for exact solver).
- ACO settings are independent and configured from the right panel via `Configure ACO`.
- Every time a new TSP instance is generated, the app automatically computes an exact/optimal route and shows it below the city map.
- Right panel includes an algorithm dropdown with implemented `ACO` and `PSO` (Bee Colony placeholder).
- After each model run, the app shows comparison cards:
	- `Convergence time (s)`
	- `Evals to convergence`
	- `Total objective evals`
- Each run is appended to `results/model_comparison_runs.csv` for cross-model benchmarking.

Quick start:

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Streamlit app:

```powershell
streamlit run web/app.py
```

Important: run that command from the project root folder `C:\Users\abhyu\Documents\coding\CS_ResearchPaper`, not from inside `web/`.

If `streamlit` is not found on PATH, use this instead:

```powershell
python -m streamlit run web/app.py
```

3. Or run a single CLI experiment:

```powershell
python -m src.runner --num-cities 20 --iterations 200 --two-opt
```

Virtual environment and activation (Windows PowerShell):

```powershell
python -m venv .venv
# Activate in PowerShell
.\.venv\Scripts\Activate.ps1
# Install dependencies
pip install -r requirements.txt
```

Virtual environment and activation (Windows CMD):

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Virtual environment and activation (macOS / Linux / WSL):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run tests (after installing requirements):

```bash
pytest -q
```

Run a parameter sweep example (will save CSV to `results/`):

```bash
python -m src.experiment
``` 

Exact solver notes:
- For very small instances, the app uses brute force.
- For larger instances (up to 50 cities), it uses an exact CP-SAT model (`ortools`) with a time limit.
- If optimality is proven, status is `OPTIMAL`; otherwise `FEASIBLE` means a valid route was found within the time limit.

Notes:
- Use `streamlit run web/app.py` to start the interactive UI and adjust parameters from the sidebar.
- Output files and experiment CSVs are saved under the `results/` folder by default.

Files:
- `src/aco.py` — AntColony implementation with optional 2-opt.
- `src/pso.py` — Particle Swarm Optimization (random-key encoding) with optional 2-opt.
- `src/exact_solver.py` — exact TSP solver (brute force + CP-SAT up to 50 cities).
- `src/utils.py` — random instance generator and distance matrix.
- `web/app.py` — Streamlit UI to run and visualize ACO/PSO with comparison metrics and CSV logging.
- `src/runner.py` — simple CLI runner that saves results to `results/`.

If you'd like, I'll implement tests, parameter-sweep harness, GIF export, and polish the UI next.

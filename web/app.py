from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st

from src.aco import AntColony
from src.exact_solver import solve_tsp_exact
from src.utils import generate_cities
from src.pso import ParticleSwarm


st.set_page_config(layout="wide", page_title="TSP Exact vs ACO")


def _init_state() -> None:
    defaults = {
        "coords": None,
        "exact_result": None,
        "needs_exact": True,
        "show_aco_settings": False,
        "aco_num_ants": 20,
        "aco_iterations": 200,
        "aco_alpha": 1.0,
        "aco_beta": 5.0,
        "aco_rho": 0.5,
        "aco_q": 100.0,
        "aco_two_opt": True,
        "aco_seed": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

st.title("TSP Visual Lab: Exact vs ACO")

# Left sidebar: TSP generation only
st.sidebar.header("TSP Instance Generator")
num_cities = st.sidebar.number_input("Number of cities", min_value=4, max_value=50, value=20)
instance_seed = st.sidebar.number_input("Instance seed (0=random)", min_value=0, value=0, step=1)
clustered = st.sidebar.checkbox("Clustered city layout", value=False)
exact_timeout = st.sidebar.slider("Exact solver time limit (seconds)", min_value=10, max_value=300, value=90)

if st.sidebar.button("Generate New TSP Instance", type="primary"):
    seed = None if instance_seed == 0 else int(instance_seed)
    st.session_state["coords"] = generate_cities(int(num_cities), seed=seed, clustered=clustered)
    st.session_state["needs_exact"] = True

if st.session_state["coords"] is None:
    seed = None if instance_seed == 0 else int(instance_seed)
    st.session_state["coords"] = generate_cities(int(num_cities), seed=seed, clustered=clustered)
    st.session_state["needs_exact"] = True

coords = st.session_state["coords"]

if st.session_state.get("needs_exact", True):
    with st.spinner("Computing exact/optimal route for this TSP instance..."):
        try:
            st.session_state["exact_result"] = solve_tsp_exact(coords, max_seconds=int(exact_timeout))
        except Exception as exc:
            st.session_state["exact_result"] = {
                "error": str(exc),
                "tour": None,
                "length": None,
                "is_optimal": False,
                "method": "failed",
                "status": "ERROR",
                "solve_time_s": 0.0,
            }
    st.session_state["needs_exact"] = False

exact_result = st.session_state.get("exact_result")


def _tour_xy(tour):
    idx = list(tour) + [tour[0]]
    xs = [float(coords[i, 0]) for i in idx]
    ys = [float(coords[i, 1]) for i in idx]
    return xs, ys


def _base_city_trace(fig: go.Figure) -> None:
    fig.add_trace(
        go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers+text",
            text=[str(i) for i in range(len(coords))],
            textposition="top center",
            name="Cities",
        )
    )


map_col, right_col = st.columns([3.2, 1.6])

with map_col:
    fig = go.Figure()
    _base_city_trace(fig)
    if exact_result and exact_result.get("tour"):
        x_opt, y_opt = _tour_xy(exact_result["tour"])
        fig.add_trace(
            go.Scatter(
                x=x_opt,
                y=y_opt,
                mode="lines",
                line=dict(width=4, color="#1f77b4"),
                name="Exact Route",
            )
        )
    fig.update_layout(title="Generated TSP Instance", height=560)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Exact/Optimal Route (Auto-computed)")
    if exact_result and exact_result.get("error"):
        st.error(f"Exact solver failed: {exact_result['error']}")
    elif exact_result:
        c1, c2, c3 = st.columns(3)
        c1.metric("Exact route length", f"{exact_result['length']:.3f}")
        c2.metric("Status", exact_result.get("status", "UNKNOWN"))
        c3.metric("Solve time (s)", f"{exact_result.get('solve_time_s', 0.0):.2f}")
        st.caption(
            f"Method: {exact_result.get('method', 'unknown')} | Optimal proven: {exact_result.get('is_optimal', False)}"
        )

with right_col:
    st.subheader("Algorithm Runner")
    algorithm = st.selectbox(
        "Choose algorithm",
        [
            "ACO (Ant Colony Optimization)",
            "Bee Colony (Coming Soon)",
            "Particle Swarm Optimization",
        ],
        index=0,
    )

    b1, b2 = st.columns(2)
    run_clicked = b1.button("Run", type="primary")
    if b2.button("Configure ACO"):
        st.session_state["show_aco_settings"] = not st.session_state["show_aco_settings"]

    if st.session_state.get("show_aco_settings", False):
        st.markdown("### ACO Settings")
        st.session_state["aco_num_ants"] = st.number_input(
            "Number of ants",
            min_value=1,
            max_value=1000,
            value=int(st.session_state["aco_num_ants"]),
        )
        st.session_state["aco_iterations"] = st.number_input(
            "Iterations",
            min_value=1,
            max_value=5000,
            value=int(st.session_state["aco_iterations"]),
        )
        st.session_state["aco_alpha"] = st.slider(
            "alpha (pheromone importance)",
            min_value=0.1,
            max_value=5.0,
            value=float(st.session_state["aco_alpha"]),
        )
        st.session_state["aco_beta"] = st.slider(
            "beta (distance heuristic importance)",
            min_value=0.1,
            max_value=10.0,
            value=float(st.session_state["aco_beta"]),
        )
        st.session_state["aco_rho"] = st.slider(
            "rho (evaporation)",
            min_value=0.01,
            max_value=0.99,
            value=float(st.session_state["aco_rho"]),
        )
        st.session_state["aco_q"] = st.number_input(
            "Q (deposit scale)",
            min_value=0.1,
            value=float(st.session_state["aco_q"]),
        )
        st.session_state["aco_two_opt"] = st.checkbox(
            "Use 2-opt local search",
            value=bool(st.session_state["aco_two_opt"]),
        )
        st.session_state["aco_seed"] = st.number_input(
            "ACO seed (0=random)",
            min_value=0,
            value=int(st.session_state["aco_seed"]),
            step=1,
        )

    # PSO settings (shown when PSO selected)
    if algorithm == "Particle Swarm Optimization":
        st.markdown("### PSO Settings")
        if "pso_num_particles" not in st.session_state:
            st.session_state["pso_num_particles"] = max(10, int(num_cities // 2))
        st.session_state["pso_num_particles"] = st.number_input(
            "Number of particles",
            min_value=2,
            max_value=2000,
            value=int(st.session_state["pso_num_particles"]),
        )
        if "pso_iterations" not in st.session_state:
            st.session_state["pso_iterations"] = 200
        st.session_state["pso_iterations"] = st.number_input(
            "Iterations",
            min_value=1,
            max_value=5000,
            value=int(st.session_state["pso_iterations"]),
        )
        if "pso_w" not in st.session_state:
            st.session_state["pso_w"] = 0.5
        st.session_state["pso_w"] = st.slider("Inertia (w)", min_value=0.0, max_value=1.5, value=float(st.session_state["pso_w"]))
        if "pso_c1" not in st.session_state:
            st.session_state["pso_c1"] = 1.5
        st.session_state["pso_c1"] = st.slider("Cognitive (c1)", min_value=0.0, max_value=3.0, value=float(st.session_state["pso_c1"]))
        if "pso_c2" not in st.session_state:
            st.session_state["pso_c2"] = 1.5
        st.session_state["pso_c2"] = st.slider("Social (c2)", min_value=0.0, max_value=3.0, value=float(st.session_state["pso_c2"]))
        if "pso_two_opt" not in st.session_state:
            st.session_state["pso_two_opt"] = True
        st.session_state["pso_two_opt"] = st.checkbox("Use 2-opt local search", value=bool(st.session_state["pso_two_opt"]))
        if "pso_seed" not in st.session_state:
            st.session_state["pso_seed"] = 0
        st.session_state["pso_seed"] = st.number_input("PSO seed (0=random)", min_value=0, value=int(st.session_state["pso_seed"]), step=1)

    result_placeholder = st.empty()
    chart_placeholder = st.empty()

    if run_clicked:
        if algorithm == "ACO (Ant Colony Optimization)":
            ac_seed = None if int(st.session_state["aco_seed"]) == 0 else int(st.session_state["aco_seed"])
            ac = AntColony(
                coords,
                n_ants=int(st.session_state["aco_num_ants"]),
                n_iterations=int(st.session_state["aco_iterations"]),
                alpha=float(st.session_state["aco_alpha"]),
                beta=float(st.session_state["aco_beta"]),
                rho=float(st.session_state["aco_rho"]),
                Q=float(st.session_state["aco_q"]),
                apply_two_opt=bool(st.session_state["aco_two_opt"]),
                seed=ac_seed,
            )
            progress = st.progress(0)

            def cb(iteration, best_len, best_tour, convergence_iteration=None):
                progress.progress(int((iteration + 1) / ac.n_iterations * 100))
                if best_tour is None:
                    return
                fx = go.Figure()
                _base_city_trace(fx)
                x_aco, y_aco = _tour_xy(best_tour)
                fx.add_trace(
                    go.Scatter(
                        x=x_aco,
                        y=y_aco,
                        mode="lines",
                        line=dict(width=4, color="#d62728"),
                        name="ACO Best",
                    )
                )
                if exact_result and exact_result.get("tour"):
                    x_opt, y_opt = _tour_xy(exact_result["tour"])
                    fx.add_trace(
                        go.Scatter(
                            x=x_opt,
                            y=y_opt,
                            mode="lines",
                            line=dict(width=2, color="#1f77b4", dash="dash"),
                            name="Exact",
                        )
                    )
                fx.update_layout(title=f"ACO progress (iteration {iteration})", height=420)
                chart_placeholder.plotly_chart(fx, width="stretch")

            run_result = ac.run(callback=cb)
            best_tour = best_len = history = convergence_iteration = best_found_iter = None
            if isinstance(run_result, tuple):
                if len(run_result) == 3:
                    best_tour, best_len, history = run_result
                elif len(run_result) == 4:
                    best_tour, best_len, history, convergence_iteration = run_result
                elif len(run_result) >= 5:
                    best_tour, best_len, history, convergence_iteration, best_found_iter = run_result[:5]

            gap_text = ""
            if exact_result and exact_result.get("length"):
                opt_len = float(exact_result["length"])
                gap = ((float(best_len) - opt_len) / opt_len) * 100.0
                gap_text = f" | Optimality gap: {gap:.3f}%"

            iter_text = ""
            if best_found_iter is not None:
                iter_text = f" | Best found at iteration: {best_found_iter}"
            elif convergence_iteration is not None:
                iter_text = f" | Converged at iteration: {convergence_iteration}"

            result_placeholder.success(
                f"ACO best length: {best_len:.3f}{iter_text}{gap_text}"
            )
        elif algorithm == "Particle Swarm Optimization":
            pso_seed = None if int(st.session_state["pso_seed"]) == 0 else int(st.session_state["pso_seed"])
            pso = ParticleSwarm(
                coords,
                n_particles=int(st.session_state["pso_num_particles"]),
                n_iterations=int(st.session_state["pso_iterations"]),
                w=float(st.session_state["pso_w"]),
                c1=float(st.session_state["pso_c1"]),
                c2=float(st.session_state["pso_c2"]),
                apply_two_opt=bool(st.session_state["pso_two_opt"]),
                seed=pso_seed,
            )
            progress = st.progress(0)

            def cb_pso(iteration, best_len, best_tour, convergence_iteration=None):
                progress.progress(int((iteration + 1) / pso.n_iterations * 100))
                if best_tour is None:
                    return
                fx = go.Figure()
                _base_city_trace(fx)
                x_pso, y_pso = _tour_xy(best_tour)
                fx.add_trace(
                    go.Scatter(
                        x=x_pso,
                        y=y_pso,
                        mode="lines",
                        line=dict(width=4, color="#2ca02c"),
                        name="PSO Best",
                    )
                )
                if exact_result and exact_result.get("tour"):
                    x_opt, y_opt = _tour_xy(exact_result["tour"])
                    fx.add_trace(
                        go.Scatter(
                            x=x_opt,
                            y=y_opt,
                            mode="lines",
                            line=dict(width=2, color="#1f77b4", dash="dash"),
                            name="Exact",
                        )
                    )
                fx.update_layout(title=f"PSO progress (iteration {iteration})", height=420)
                chart_placeholder.plotly_chart(fx, width="stretch")

            run_result = pso.run(callback=cb_pso)
            best_tour = best_len = history = convergence_iteration = best_found_iter = None
            if isinstance(run_result, tuple):
                if len(run_result) == 3:
                    best_tour, best_len, history = run_result
                elif len(run_result) == 4:
                    best_tour, best_len, history, convergence_iteration = run_result
                elif len(run_result) >= 5:
                    best_tour, best_len, history, convergence_iteration, best_found_iter = run_result[:5]

            gap_text = ""
            if exact_result and exact_result.get("length"):
                opt_len = float(exact_result["length"])
                gap = ((float(best_len) - opt_len) / opt_len) * 100.0
                gap_text = f" | Optimality gap: {gap:.3f}%"

            iter_text = ""
            if best_found_iter is not None:
                iter_text = f" | Best found at iteration: {best_found_iter}"
            elif convergence_iteration is not None:
                iter_text = f" | Converged at iteration: {convergence_iteration}"

            result_placeholder.success(
                f"PSO best length: {best_len:.3f}{iter_text}{gap_text}"
            )
        else:
            st.info("Selected algorithm is a placeholder. ACO and PSO are implemented; others will be added next.")

from pathlib import Path
import sys
import inspect
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st

from src.aco import AntColony
from src.abc import ArtificialBeeColony
from src.exact_solver import solve_tsp_exact
from src.ga import GeneticAlgorithm
from src.utils import generate_cities
from src.pso import ParticleSwarm


st.set_page_config(layout="wide", page_title="TSP Visual Lab")


def _run_optimizer_with_optional_stats(optimizer, callback, target_length=None):
    run_fn = getattr(optimizer, "run")
    try:
        sig = inspect.signature(run_fn)
        kwargs = {"callback": callback}
        if "return_stats" in sig.parameters:
            kwargs["return_stats"] = True
        if target_length is not None and "target_length" in sig.parameters:
            kwargs["target_length"] = target_length
        if target_length is not None and "target_tolerance" in sig.parameters:
            kwargs["target_tolerance"] = 1e-9
        return run_fn(**kwargs)
    except (TypeError, ValueError):
        # If signature inspection fails for any reason, fall back safely.
        pass
    return run_fn(callback=callback)


def _algorithm_short_name(algorithm: str) -> str:
    if algorithm == "ACO (Ant Colony Optimization)":
        return "ACO"
    if algorithm == "Artificial Bee Colony":
        return "ABC"
    if algorithm == "Genetic Algorithm":
        return "GA"
    if algorithm == "Particle Swarm Optimization":
        return "PSO"
    return "Model"


def _fallback_stats(model: str, history, best_len, best_found_iter, convergence_iteration, elapsed_s, population_size):
    hist_len = len(history) if history else 0
    if hist_len == 0:
        best_iter = 0
    elif best_found_iter is not None:
        best_iter = int(best_found_iter)
    elif convergence_iteration is not None:
        best_iter = int(convergence_iteration)
    else:
        best_iter = next((i for i, v in enumerate(history) if abs(float(v) - float(best_len)) < 1e-12), hist_len - 1)

    if hist_len > 0:
        frac = float(best_iter + 1) / float(hist_len)
    else:
        frac = 1.0
    convergence_time_s = float(max(0.0, elapsed_s * min(1.0, max(0.0, frac))))

    if model == "ACO":
        objective_evals_total = int(max(1, population_size * max(1, hist_len)))
        objective_evals_to_convergence = int(max(1, population_size * (best_iter + 1)))
    else:
        # PSO includes initial particle evaluation before iteration loop.
        objective_evals_total = int(max(1, population_size * (max(1, hist_len) + 1)))
        objective_evals_to_convergence = int(max(1, population_size * (best_iter + 1)))

    return {
        "run_time_s": float(elapsed_s),
        "convergence_time_s": convergence_time_s,
        "objective_evals_total": objective_evals_total,
        "objective_evals_to_convergence": objective_evals_to_convergence,
        "best_found_iteration": int(best_iter),
    }


def _init_state() -> None:
    defaults = {
        "coords": None,
        "exact_result": None,
        "needs_exact": True,
        "show_algorithm_settings": False,
        "selected_algorithm_prev": None,
        "last_run": None,
        "aco_num_ants": 20,
        "aco_iterations": 200,
        "aco_alpha": 1.0,
        "aco_beta": 5.0,
        "aco_rho": 0.5,
        "aco_q": 100.0,
        "aco_two_opt": True,
        "aco_seed": 0,
        "abc_num_food_sources": 20,
        "abc_iterations": 200,
        "abc_limit": 60,
        "abc_two_opt": True,
        "abc_seed": 0,
        "ga_population_size": 30,
        "ga_iterations": 200,
        "ga_crossover_rate": 0.9,
        "ga_mutation_rate": 0.2,
        "ga_elite_fraction": 0.1,
        "ga_tournament_size": 3,
        "ga_two_opt": True,
        "ga_seed": 0,
        "pso_num_particles": 20,
        "pso_iterations": 200,
        "pso_w": 0.5,
        "pso_c1": 1.5,
        "pso_c2": 1.5,
        "pso_two_opt": True,
        "pso_seed": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

st.title("TSP Visual Lab: ACO / ABC / GA / PSO Explorer")

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


def _build_run_insights(run_data: dict) -> list[str]:
    insights = []
    gap = run_data.get("optimality_gap_pct")
    if gap is None:
        insights.append("No optimality-gap insight yet because exact baseline is unavailable for this run.")
    elif gap <= 1.0:
        insights.append(f"Solution quality is strong: gap is {gap:.3f}%, which is near-optimal.")
    elif gap <= 3.0:
        insights.append(f"Solution quality is moderate: gap is {gap:.3f}%. Try more iterations for tighter routes.")
    else:
        insights.append(f"Solution quality is weak: gap is {gap:.3f}%. Increase iterations/population and rerun.")

    iterations = int(run_data.get("iterations", 1))
    best_iter = int(run_data.get("best_found_iteration", 0))
    progress_ratio = (best_iter + 1) / max(1, iterations)
    if progress_ratio <= 0.25:
        insights.append("Convergence happened early in the run, indicating fast search dynamics for this instance.")
    elif progress_ratio <= 0.70:
        insights.append("Convergence happened mid-run, indicating balanced exploration and exploitation.")
    else:
        insights.append("Convergence happened late; this model needed most iterations to stabilize.")

    conv_time = float(run_data.get("convergence_time_s", 0.0))
    evals_conv = int(run_data.get("objective_evals_to_convergence", 0))
    eval_rate = evals_conv / max(conv_time, 1e-9)
    insights.append(f"Compute intensity to convergence: {eval_rate:,.0f} objective evals/second.")
    return insights


map_col, right_col = st.columns([1.15, 2.35])

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
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Exact Baseline")
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
    st.subheader("Algorithm Workspace")
    algorithm = st.selectbox(
        "Choose algorithm",
        [
            "ACO (Ant Colony Optimization)",
            "Artificial Bee Colony",
            "Genetic Algorithm",
            "Particle Swarm Optimization",
        ],
        index=0,
    )

    b1, b2 = st.columns(2)
    run_clicked = b1.button("Run Algorithm", type="primary", use_container_width=True)

    prev_algorithm = st.session_state.get("selected_algorithm_prev")
    if prev_algorithm is not None and prev_algorithm != algorithm:
        st.session_state["show_algorithm_settings"] = False
    st.session_state["selected_algorithm_prev"] = algorithm

    configure_label = f"Configure {_algorithm_short_name(algorithm)}"
    if b2.button(configure_label):
        st.session_state["show_algorithm_settings"] = not st.session_state.get("show_algorithm_settings", False)

    if st.session_state.get("show_algorithm_settings", False):
        st.markdown("---")
        if algorithm == "ACO (Ant Colony Optimization)":
            st.markdown("### ACO Settings")
            st.session_state["aco_num_ants"] = st.number_input(
                "Number of ants",
                min_value=1,
                max_value=1000,
                value=int(st.session_state["aco_num_ants"]),
                key="aco_num_ants_input",
            )
            st.session_state["aco_iterations"] = st.number_input(
                "Iterations",
                min_value=1,
                max_value=5000,
                value=int(st.session_state["aco_iterations"]),
                key="aco_iterations_input",
            )
            st.session_state["aco_alpha"] = st.slider(
                "alpha (pheromone importance)",
                min_value=0.1,
                max_value=5.0,
                value=float(st.session_state["aco_alpha"]),
                key="aco_alpha_input",
            )
            st.session_state["aco_beta"] = st.slider(
                "beta (distance heuristic importance)",
                min_value=0.1,
                max_value=10.0,
                value=float(st.session_state["aco_beta"]),
                key="aco_beta_input",
            )
            st.session_state["aco_rho"] = st.slider(
                "rho (evaporation)",
                min_value=0.01,
                max_value=0.99,
                value=float(st.session_state["aco_rho"]),
                key="aco_rho_input",
            )
            st.session_state["aco_q"] = st.number_input(
                "Q (deposit scale)",
                min_value=0.1,
                value=float(st.session_state["aco_q"]),
                key="aco_q_input",
            )
            st.session_state["aco_two_opt"] = st.checkbox(
                "Use 2-opt local search",
                value=bool(st.session_state["aco_two_opt"]),
                key="aco_two_opt_input",
            )
            st.session_state["aco_seed"] = st.number_input(
                "ACO seed (0=random)",
                min_value=0,
                value=int(st.session_state["aco_seed"]),
                step=1,
                key="aco_seed_input",
            )
        elif algorithm == "Artificial Bee Colony":
            st.markdown("### ABC Settings")
            st.session_state["abc_num_food_sources"] = st.number_input(
                "Number of food sources",
                min_value=2,
                max_value=2000,
                value=int(st.session_state["abc_num_food_sources"]),
                key="abc_num_food_sources_input",
            )
            st.session_state["abc_iterations"] = st.number_input(
                "Iterations",
                min_value=1,
                max_value=5000,
                value=int(st.session_state["abc_iterations"]),
                key="abc_iterations_input",
            )
            st.session_state["abc_limit"] = st.number_input(
                "Scout limit",
                min_value=1,
                max_value=20000,
                value=int(st.session_state["abc_limit"]),
                key="abc_limit_input",
                help="How many failed attempts a food source can survive before being replaced by a scout bee.",
            )
            st.session_state["abc_two_opt"] = st.checkbox(
                "Use 2-opt local search",
                value=bool(st.session_state["abc_two_opt"]),
                key="abc_two_opt_input",
            )
            st.session_state["abc_seed"] = st.number_input(
                "ABC seed (0=random)",
                min_value=0,
                value=int(st.session_state["abc_seed"]),
                step=1,
                key="abc_seed_input",
            )
        elif algorithm == "Genetic Algorithm":
            st.markdown("### GA Settings")
            st.session_state["ga_population_size"] = st.number_input(
                "Population size",
                min_value=4,
                max_value=5000,
                value=int(st.session_state["ga_population_size"]),
                key="ga_population_size_input",
            )
            st.session_state["ga_iterations"] = st.number_input(
                "Iterations",
                min_value=1,
                max_value=5000,
                value=int(st.session_state["ga_iterations"]),
                key="ga_iterations_input",
            )
            st.session_state["ga_crossover_rate"] = st.slider(
                "Crossover rate",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["ga_crossover_rate"]),
                key="ga_crossover_rate_input",
            )
            st.session_state["ga_mutation_rate"] = st.slider(
                "Mutation rate",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["ga_mutation_rate"]),
                key="ga_mutation_rate_input",
            )
            st.session_state["ga_elite_fraction"] = st.slider(
                "Elite fraction",
                min_value=0.0,
                max_value=0.5,
                value=float(st.session_state["ga_elite_fraction"]),
                key="ga_elite_fraction_input",
            )
            st.session_state["ga_tournament_size"] = st.number_input(
                "Tournament size",
                min_value=2,
                max_value=50,
                value=int(st.session_state["ga_tournament_size"]),
                key="ga_tournament_size_input",
            )
            st.session_state["ga_two_opt"] = st.checkbox(
                "Use 2-opt local search",
                value=bool(st.session_state["ga_two_opt"]),
                key="ga_two_opt_input",
            )
            st.session_state["ga_seed"] = st.number_input(
                "GA seed (0=random)",
                min_value=0,
                value=int(st.session_state["ga_seed"]),
                step=1,
                key="ga_seed_input",
            )
        elif algorithm == "Particle Swarm Optimization":
            st.markdown("### PSO Settings")
            st.session_state["pso_num_particles"] = st.number_input(
                "Number of particles",
                min_value=2,
                max_value=2000,
                value=int(st.session_state["pso_num_particles"]),
                key="pso_num_particles_input",
            )
            st.session_state["pso_iterations"] = st.number_input(
                "Iterations",
                min_value=1,
                max_value=5000,
                value=int(st.session_state["pso_iterations"]),
                key="pso_iterations_input",
            )
            st.session_state["pso_w"] = st.slider("Inertia (w)", min_value=0.0, max_value=1.5, value=float(st.session_state["pso_w"]), key="pso_w_input")
            st.session_state["pso_c1"] = st.slider("Cognitive (c1)", min_value=0.0, max_value=3.0, value=float(st.session_state["pso_c1"]), key="pso_c1_input")
            st.session_state["pso_c2"] = st.slider("Social (c2)", min_value=0.0, max_value=3.0, value=float(st.session_state["pso_c2"]), key="pso_c2_input")
            st.session_state["pso_two_opt"] = st.checkbox("Use 2-opt local search", value=bool(st.session_state["pso_two_opt"]), key="pso_two_opt_input")
            st.session_state["pso_seed"] = st.number_input("PSO seed (0=random)", min_value=0, value=int(st.session_state["pso_seed"]), step=1, key="pso_seed_input")

    st.markdown("---")
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

            run_start = time.perf_counter()
            run_result = _run_optimizer_with_optional_stats(ac, cb, target_length=float(exact_result["length"]) if exact_result and exact_result.get("length") else None)
            elapsed_s = time.perf_counter() - run_start
            best_tour = best_len = history = convergence_iteration = best_found_iter = stats = None
            if isinstance(run_result, tuple):
                if len(run_result) == 3:
                    best_tour, best_len, history = run_result
                elif len(run_result) == 4:
                    best_tour, best_len, history, convergence_iteration = run_result
                elif len(run_result) >= 5:
                    best_tour, best_len, history, convergence_iteration, best_found_iter = run_result[:5]
                if len(run_result) >= 6:
                    stats = run_result[5]
            if stats is None:
                stats = _fallback_stats(
                    model="ACO",
                    history=history,
                    best_len=best_len,
                    best_found_iter=best_found_iter,
                    convergence_iteration=convergence_iteration,
                    elapsed_s=elapsed_s,
                    population_size=int(st.session_state["aco_num_ants"]),
                )

            gap_text = ""
            gap_value = None
            if exact_result and exact_result.get("length"):
                opt_len = float(exact_result["length"])
                gap = ((float(best_len) - opt_len) / opt_len) * 100.0
                gap_text = f" | Optimality gap: {gap:.3f}%"
                gap_value = float(gap)

            iter_text = ""
            if best_found_iter is not None:
                iter_text = f" | Best found at iteration: {best_found_iter}"
            elif convergence_iteration is not None:
                iter_text = f" | Converged at iteration: {convergence_iteration}"

            run_summary = {
                "model": "ACO",
                "best_len": float(best_len),
                "optimality_gap_pct": gap_value,
                "best_found_iteration": int(stats["best_found_iteration"]),
                "convergence_iteration": int(convergence_iteration) if convergence_iteration is not None else None,
                "convergence_time_s": float(stats["convergence_time_s"]),
                "objective_evals_to_convergence": int(stats["objective_evals_to_convergence"]),
                "objective_evals_total": int(stats["objective_evals_total"]),
                "run_time_s": float(stats["run_time_s"]),
                "iterations_executed": int(stats.get("iterations_executed", st.session_state["aco_iterations"])),
                "stopped_early": bool(stats.get("stopped_early", False)),
                "iterations": int(st.session_state["aco_iterations"]),
                "note": f"Best length {best_len:.3f}{iter_text}{gap_text}",
            }
            st.session_state["last_run"] = run_summary

        elif algorithm == "Artificial Bee Colony":
            abc_seed = None if int(st.session_state.get("abc_seed", 0)) == 0 else int(st.session_state.get("abc_seed", 0))
            abc = ArtificialBeeColony(
                coords,
                n_food_sources=int(st.session_state.get("abc_num_food_sources", 20)),
                n_iterations=int(st.session_state.get("abc_iterations", 200)),
                limit=int(st.session_state.get("abc_limit", 60)),
                apply_two_opt=bool(st.session_state.get("abc_two_opt", True)),
                seed=abc_seed,
            )
            progress = st.progress(0)

            def cb_abc(iteration, best_len, best_tour, convergence_iteration=None):
                progress.progress(int((iteration + 1) / abc.n_iterations * 100))
                if best_tour is None:
                    return
                fx = go.Figure()
                _base_city_trace(fx)
                x_abc, y_abc = _tour_xy(best_tour)
                fx.add_trace(
                    go.Scatter(
                        x=x_abc,
                        y=y_abc,
                        mode="lines",
                        line=dict(width=4, color="#ff7f0e"),
                        name="ABC Best",
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
                fx.update_layout(title=f"ABC progress (iteration {iteration})", height=420)
                chart_placeholder.plotly_chart(fx, width="stretch")

            run_start = time.perf_counter()
            run_result = _run_optimizer_with_optional_stats(abc, cb_abc, target_length=float(exact_result["length"]) if exact_result and exact_result.get("length") else None)
            elapsed_s = time.perf_counter() - run_start
            best_tour = best_len = history = convergence_iteration = best_found_iter = stats = None
            if isinstance(run_result, tuple):
                if len(run_result) == 3:
                    best_tour, best_len, history = run_result
                elif len(run_result) == 4:
                    best_tour, best_len, history, convergence_iteration = run_result
                elif len(run_result) >= 5:
                    best_tour, best_len, history, convergence_iteration, best_found_iter = run_result[:5]
                if len(run_result) >= 6:
                    stats = run_result[5]
            if stats is None:
                stats = _fallback_stats(
                    model="ABC",
                    history=history,
                    best_len=best_len,
                    best_found_iter=best_found_iter,
                    convergence_iteration=convergence_iteration,
                    elapsed_s=elapsed_s,
                    population_size=int(st.session_state.get("abc_num_food_sources", 20)),
                )

            gap_text = ""
            gap_value = None
            if exact_result and exact_result.get("length"):
                opt_len = float(exact_result["length"])
                gap = ((float(best_len) - opt_len) / opt_len) * 100.0
                gap_text = f" | Optimality gap: {gap:.3f}%"
                gap_value = float(gap)

            iter_text = ""
            if best_found_iter is not None:
                iter_text = f" | Best found at iteration: {best_found_iter}"
            elif convergence_iteration is not None:
                iter_text = f" | Converged at iteration: {convergence_iteration}"

            run_summary = {
                "model": "ABC",
                "best_len": float(best_len),
                "optimality_gap_pct": gap_value,
                "best_found_iteration": int(stats["best_found_iteration"]),
                "convergence_iteration": int(convergence_iteration) if convergence_iteration is not None else None,
                "convergence_time_s": float(stats["convergence_time_s"]),
                "objective_evals_to_convergence": int(stats["objective_evals_to_convergence"]),
                "objective_evals_total": int(stats["objective_evals_total"]),
                "run_time_s": float(stats["run_time_s"]),
                "iterations_executed": int(stats.get("iterations_executed", st.session_state.get("abc_iterations", 200))),
                "stopped_early": bool(stats.get("stopped_early", False)),
                "iterations": int(st.session_state.get("abc_iterations", 200)),
                "note": f"Best length {best_len:.3f}{iter_text}{gap_text}",
            }
            st.session_state["last_run"] = run_summary

        elif algorithm == "Genetic Algorithm":
            ga_seed = None if int(st.session_state.get("ga_seed", 0)) == 0 else int(st.session_state.get("ga_seed", 0))
            ga = GeneticAlgorithm(
                coords,
                n_population=int(st.session_state.get("ga_population_size", 30)),
                n_iterations=int(st.session_state.get("ga_iterations", 200)),
                crossover_rate=float(st.session_state.get("ga_crossover_rate", 0.9)),
                mutation_rate=float(st.session_state.get("ga_mutation_rate", 0.2)),
                elite_fraction=float(st.session_state.get("ga_elite_fraction", 0.1)),
                tournament_size=int(st.session_state.get("ga_tournament_size", 3)),
                apply_two_opt=bool(st.session_state.get("ga_two_opt", True)),
                seed=ga_seed,
            )
            progress = st.progress(0)

            def cb_ga(iteration, best_len, best_tour, convergence_iteration=None):
                progress.progress(int((iteration + 1) / ga.n_iterations * 100))
                if best_tour is None:
                    return
                fx = go.Figure()
                _base_city_trace(fx)
                x_ga, y_ga = _tour_xy(best_tour)
                fx.add_trace(
                    go.Scatter(
                        x=x_ga,
                        y=y_ga,
                        mode="lines",
                        line=dict(width=4, color="#9467bd"),
                        name="GA Best",
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
                fx.update_layout(title=f"GA progress (iteration {iteration})", height=420)
                chart_placeholder.plotly_chart(fx, width="stretch")

            run_start = time.perf_counter()
            run_result = _run_optimizer_with_optional_stats(ga, cb_ga, target_length=float(exact_result["length"]) if exact_result and exact_result.get("length") else None)
            elapsed_s = time.perf_counter() - run_start
            best_tour = best_len = history = convergence_iteration = best_found_iter = stats = None
            if isinstance(run_result, tuple):
                if len(run_result) == 3:
                    best_tour, best_len, history = run_result
                elif len(run_result) == 4:
                    best_tour, best_len, history, convergence_iteration = run_result
                elif len(run_result) >= 5:
                    best_tour, best_len, history, convergence_iteration, best_found_iter = run_result[:5]
                if len(run_result) >= 6:
                    stats = run_result[5]
            if stats is None:
                stats = _fallback_stats(
                    model="GA",
                    history=history,
                    best_len=best_len,
                    best_found_iter=best_found_iter,
                    convergence_iteration=convergence_iteration,
                    elapsed_s=elapsed_s,
                    population_size=int(st.session_state.get("ga_population_size", 30)),
                )

            gap_text = ""
            gap_value = None
            if exact_result and exact_result.get("length"):
                opt_len = float(exact_result["length"])
                gap = ((float(best_len) - opt_len) / opt_len) * 100.0
                gap_text = f" | Optimality gap: {gap:.3f}%"
                gap_value = float(gap)

            iter_text = ""
            if best_found_iter is not None:
                iter_text = f" | Best found at iteration: {best_found_iter}"
            elif convergence_iteration is not None:
                iter_text = f" | Converged at iteration: {convergence_iteration}"

            run_summary = {
                "model": "GA",
                "best_len": float(best_len),
                "optimality_gap_pct": gap_value,
                "best_found_iteration": int(stats["best_found_iteration"]),
                "convergence_iteration": int(convergence_iteration) if convergence_iteration is not None else None,
                "convergence_time_s": float(stats["convergence_time_s"]),
                "objective_evals_to_convergence": int(stats["objective_evals_to_convergence"]),
                "objective_evals_total": int(stats["objective_evals_total"]),
                "run_time_s": float(stats["run_time_s"]),
                "iterations_executed": int(stats.get("iterations_executed", st.session_state.get("ga_iterations", 200))),
                "stopped_early": bool(stats.get("stopped_early", False)),
                "iterations": int(st.session_state.get("ga_iterations", 200)),
                "note": f"Best length {best_len:.3f}{iter_text}{gap_text}",
            }
            st.session_state["last_run"] = run_summary

        elif algorithm == "Particle Swarm Optimization":
            pso_seed = None if int(st.session_state.get("pso_seed", 0)) == 0 else int(st.session_state.get("pso_seed", 0))
            pso = ParticleSwarm(
                coords,
                n_particles=int(st.session_state.get("pso_num_particles", 20)),
                n_iterations=int(st.session_state.get("pso_iterations", 200)),
                w=float(st.session_state.get("pso_w", 0.5)),
                c1=float(st.session_state.get("pso_c1", 1.5)),
                c2=float(st.session_state.get("pso_c2", 1.5)),
                apply_two_opt=bool(st.session_state.get("pso_two_opt", True)),
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

            run_start = time.perf_counter()
            run_result = _run_optimizer_with_optional_stats(pso, cb_pso, target_length=float(exact_result["length"]) if exact_result and exact_result.get("length") else None)
            elapsed_s = time.perf_counter() - run_start
            best_tour = best_len = history = convergence_iteration = best_found_iter = stats = None
            if isinstance(run_result, tuple):
                if len(run_result) == 3:
                    best_tour, best_len, history = run_result
                elif len(run_result) == 4:
                    best_tour, best_len, history, convergence_iteration = run_result
                elif len(run_result) >= 5:
                    best_tour, best_len, history, convergence_iteration, best_found_iter = run_result[:5]
                if len(run_result) >= 6:
                    stats = run_result[5]
            if stats is None:
                stats = _fallback_stats(
                    model="PSO",
                    history=history,
                    best_len=best_len,
                    best_found_iter=best_found_iter,
                    convergence_iteration=convergence_iteration,
                    elapsed_s=elapsed_s,
                    population_size=int(st.session_state.get("pso_num_particles", 20)),
                )

            gap_text = ""
            gap_value = None
            if exact_result and exact_result.get("length"):
                opt_len = float(exact_result["length"])
                gap = ((float(best_len) - opt_len) / opt_len) * 100.0
                gap_text = f" | Optimality gap: {gap:.3f}%"
                gap_value = float(gap)

            iter_text = ""
            if best_found_iter is not None:
                iter_text = f" | Best found at iteration: {best_found_iter}"
            elif convergence_iteration is not None:
                iter_text = f" | Converged at iteration: {convergence_iteration}"

            run_summary = {
                "model": "PSO",
                "best_len": float(best_len),
                "optimality_gap_pct": gap_value,
                "best_found_iteration": int(stats["best_found_iteration"]),
                "convergence_iteration": int(convergence_iteration) if convergence_iteration is not None else None,
                "convergence_time_s": float(stats["convergence_time_s"]),
                "objective_evals_to_convergence": int(stats["objective_evals_to_convergence"]),
                "objective_evals_total": int(stats["objective_evals_total"]),
                "run_time_s": float(stats["run_time_s"]),
                "iterations_executed": int(stats.get("iterations_executed", st.session_state.get("pso_iterations", 200))),
                "stopped_early": bool(stats.get("stopped_early", False)),
                "iterations": int(st.session_state.get("pso_iterations", 200)),
                "note": f"Best length {best_len:.3f}{iter_text}{gap_text}",
            }
            st.session_state["last_run"] = run_summary

        else:
            st.info("Selected algorithm is not available.")

    run_data = st.session_state.get("last_run")
    if run_data:
        st.markdown("### Latest Run Summary")
        st.success(run_data.get("note", "Run complete."))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model", run_data["model"])
        m2.metric("Best route length", f"{run_data['best_len']:.3f}")
        if run_data.get("optimality_gap_pct") is None:
            m3.metric("Optimality gap", "N/A")
        else:
            m3.metric("Optimality gap", f"{run_data['optimality_gap_pct']:.3f}%")
        m4.metric("Convergence time", f"{run_data['convergence_time_s']:.4f} s")

        m5, m6, m7 = st.columns(3)
        m5.metric("Best found iteration", f"{run_data['best_found_iteration']}")
        m6.metric("Evals to convergence", f"{run_data['objective_evals_to_convergence']:,}")
        m7.metric("Total evals", f"{run_data['objective_evals_total']:,}")

        st.markdown("### Stop Condition")
        if run_data.get("stopped_early"):
            st.success(
                f"Stopped early after {run_data['iterations_executed']} iterations because the model matched the exact route length within tolerance."
            )
        else:
            st.info(
                f"No exact match was reached, so the model used the full budget of {run_data['iterations']} iterations."
            )

        st.markdown("### Insights")
        insights = _build_run_insights(run_data)
        for item in insights:
            st.markdown(f"- {item}")
    else:
        st.info("Run an algorithm to see formatted summary cards and performance insights.")

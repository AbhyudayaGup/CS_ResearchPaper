from __future__ import annotations

from pathlib import Path
import threading
import time
import sys
import inspect
from html import escape
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.comparison import AVAILABLE_ALGORITHMS, build_insights, build_model_summary, parse_int_list, run_comparison_batch


st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .hero-card {
            background: linear-gradient(135deg, rgba(18,24,38,0.96), rgba(28,38,62,0.96));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 28px 30px;
            box-shadow: 0 18px 60px rgba(0,0,0,0.28);
            margin-bottom: 18px;
        }
        .hero-title {
            font-size: 2.05rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin: 0 0 8px 0;
            color: #f5f7fb;
        }
        .hero-subtitle {
            color: rgba(245,247,251,0.75);
            font-size: 0.98rem;
            line-height: 1.55;
            max-width: 980px;
        }
        .pill-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 16px;
        }
        .pill {
            background: rgba(255,255,255,0.07);
            color: #eaf0ff;
            border: 1px solid rgba(255,255,255,0.09);
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 0.86rem;
        }
        .panel-card {
            background: rgba(12, 16, 25, 0.9);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 18px 18px 12px 18px;
            margin-bottom: 16px;
            box-shadow: 0 12px 32px rgba(0,0,0,0.18);
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(22,28,44,0.98), rgba(15,20,32,0.98));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 14px 16px;
        }
        .metric-label {
            font-size: 0.76rem;
            color: rgba(235,241,255,0.68);
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 1.55rem;
            font-weight: 800;
            color: #ffffff;
        }
        .metric-note {
            margin-top: 6px;
            color: rgba(235,241,255,0.68);
            font-size: 0.86rem;
        }
        .race-card {
            background: linear-gradient(135deg, rgba(10,16,27,0.96), rgba(22,31,50,0.96));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 16px 18px 18px 18px;
            margin-bottom: 14px;
            box-shadow: 0 14px 36px rgba(0,0,0,0.22);
        }
        .race-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            color: #eef4ff;
            font-weight: 800;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }
        .race-subtitle {
            color: rgba(235,241,255,0.72);
            font-size: 0.88rem;
            margin-bottom: 12px;
        }
        .race-track {
            position: relative;
            height: 20px;
            border-radius: 999px;
            overflow: hidden;
            background: linear-gradient(90deg, rgba(255,255,255,0.04), rgba(255,255,255,0.11), rgba(255,255,255,0.04));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
        }
        .race-track::before {
            content: "";
            position: absolute;
            inset: 0;
            background: repeating-linear-gradient(
                90deg,
                rgba(255,255,255,0.04) 0,
                rgba(255,255,255,0.04) 16px,
                transparent 16px,
                transparent 32px
            );
            animation: track-shift 1.1s linear infinite;
        }
        .race-fill {
            position: absolute;
            inset: 0 auto 0 0;
            background: linear-gradient(90deg, #34d399 0%, #fbbf24 52%, #fb7185 100%);
            box-shadow: 0 0 18px rgba(52,211,153,0.45);
        }
        .race-runner {
            position: absolute;
            top: -8px;
            transform: translateX(-50%);
            font-size: 1.15rem;
            filter: drop-shadow(0 2px 8px rgba(0,0,0,0.45));
            animation: runner-bob 0.8s ease-in-out infinite alternate;
        }
        .race-finish {
            position: absolute;
            top: -10px;
            right: 4px;
            font-size: 1.15rem;
            filter: drop-shadow(0 2px 8px rgba(0,0,0,0.45));
        }
        .race-meter {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-top: 8px;
            color: rgba(235,241,255,0.8);
            font-size: 0.84rem;
        }
        .race-flag {
            color: #fde68a;
            font-weight: 700;
        }
        @keyframes runner-bob {
            from { transform: translateX(-50%) translateY(0); }
            to { transform: translateX(-50%) translateY(-3px); }
        }
        @keyframes track-shift {
            from { transform: translateX(0); }
            to { transform: translateX(-32px); }
        }
        .table-wrap {
            overflow-x: auto;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(10,14,22,0.92);
        }
        table.comparison-table {
            width: 100%;
            border-collapse: collapse;
            color: #eff3fb;
            font-size: 0.88rem;
        }
        table.comparison-table th,
        table.comparison-table td {
            padding: 12px 10px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            white-space: nowrap;
        }
        table.comparison-table thead th {
            position: sticky;
            top: 0;
            background: #101725;
            z-index: 1;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.74rem;
            color: rgba(239,243,251,0.78);
        }
        table.comparison-table tbody tr:nth-child(odd) {
            background: rgba(255,255,255,0.02);
        }
        .badge-ok { color: #72ef9f; font-weight: 700; }
        .badge-warn { color: #ffd166; font-weight: 700; }
        .badge-off { color: #9aa6c1; font-weight: 700; }
        .insight-list li {
            margin-bottom: 10px;
            color: rgba(239,243,251,0.92);
            line-height: 1.55;
        }
        .muted { color: rgba(239,243,251,0.7); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _parse_models() -> list[str]:
    st.markdown("### Models to compare")
    st.caption("Each model is shown explicitly so the full set stays visible even when all are selected.")
    selected: list[str] = []
    for row_index in range(0, len(AVAILABLE_ALGORITHMS), 2):
        columns = st.columns(2)
        for column, spec in zip(columns, AVAILABLE_ALGORITHMS[row_index : row_index + 2]):
            with column:
                label = f"{spec.label} ({'ready' if spec.runnable else 'coming soon'})"
                checked = st.checkbox(label, value=spec.runnable, key=f"model_{spec.label.lower()}_selected")
                if checked:
                    selected.append(spec.label)
    return selected


def _render_metric_cards(summary_rows: list[dict]) -> None:
    if not summary_rows:
        st.info("Run a batch to generate summary cards.")
        return
    best_time = min(summary_rows, key=lambda row: row["avg_convergence_time_s"])
    best_gap = min(summary_rows, key=lambda row: row["avg_gap_pct"])
    best_evals = min(summary_rows, key=lambda row: row["avg_evals_to_convergence"])
    best_hits = max(summary_rows, key=lambda row: row["optimal_hits"])
    total_runs = sum(row["runs"] for row in summary_rows)
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">Fastest Average Convergence</div><div class="metric-value">{escape(best_time['model'])}</div><div class="metric-note">{best_time['avg_convergence_time_s']:.4f} s</div></div>
            <div class="metric-card"><div class="metric-label">Best Average Quality</div><div class="metric-value">{escape(best_gap['model'])}</div><div class="metric-note">{best_gap['avg_gap_pct']:.3f}% gap</div></div>
            <div class="metric-card"><div class="metric-label">Lowest Eval Effort</div><div class="metric-value">{escape(best_evals['model'])}</div><div class="metric-note">{best_evals['avg_evals_to_convergence']:.0f} evals</div></div>
            <div class="metric-card"><div class="metric-label">Total Completed Runs</div><div class="metric-value">{total_runs}</div><div class="metric-note">Across all selected configurations</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Most exact matches: {best_hits['model']} ({best_hits['optimal_hits']} / {best_hits['runs']})."
    )


def _render_table(rows: list[dict], selected_models: list[str]) -> None:
    if not rows:
        st.warning("No completed runs to display.")
        return

    # Build grouped table by scenario, then one column block per model.
    grouped: dict[tuple[int, int], dict[str, dict]] = {}
    for row in rows:
        key = (int(row["city_count"]), int(row["config_value"]))
        grouped.setdefault(key, {})[row["model"]] = row

    header_models = [model for model in selected_models]
    html = ["<div class='table-wrap'><table class='comparison-table'>"]
    html.append("<thead>")
    html.append("<tr><th rowspan='2'>Cities</th><th rowspan='2'>Config</th><th rowspan='2'>Exact Length</th><th rowspan='2'>Exact Status</th>")
    for model in header_models:
        html.append(f"<th colspan='6'>{escape(model)}</th>")
    html.append("</tr><tr>")
    for _ in header_models:
        html.append("<th>Time (s)</th><th>Converged</th><th>Early Stop</th><th>Iters Used</th><th>Evals</th><th>Gap %</th>")
    html.append("</tr></thead><tbody>")

    for (city_count, config_value), model_map in sorted(grouped.items()):
        exact_row = next((row for row in rows if int(row["city_count"]) == city_count and int(row["config_value"]) == config_value), None)
        exact_length = "N/A"
        exact_status = "N/A"
        if exact_row is not None and exact_row.get("exact_length") is not None:
            exact_length = f"{float(exact_row['exact_length']):.3f}"
            exact_status = str(exact_row.get("exact_status", "N/A"))

        html.append(f"<tr><td>{city_count}</td><td>{config_value}</td><td>{exact_length}</td><td>{escape(exact_status)}</td>")
        for model in header_models:
            model_row = model_map.get(model)
            if model_row is None:
                html.append("<td colspan='4' class='badge-off'>N/A</td>")
                continue
            status = model_row.get("status")
            if status != "complete":
                html.append(f"<td colspan='6' class='badge-off'>{escape(str(status))}</td>")
                continue
            converged = "Yes" if model_row.get("optimal_match") else "No"
            converged_class = "badge-ok" if model_row.get("optimal_match") else "badge-warn"
            gap_value = model_row.get("optimality_gap_pct")
            gap_text = "N/A" if gap_value is None else f"{float(gap_value):.3f}"
            early_stop = "Yes" if model_row.get("stopped_early") else "No"
            early_class = "badge-ok" if model_row.get("stopped_early") else "badge-off"
            iters_used = int(model_row.get("iterations_executed") or 0)
            html.append(
                f"<td>{float(model_row['run_time_s']):.4f}</td><td class='{converged_class}'>{converged}</td><td class='{early_class}'>{early_stop}</td><td>{iters_used}</td><td>{int(model_row['objective_evals_to_convergence']):,}</td><td>{gap_text}</td>"
            )
        html.append("</tr>")

    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_insights(rows: list[dict]) -> None:
    insights = build_insights(rows)
    st.markdown("### What the report says")
    st.markdown("<ul class='insight-list'>" + "".join(f"<li>{escape(text)}</li>" for text in insights) + "</ul>", unsafe_allow_html=True)


def _render_race_banner(target, progress_pct: float, headline: str, detail: str, batch_label: str, batch_suffix: str = "") -> None:
    progress_pct = max(0.0, min(100.0, progress_pct))
    runner_left = max(2.0, min(98.0, progress_pct))
    target.markdown(
        f"""
        <div class="race-card">
            <div class="race-title">
                <span>🏃 {escape(headline)}</span>
                <span class="race-flag">{progress_pct:.1f}%</span>
            </div>
            <div class="race-subtitle">{escape(detail)}</div>
            <div class="race-track">
                <div class="race-fill" style="width: {progress_pct:.1f}%;"></div>
                <div class="race-runner" style="left: {runner_left:.1f}%;">🏃‍➡️</div>
                <div class="race-finish">🏁</div>
            </div>
            <div class="race-meter">
                <span>{escape(batch_label)}</span>
                <span>{escape(batch_suffix)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _call_run_comparison_batch(**kwargs):
    signature = inspect.signature(run_comparison_batch)
    filtered = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return run_comparison_batch(**filtered)


_inject_css()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Model Comparison Dashboard</div>
        <div class="hero-subtitle">
            Run the active algorithms across multiple TSP sizes and parameter configurations, then inspect one consolidated report.
            The dashboard is designed to stay future-proof: new algorithms added to the registry will show up here automatically.
        </div>
        <div class="pill-row">
            <span class="pill">4 default city sizes: 10, 20, 30, 40</span>
            <span class="pill">3 default configs per runnable model: 10, 20, 40</span>
            <span class="pill">12 runs per model at defaults</span>
            <span class="pill">No CSV logging</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

selected_models = _parse_models()
available_models = {spec.label: spec for spec in AVAILABLE_ALGORITHMS}
selected_runnable = [label for label in selected_models if available_models[label].runnable]
selected_unavailable = [label for label in selected_models if not available_models[label].runnable]

left, right = st.columns([1.15, 1.4])
with left:
    st.markdown("### Batch settings")
    city_sizes_raw = st.text_input("City sizes", value="10, 20, 30, 40", help="Comma-separated list of TSP sizes to run.")
    aco_configs_raw = st.text_input("ACO ants", value="10, 20, 40", help="Comma-separated list of ant counts.")
    abc_configs_raw = st.text_input("ABC food sources", value="10, 20, 40", help="Comma-separated list of food-source counts.")
    ga_configs_raw = st.text_input("GA population", value="10, 20, 40", help="Comma-separated list of population sizes.")
    pso_configs_raw = st.text_input("PSO particles", value="10, 20, 40", help="Comma-separated list of particle counts.")
    iterations = st.number_input("Iterations per run", min_value=1, max_value=5000, value=200, step=10)
    exact_timeout = st.slider("Exact solver time limit (seconds)", min_value=10, max_value=300, value=60)
    base_seed = st.number_input("Base random seed", min_value=0, value=7, step=1)
    clustered = st.checkbox("Clustered city layouts", value=False)
    aco_two_opt = st.checkbox("Use 2-opt for ACO", value=True)
    abc_two_opt = st.checkbox("Use 2-opt for ABC", value=True)
    ga_two_opt = st.checkbox("Use 2-opt for GA", value=True)
    pso_two_opt = st.checkbox("Use 2-opt for PSO", value=True)
    target_gap_pct = st.number_input("Early stop gap threshold (%)", min_value=0.0, max_value=25.0, value=0.0, step=0.1, help="Stop as soon as a model gets within this gap of the exact solver. Use 0.0 to stop only on an exact match.")

    st.markdown("### ACO fine-tuning")
    aco_alpha = st.slider("alpha", min_value=0.1, max_value=5.0, value=1.0)
    aco_beta = st.slider("beta", min_value=0.1, max_value=10.0, value=5.0)
    aco_rho = st.slider("rho", min_value=0.01, max_value=0.99, value=0.5)
    aco_q = st.number_input("Q", min_value=0.1, value=100.0)

    st.markdown("### ABC fine-tuning")
    abc_limit = st.number_input("Scout limit", min_value=1, max_value=20000, value=60)

    st.markdown("### GA fine-tuning")
    ga_crossover_rate = st.slider("Crossover rate", min_value=0.0, max_value=1.0, value=0.9)
    ga_mutation_rate = st.slider("Mutation rate", min_value=0.0, max_value=1.0, value=0.2)
    ga_elite_fraction = st.slider("Elite fraction", min_value=0.0, max_value=0.5, value=0.1)
    ga_tournament_size = st.number_input("Tournament size", min_value=2, max_value=50, value=3)

    st.markdown("### PSO fine-tuning")
    pso_w = st.slider("w", min_value=0.0, max_value=1.5, value=0.5)
    pso_c1 = st.slider("c1", min_value=0.0, max_value=3.0, value=1.5)
    pso_c2 = st.slider("c2", min_value=0.0, max_value=3.0, value=1.5)

    run = st.button("Run comparison batch", type="primary", use_container_width=True)

with right:
    st.markdown("### Selection summary")
    st.write("Available models:")
    for spec in AVAILABLE_ALGORITHMS:
        state = "ready" if spec.runnable else "coming soon"
        st.write(f"- {spec.label}: {state} - {spec.description}")

    st.markdown("### Effective run matrix")
    city_sizes = parse_int_list(city_sizes_raw, fallback=[10, 20, 30, 40])
    aco_configs = parse_int_list(aco_configs_raw, fallback=[10, 20, 40])
    abc_configs = parse_int_list(abc_configs_raw, fallback=[10, 20, 40])
    ga_configs = parse_int_list(ga_configs_raw, fallback=[10, 20, 40])
    pso_configs = parse_int_list(pso_configs_raw, fallback=[10, 20, 40])
    estimated_runs = 0
    if "ACO" in selected_models:
        estimated_runs += len(city_sizes) * len(aco_configs)
    if "ABC" in selected_models:
        estimated_runs += len(city_sizes) * len(abc_configs)
    if "GA" in selected_models:
        estimated_runs += len(city_sizes) * len(ga_configs)
    if "PSO" in selected_models:
        estimated_runs += len(city_sizes) * len(pso_configs)
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="metric-label">City sizes</div>
            <div class="metric-value">{', '.join(map(str, city_sizes))}</div>
            <div class="metric-note">{len(city_sizes)} sizes selected</div>
        </div>
        <div class="panel-card">
            <div class="metric-label">ACO configs</div>
            <div class="metric-value">{', '.join(map(str, aco_configs))}</div>
            <div class="metric-note">{len(aco_configs)} ant-count settings</div>
        </div>
        <div class="panel-card">
            <div class="metric-label">ABC configs</div>
            <div class="metric-value">{', '.join(map(str, abc_configs))}</div>
            <div class="metric-note">{len(abc_configs)} food-source settings</div>
        </div>
        <div class="panel-card">
            <div class="metric-label">GA configs</div>
            <div class="metric-value">{', '.join(map(str, ga_configs))}</div>
            <div class="metric-note">{len(ga_configs)} population settings</div>
        </div>
        <div class="panel-card">
            <div class="metric-label">PSO configs</div>
            <div class="metric-value">{', '.join(map(str, pso_configs))}</div>
            <div class="metric-note">{len(pso_configs)} particle-count settings</div>
        </div>
        <div class="panel-card">
            <div class="metric-label">Estimated completed runs</div>
            <div class="metric-value">{estimated_runs}</div>
            <div class="metric-note">Only runnable models are executed.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if selected_unavailable:
    st.warning(
        "Some selected models are not implemented yet: " + ", ".join(selected_unavailable) + ". They will appear in the report as unavailable until runners are added."
    )

if run:
    if not selected_models:
        st.error("Select at least one model.")
        st.stop()
    if not city_sizes:
        st.error("Provide at least one city size.")
        st.stop()

    aco_settings = {
        "two_opt": aco_two_opt,
        "alpha": aco_alpha,
        "beta": aco_beta,
        "rho": aco_rho,
        "Q": aco_q,
    }
    abc_settings = {
        "two_opt": abc_two_opt,
        "limit": abc_limit,
    }
    ga_settings = {
        "two_opt": ga_two_opt,
        "crossover_rate": ga_crossover_rate,
        "mutation_rate": ga_mutation_rate,
        "elite_fraction": ga_elite_fraction,
        "tournament_size": ga_tournament_size,
    }
    pso_settings = {
        "two_opt": pso_two_opt,
        "w": pso_w,
        "c1": pso_c1,
        "c2": pso_c2,
    }

    model_config_values = {
        "ACO": aco_configs,
        "ABC": abc_configs,
        "GA": ga_configs,
        "PSO": pso_configs,
    }

    estimated_runs = 0
    for model_label in selected_models:
        estimated_runs += len(city_sizes) * len(model_config_values.get(model_label, [0]))

    progress_area = st.container()
    overall_text = progress_area.empty()
    overall_bar = progress_area.empty()
    current_text = progress_area.empty()
    current_bar = progress_area.empty()
    current_banner = progress_area.empty()

    rows: list[dict] = []
    progress_state = {
        "event": {
            "type": "idle",
            "batch_total": estimated_runs,
            "batch_completed": 0,
            "batch_progress": 0.0,
            "current_progress": 0.0,
        },
        "report": None,
        "error": None,
        "done": False,
    }
    progress_lock = threading.Lock()

    def update_progress(event: dict) -> None:
        with progress_lock:
            progress_state["event"] = dict(event)

    def _run_batch_worker() -> None:
        try:
            report = _call_run_comparison_batch(
                selected_models=selected_models,
                city_sizes=city_sizes,
                model_config_values=model_config_values,
                iterations=int(iterations),
                base_seed=None if int(base_seed) == 0 else int(base_seed),
                clustered=clustered,
                exact_timeout=int(exact_timeout),
                aco_settings=aco_settings,
                abc_settings=abc_settings,
                ga_settings=ga_settings,
                pso_settings=pso_settings,
                target_gap_pct=float(target_gap_pct),
                progress_callback=update_progress,
            )
            with progress_lock:
                progress_state["report"] = report
        except Exception as exc:
            with progress_lock:
                progress_state["error"] = exc
        finally:
            with progress_lock:
                progress_state["done"] = True

    worker = threading.Thread(target=_run_batch_worker, daemon=True)
    worker.start()

    animation_frames = ["🏃‍➡️", "🏃", "🏃‍♀️", "🏃‍♂️"]
    animation_index = 0
    while worker.is_alive():
        with progress_lock:
            event = dict(progress_state["event"])

        batch_total = max(1, int(event.get("batch_total", estimated_runs)))
        batch_completed = min(batch_total, int(event.get("batch_completed", 0)))
        batch_progress = float(event.get("batch_progress", batch_completed / batch_total))
        current_progress = float(event.get("current_progress", 0.0))
        headline = "Batch comparison is running"
        detail = "The models are sprinting toward the finish line."
        batch_label = f"{batch_completed}/{batch_total} runs complete"
        batch_suffix = f"{animation_frames[animation_index % len(animation_frames)]} in motion"

        if event.get("type") == "run_started":
            headline = f"{event.get('model', 'Model')} is on the track"
            detail = f"{event.get('city_count')} cities · config {event.get('config_value')} · exact baseline: {event.get('exact_status')}"
            batch_suffix = "next run queued"
        elif event.get("type") == "iteration":
            headline = f"{event.get('model', 'Model')} is sprinting"
            detail = f"Iteration {event.get('iteration')}/{event.get('iterations_total')} · current best gap is closing"
            batch_suffix = f"{current_progress * 100.0:.1f}% of this run"
        elif event.get("type") == "run_completed":
            headline = f"{event.get('model', 'Model')} crossed the line"
            detail = f"Status: {event.get('status')} · early stop: {event.get('stopped_early')}"
            batch_suffix = "run locked in"

        overall_text.markdown(f"**Batch progress:** {batch_completed}/{batch_total} runs complete")
        overall_bar.progress(int(round(batch_progress * 100.0)))
        current_text.markdown(f"**Current run:** {headline} · {detail}")
        current_bar.progress(int(round(max(0.0, min(1.0, current_progress)) * 100.0)))
        _render_race_banner(current_banner, batch_progress * 100.0, headline, detail, batch_label, batch_suffix)

        animation_index += 1
        time.sleep(0.08)

    worker.join()
    with progress_lock:
        if progress_state["error"] is not None:
            raise progress_state["error"]
        report = progress_state["report"]

    rows = report["rows"]
    _render_race_banner(current_banner, 100.0, "All batch runs finished", "The report is ready.", f"{estimated_runs}/{estimated_runs} runs complete", "full completion")

    st.markdown("### Batch summary")
    summary = build_model_summary(rows)
    _render_metric_cards(summary)
    _render_table(rows, [label for label in selected_models if available_models[label].runnable])
    st.markdown("### Model insights")
    _render_insights(rows)

    st.markdown(
        """
        <div class="panel-card">
            <div class="metric-label">What to look for</div>
            <div class="metric-note">
                Lower convergence time means the model becomes useful sooner. Lower evaluation count means less computation to settle.
                Lower gap means better route quality against the exact baseline. If a model wins on quality for small sizes but loses on time for larger sizes,
                that usually means it is exploiting local structure more aggressively but paying a higher search cost.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Choose models and run the batch to generate the full comparison report.")

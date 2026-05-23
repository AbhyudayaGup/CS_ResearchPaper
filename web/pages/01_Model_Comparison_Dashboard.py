from __future__ import annotations

from pathlib import Path
import sys
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
    labels = [spec.label for spec in AVAILABLE_ALGORITHMS]
    default = [spec.label for spec in AVAILABLE_ALGORITHMS if spec.runnable]
    selected = st.multiselect(
        "Models to compare",
        options=labels,
        default=default,
        format_func=lambda label: f"{label} ({'ready' if next(spec for spec in AVAILABLE_ALGORITHMS if spec.label == label).runnable else 'coming soon'})",
    )
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
        html.append(f"<th colspan='4'>{escape(model)}</th>")
    html.append("</tr><tr>")
    for _ in header_models:
        html.append("<th>Time (s)</th><th>Converged</th><th>Evals</th><th>Gap %</th>")
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
                html.append(f"<td colspan='4' class='badge-off'>{escape(str(status))}</td>")
                continue
            converged = "Yes" if model_row.get("optimal_match") else "No"
            converged_class = "badge-ok" if model_row.get("optimal_match") else "badge-warn"
            gap_value = model_row.get("optimality_gap_pct")
            gap_text = "N/A" if gap_value is None else f"{float(gap_value):.3f}"
            html.append(
                f"<td>{float(model_row['run_time_s']):.4f}</td><td class='{converged_class}'>{converged}</td><td>{int(model_row['objective_evals_to_convergence']):,}</td><td>{gap_text}</td>"
            )
        html.append("</tr>")

    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_insights(rows: list[dict]) -> None:
    insights = build_insights(rows)
    st.markdown("### What the report says")
    st.markdown("<ul class='insight-list'>" + "".join(f"<li>{escape(text)}</li>" for text in insights) + "</ul>", unsafe_allow_html=True)


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
            <span class="pill">3 default ACO / PSO configs: 10, 20, 40</span>
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
    pso_configs_raw = st.text_input("PSO particles", value="10, 20, 40", help="Comma-separated list of particle counts.")
    iterations = st.number_input("Iterations per run", min_value=1, max_value=5000, value=200, step=10)
    exact_timeout = st.slider("Exact solver time limit (seconds)", min_value=10, max_value=300, value=60)
    base_seed = st.number_input("Base random seed", min_value=0, value=7, step=1)
    clustered = st.checkbox("Clustered city layouts", value=False)
    aco_two_opt = st.checkbox("Use 2-opt for ACO", value=True)
    pso_two_opt = st.checkbox("Use 2-opt for PSO", value=True)

    st.markdown("### ACO fine-tuning")
    aco_alpha = st.slider("alpha", min_value=0.1, max_value=5.0, value=1.0)
    aco_beta = st.slider("beta", min_value=0.1, max_value=10.0, value=5.0)
    aco_rho = st.slider("rho", min_value=0.01, max_value=0.99, value=0.5)
    aco_q = st.number_input("Q", min_value=0.1, value=100.0)

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
    pso_configs = parse_int_list(pso_configs_raw, fallback=[10, 20, 40])
    estimated_runs = 0
    if "ACO" in selected_models:
        estimated_runs += len(city_sizes) * len(aco_configs)
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
    pso_settings = {
        "two_opt": pso_two_opt,
        "w": pso_w,
        "c1": pso_c1,
        "c2": pso_c2,
    }

    with st.spinner("Running batch comparison..."):
        model_config_values = {
            "ACO": aco_configs,
            "PSO": pso_configs,
            "Bee Colony": [0],
        }
        report = run_comparison_batch(
            selected_models,
            city_sizes,
            model_config_values,
            iterations=int(iterations),
            base_seed=None if int(base_seed) == 0 else int(base_seed),
            clustered=clustered,
            exact_timeout=int(exact_timeout),
            aco_settings=aco_settings,
            pso_settings=pso_settings,
        )
        rows = report["rows"]

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

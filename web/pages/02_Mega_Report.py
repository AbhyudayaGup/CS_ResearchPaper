from __future__ import annotations

from html import escape
from pathlib import Path
import sys

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mega_report import (  # noqa: E402
    available_report_paths,
    city_summary_frame,
    load_report_json,
    report_rows_frame,
    report_summary_frame,
)


st.set_page_config(page_title="Mega Report", layout="wide")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .hero-card {
            background: linear-gradient(135deg, rgba(12,16,24,0.98), rgba(24,31,48,0.98));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 26px 28px;
            box-shadow: 0 18px 50px rgba(0,0,0,0.22);
            margin-bottom: 18px;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #f6f8fc;
            margin-bottom: 8px;
        }
        .hero-subtitle {
            color: rgba(246,248,252,0.74);
            line-height: 1.55;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(18,24,37,0.98), rgba(12,16,25,0.98));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 14px 16px;
        }
        .metric-label {
            font-size: 0.75rem;
            color: rgba(233,239,247,0.68);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #fff;
        }
        .metric-note {
            margin-top: 6px;
            color: rgba(233,239,247,0.72);
            font-size: 0.86rem;
        }
        .panel-card {
            background: rgba(12,16,25,0.92);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 18px;
            margin-bottom: 16px;
        }
        .insight-list li {
            margin-bottom: 10px;
            color: rgba(240,244,250,0.94);
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_cards(summary_rows: list[dict]) -> None:
    if not summary_rows:
        st.info("No completed rows were found in this report.")
        return

    def _best_min(key: str) -> str:
        numeric_rows = []
        for row in summary_rows:
            try:
                numeric_rows.append((float(row[key]), row["model"]))
            except Exception:
                continue
        if not numeric_rows:
            return "N/A"
        best_value = min(val for val, _ in numeric_rows)
        winners = [model for val, model in numeric_rows if abs(val - best_value) <= 1e-12]
        return ", ".join(winners)

    def _best_max(key: str) -> str:
        numeric_rows = []
        for row in summary_rows:
            try:
                numeric_rows.append((float(row[key]), row["model"]))
            except Exception:
                continue
        if not numeric_rows:
            return "N/A"
        best_value = max(val for val, _ in numeric_rows)
        winners = [model for val, model in numeric_rows if abs(val - best_value) <= 1e-12]
        return ", ".join(winners)

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-label">Best average quality</div><div class="metric-value">{escape(_best_min('avg_gap_pct'))}</div><div class="metric-note">Lower gap is better</div></div>
            <div class="metric-card"><div class="metric-label">Fastest convergence</div><div class="metric-value">{escape(_best_min('avg_convergence_time_s'))}</div><div class="metric-note">Lower time is better</div></div>
            <div class="metric-card"><div class="metric-label">Lowest total runtime</div><div class="metric-value">{escape(_best_min('avg_run_time_s'))}</div><div class="metric-note">Wall-clock performance</div></div>
            <div class="metric-card"><div class="metric-label">Highest exact-match rate</div><div class="metric-value">{escape(_best_max('exact_match_rate'))}</div><div class="metric-note">Most reliable route matches</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_charts(summary: list[dict], city_summary) -> None:
    if summary:
        summary_sorted = sorted(summary, key=lambda row: row["model"])
        models = [row["model"] for row in summary_sorted]
        fig_gap = go.Figure()
        fig_gap.add_bar(x=models, y=[float(row.get("avg_gap_pct", 0.0)) for row in summary_sorted], marker_color="#f97316")
        fig_gap.update_layout(title="Average Gap by Model", xaxis_title="Model", yaxis_title="Gap %", template="plotly_dark", height=360)

        fig_time = go.Figure()
        fig_time.add_bar(x=models, y=[float(row.get("avg_convergence_time_s", 0.0)) for row in summary_sorted], marker_color="#38bdf8")
        fig_time.update_layout(title="Average Convergence Time", xaxis_title="Model", yaxis_title="Seconds", template="plotly_dark", height=360)

        fig_evals = go.Figure()
        fig_evals.add_bar(x=models, y=[float(row.get("avg_evals_to_convergence", 0.0)) for row in summary_sorted], marker_color="#a78bfa")
        fig_evals.update_layout(title="Average Evaluations to Convergence", xaxis_title="Model", yaxis_title="Evaluations", template="plotly_dark", height=360)

        fig_rate = go.Figure()
        fig_rate.add_bar(x=models, y=[float(row.get("exact_match_rate", 0.0)) * 100.0 for row in summary_sorted], marker_color="#34d399")
        fig_rate.update_layout(title="Exact Match Rate", xaxis_title="Model", yaxis_title="Percent", template="plotly_dark", height=360)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_gap, use_container_width=True)
            st.plotly_chart(fig_time, use_container_width=True)
        with col2:
            st.plotly_chart(fig_evals, use_container_width=True)
            st.plotly_chart(fig_rate, use_container_width=True)

    if city_summary is not None and len(city_summary) > 0:
        fig_city_gap = go.Figure()
        for model in sorted(city_summary["model"].unique()):
            subset = city_summary[city_summary["model"] == model].sort_values("city_count")
            fig_city_gap.add_trace(
                go.Scatter(
                    x=subset["city_count"],
                    y=subset["avg_gap_pct"],
                    mode="lines+markers",
                    name=model,
                )
            )
        fig_city_gap.update_layout(title="Gap by City Size", xaxis_title="Cities", yaxis_title="Gap %", template="plotly_dark", height=360)
        st.plotly_chart(fig_city_gap, use_container_width=True)


def _report_label(report_path: Path) -> str:
    report = load_report_json(report_path)
    report_id = str(report.get("report_id") or report_path.parent.name)
    variant = str(report.get("variant", "standard")).title()
    created_at = str(report.get("created_at", ""))
    created_label = created_at.replace("T", " ").replace("Z", "")[:19]
    return f"{created_label} · {variant} · {report_id}"


_inject_css()

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Mega Report</div>
        <div class="hero-subtitle">
            This page shows the latest long-form benchmark report. Use the dashboard to generate a new report,
            then come back here to inspect the summary, charts, and downloads.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

session_report_path = st.session_state.get("mega_report_path")
report_paths = available_report_paths()
if session_report_path:
    session_path = Path(session_report_path)
    if session_path.exists() and session_path not in report_paths:
        report_paths.insert(0, session_path)
if not report_paths:
    st.warning("No mega report has been generated yet.")
    st.info("Go to the Model Comparison Dashboard, generate a mega report, and return here to review it.")
    st.stop()

default_report_index = 0
if session_report_path:
    session_path = Path(session_report_path)
    if session_path in report_paths:
        default_report_index = report_paths.index(session_path)
selected_report_path = st.selectbox(
    "Mega report to display",
    options=report_paths,
    index=default_report_index,
    format_func=_report_label,
    help="Choose which saved mega report powers the summary, charts, tables, and downloads below.",
)
report_path = Path(selected_report_path)
report = load_report_json(report_path)
summary = report_summary_frame(report).to_dict(orient="records")
city_summary = city_summary_frame(report)
rows_frame = report_rows_frame(report)

st.markdown(
    f"""
    <div class="panel-card">
        <div class="metric-label">Loaded report</div>
        <div class="metric-value">{escape(report.get('variant', 'standard').title())} TSP</div>
        <div class="metric-note">Created {escape(str(report.get('created_at', 'unknown')))} · {int(report.get('settings', {}).get('scenario_count', 0))} scenarios · {int(report.get('settings', {}).get('task_count', 0))} algorithm runs</div>
    </div>
    """,
    unsafe_allow_html=True,
)

_summary_cards(summary)

st.markdown("### Downloads")
pdf_path = Path(report.get("files", {}).get("pdf", ""))
json_path = Path(report.get("files", {}).get("json", ""))
if pdf_path.exists():
    st.download_button(
        "Download PDF",
        data=pdf_path.read_bytes(),
        file_name=pdf_path.name,
        mime="application/pdf",
        use_container_width=True,
    )
if json_path.exists():
    st.download_button(
        "Download JSON",
        data=json_path.read_bytes(),
        file_name=json_path.name,
        mime="application/json",
        use_container_width=True,
    )

st.markdown("### Charts")
gap_view_mode = st.selectbox(
    "Gap by city size view",
    ["summary city sizes", "detailed 30-40 sweep"],
    index=0,
    help="Switch only the gap-by-city-size chart between the default report data and the detailed sweep, if it was generated.",
)
city_summary_view = city_summary_frame(report, use_detailed=gap_view_mode.startswith("detailed"))
if gap_view_mode.startswith("detailed") and city_summary_view.equals(city_summary):
    st.info("This report does not include a detailed 30-40 sweep, so the chart is using the summary city sizes.")
_render_summary_charts(summary, city_summary_view)

st.markdown("### Insights")
if report.get("insights"):
    st.markdown("<ul class='insight-list'>" + "".join(f"<li>{escape(text)}</li>" for text in report["insights"]) + "</ul>", unsafe_allow_html=True)
else:
    st.info("No insights were generated for this report.")

st.markdown("### Scenario details")
if not rows_frame.empty:
    display_cols = [col for col in ["scenario_id", "city_count", "model", "config_value", "run_time_s", "convergence_time_s", "objective_evals_to_convergence", "optimality_gap_pct", "stopped_early", "exact_match_rate"] if col in rows_frame.columns]
    st.dataframe(rows_frame[display_cols].sort_values(["scenario_id", "model"]), use_container_width=True)
else:
    st.info("This report does not contain per-run rows.")

st.caption(f"Report file: {report_path}")

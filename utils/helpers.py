from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model.readiness import compute_readiness_score
from utils.config import (
    ARTIFACT_FILES,
    DATASET_PATH,
    FORM_FIELD_GROUPS,
    INPUT_FEATURES,
    NUMERIC_FEATURES,
)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_metrics() -> Dict[str, Any]:
    return {
        "placement": load_json(ARTIFACT_FILES["placement_metrics"]),
        "role": load_json(ARTIFACT_FILES["role_metrics"]),
        "metadata": load_json(ARTIFACT_FILES["metadata"]),
    }


def load_feature_reports() -> Dict[str, Any]:
    return {
        "placement": joblib.load(ARTIFACT_FILES["selected_features_placement"]),
        "role": joblib.load(ARTIFACT_FILES["selected_features_role"]),
    }


def form_schema() -> List[Dict[str, Any]]:
    return FORM_FIELD_GROUPS


@lru_cache(maxsize=3)
def dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def _chart_html(fig: go.Figure, include_plotly: bool = False) -> str:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=34, r=20, t=48, b=34),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial", size=12, color="#1f2937"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=True if include_plotly else False,
        config={"displayModeBar": False, "responsive": True},
    )


def _readiness_frame(df: pd.DataFrame) -> pd.DataFrame:
    scores = df.apply(lambda row: compute_readiness_score(row.to_dict())["score"], axis=1)
    frame = df.copy()
    frame["readiness_score"] = scores
    return frame


def dashboard_context() -> Dict[str, Any]:
    df = dataset()
    metrics = load_metrics()
    feature_reports = load_feature_reports()
    readiness_df = _readiness_frame(df)
    placement_labeled = df.assign(
        placement_label=df["placement_status"].map({1: "Placed", 0: "Not Placed"})
    )

    chart_items: List[Dict[str, str]] = []

    placement_counts = placement_labeled["placement_label"].value_counts().reset_index()
    placement_counts.columns = ["Placement", "Students"]
    fig = px.pie(
        placement_counts,
        names="Placement",
        values="Students",
        hole=0.52,
        color="Placement",
        color_discrete_map={"Placed": "#16a34a", "Not Placed": "#dc2626"},
        title="Placement Distribution",
    )
    chart_items.append({"title": "Placement Distribution", "html": _chart_html(fig, include_plotly=True)})

    role_counts = df["recommended_role"].value_counts().head(16).sort_values().reset_index()
    role_counts.columns = ["Role", "Students"]
    fig = px.bar(role_counts, x="Students", y="Role", orientation="h", title="Recommended Role Distribution", color="Students", color_continuous_scale="Teal")
    chart_items.append({"title": "Role Distribution", "html": _chart_html(fig)})

    importance = pd.DataFrame(feature_reports["placement"]["top_features"][:14]).sort_values("combined_score")
    fig = px.bar(
        importance,
        x="combined_score",
        y="feature",
        orientation="h",
        title="Top Placement Feature Importance",
        color="combined_score",
        color_continuous_scale="Sunset",
    )
    chart_items.append({"title": "Placement Feature Importance", "html": _chart_html(fig)})

    role_importance = pd.DataFrame(feature_reports["role"]["top_features"][:14]).sort_values("combined_score")
    fig = px.bar(
        role_importance,
        x="combined_score",
        y="feature",
        orientation="h",
        title="Top Role Recommendation Feature Importance",
        color="combined_score",
        color_continuous_scale="Viridis",
    )
    chart_items.append({"title": "Role Feature Importance", "html": _chart_html(fig)})

    internship_impact = (
        df.groupby("internships_count")["placement_status"]
        .agg(["mean", "count"])
        .reset_index()
        .query("count >= 20")
    )
    internship_impact["placement_rate"] = internship_impact["mean"] * 100
    fig = px.line(
        internship_impact,
        x="internships_count",
        y="placement_rate",
        markers=True,
        title="Internships Impact On Placement",
        labels={"internships_count": "Internships", "placement_rate": "Placement Rate (%)"},
    )
    fig.update_traces(line_color="#0f766e")
    chart_items.append({"title": "Internships Impact", "html": _chart_html(fig)})

    project_impact = (
        df.groupby("projects_count")["placement_status"]
        .agg(["mean", "count"])
        .reset_index()
        .query("count >= 20")
    )
    project_impact["placement_rate"] = project_impact["mean"] * 100
    fig = px.area(
        project_impact,
        x="projects_count",
        y="placement_rate",
        title="Projects Impact On Placement",
        labels={"projects_count": "Projects", "placement_rate": "Placement Rate (%)"},
    )
    fig.update_traces(line_color="#2563eb", fillcolor="rgba(37,99,235,0.18)")
    chart_items.append({"title": "Projects Impact", "html": _chart_html(fig)})

    fig = px.histogram(
        placement_labeled,
        x="aptitude_score",
        color="placement_label",
        nbins=28,
        barmode="overlay",
        opacity=0.72,
        title="Aptitude Score Distribution By Placement",
        color_discrete_map={"Placed": "#16a34a", "Not Placed": "#dc2626"},
    )
    chart_items.append({"title": "Aptitude Distribution", "html": _chart_html(fig)})

    fig = px.histogram(
        readiness_df.assign(placement_label=readiness_df["placement_status"].map({1: "Placed", 0: "Not Placed"})),
        x="readiness_score",
        color="placement_label",
        nbins=24,
        barmode="overlay",
        opacity=0.74,
        title="Readiness Score Distribution",
        labels={"color": "Placement"},
        color_discrete_map={"Placed": "#16a34a", "Not Placed": "#dc2626"},
    )
    chart_items.append({"title": "Readiness Distribution", "html": _chart_html(fig)})

    corr_features = [
        "cgpa",
        "aptitude_score",
        "programming_skills",
        "communication_skills",
        "technical_test_score",
        "mock_interview_score",
        "projects_count",
        "internships_count",
        "resume_score",
        "backlogs",
        "placement_status",
    ]
    corr = df[corr_features].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
        zmin=-1,
        zmax=1,
    )
    chart_items.append({"title": "Correlation Heatmap", "html": _chart_html(fig)})

    role_pattern_features = [
        "programming_skills",
        "communication_skills",
        "logical_reasoning",
        "ai_interest",
        "web_interest",
        "cloud_interest",
        "security_interest",
        "devops_interest",
    ]
    top_roles = df["recommended_role"].value_counts().head(12).index
    role_patterns = df[df["recommended_role"].isin(top_roles)].groupby("recommended_role")[role_pattern_features].mean().round(1)
    fig = px.imshow(
        role_patterns,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="Role-Wise Readiness Patterns",
    )
    chart_items.append({"title": "Role-Wise Patterns", "html": _chart_html(fig)})

    stats = {
        "students": f"{len(df):,}",
        "placement_rate": f"{df['placement_status'].mean() * 100:.1f}%",
        "roles": df["recommended_role"].nunique(),
        "avg_readiness": f"{readiness_df['readiness_score'].mean():.1f}",
    }

    return {
        "charts": chart_items,
        "stats": stats,
        "metrics": metrics,
    }


def _rate_table(df: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
    table = (
        df.groupby(column)
        .agg(students=("student_id", "count"), placement_rate=("placement_status", "mean"))
        .reset_index()
        .sort_values("placement_rate", ascending=False)
    )
    table["placement_rate"] = (table["placement_rate"] * 100).round(1)
    return table.to_dict(orient="records")


def feature_insight_context() -> Dict[str, Any]:
    df = dataset()
    reports = load_feature_reports()
    metrics = load_metrics()
    labeled = df.assign(placement_label=df["placement_status"].map({1: "Placed", 0: "Not Placed"}))

    comparison_features = [
        "cgpa",
        "aptitude_score",
        "programming_skills",
        "communication_skills",
        "technical_test_score",
        "mock_interview_score",
        "projects_count",
        "internships_count",
        "certifications_count",
        "resume_score",
        "backlogs",
        "gap_years",
    ]
    placed_patterns = (
        labeled.groupby("placement_label")[comparison_features]
        .mean()
        .T.round(2)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    if {"Placed", "Not Placed"}.issubset(placed_patterns.columns):
        placed_patterns["gap"] = (placed_patterns["Placed"] - placed_patterns["Not Placed"]).round(2)
        placed_patterns = placed_patterns.sort_values("gap", ascending=False)

    role_skill_features = [
        "programming_skills",
        "core_cs_fundamentals",
        "communication_skills",
        "logical_reasoning",
        "technical_test_score",
        "projects_count",
        "internships_count",
        "mobile_interest",
        "web_interest",
        "ai_interest",
        "cloud_interest",
        "data_interest",
        "security_interest",
        "devops_interest",
    ]
    role_patterns = (
        df.groupby("recommended_role")[role_skill_features]
        .mean()
        .round(1)
        .reset_index()
    )

    insights = [
        "Technical test score, programming skill, aptitude, and mock interview performance are the strongest placement signals.",
        "Backlogs, gap years, attendance risk, and low resume strength consistently reduce placement probability.",
        "Role prediction is driven mainly by interest clusters combined with technical proof such as projects, GitHub activity, and core CS fundamentals.",
        "Internships and projects show a clear upward placement-rate pattern because they act as practical readiness signals.",
    ]

    return {
        "reports": reports,
        "metrics": metrics,
        "placed_patterns": placed_patterns.to_dict(orient="records"),
        "role_patterns": role_patterns.to_dict(orient="records"),
        "college_tier_table": _rate_table(df, "college_tier"),
        "degree_table": _rate_table(df, "degree_type"),
        "specialization_table": _rate_table(df, "specialization"),
        "insights": insights,
        "input_features": INPUT_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
    }

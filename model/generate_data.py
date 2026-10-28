from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from model.readiness import compute_readiness_score, readiness_level
from utils.config import (
    COLLEGE_TIER_OPTIONS,
    DATASET_PATH,
    DATASET_ROWS,
    DEGREE_OPTIONS,
    DOMAIN_OPTIONS,
    GENDER_OPTIONS,
    HERO_IMAGE_PATH,
    PLOT_DIR,
    ROLE_LABELS,
    SAMPLE_BATCH_PATH,
    SPECIALIZATION_OPTIONS,
    ensure_directories,
)


def _clip(values, low=0, high=100):
    return np.clip(values, low, high)


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def _score(rng, base, noise=8, low=0, high=100):
    return _clip(base + rng.normal(0, noise, len(base)), low, high)


def _domain_interest(rng, preferred_domains: np.ndarray, specializations: np.ndarray) -> Dict[str, np.ndarray]:
    n_rows = len(preferred_domains)
    interests = {
        "mobile_interest": rng.normal(42, 14, n_rows),
        "web_interest": rng.normal(48, 15, n_rows),
        "ai_interest": rng.normal(42, 16, n_rows),
        "cloud_interest": rng.normal(40, 15, n_rows),
        "backend_interest": rng.normal(47, 15, n_rows),
        "frontend_interest": rng.normal(45, 15, n_rows),
        "data_interest": rng.normal(43, 16, n_rows),
        "security_interest": rng.normal(35, 15, n_rows),
        "devops_interest": rng.normal(35, 15, n_rows),
    }
    boosts = {
        "Mobile Apps": ["mobile_interest", "frontend_interest", "web_interest"],
        "Web Development": ["web_interest", "frontend_interest", "backend_interest"],
        "AI/ML": ["ai_interest", "data_interest", "backend_interest"],
        "Data Analytics": ["data_interest", "ai_interest", "presentation_skills"],
        "Cloud/DevOps": ["cloud_interest", "devops_interest", "backend_interest"],
        "Cybersecurity": ["security_interest", "backend_interest", "cloud_interest"],
        "Software Engineering": ["backend_interest", "web_interest", "frontend_interest"],
        "QA/Testing": ["web_interest", "backend_interest", "data_interest"],
    }

    for domain, keys in boosts.items():
        mask = preferred_domains == domain
        for key in keys:
            if key in interests:
                interests[key][mask] += rng.normal(24, 8, mask.sum())

    ai_mask = np.isin(specializations, ["AI & Data Science", "Data Science"])
    cyber_mask = specializations == "Cybersecurity"
    cs_mask = np.isin(specializations, ["Computer Science", "Information Technology"])
    ece_mask = specializations == "Electronics & Communication"
    interests["ai_interest"][ai_mask] += 14
    interests["data_interest"][ai_mask] += 13
    interests["security_interest"][cyber_mask] += 22
    interests["backend_interest"][cs_mask] += 8
    interests["web_interest"][cs_mask] += 7
    interests["cloud_interest"][ece_mask] += 5

    return {key: _clip(value) for key, value in interests.items()}


def _role_for_row(row: Dict[str, float], rng) -> str:
    preferred = row["preferred_domain"]
    pref_boost = lambda domain: 28 if preferred == domain else 0
    spec = row["specialization"]
    ai_spec = 18 if spec in {"AI & Data Science", "Data Science"} else 0
    cyber_spec = 20 if spec == "Cybersecurity" else 0
    cs_spec = 10 if spec in {"Computer Science", "Information Technology"} else 0

    scores = {
        "Android Developer": 0.44 * row["mobile_interest"] + 0.18 * row["programming_skills"] + 3.2 * row["projects_count"] + 0.06 * row["github_activity_score"] + pref_boost("Mobile Apps"),
        "Flutter Developer": 0.38 * row["mobile_interest"] + 0.16 * row["frontend_interest"] + 0.16 * row["programming_skills"] + 2.8 * row["projects_count"] + pref_boost("Mobile Apps"),
        "React Native Developer": 0.31 * row["mobile_interest"] + 0.22 * row["frontend_interest"] + 0.18 * row["web_interest"] + 0.14 * row["programming_skills"] + pref_boost("Mobile Apps"),
        "Web Developer": 0.34 * row["web_interest"] + 0.22 * row["frontend_interest"] + 0.16 * row["backend_interest"] + 0.13 * row["programming_skills"] + pref_boost("Web Development"),
        "Frontend Developer": 0.36 * row["frontend_interest"] + 0.22 * row["web_interest"] + 0.12 * row["presentation_skills"] + 0.12 * row["programming_skills"] + pref_boost("Web Development"),
        "Backend Developer": 0.36 * row["backend_interest"] + 0.20 * row["programming_skills"] + 0.16 * row["core_cs_fundamentals"] + 0.13 * row["problem_solving"] + cs_spec,
        "Full Stack Developer": 0.22 * row["web_interest"] + 0.22 * row["frontend_interest"] + 0.24 * row["backend_interest"] + 0.16 * row["programming_skills"] + 2.2 * row["projects_count"] + pref_boost("Software Engineering"),
        "AI Engineer": 0.36 * row["ai_interest"] + 0.18 * row["programming_skills"] + 0.16 * row["logical_reasoning"] + 0.12 * row["technical_test_score"] + ai_spec + pref_boost("AI/ML"),
        "Data Scientist": 0.34 * row["data_interest"] + 0.19 * row["ai_interest"] + 0.17 * row["aptitude_score"] + 0.12 * row["logical_reasoning"] + ai_spec + pref_boost("Data Analytics"),
        "ML Engineer": 0.32 * row["ai_interest"] + 0.22 * row["programming_skills"] + 0.14 * row["core_cs_fundamentals"] + 0.10 * row["github_activity_score"] + ai_spec + pref_boost("AI/ML"),
        "Data Analyst": 0.40 * row["data_interest"] + 0.16 * row["aptitude_score"] + 0.13 * row["presentation_skills"] + 0.10 * row["communication_skills"] + pref_boost("Data Analytics"),
        "Cloud Engineer": 0.40 * row["cloud_interest"] + 0.19 * row["backend_interest"] + 0.13 * row["core_cs_fundamentals"] + 2.4 * row["certifications_count"] + pref_boost("Cloud/DevOps"),
        "DevOps Engineer": 0.38 * row["devops_interest"] + 0.24 * row["cloud_interest"] + 0.12 * row["github_activity_score"] + 0.11 * row["problem_solving"] + pref_boost("Cloud/DevOps"),
        "Cybersecurity Analyst": 0.42 * row["security_interest"] + 0.16 * row["core_cs_fundamentals"] + 0.14 * row["logical_reasoning"] + 2.4 * row["certifications_count"] + cyber_spec + pref_boost("Cybersecurity"),
        "QA Engineer": 0.20 * row["technical_test_score"] + 0.19 * row["problem_solving"] + 0.16 * row["communication_skills"] + 0.14 * row["aptitude_score"] + pref_boost("QA/Testing"),
        "Software Engineer": 0.24 * row["programming_skills"] + 0.20 * row["problem_solving"] + 0.18 * row["core_cs_fundamentals"] + 0.14 * row["technical_test_score"] + cs_spec + pref_boost("Software Engineering"),
    }
    for role in ROLE_LABELS:
        scores[role] += rng.normal(0, 2.8)
    return max(scores, key=scores.get)


def generate_dataset(n_rows: int = DATASET_ROWS, random_state: int = 42) -> pd.DataFrame:
    ensure_directories()
    rng = np.random.default_rng(random_state)

    genders = rng.choice(GENDER_OPTIONS, n_rows, p=[0.53, 0.45, 0.02])
    degree_types = rng.choice(DEGREE_OPTIONS, n_rows, p=[0.43, 0.13, 0.17, 0.08, 0.09, 0.05, 0.05])
    specializations = rng.choice(
        SPECIALIZATION_OPTIONS,
        n_rows,
        p=[0.32, 0.18, 0.11, 0.08, 0.07, 0.10, 0.05, 0.05, 0.04],
    )
    college_tiers = rng.choice(COLLEGE_TIER_OPTIONS, n_rows, p=[0.18, 0.45, 0.37])
    preferred_domains = rng.choice(DOMAIN_OPTIONS, n_rows, p=[0.11, 0.19, 0.15, 0.12, 0.12, 0.09, 0.16, 0.06])

    tier_effect = np.select(
        [college_tiers == "Tier 1", college_tiers == "Tier 2", college_tiers == "Tier 3"],
        [0.10, 0.00, -0.08],
    )
    specialization_effect = np.select(
        [
            np.isin(specializations, ["Computer Science", "Information Technology", "AI & Data Science", "Data Science"]),
            specializations == "Cybersecurity",
            specializations == "Electronics & Communication",
        ],
        [0.08, 0.06, 0.02],
        default=-0.05,
    )
    ability = np.clip(rng.normal(0.58, 0.16, n_rows) + tier_effect + specialization_effect, 0.12, 0.98)
    engagement = np.clip(rng.normal(0.55, 0.18, n_rows) + 0.35 * (ability - 0.55), 0.05, 1.0)

    age = np.clip(np.rint(rng.normal(21.4, 1.5, n_rows)), 18, 28).astype(int)
    tenth = _score(rng, 49 + 43 * ability + 4 * tier_effect, 5, 35, 99)
    twelfth = _score(rng, 47 + 44 * ability + 4 * tier_effect, 6, 35, 99)
    graduation = _score(rng, 48 + 42 * ability + 5 * tier_effect, 6, 35, 98)
    cgpa = np.round(_clip(5.0 + 4.35 * ability + 0.25 * tier_effect + rng.normal(0, 0.34, n_rows), 4.5, 10), 2)
    diploma = np.where(
        degree_types == "Diploma",
        _score(rng, 50 + 40 * ability, 6, 35, 98),
        0,
    )

    interests = _domain_interest(rng, preferred_domains, specializations)

    programming = _score(
        rng,
        36 + 42 * ability + 0.12 * interests["backend_interest"] + 0.08 * interests["web_interest"] + 0.08 * interests["ai_interest"],
        8,
    )
    logical = _score(rng, 38 + 45 * ability + 0.10 * interests["data_interest"] + 0.05 * interests["security_interest"], 8)
    aptitude = _score(rng, 39 + 43 * ability + 0.12 * logical, 7)
    core_cs = _score(rng, 35 + 42 * ability + 0.20 * programming + 0.08 * interests["backend_interest"], 8)
    problem_solving = _score(rng, 34 + 39 * ability + 0.18 * programming + 0.12 * logical, 8)
    communication = _score(rng, 39 + 34 * ability + 0.16 * engagement + 0.08 * interests["frontend_interest"], 9)
    english = _score(rng, 42 + 34 * ability + 0.12 * communication, 8)
    teamwork = _score(rng, 45 + 32 * ability + 0.13 * engagement + 0.08 * communication, 8)
    leadership = _score(rng, 35 + 30 * ability + 0.18 * engagement + 0.09 * teamwork, 10)
    presentation = _score(rng, 36 + 32 * ability + 0.20 * communication + 0.06 * interests["frontend_interest"], 8)

    projects = np.clip(
        rng.poisson(0.8 + 2.7 * ability + 1.0 * engagement + programming / 70.0),
        0,
        10,
    ).astype(int)
    internships = np.clip(
        rng.poisson(0.15 + 1.5 * ability + 0.35 * engagement + 0.10 * projects),
        0,
        5,
    ).astype(int)
    certifications = np.clip(
        rng.poisson(0.45 + 1.8 * engagement + 0.55 * ability + 0.08 * projects),
        0,
        8,
    ).astype(int)
    hackathons = np.clip(rng.poisson(0.25 + 1.3 * engagement + programming / 95.0), 0, 7).astype(int)

    github = _score(rng, 20 + 0.44 * programming + 4.3 * projects + 3.2 * hackathons, 9)
    linkedin = _score(rng, 30 + 0.20 * communication + 0.18 * presentation + 4.5 * internships + 2.2 * certifications, 9)
    resume = _score(rng, 31 + 0.16 * graduation + 0.20 * communication + 0.16 * programming + 3.5 * internships, 9)
    mock_interview = _score(rng, 30 + 0.23 * communication + 0.18 * programming + 0.18 * aptitude + 0.12 * presentation, 8)
    technical_test = _score(rng, 28 + 0.30 * programming + 0.23 * core_cs + 0.18 * logical + 0.10 * problem_solving, 8)
    training_attendance = _score(rng, 58 + 30 * engagement + 6 * ability, 8, 35, 100)

    backlog_lambda = np.clip(1.55 - 1.65 * ability - 0.25 * tier_effect, 0.02, 2.2)
    backlogs = np.clip(rng.poisson(backlog_lambda), 0, 8).astype(int)
    gap_years = rng.choice([0, 1, 2, 3], n_rows, p=[0.83, 0.12, 0.04, 0.01])
    disciplinary_flags = rng.choice([0, 1, 2], n_rows, p=[0.955, 0.038, 0.007])
    poor_attendance_flag = ((training_attendance < 62) | (rng.random(n_rows) < 0.045)).astype(int)

    logit = (
        -3.05
        + 0.70 * ((cgpa - 6.5) / 1.7)
        + 0.42 * ((aptitude - 58) / 16)
        + 0.44 * ((programming - 58) / 16)
        + 0.35 * ((communication - 58) / 16)
        + 0.48 * ((technical_test - 58) / 16)
        + 0.30 * ((mock_interview - 58) / 16)
        + 0.23 * internships
        + 0.11 * projects
        + 0.07 * certifications
        + 0.005 * (resume - 60)
        + 0.45 * tier_effect
        - 0.52 * backlogs
        - 0.55 * gap_years
        - 1.00 * disciplinary_flags
        - 0.72 * poor_attendance_flag
        + rng.normal(0, 0.45, n_rows)
    )
    placement_probability = _sigmoid(logit)
    placement_status = (rng.random(n_rows) < placement_probability).astype(int)

    rows: List[Dict[str, float]] = []
    for idx in range(n_rows):
        row = {
            "student_id": f"SH{idx + 1:05d}",
            "gender": genders[idx],
            "age": int(age[idx]),
            "tenth_percentage": round(float(tenth[idx]), 2),
            "twelfth_percentage": round(float(twelfth[idx]), 2),
            "diploma_percentage": round(float(diploma[idx]), 2),
            "graduation_percentage": round(float(graduation[idx]), 2),
            "cgpa": float(cgpa[idx]),
            "degree_type": degree_types[idx],
            "specialization": specializations[idx],
            "college_tier": college_tiers[idx],
            "communication_skills": round(float(communication[idx]), 2),
            "logical_reasoning": round(float(logical[idx]), 2),
            "aptitude_score": round(float(aptitude[idx]), 2),
            "programming_skills": round(float(programming[idx]), 2),
            "core_cs_fundamentals": round(float(core_cs[idx]), 2),
            "problem_solving": round(float(problem_solving[idx]), 2),
            "english_proficiency": round(float(english[idx]), 2),
            "teamwork": round(float(teamwork[idx]), 2),
            "leadership": round(float(leadership[idx]), 2),
            "presentation_skills": round(float(presentation[idx]), 2),
            "projects_count": int(projects[idx]),
            "internships_count": int(internships[idx]),
            "certifications_count": int(certifications[idx]),
            "hackathons_count": int(hackathons[idx]),
            "github_activity_score": round(float(github[idx]), 2),
            "linkedin_strength_score": round(float(linkedin[idx]), 2),
            "resume_score": round(float(resume[idx]), 2),
            "mock_interview_score": round(float(mock_interview[idx]), 2),
            "technical_test_score": round(float(technical_test[idx]), 2),
            "training_attendance_percentage": round(float(training_attendance[idx]), 2),
            "backlogs": int(backlogs[idx]),
            "gap_years": int(gap_years[idx]),
            "disciplinary_flags": int(disciplinary_flags[idx]),
            "poor_attendance_flag": int(poor_attendance_flag[idx]),
            "preferred_domain": preferred_domains[idx],
            "mobile_interest": round(float(interests["mobile_interest"][idx]), 2),
            "web_interest": round(float(interests["web_interest"][idx]), 2),
            "ai_interest": round(float(interests["ai_interest"][idx]), 2),
            "cloud_interest": round(float(interests["cloud_interest"][idx]), 2),
            "backend_interest": round(float(interests["backend_interest"][idx]), 2),
            "frontend_interest": round(float(interests["frontend_interest"][idx]), 2),
            "data_interest": round(float(interests["data_interest"][idx]), 2),
            "security_interest": round(float(interests["security_interest"][idx]), 2),
            "devops_interest": round(float(interests["devops_interest"][idx]), 2),
            "placement_status": int(placement_status[idx]),
            "placement_probability_proxy": round(float(placement_probability[idx]), 4),
        }
        row["recommended_role"] = _role_for_row(row, rng)
        row["readiness_level"] = readiness_level(compute_readiness_score(row)["score"])
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def save_dataset(n_rows: int = DATASET_ROWS, random_state: int = 42) -> pd.DataFrame:
    df = generate_dataset(n_rows=n_rows, random_state=random_state)
    df.to_csv(DATASET_PATH, index=False)
    sample = df.drop(columns=["placement_status", "recommended_role", "placement_probability_proxy", "readiness_level"]).head(12)
    sample.to_csv(SAMPLE_BATCH_PATH, index=False)
    return df


def create_hero_asset() -> None:
    if HERO_IMAGE_PATH.exists():
        return

    mpl_config = PLOT_DIR / "mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    ensure_directories()
    fig, ax = plt.subplots(figsize=(14, 7), dpi=140)
    ax.set_axis_off()
    gradient = np.linspace(0, 1, 900)
    gradient = np.vstack((gradient, gradient))
    ax.imshow(gradient, aspect="auto", cmap="viridis", extent=[0, 14, 0, 7], alpha=0.88)
    ax.add_patch(FancyBboxPatch((0.7, 0.8), 5.0, 5.2, boxstyle="round,pad=0.04,rounding_size=0.12", facecolor="#ffffff", alpha=0.92, linewidth=0))
    ax.add_patch(FancyBboxPatch((6.2, 1.1), 3.0, 4.6, boxstyle="round,pad=0.04,rounding_size=0.12", facecolor="#111827", alpha=0.82, linewidth=0))
    ax.add_patch(FancyBboxPatch((9.7, 1.0), 3.4, 4.9, boxstyle="round,pad=0.04,rounding_size=0.12", facecolor="#ffffff", alpha=0.88, linewidth=0))
    colors = ["#16a34a", "#f59e0b", "#2563eb", "#dc2626", "#0f766e"]
    bars = [3.4, 4.5, 2.6, 3.8, 5.0]
    for i, height in enumerate(bars):
        ax.add_patch(FancyBboxPatch((1.1 + i * 0.75, 1.25), 0.38, height, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=colors[i], alpha=0.88, linewidth=0))
    x = np.linspace(6.55, 8.85, 7)
    y = np.array([1.7, 2.4, 2.0, 3.3, 3.1, 4.2, 4.8])
    ax.plot(x, y, color="#67e8f9", linewidth=4)
    ax.scatter(x, y, s=42, color="#fef3c7")
    for i in range(4):
        ax.add_patch(FancyBboxPatch((10.15, 1.45 + i * 1.0), 2.5, 0.38, boxstyle="round,pad=0.03,rounding_size=0.12", facecolor=colors[i], alpha=0.82, linewidth=0))
    ax.text(1.05, 6.2, "SMART HIRE", fontsize=22, fontweight="bold", color="#111827")
    ax.text(1.05, 5.85, "Placement prediction and role intelligence", fontsize=10, color="#334155")
    fig.savefig(HERO_IMAGE_PATH, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    save_dataset()
    create_hero_asset()

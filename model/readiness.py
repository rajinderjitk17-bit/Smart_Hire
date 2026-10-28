from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def _float(profile: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = profile.get(key, default)
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _count_score(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return _bounded((value / cap) * 100.0)


def readiness_level(score: float) -> str:
    if score >= 82:
        return "Excellent"
    if score >= 68:
        return "Good"
    if score >= 50:
        return "Moderate"
    return "Needs Improvement"


def readiness_badge_class(level: str) -> str:
    return {
        "Excellent": "success",
        "Good": "primary",
        "Moderate": "warning",
        "Needs Improvement": "danger",
    }.get(level, "secondary")


def compute_readiness_score(profile: Dict[str, Any]) -> Dict[str, Any]:
    cgpa_score = _bounded((_float(profile, "cgpa", 0) / 10.0) * 100.0)
    internships_score = _count_score(_float(profile, "internships_count"), 4)
    projects_score = _count_score(_float(profile, "projects_count"), 6)
    certifications_score = _count_score(_float(profile, "certifications_count"), 5)

    weighted_score = (
        0.12 * cgpa_score
        + 0.12 * _float(profile, "aptitude_score")
        + 0.11 * _float(profile, "communication_skills")
        + 0.13 * _float(profile, "programming_skills")
        + 0.14 * _float(profile, "technical_test_score")
        + 0.10 * _float(profile, "mock_interview_score")
        + 0.08 * internships_score
        + 0.08 * projects_score
        + 0.05 * certifications_score
        + 0.07 * _float(profile, "resume_score")
    )

    penalty = (
        4.0 * _float(profile, "backlogs")
        + 5.0 * _float(profile, "gap_years")
        + 8.0 * _float(profile, "disciplinary_flags")
        + 6.0 * _float(profile, "poor_attendance_flag")
    )
    score = round(_bounded(weighted_score - penalty), 1)
    level = readiness_level(score)
    return {
        "score": score,
        "level": level,
        "badge_class": readiness_badge_class(level),
    }


def strengths_and_weaknesses(profile: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    checks = [
        ("cgpa", 8.2, 6.5, "strong CGPA", "CGPA below preferred hiring range"),
        ("aptitude_score", 75, 55, "strong aptitude score", "aptitude score needs work"),
        ("communication_skills", 76, 55, "clear communication skills", "communication score is holding the profile back"),
        ("programming_skills", 76, 55, "solid programming skill signal", "programming fundamentals need improvement"),
        ("core_cs_fundamentals", 75, 55, "good CS fundamentals", "core CS fundamentals need practice"),
        ("technical_test_score", 76, 55, "strong technical test performance", "technical test score is below placement-ready level"),
        ("mock_interview_score", 74, 55, "good interview readiness", "mock interview performance needs focused practice"),
        ("resume_score", 76, 55, "well-positioned resume", "resume score needs improvement"),
        ("github_activity_score", 72, 45, "active GitHub/project footprint", "GitHub activity is light"),
    ]

    strengths: List[str] = []
    weaknesses: List[str] = []

    for key, high, low, strength_text, weakness_text in checks:
        value = _float(profile, key)
        if value >= high:
            strengths.append(strength_text)
        elif value < low:
            weaknesses.append(weakness_text)

    if _float(profile, "projects_count") >= 4:
        strengths.append("good project exposure")
    elif _float(profile, "projects_count") <= 1:
        weaknesses.append("limited project portfolio")

    if _float(profile, "internships_count") >= 2:
        strengths.append("meaningful internship exposure")
    elif _float(profile, "internships_count") == 0:
        weaknesses.append("no internship exposure yet")

    if _float(profile, "certifications_count") >= 3:
        strengths.append("certification-backed learning")
    elif _float(profile, "certifications_count") == 0:
        weaknesses.append("few external certifications")

    if _float(profile, "backlogs") > 0:
        weaknesses.append("active or historical backlog risk")
    if _float(profile, "gap_years") > 0:
        weaknesses.append("gap year risk should be explained well")
    if _float(profile, "poor_attendance_flag") > 0:
        weaknesses.append("attendance risk is visible")
    if _float(profile, "disciplinary_flags") > 0:
        weaknesses.append("disciplinary record requires careful handling")

    if not strengths:
        strengths.append("balanced baseline across several employability areas")
    if not weaknesses:
        weaknesses.append("no major weak area detected from the submitted profile")

    return strengths[:7], weaknesses[:7]


def explanation_text(
    profile: Dict[str, Any],
    placement_label: str,
    placement_probability: float,
    readiness: Dict[str, Any],
    top_roles: Iterable[Dict[str, Any]],
) -> str:
    role_names = ", ".join(role["role"] for role in list(top_roles)[:2])
    technical_signal = (
        _float(profile, "programming_skills")
        + _float(profile, "technical_test_score")
        + _float(profile, "core_cs_fundamentals")
    ) / 3.0
    hiring_signal = (
        _float(profile, "communication_skills")
        + _float(profile, "resume_score")
        + _float(profile, "mock_interview_score")
    ) / 3.0

    if placement_label == "Placed":
        tone = "The profile is above the learned hiring threshold"
    else:
        tone = "The profile is currently below the learned hiring threshold"

    if technical_signal >= 72 and hiring_signal >= 68:
        reason = "because both technical depth and hiring-facing signals are strong."
    elif technical_signal >= hiring_signal:
        reason = "with technical indicators stronger than communication, resume, and interview signals."
    else:
        reason = "with hiring-facing indicators stronger than the current technical depth."

    return (
        f"{tone} with a {placement_probability:.1f}% placement probability. "
        f"Readiness is {readiness['level']} at {readiness['score']:.1f}/100, {reason} "
        f"The closest role direction is {role_names}."
    )

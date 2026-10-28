from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
ARTIFACT_DIR = MODEL_DIR / "artifacts"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
EXPORT_DIR = STATIC_DIR / "exports"
PLOT_DIR = STATIC_DIR / "plots"
IMAGE_DIR = STATIC_DIR / "img"

DATASET_PATH = DATA_DIR / "smart_hire_dataset.csv"
SAMPLE_BATCH_PATH = DATA_DIR / "sample_batch_students.csv"
HERO_IMAGE_PATH = IMAGE_DIR / "smart_hire_analytics_hero.png"

RANDOM_STATE = 42
DATASET_ROWS = 5000

NUMERIC_FEATURES = [
    "age",
    "tenth_percentage",
    "twelfth_percentage",
    "diploma_percentage",
    "graduation_percentage",
    "cgpa",
    "communication_skills",
    "logical_reasoning",
    "aptitude_score",
    "programming_skills",
    "core_cs_fundamentals",
    "problem_solving",
    "english_proficiency",
    "teamwork",
    "leadership",
    "presentation_skills",
    "projects_count",
    "internships_count",
    "certifications_count",
    "hackathons_count",
    "github_activity_score",
    "linkedin_strength_score",
    "resume_score",
    "mock_interview_score",
    "technical_test_score",
    "training_attendance_percentage",
    "backlogs",
    "gap_years",
    "disciplinary_flags",
    "poor_attendance_flag",
    "mobile_interest",
    "web_interest",
    "ai_interest",
    "cloud_interest",
    "backend_interest",
    "frontend_interest",
    "data_interest",
    "security_interest",
    "devops_interest",
]

CATEGORICAL_FEATURES = [
    "gender",
    "degree_type",
    "specialization",
    "college_tier",
    "preferred_domain",
]

INPUT_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

TARGET_COLUMNS = [
    "placement_status",
    "recommended_role",
    "placement_probability_proxy",
    "readiness_level",
]

GENDER_OPTIONS = ["Male", "Female", "Other"]
DEGREE_OPTIONS = ["B.Tech", "B.E.", "BCA", "B.Sc", "MCA", "M.Tech", "Diploma"]
SPECIALIZATION_OPTIONS = [
    "Computer Science",
    "Information Technology",
    "AI & Data Science",
    "Data Science",
    "Cybersecurity",
    "Electronics & Communication",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
]
COLLEGE_TIER_OPTIONS = ["Tier 1", "Tier 2", "Tier 3"]
DOMAIN_OPTIONS = [
    "Mobile Apps",
    "Web Development",
    "AI/ML",
    "Data Analytics",
    "Cloud/DevOps",
    "Cybersecurity",
    "Software Engineering",
    "QA/Testing",
]

ROLE_LABELS = [
    "Android Developer",
    "Flutter Developer",
    "React Native Developer",
    "Web Developer",
    "Frontend Developer",
    "Backend Developer",
    "Full Stack Developer",
    "AI Engineer",
    "Data Scientist",
    "ML Engineer",
    "Data Analyst",
    "Cloud Engineer",
    "DevOps Engineer",
    "Cybersecurity Analyst",
    "QA Engineer",
    "Software Engineer",
]

DEFAULT_PROFILE = {
    "gender": "Male",
    "age": 21,
    "tenth_percentage": 78,
    "twelfth_percentage": 76,
    "diploma_percentage": 0,
    "graduation_percentage": 74,
    "cgpa": 7.4,
    "degree_type": "B.Tech",
    "specialization": "Computer Science",
    "college_tier": "Tier 2",
    "communication_skills": 70,
    "logical_reasoning": 72,
    "aptitude_score": 70,
    "programming_skills": 72,
    "core_cs_fundamentals": 70,
    "problem_solving": 72,
    "english_proficiency": 70,
    "teamwork": 74,
    "leadership": 64,
    "presentation_skills": 68,
    "projects_count": 3,
    "internships_count": 1,
    "certifications_count": 2,
    "hackathons_count": 1,
    "github_activity_score": 62,
    "linkedin_strength_score": 62,
    "resume_score": 68,
    "mock_interview_score": 66,
    "technical_test_score": 70,
    "training_attendance_percentage": 82,
    "backlogs": 0,
    "gap_years": 0,
    "disciplinary_flags": 0,
    "poor_attendance_flag": 0,
    "preferred_domain": "Software Engineering",
    "mobile_interest": 45,
    "web_interest": 70,
    "ai_interest": 55,
    "cloud_interest": 50,
    "backend_interest": 70,
    "frontend_interest": 65,
    "data_interest": 55,
    "security_interest": 40,
    "devops_interest": 45,
}

COUNT_FIELDS = {
    "projects_count",
    "internships_count",
    "certifications_count",
    "hackathons_count",
    "backlogs",
    "gap_years",
    "disciplinary_flags",
    "poor_attendance_flag",
}

PERCENT_OR_SCORE_FIELDS = [
    "tenth_percentage",
    "twelfth_percentage",
    "diploma_percentage",
    "graduation_percentage",
    "communication_skills",
    "logical_reasoning",
    "aptitude_score",
    "programming_skills",
    "core_cs_fundamentals",
    "problem_solving",
    "english_proficiency",
    "teamwork",
    "leadership",
    "presentation_skills",
    "github_activity_score",
    "linkedin_strength_score",
    "resume_score",
    "mock_interview_score",
    "technical_test_score",
    "training_attendance_percentage",
    "mobile_interest",
    "web_interest",
    "ai_interest",
    "cloud_interest",
    "backend_interest",
    "frontend_interest",
    "data_interest",
    "security_interest",
    "devops_interest",
]

FORM_FIELD_GROUPS = [
    {
        "title": "Basic Profile",
        "fields": [
            {"name": "gender", "label": "Gender", "type": "select", "options": GENDER_OPTIONS},
            {"name": "age", "label": "Age", "type": "number", "min": 18, "max": 30, "step": 1},
            {"name": "degree_type", "label": "Degree Type", "type": "select", "options": DEGREE_OPTIONS},
            {
                "name": "specialization",
                "label": "Specialization",
                "type": "select",
                "options": SPECIALIZATION_OPTIONS,
            },
            {
                "name": "college_tier",
                "label": "College Tier",
                "type": "select",
                "options": COLLEGE_TIER_OPTIONS,
            },
        ],
    },
    {
        "title": "Academic Performance",
        "fields": [
            {"name": "tenth_percentage", "label": "10th Percentage", "type": "number", "min": 0, "max": 100, "step": 0.1},
            {"name": "twelfth_percentage", "label": "12th Percentage", "type": "number", "min": 0, "max": 100, "step": 0.1},
            {"name": "diploma_percentage", "label": "Diploma Percentage", "type": "number", "min": 0, "max": 100, "step": 0.1},
            {"name": "graduation_percentage", "label": "Graduation Percentage", "type": "number", "min": 0, "max": 100, "step": 0.1},
            {"name": "cgpa", "label": "CGPA", "type": "number", "min": 0, "max": 10, "step": 0.01},
        ],
    },
    {
        "title": "Skills And Employability",
        "fields": [
            {"name": "communication_skills", "label": "Communication", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "logical_reasoning", "label": "Logical Reasoning", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "aptitude_score", "label": "Aptitude", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "programming_skills", "label": "Programming", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "core_cs_fundamentals", "label": "Core CS", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "problem_solving", "label": "Problem Solving", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "english_proficiency", "label": "English", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "teamwork", "label": "Teamwork", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "leadership", "label": "Leadership", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "presentation_skills", "label": "Presentation", "type": "range", "min": 0, "max": 100, "step": 1},
        ],
    },
    {
        "title": "Experience Signals",
        "fields": [
            {"name": "projects_count", "label": "Projects", "type": "number", "min": 0, "max": 15, "step": 1},
            {"name": "internships_count", "label": "Internships", "type": "number", "min": 0, "max": 8, "step": 1},
            {"name": "certifications_count", "label": "Certifications", "type": "number", "min": 0, "max": 12, "step": 1},
            {"name": "hackathons_count", "label": "Hackathons", "type": "number", "min": 0, "max": 10, "step": 1},
            {"name": "github_activity_score", "label": "GitHub Activity", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "linkedin_strength_score", "label": "LinkedIn Strength", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "resume_score", "label": "Resume Score", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "mock_interview_score", "label": "Mock Interview", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "technical_test_score", "label": "Technical Test", "type": "range", "min": 0, "max": 100, "step": 1},
            {
                "name": "training_attendance_percentage",
                "label": "Training Attendance",
                "type": "number",
                "min": 0,
                "max": 100,
                "step": 0.1,
            },
        ],
    },
    {
        "title": "Risk Factors",
        "fields": [
            {"name": "backlogs", "label": "Backlogs", "type": "number", "min": 0, "max": 12, "step": 1},
            {"name": "gap_years", "label": "Gap Years", "type": "number", "min": 0, "max": 5, "step": 1},
            {"name": "disciplinary_flags", "label": "Disciplinary Flags", "type": "number", "min": 0, "max": 3, "step": 1},
            {"name": "poor_attendance_flag", "label": "Poor Attendance Flag", "type": "number", "min": 0, "max": 1, "step": 1},
        ],
    },
    {
        "title": "Domain Interests",
        "fields": [
            {
                "name": "preferred_domain",
                "label": "Preferred Domain",
                "type": "select",
                "options": DOMAIN_OPTIONS,
            },
            {"name": "mobile_interest", "label": "Mobile", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "web_interest", "label": "Web", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "ai_interest", "label": "AI/ML", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "cloud_interest", "label": "Cloud", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "backend_interest", "label": "Backend", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "frontend_interest", "label": "Frontend", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "data_interest", "label": "Data", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "security_interest", "label": "Security", "type": "range", "min": 0, "max": 100, "step": 1},
            {"name": "devops_interest", "label": "DevOps", "type": "range", "min": 0, "max": 100, "step": 1},
        ],
    },
]

ARTIFACT_FILES = {
    "placement_model": ARTIFACT_DIR / "placement_model.pkl",
    "role_model": ARTIFACT_DIR / "role_model.pkl",
    "placement_preprocessor": ARTIFACT_DIR / "placement_preprocessor.pkl",
    "role_preprocessor": ARTIFACT_DIR / "role_preprocessor.pkl",
    "selected_features_placement": ARTIFACT_DIR / "selected_features_placement.pkl",
    "selected_features_role": ARTIFACT_DIR / "selected_features_role.pkl",
    "placement_metrics": ARTIFACT_DIR / "placement_metrics.json",
    "role_metrics": ARTIFACT_DIR / "role_metrics.json",
    "metadata": ARTIFACT_DIR / "metadata.json",
}


def ensure_directories() -> None:
    for path in [DATA_DIR, ARTIFACT_DIR, UPLOAD_DIR, EXPORT_DIR, PLOT_DIR, IMAGE_DIR]:
        path.mkdir(parents=True, exist_ok=True)

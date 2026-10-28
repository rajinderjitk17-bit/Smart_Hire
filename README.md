# SMART HIRE - Student Placement Prediction System

SMART HIRE is an end-to-end Flask and machine learning project that predicts student placement likelihood and recommends suitable tech job roles. It includes synthetic dataset generation, preprocessing, feature selection, model comparison, saved artifacts, a polished dashboard UI, manual prediction, and batch CSV prediction.

## Key Features

- Placement prediction with probability and confidence score
- Top 3 tech role recommendations
- Custom readiness score from 0 to 100
- Strengths, weak areas, and short explanation for each profile
- Synthetic dataset generation with 5,000 realistic student records
- Binary placement model and multiclass role model
- Model comparison across Logistic Regression, tree models, Random Forest, Extra Trees, and Gradient Boosting
- Feature selection with SelectKBest and ExtraTrees feature importance
- Interactive analytics dashboard with Plotly charts
- Feature insight page with selected, dropped, and ranked features
- Batch CSV upload with downloadable prediction results
- Auto-bootstrap on app startup when data or model artifacts are missing

## Tech Stack

- Python
- Flask
- pandas, NumPy
- scikit-learn
- joblib
- Plotly
- matplotlib and seaborn
- Bootstrap, HTML, CSS, JavaScript

## Project Structure

```text
smart_hire/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── smart_hire_dataset.csv
│   └── sample_batch_students.csv
├── model/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── feature_selection.py
│   ├── train_placement_model.py
│   ├── train_role_model.py
│   ├── evaluate_models.py
│   ├── inference.py
│   ├── readiness.py
│   └── artifacts/
│       ├── placement_model.pkl
│       ├── role_model.pkl
│       ├── placement_preprocessor.pkl
│       ├── role_preprocessor.pkl
│       ├── selected_features_placement.pkl
│       ├── selected_features_role.pkl
│       ├── placement_metrics.json
│       ├── role_metrics.json
│       └── metadata.json
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── img/
│   ├── uploads/
│   └── exports/
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   ├── analytics.html
│   ├── feature_insights.html
│   ├── about.html
│   └── upload.html
└── utils/
    ├── __init__.py
    ├── config.py
    └── helpers.py
```

## Installation

```bash
cd smart_hire
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Run The Project

The app automatically creates the dataset, trains models, saves artifacts, and generates the hero image if anything is missing.

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Train Or Rebuild Artifacts Manually

```bash
python -m model.evaluate_models
```

This regenerates:

- `data/smart_hire_dataset.csv`
- `data/sample_batch_students.csv`
- placement and role model pickle files
- feature selection artifacts
- metrics JSON files
- metadata JSON

## ML Workflow

1. Generate a realistic synthetic student employability dataset.
2. Split data into training and test sets.
3. Scale numeric features and one-hot encode categorical features.
4. Apply `SelectKBest(f_classif)` for feature selection.
5. Train multiple models for placement prediction.
6. Train multiple models for role recommendation.
7. Select the best model based on blended evaluation scores.
8. Save trained sklearn pipelines with joblib.
9. Use saved pipelines for manual and batch web inference.

## Pages And Routes

- `/` - Home and project summary
- `/predict` - Manual student prediction form
- `/result` - Latest prediction result
- `/analytics` - Dashboard with charts and model metrics
- `/features` - Feature selection and pattern insights
- `/upload` - Batch CSV prediction
- `/about` - Project explanation and methodology
- `/sample-batch` - Download sample batch CSV

## Dataset Design

The generated dataset includes academic performance, skills, employability signals, experience, risk factors, and domain interests. Targets are created with meaningful rules and controlled noise:

- Strong aptitude, programming, communication, technical test scores, projects, internships, and resume strength increase placement likelihood.
- Backlogs, gap years, disciplinary flags, and attendance risks reduce placement likelihood.
- Role labels depend on skill-interest patterns such as AI interest plus logical reasoning for AI/ML roles, mobile interest plus programming for mobile roles, and cloud/devops interest plus backend signals for cloud roles.

## Batch Prediction

Use the sample CSV from the Batch page or `data/sample_batch_students.csv`. Uploaded files can include all input features. Missing columns are filled with sensible defaults from the shared config.

## Future Improvements

- Add authentication for placement cell users
- Add database-backed prediction history
- Add SHAP explanations for local interpretability
- Add real placement data ingestion when available
- Add model drift reports across placement seasons

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

from model.evaluate_models import ensure_project_ready
from model.inference import predict_batch, predict_student
from utils.config import (
    DATASET_PATH,
    DEFAULT_PROFILE,
    EXPORT_DIR,
    FORM_FIELD_GROUPS,
    INPUT_FEATURES,
    SAMPLE_BATCH_PATH,
    UPLOAD_DIR,
    ensure_directories,
)
from utils.helpers import dashboard_context, feature_insight_context, form_schema, load_metrics


ensure_directories()
ensure_project_ready(force=False)

app = Flask(__name__)
app.secret_key = "smart-hire-local-demo-secret"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


@app.context_processor
def inject_globals():
    metrics = load_metrics()
    return {
        "app_name": "SMART HIRE",
        "metadata": metrics["metadata"],
        "current_year": datetime.now().year,
    }


@app.route("/")
def index():
    df = pd.read_csv(DATASET_PATH)
    metrics = load_metrics()
    stats = {
        "students": f"{len(df):,}",
        "placement_rate": f"{df['placement_status'].mean() * 100:.1f}%",
        "roles": df["recommended_role"].nunique(),
        "placement_model": metrics["placement"]["best_model"],
        "role_model": metrics["role"]["best_model"],
    }
    return render_template("index.html", stats=stats, metrics=metrics)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        result = predict_student(request.form.to_dict())
        session["latest_result"] = result
        return render_template("result.html", result=result)
    return render_template(
        "predict.html",
        field_groups=form_schema(),
        defaults=DEFAULT_PROFILE,
    )


@app.route("/result")
def result():
    latest_result = session.get("latest_result")
    if not latest_result:
        flash("Run a student profile prediction first.", "warning")
        return redirect(url_for("predict"))
    return render_template("result.html", result=latest_result)


@app.route("/analytics")
def analytics():
    context = dashboard_context()
    return render_template("analytics.html", **context)


@app.route("/features")
def features():
    context = feature_insight_context()
    return render_template("feature_insights.html", **context)


@app.route("/about")
def about():
    metrics = load_metrics()
    return render_template("about.html", metrics=metrics, input_features=INPUT_FEATURES)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    result_table = None
    download_filename = None
    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or not file.filename:
            flash("Choose a CSV file before running batch prediction.", "danger")
            return redirect(url_for("upload"))

        filename = secure_filename(file.filename)
        upload_path = UPLOAD_DIR / filename
        file.save(upload_path)

        try:
            input_frame = pd.read_csv(upload_path)
            output_frame = predict_batch(input_frame)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_filename = f"smart_hire_predictions_{timestamp}.csv"
            output_frame.to_csv(EXPORT_DIR / download_filename, index=False)
            result_table = output_frame.head(30).to_dict(orient="records")
            flash(f"Batch prediction completed for {len(output_frame)} students.", "success")
        except Exception as exc:
            flash(f"Batch prediction failed: {exc}", "danger")
            return redirect(url_for("upload"))

    return render_template(
        "upload.html",
        result_table=result_table,
        download_filename=download_filename,
        required_columns=INPUT_FEATURES,
        sample_exists=Path(SAMPLE_BATCH_PATH).exists(),
    )


@app.route("/download/<path:filename>")
def download(filename):
    return send_from_directory(EXPORT_DIR, filename, as_attachment=True)


@app.route("/sample-batch")
def sample_batch():
    return send_from_directory(SAMPLE_BATCH_PATH.parent, SAMPLE_BATCH_PATH.name, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)

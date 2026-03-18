from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, send_file, url_for

from attendance import AttendanceManager
from config import BASE_PATH, DEFAULT_SETTINGS
from index import ProjectIndex
from io_utils import build_project_paths, ensure_photo_date_dir, iter_person_dirs, list_image_files, load_json
from processor import Processor


def create_app(root_directory: str | None = None) -> Flask:
    app = Flask(__name__, template_folder=str(BASE_PATH / "templates"))
    app.config["SECRET_KEY"] = os.environ.get("FACE_ATTENDANCE_SECRET", "face-attendance-secret")
    app.config["ROOT_PROJECTS"] = str(Path(root_directory or Path.cwd() / "root_projects").resolve())

    processor = Processor(DEFAULT_SETTINGS)

    @app.get("/")
    def home():
        project_index = ProjectIndex(app.config["ROOT_PROJECTS"])
        return render_template("index.html", projects=project_index.list_projects(), root=app.config["ROOT_PROJECTS"])

    @app.post("/projects")
    def create_project():
        project_name = request.form.get("project_name", "").strip()
        if not project_name:
            flash("Project name is required.", "error")
            return redirect(url_for("home"))
        ProjectIndex(app.config["ROOT_PROJECTS"]).create_project(project_name)
        flash(f"Project {project_name} created.", "success")
        return redirect(url_for("view_project", project_name=project_name))

    @app.get("/projects/<project_name>")
    def view_project(project_name: str):
        paths = build_project_paths(app.config["ROOT_PROJECTS"], project_name, DEFAULT_SETTINGS)
        if not paths.project_dir.exists():
            flash("Project not found.", "error")
            return redirect(url_for("home"))
        attendance_html = AttendanceManager(paths).as_html()
        people = []
        for person_dir in iter_person_dirs(paths):
            metadata = load_json(person_dir / DEFAULT_SETTINGS.metadata_filename, default={"display_name": person_dir.name})
            crops = [
                file_path.name
                for file_path in list_image_files(person_dir, DEFAULT_SETTINGS)
                if file_path.name != DEFAULT_SETTINGS.canonical_filename
            ]
            people.append(
                {
                    "name": person_dir.name,
                    "display_name": metadata.get("display_name", person_dir.name),
                    "canonical_exists": (person_dir / DEFAULT_SETTINGS.canonical_filename).exists(),
                    "crops": crops,
                    "metadata": metadata,
                }
            )
        photo_dates = sorted([path.name for path in paths.photos_dir.iterdir() if path.is_dir()]) if paths.photos_dir.exists() else []
        return render_template(
            "project.html",
            project_name=project_name,
            root=app.config["ROOT_PROJECTS"],
            people=people,
            photo_dates=photo_dates,
            attendance_html=attendance_html,
        )

    @app.post("/projects/<project_name>/upload")
    def upload_images(project_name: str):
        date_str = request.form.get("date", "").strip()
        files = request.files.getlist("images")
        if not date_str or not files:
            flash("Date and at least one image are required.", "error")
            return redirect(url_for("view_project", project_name=project_name))
        paths = build_project_paths(app.config["ROOT_PROJECTS"], project_name, DEFAULT_SETTINGS)
        target_dir = ensure_photo_date_dir(paths, date_str)
        saved = 0
        for uploaded in files:
            if not uploaded.filename:
                continue
            destination = target_dir / Path(uploaded.filename).name
            uploaded.save(destination)
            saved += 1
        flash(f"Uploaded {saved} image(s) to {date_str}.", "success")
        return redirect(url_for("view_project", project_name=project_name))

    @app.post("/projects/<project_name>/process")
    def process_project(project_name: str):
        date_str = request.form.get("date", "").strip()
        if not date_str:
            flash("Processing date is required.", "error")
            return redirect(url_for("view_project", project_name=project_name))
        try:
            summary = processor.process(app.config["ROOT_PROJECTS"], project_name, date_str)
            flash(
                f"Processed {summary.images_processed} image(s), kept {summary.detections_kept} face(s), "
                f"created {summary.new_people} new people.",
                "success",
            )
        except Exception as exc:
            flash(f"Processing failed: {exc}", "error")
        return redirect(url_for("view_project", project_name=project_name))

    @app.post("/projects/<project_name>/rename")
    def rename_person(project_name: str):
        current_name = request.form.get("current_name", "").strip()
        new_name = request.form.get("new_name", "").strip()
        if not current_name or not new_name:
            flash("Both current and new names are required.", "error")
            return redirect(url_for("view_project", project_name=project_name))
        try:
            processor.rename_person(app.config["ROOT_PROJECTS"], project_name, current_name, new_name)
            flash(f"Renamed {current_name} to {new_name}.", "success")
        except Exception as exc:
            flash(f"Rename failed: {exc}", "error")
        return redirect(url_for("view_project", project_name=project_name))

    @app.get("/projects/<project_name>/people/<person_name>/image/<filename>")
    def person_image(project_name: str, person_name: str, filename: str):
        paths = build_project_paths(app.config["ROOT_PROJECTS"], project_name, DEFAULT_SETTINGS)
        return send_file(paths.people_dir / person_name / filename)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)

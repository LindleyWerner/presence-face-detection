# Face Detection Attendance System

A Windows-friendly Python application for detecting faces in project photo batches, matching them against known people, creating new identities when needed, and maintaining a single attendance sheet per project.

This project includes:

- A CLI processor for date-based image batches
- A merge tool for correcting duplicated identities
- A Flask web interface for project management and manual review
- Project-level attendance CSV generation
- Logging for detections, matches, and processing events
- A code structure designed to be packaged later with PyInstaller

## Features

- Processes photos organized by project and date
- Prioritizes face detection recall over precision
- Uses RetinaFace first and MTCNN as fallback
- Generates ArcFace embeddings for recognition
- Matches existing people with cosine distance
- Creates new `person_XXXX` folders automatically when no match is found
- Saves face crops and updates per-person embeddings
- Chooses a canonical face image based on detection quality
- Maintains a single attendance CSV per project
- Supports manual correction by renaming or merging people
- Includes a simple Flask frontend for day-to-day use
- Uses `pathlib` and Windows-safe paths throughout

## Tech Stack

### Core

- Python 3.10+ recommended
- Flask
- DeepFace
- ArcFace
- RetinaFace
- MTCNN

### Data and Processing

- NumPy
- Pandas
- OpenCV

### Packaging

- PyInstaller

## Project Structure

```text
root_projects/
    Project_A/
        project_attendance.csv
        processing.log

        people/
            person_0001/
                embeddings.npy
                metadata.json
                canonical.jpg
                image1_face_01.jpg

        photos/
            2025-08-15/
                image1.jpg
                image2.jpg
```

## Repository Structure

```text
FaceDetection/
    app.py
    attendance.py
    build_exe.md
    config.py
    face_engine.py
    index.py
    io_utils.py
    main.py
    merge_people.py
    processor.py
    requirements.txt
    README.md
    templates/
    tests/
```

## How It Works

### Processing flow

1. The processor reads images from `root_projects/<project>/photos/<YYYY-MM-DD>/`
2. It tries face detection with RetinaFace first
3. If needed, it also uses MTCNN as a fallback backend
4. For each face, it generates an ArcFace embedding
5. The embedding is normalized and compared with saved embeddings using cosine distance
6. If the best match is within threshold, the face is assigned to that person
7. Otherwise, a new `person_XXXX` folder is created
8. The face crop, embeddings, metadata, and canonical image are updated
9. The project attendance CSV is updated with presence `1` for that date
10. Processing events are appended to `processing.log`

### Canonical image strategy

Canonical face selection uses:

```text
quality = face_area * detection_confidence
```

This means larger and more confident detections are preferred as the representative image for each person.

## Configuration

Main configuration values are defined in [config.py](D:\Code\FaceDetection\config.py):

```python
DETECTORS = ["retinaface", "mtcnn"]
RECOGNITION_MODEL = "ArcFace"
COSINE_THRESHOLD = 0.45
MIN_FACE_SIZE = 20
```

You can adjust these defaults if you want to trade off recall, precision, or filtering behavior.

## Requirements

- Windows 10 or Windows 11
- Python installed locally
- A working virtual environment
- DeepFace-compatible dependencies installed
- `tf-keras` installed when using TensorFlow 2.20

Install dependencies with:

```bash
pip install -r requirements.txt
```

If DeepFace reports that TensorFlow requires `tf-keras`, run:

```bash
pip install tf-keras
```

## Windows Setup

### 1. Create and activate a virtual environment

In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Confirm Python is working

```powershell
python --version
pip --version
```

## Running the CLI Manually

The main processing entry point is [main.py](D:\Code\FaceDetection\main.py).

### Basic command

```powershell
python main.py --root "D:\root_projects" --project "Project_A" --date 2025-08-15
```

### Parameters

- `--root`: root folder containing all projects
- `--project`: project folder name
- `--date`: date folder under `photos`, in `YYYY-MM-DD` format

### Example

If your images are stored here:

```text
D:\root_projects\Project_A\photos\2025-08-15\
```

run:

```powershell
python main.py --root "D:\root_projects" --project "Project_A" --date 2025-08-15
```

## Running the Merge Script Manually

The merge utility is [merge_people.py](D:\Code\FaceDetection\merge_people.py).

It merges one person into another, combines embeddings, moves crops, merges attendance using OR logic, and removes the source folder.

### Explicit project command

```powershell
python merge_people.py person_0001 person_0002 --root "D:\root_projects" --project "Project_A"
```

This keeps `person_0001` and merges `person_0002` into it.

### Alternate usage

If you run the command from inside the project folder, the script can infer the project and root from the current directory.

```powershell
cd D:\root_projects\Project_A
python D:\Code\FaceDetection\merge_people.py person_0001 person_0002
```

## Running the Flask App on Windows

The web interface entry point is [app.py](D:\Code\FaceDetection\app.py).

### Start the development server

```powershell
python app.py
```

By default, the app runs on:

```text
http://127.0.0.1:5000
```

### What you can do in the web UI

- Create a new project
- Upload images grouped by date
- Trigger processing for a selected date
- View the attendance table
- Browse detected people
- Rename people
- Preview canonical images and saved face crops

### Using a custom projects root

If you want to run the Flask app with a different projects root, you can import `create_app()` from [app.py](D:\Code\FaceDetection\app.py) and pass a custom directory.

Example:

```python
from app import create_app

app = create_app(r"D:\root_projects")
app.run(host="127.0.0.1", port=5000, debug=True)
```

## Attendance CSV Format

Each project has a single CSV file:

```text
project_attendance.csv
```

Example:

```csv
Name,2025-08-01,2025-08-15
person_0001,1,1
person_0002,0,1
```

Rules:

- New date columns are added automatically
- A person is marked `1` if detected at least once on that date
- Merge operations preserve attendance using OR logic

## Logging

Each project has a processing log:

```text
processing.log
```

Typical events include:

- `MATCH`
- `NEW`
- `SKIP_SMALL`
- `EMBED_FAIL`
- `RENAME`
- `MERGE`
- `SUMMARY`

Example log lines:

```text
MATCH person_0001 | img.jpg | dist=0.3200 | conf=0.9800 | backend=retinaface | box=(10, 20, 80, 80)
NEW person_0005 | img.jpg | dist=0.6700 | conf=0.9100 | backend=mtcnn | box=(45, 32, 60, 60)
```

## Testing

The repository includes a small `unittest` suite for attendance logic, merge behavior, and Flask smoke coverage.

Run tests with:

```powershell
python -m unittest discover -s tests -v
```

## Notes About DeepFace and Models

- DeepFace may download or initialize model assets depending on your environment
- First-time model setup can be slower
- Runtime behavior can vary depending on your installed TensorFlow/OpenCV stack
- If you use TensorFlow 2.20, install `tf-keras` in the same environment
- For packaging, always test on the same Windows version you plan to distribute to

## Building an Executable

If you want to package the project into a Windows executable, see [build_exe.md](D:\Code\FaceDetection\build_exe.md).

That guide covers:

- PyInstaller setup
- CLI build commands
- Flask build commands
- handling bundled templates
- notes for DeepFace-related packaging issues

## Development Notes

- Paths are handled with `pathlib`
- Imports are intentionally explicit to reduce packaging issues
- Runtime base paths are prepared for frozen executable mode
- The architecture separates configuration, I/O, attendance, recognition, processing, and web concerns

## Current Modules

- [config.py](D:\Code\FaceDetection\config.py): central settings and frozen-app base path handling
- [io_utils.py](D:\Code\FaceDetection\io_utils.py): project path helpers, JSON helpers, image discovery, logging
- [attendance.py](D:\Code\FaceDetection\attendance.py): attendance CSV management
- [face_engine.py](D:\Code\FaceDetection\face_engine.py): detection, embedding, matching
- [processor.py](D:\Code\FaceDetection\processor.py): main processing workflow, rename, merge
- [index.py](D:\Code\FaceDetection\index.py): project index and creation helpers
- [main.py](D:\Code\FaceDetection\main.py): CLI processing entry point
- [merge_people.py](D:\Code\FaceDetection\merge_people.py): CLI merge entry point
- [app.py](D:\Code\FaceDetection\app.py): Flask application

## Roadmap Ideas

- Add a dedicated merge action in the web UI
- Add person search and filtering
- Add batch project processing
- Add stronger validation for uploaded files
- Add thumbnail galleries for more saved face crops
- Add FAISS-based matching for larger datasets
- Add export/reporting improvements

## License

Add your preferred license here before publishing publicly.



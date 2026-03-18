# Building a Windows executable with PyInstaller

## 1. Install dependencies

```bash
pip install -r requirements.txt
pip install pyinstaller
```

If DeepFace asks for additional backend packages in your environment, install them before freezing.
With TensorFlow 2.20, make sure `tf-keras` is installed as well:

```bash
pip install tf-keras
```

## 2. Build the CLI executable

```bash
pyinstaller --onefile --name face_attendance main.py
```

Use the generated executable from `dist\face_attendance.exe`.

## 3. Build the Flask UI executable

```bash
pyinstaller --onefile --name face_attendance_web --add-data "templates;templates" app.py
```

## 4. Notes for DeepFace and frozen apps

- This project keeps imports explicit at module level where practical and centralizes runtime paths through `config.py`.
- `config.py` uses:

```python
import sys
if getattr(sys, "frozen", False):
    BASE_PATH = sys._MEIPASS
else:
    BASE_PATH = Path(__file__).parent
```

- If PyInstaller misses model-related modules on your machine, rebuild with collection flags such as:

```bash
pyinstaller --onefile --collect-all deepface --collect-all retinaface --collect-all mtcnn main.py
```

- For the web build, keep `templates` bundled with `--add-data`.
- Always test the executable on a clean Windows machine after build, because DeepFace backend requirements vary by installed runtime.

## 5. Example CLI usage after build

```bash
face_attendance.exe --root "D:\root_projects" --project "Project_A" --date 2025-08-15
```

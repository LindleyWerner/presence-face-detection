from __future__ import annotations

import pandas as pd

from io_utils import ProjectPaths


class AttendanceManager:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def load(self) -> pd.DataFrame:
        if not self.paths.attendance_csv.exists():
            return pd.DataFrame(columns=["Name"])
        frame = pd.read_csv(self.paths.attendance_csv)
        if "Name" not in frame.columns:
            frame.insert(0, "Name", "")
        return frame

    def save(self, frame: pd.DataFrame) -> None:
        frame.to_csv(self.paths.attendance_csv, index=False)

    def ensure_date_column(self, frame: pd.DataFrame, date_str: str) -> pd.DataFrame:
        if date_str not in frame.columns:
            frame[date_str] = 0
        return frame

    def mark_present(self, person_name: str, date_str: str) -> None:
        frame = self.load()
        frame = self.ensure_date_column(frame, date_str)
        if person_name not in frame["Name"].values:
            new_row = {column: 0 for column in frame.columns}
            new_row["Name"] = person_name
            frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
        frame.loc[frame["Name"] == person_name, date_str] = 1
        self.save(frame)

    def rename_person(self, old_name: str, new_name: str) -> None:
        frame = self.load()
        if old_name in frame["Name"].values:
            frame.loc[frame["Name"] == old_name, "Name"] = new_name
            self.save(frame)

    def merge_people(self, target_name: str, source_name: str) -> None:
        frame = self.load()
        if target_name not in frame["Name"].values:
            return
        date_columns = [column for column in frame.columns if column != "Name"]
        if source_name in frame["Name"].values:
            source_row = frame.loc[frame["Name"] == source_name, date_columns].fillna(0)
            target_row = frame.loc[frame["Name"] == target_name, date_columns].fillna(0)
            merged = source_row.astype(int).reset_index(drop=True).combine(
                target_row.astype(int).reset_index(drop=True),
                lambda left, right: left.where(left >= right, right),
            ).iloc[0]
            for column in date_columns:
                frame.loc[frame["Name"] == target_name, column] = int(merged[column])
            frame = frame.loc[frame["Name"] != source_name].reset_index(drop=True)
            self.save(frame)

    def as_html(self) -> str:
        frame = self.load()
        if frame.empty:
            return "<p>No attendance records yet.</p>"
        return frame.to_html(index=False, classes=["attendance-table"], border=0)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Iterable

from .resume import current_task_id


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    fingerprint: str
    status: str
    started_at: str | None
    ended_at: str | None
    exit_code: int | None
    log_path: str | None
    message: str | None
    resume_fingerprint: str | None = None
    portable_checkpoint: str | None = None


class StateStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS task_runs (
                task_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                exit_code INTEGER,
                log_path TEXT,
                message TEXT,
                resume_fingerprint TEXT
            )
            """
        )
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(task_runs)")}
        if "resume_fingerprint" not in columns:
            self.connection.execute("ALTER TABLE task_runs ADD COLUMN resume_fingerprint TEXT")
        if "portable_checkpoint" not in columns:
            self.connection.execute("ALTER TABLE task_runs ADD COLUMN portable_checkpoint TEXT")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS workflow_metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        # Rename database keys only; keep timestamps, hashes, status and old log
        # locators intact. Never overwrite a record already under the new name.
        aliases = [(current_task_id(row[0]), row[0]) for row in
                   self.connection.execute("SELECT task_id FROM task_runs").fetchall()
                   if current_task_id(row[0]) != row[0]]
        self.connection.executemany(
            "UPDATE OR IGNORE task_runs SET task_id=? WHERE task_id=?", aliases,
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get_metadata(self, name: str) -> str | None:
        row = self.connection.execute("SELECT value FROM workflow_metadata WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    def set_metadata(self, name: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO workflow_metadata(name, value) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET value=excluded.value", (name, value),
        )
        self.connection.commit()

    def get(self, task_id: str) -> TaskRecord | None:
        row = self.connection.execute(
            "SELECT task_id, fingerprint, status, started_at, ended_at, exit_code, log_path, message, resume_fingerprint, portable_checkpoint "
            "FROM task_runs WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return TaskRecord(*row) if row else None

    def start(self, task_id: str, fingerprint: str, log_path: Path,
              resume_fingerprint: str | None = None,
              portable_checkpoint: str | None = None) -> None:
        self.connection.execute(
            """
            INSERT INTO task_runs(task_id, fingerprint, status, started_at, ended_at, exit_code, log_path, message, resume_fingerprint, portable_checkpoint)
            VALUES (?, ?, 'running', ?, NULL, NULL, ?, NULL, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                status='running',
                started_at=excluded.started_at,
                ended_at=NULL,
                exit_code=NULL,
                log_path=excluded.log_path,
                message=NULL,
                resume_fingerprint=excluded.resume_fingerprint,
                portable_checkpoint=excluded.portable_checkpoint
            """,
            (task_id, fingerprint, utc_now(), str(log_path), resume_fingerprint, portable_checkpoint),
        )
        self.connection.commit()

    def finish(self, task_id: str, status: str, exit_code: int, message: str = "") -> None:
        self.connection.execute(
            "UPDATE task_runs SET status=?, ended_at=?, exit_code=?, message=? WHERE task_id=?",
            (status, utc_now(), exit_code, message, task_id),
        )
        self.connection.commit()

    def adopt(self, task_id: str, fingerprint: str, message: str,
              resume_fingerprint: str | None = None,
              portable_checkpoint: str | None = None) -> None:
        """Record complete pre-existing outputs as a successful reusable task."""
        timestamp = utc_now()
        self.connection.execute(
            """
            INSERT INTO task_runs(
                task_id, fingerprint, status, started_at, ended_at,
                exit_code, log_path, message, resume_fingerprint, portable_checkpoint
            )
            VALUES (?, ?, 'success', ?, ?, 0, NULL, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                status='success',
                started_at=excluded.started_at,
                ended_at=excluded.ended_at,
                exit_code=0,
                log_path=NULL,
                message=excluded.message,
                resume_fingerprint=excluded.resume_fingerprint,
                portable_checkpoint=excluded.portable_checkpoint
            """,
            (task_id, fingerprint, timestamp, timestamp, message, resume_fingerprint, portable_checkpoint),
        )
        self.connection.commit()

    def set_resume_fingerprint(self, task_id: str, fingerprint: str) -> None:
        """Upgrade a reusable record without rewriting execution provenance."""
        self.connection.execute(
            "UPDATE task_runs SET resume_fingerprint=? WHERE task_id=?",
            (fingerprint, task_id),
        )
        self.connection.commit()

    def set_portable_checkpoint(self, task_id: str, checkpoint: str) -> None:
        self.connection.execute(
            "UPDATE task_runs SET portable_checkpoint=? WHERE task_id=?",
            (checkpoint, task_id),
        )
        self.connection.commit()

    def relocate_log(self, task_id: str, log_path: Path) -> None:
        """Change the log locator, never its contents or execution provenance."""
        self.connection.execute(
            "UPDATE task_runs SET log_path=? WHERE task_id=?", (str(log_path), task_id),
        )
        self.connection.commit()

    def records(self) -> Iterable[TaskRecord]:
        rows = self.connection.execute(
            "SELECT task_id, fingerprint, status, started_at, ended_at, exit_code, log_path, message, resume_fingerprint, portable_checkpoint "
            "FROM task_runs ORDER BY task_id"
        )
        for row in rows:
            yield TaskRecord(*row)

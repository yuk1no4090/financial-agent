from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from deerflow.agents.memory.memory_schema import MemoryRecord
from deerflow.agents.memory.storage import utc_now_iso_z
from deerflow.config.paths import get_paths

logger = logging.getLogger(__name__)


class MemoryRecordStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else get_paths().base_dir / "memory_records.jsonl"
        self._lock = threading.Lock()
        self._records_cache: list[MemoryRecord] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_parent(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _read_records_from_disk(self) -> list[MemoryRecord]:
        if not self._path.exists():
            return []

        records: list[MemoryRecord] = []
        try:
            with self._path.open(encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError:
                        logger.warning("Skipping invalid memory record line in %s", self._path)
                        continue
                    if not isinstance(payload, dict):
                        continue
                    record = MemoryRecord.from_dict(payload)
                    if record.id and record.content:
                        records.append(record)
        except OSError:
            logger.exception("Failed to read memory records from %s", self._path)
            return []

        return records

    def list_records(self) -> list[MemoryRecord]:
        with self._lock:
            if self._records_cache is not None:
                return list(self._records_cache)

        records = self._read_records_from_disk()
        with self._lock:
            self._records_cache = list(records)
        return records

    def replace_all(self, records: list[MemoryRecord]) -> None:
        self._ensure_parent()
        temp_path = self._path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        temp_path.replace(self._path)
        self._records_cache = list(records)

    def add(self, record: MemoryRecord) -> MemoryRecord:
        with self._lock:
            records = list(self._records_cache) if self._records_cache is not None else self._read_records_from_disk()
            records.append(record)
            self.replace_all(records)
        return record

    def update(self, record_id: str, **fields: object) -> MemoryRecord | None:
        with self._lock:
            records = list(self._records_cache) if self._records_cache is not None else self._read_records_from_disk()
            updated: MemoryRecord | None = None
            for idx, record in enumerate(records):
                if record.id != record_id:
                    continue
                payload = record.to_dict()
                payload.update(fields)
                payload["updated_at"] = utc_now_iso_z()
                updated = MemoryRecord.from_dict(payload)
                records[idx] = updated
                break
            if updated is None:
                return None
            self.replace_all(records)
            return updated

    def delete(self, record_id: str) -> bool:
        with self._lock:
            records = list(self._records_cache) if self._records_cache is not None else self._read_records_from_disk()
            filtered = [record for record in records if record.id != record_id]
            if len(filtered) == len(records):
                return False
            self.replace_all(filtered)
            return True

    def list_by_thread(self, thread_id: str) -> list[MemoryRecord]:
        return [record for record in self.list_records() if record.thread_id == thread_id]


_memory_record_store: MemoryRecordStore | None = None
_memory_record_store_lock = threading.Lock()


def get_memory_record_store() -> MemoryRecordStore:
    global _memory_record_store
    if _memory_record_store is not None:
        return _memory_record_store

    with _memory_record_store_lock:
        if _memory_record_store is None:
            _memory_record_store = MemoryRecordStore()
        return _memory_record_store

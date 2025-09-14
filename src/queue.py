# src/queue.py
import queue
import threading
import time
from typing import Any, Dict, Callable, Optional
from dataclasses import dataclass, field

@dataclass
class Job:
    id: str
    payload: Dict[str, Any]
    status: str = "queued"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

class InMemoryQueue:
    def __init__(self):
        self.q = queue.Queue()
        self.store: Dict[str, Job] = {}
        self.worker_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def submit(self, job: Job):
        self.store[job.id] = job
        self.q.put(job.id)

    def get(self, job_id: str) -> Optional[Job]:
        return self.store.get(job_id)

    def run_worker(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        def loop():
            while not self._stop.is_set():
                try:
                    job_id = self.q.get(timeout=0.1)
                except queue.Empty:
                    continue
                job = self.store[job_id]
                job.status = "running"
                job.started_at = time.time()
                try:
                    job.result = handler(job.payload)
                    job.status = "done"
                except Exception as e:
                    job.status = "error"
                    job.error = str(e)
                finally:
                    job.finished_at = time.time()
                    self.q.task_done()
        self.worker_thread = threading.Thread(target=loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        self._stop.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1)


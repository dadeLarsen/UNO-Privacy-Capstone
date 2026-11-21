"""In-memory policy and job store. Nothing is written to disk (FR-22, NFR-02)."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from .errors import NotFound, PolicyTooLong
from .schemas import (
    MAX_POLICY_CHARACTERS,
    AnalysisPhase,
    AnalysisResults,
    ApiErrorInfo,
    Classification,
    GenerationResult,
    Job,
    JobProgress,
    JobStatus,
    Policy,
    PolicySource,
    RewriteResult,
    SummaryResult,
)


@dataclass
class JobRecord:
    job: Job
    policy_id: UUID
    cancel: threading.Event = field(default_factory=threading.Event)


@dataclass
class PolicyRecord:
    policy: Policy
    text: str
    classification: Classification | None = None
    latest_summary: SummaryResult | None = None
    latest_rewrite: RewriteResult | None = None


class Store:
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = threading.RLock()
        self._policies: dict[UUID, PolicyRecord] = {}
        self._jobs: dict[UUID, JobRecord] = {}

    def create_policy(
        self,
        text: str,
        source: PolicySource,
        filename: str | None = None,
    ) -> PolicyRecord:
        # Pasted and uploaded text both arrive here, so both get the same answer.
        if len(text) > MAX_POLICY_CHARACTERS:
            raise PolicyTooLong
        policy = Policy(
            id=uuid4(),
            source=source,
            filename=filename,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            characters=len(text),
        )
        record = PolicyRecord(policy=policy, text=text)
        with self._lock:
            self._expire_locked()
            self._policies[policy.id] = record
        return record

    def get_policy(self, policy_id: UUID) -> PolicyRecord:
        with self._lock:
            self._expire_locked()
            record = self._policies.get(policy_id)
            if record is None:
                raise NotFound("Policy")
            return record

    def set_classification(self, policy_id: UUID, classification: Classification) -> None:
        with self._lock:
            self.get_policy(policy_id).classification = classification

    def create_job(self, policy_id: UUID, kind: str) -> JobRecord:
        job = Job(
            id=uuid4(),
            policy_id=policy_id,
            kind=kind,
            status=JobStatus.QUEUED,
            progress=JobProgress(stage="queued", completed=0, total=1),
        )
        record = JobRecord(job=job, policy_id=policy_id)
        with self._lock:
            self.get_policy(policy_id)
            self._jobs[job.id] = record
        return record

    def get_job(self, job_id: UUID) -> JobRecord:
        with self._lock:
            self._expire_locked()
            record = self._jobs.get(job_id)
            if record is None:
                raise NotFound("Job")
            return record

    def set_progress(self, job_id: UUID, stage: str, completed: int, total: int, message: str | None = None) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.job.status in (JobStatus.SUCCEEDED, JobStatus.CANCELLED, JobStatus.FAILED):
                return
            record.job.progress = JobProgress(
                stage=stage,
                completed=completed,
                total=total,
                message=message,
            )
            record.job.status = JobStatus.RUNNING
            record.job.updated_at = datetime.now(timezone.utc)

    def finish_job(
        self,
        job_id: UUID,
        *,
        status: JobStatus,
        result: GenerationResult | None = None,
        error: ApiErrorInfo | None = None,
        progress: JobProgress | None = None,
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.job.status in (JobStatus.SUCCEEDED, JobStatus.CANCELLED, JobStatus.FAILED):
                return
            record.job.status = status
            record.job.result = result
            record.job.error = error
            if progress is not None:
                record.job.progress = progress
            record.job.updated_at = datetime.now(timezone.utc)
            if status is JobStatus.SUCCEEDED and result is not None:
                policy = self._policies.get(record.policy_id)
                if policy is not None:
                    if isinstance(result, SummaryResult):
                        policy.latest_summary = result
                    elif isinstance(result, RewriteResult):
                        policy.latest_rewrite = result

    def cancel_job(self, job_id: UUID) -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
                return False
            record.cancel.set()
            record.job.status = JobStatus.CANCELLED
            record.job.progress = JobProgress(stage="cancelled", completed=0, total=1)
            record.job.updated_at = datetime.now(timezone.utc)
            return True

    def snapshot(self, policy_id: UUID) -> AnalysisResults:
        with self._lock:
            record = self.get_policy(policy_id)
            jobs = [item.job for item in self._jobs.values() if item.policy_id == policy_id]
            classification = record.classification
            phase = AnalysisPhase.AWAITING_CLASSIFICATION
            if classification is not None:
                phase = AnalysisPhase.CLASSIFIED
                if any(job.status in (JobStatus.QUEUED, JobStatus.RUNNING) for job in jobs):
                    phase = AnalysisPhase.GENERATING
                if record.latest_summary is not None or record.latest_rewrite is not None:
                    phase = AnalysisPhase.RESULTS_READY
            return AnalysisResults(
                policy=record.policy.model_copy(deep=True),
                phase=phase,
                classification=classification.model_copy(deep=True) if classification else None,
                summaries=list(record.latest_summary.summaries) if record.latest_summary else [],
                rewrites=list(record.latest_rewrite.rewrites) if record.latest_rewrite else [],
            )

    def _expire_locked(self) -> None:
        cutoff = datetime.now(timezone.utc) - self._ttl
        stale = {pid for pid, record in self._policies.items() if record.policy.created_at < cutoff}
        for policy_id in stale:
            del self._policies[policy_id]
        for job_id, record in list(self._jobs.items()):
            if record.policy_id in stale:
                del self._jobs[job_id]

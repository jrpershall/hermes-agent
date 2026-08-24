"""Scheduler-owned managed-execution provenance for cron workers.

Contract under test (declared by a job's ``managed_execution_context``):

* A model worker for a managed job must not begin until the scheduler has
  durably bound it to exactly one execution row — lane, ``worker_started_at``,
  a session-identity digest and an execution-capability digest — and the
  worker process receives the exact execution ID plus the RAW capability
  through the configured environment variable names.
* Only digests ever enter SQLite; the raw capability never appears in the
  ledger, the job output, logs, or error text.
* A script gate that returns ``wakeAgent=false`` produces NO worker evidence
  and no provider call, and the gate script itself never sees the binding.
* Malformed declarations and ledger write failures stop BEFORE dispatch.
* Terminalization and interrupted-execution recovery preserve the binding.
* Jobs that do not declare a managed context keep today's ledger shape.

The downstream consumer is an external controller that reads the ledger with
a fixed column projection and resolves exactly one running bound row per
job/lane (``_trusted_fixer_principal``).  Its invariant is encoded here so a
Hermes change that breaks it fails in Hermes CI, not in production.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from cron import executions, scheduler
from tools.environments import local

ID_ENV = "HERMES_MANAGED_EXECUTION_ID"
CAP_ENV = "HERMES_MANAGED_EXECUTION_CAPABILITY"
CONTEXT = {
    "lane_id": "repair-3",
    "execution_id_env": ID_ENV,
    "execution_capability_env": CAP_ENV,
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
# Any base64url run long enough to be a capability — must never leak.
TOKEN_LIKE_RE = re.compile(r"[A-Za-z0-9_-]{40,}")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _principal_never_outlives_a_test():
    """Attribute a leak to the test that caused it, not to the next one."""
    assert local._MANAGED_EXECUTION_ENV.get() is None
    yield
    assert local._MANAGED_EXECUTION_ENV.get() is None


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with a scripts dir, ledger, and cron store."""
    root = tmp_path / "home"
    (root / "scripts").mkdir(parents=True)
    (root / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(scheduler, "_get_hermes_home", lambda: root)
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", root / "cron" / "executions.db")
    return root


def _write_gate(home: Path, wake: bool) -> str:
    """Gate script that also records the env it saw, for leak assertions."""
    probe = home / "gate-env.json"
    script = home / "scripts" / "gate.py"
    script.write_text(
        "import json, os, sys\n"
        f"open({str(probe)!r}, 'w').write(json.dumps({{k: os.environ.get(k) for k in "
        f"({ID_ENV!r}, {CAP_ENV!r})}}))\n"
        f"print(json.dumps({{'wakeAgent': {wake!r}, 'data': {{'tick': 1}}}}))\n",
        encoding="utf-8",
    )
    return "gate.py"


class _RunJobStubs:
    """Patch run_job's provider/config seams so it runs without credentials.

    Mirrors ``tests/cron/test_cron_workdir.py::TestRunJobTerminalCwd``; the
    script gate, ledger, and env plumbing are deliberately left REAL.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        self.observed: dict = {}
        observed = self.observed

        class FakeAgent:
            def __init__(self, **kwargs):
                observed["constructed"] = True
                observed["session_id"] = kwargs.get("session_id")
                # Evidence visible to the worker at construction time.
                observed["env_at_init"] = local.build_subprocess_env({})
                observed["ledger_at_init"] = executions.list_executions(limit=50)

            def run_conversation(self, *_a, **_kw):
                observed["env_terminal"] = local.build_subprocess_env({})
                observed["env_cli_executor"] = local.hermes_subprocess_env()
                return {"final_response": "[SILENT]", "messages": []}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

        fake_mod = type(sys)("run_agent")
        fake_mod.AIAgent = FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

        from hermes_cli import runtime_provider as _rtp

        monkeypatch.setattr(
            _rtp,
            "resolve_runtime_provider",
            lambda **_kw: {
                "provider": "test",
                "api_key": "k",
                "base_url": "http://test.local",
                "api_mode": "chat_completions",
            },
        )
        monkeypatch.setattr(
            scheduler, "_build_job_prompt", lambda job, prerun_script=None, **kw: "hi"
        )
        monkeypatch.setattr(scheduler, "_resolve_origin", lambda job: None)
        monkeypatch.setattr(scheduler, "_resolve_delivery_target", lambda job: None)
        monkeypatch.setattr(
            scheduler, "_resolve_cron_enabled_toolsets", lambda job, cfg: None
        )
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)

        # run_one_job seams that would otherwise touch the real store.
        monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)
        monkeypatch.setattr(
            scheduler,
            "save_job_output",
            lambda _job_id, doc, *_a, **_kw: (
                observed.setdefault("docs", []).append(doc) or (tmp_path / "out.md")
            ),
        )
        monkeypatch.setattr(
            scheduler,
            "mark_job_run",
            lambda job_id, success, error=None, **_kw: observed.setdefault(
                "runs", []
            ).append((job_id, success, error)),
        )


def _managed_job(script: str, **extra) -> dict:
    return {
        "id": "managed-job",
        "name": "managed job",
        "prompt": "do the work",
        "script": script,
        "schedule_display": "manual",
        "managed_execution_context": dict(CONTEXT),
        **extra,
    }


# ---------------------------------------------------------------------------
# 1. managed wake-agent execution gets one durable principal
# ---------------------------------------------------------------------------


def test_managed_wake_agent_execution_gets_one_durable_principal(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    stubs = _RunJobStubs(monkeypatch, tmp_path)
    job = _managed_job(_write_gate(home, wake=True))

    with caplog.at_level("DEBUG"):
        assert scheduler.run_one_job(job) is True

    observed = stubs.observed
    assert observed.get("constructed") is True

    # The worker saw the binding through BOTH spawn surfaces: the terminal
    # tool env and the model-driving CLI executor env (ACP / codex / claude).
    for env in (
        observed["env_at_init"],
        observed["env_terminal"],
        observed["env_cli_executor"],
    ):
        assert env[ID_ENV], env
        assert env[CAP_ENV], env
    execution_id = observed["env_terminal"][ID_ENV]
    capability = observed["env_terminal"][CAP_ENV]
    assert observed["env_cli_executor"][ID_ENV] == execution_id
    assert observed["env_cli_executor"][CAP_ENV] == capability
    assert 16 <= len(capability) <= 512

    # Exactly one execution row, bound BEFORE the agent was constructed.
    rows = observed["ledger_at_init"]
    assert len(rows) == 1
    bound = rows[0]
    assert bound["id"] == execution_id
    assert bound["job_id"] == "managed-job"
    assert bound["status"] == "running"
    assert bound["lane_id"] == "repair-3"
    assert bound["worker_started_at"]
    assert bound["worker_started_at"] >= bound["started_at"]
    assert SHA256_RE.match(bound["session_sha256"])
    assert SHA256_RE.match(bound["capability_sha256"])
    assert bound["capability_sha256"] == hashlib.sha256(capability.encode()).hexdigest()
    # Session identity is derived from the exact execution + cron session.
    assert (
        bound["session_sha256"]
        == hashlib.sha256(
            f"{execution_id}:{observed['session_id']}".encode()
        ).hexdigest()
    )

    # The raw capability never leaves the worker context.
    persisted = executions.latest_execution("managed-job")
    assert persisted["status"] == "completed"
    assert persisted["lane_id"] == "repair-3"
    assert persisted["worker_started_at"] == bound["worker_started_at"]
    assert capability not in json.dumps(persisted)
    with sqlite3.connect(executions.EXECUTIONS_FILE) as db:
        dump = "\n".join(db.iterdump())
    assert capability not in dump
    assert capability not in "\n".join(observed.get("docs", []))
    assert capability not in caplog.text
    assert not any(capability in str(r) for r in observed.get("runs", []))

    # The gate script ran before binding and saw nothing.
    assert json.loads((home / "gate-env.json").read_text()) == {
        ID_ENV: None,
        CAP_ENV: None,
    }

    # Nothing leaks out of the job's context afterwards.
    assert ID_ENV not in local.build_subprocess_env({})
    assert CAP_ENV not in local.hermes_subprocess_env()
    assert ID_ENV not in os.environ and CAP_ENV not in os.environ


# ---------------------------------------------------------------------------
# 2. script-only skip creates no worker principal
# ---------------------------------------------------------------------------


def test_script_only_skip_creates_no_worker_principal(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = _RunJobStubs(monkeypatch, tmp_path)
    job = _managed_job(_write_gate(home, wake=False))

    assert scheduler.run_one_job(job) is True

    assert "constructed" not in stubs.observed  # no provider call
    rows = executions.list_executions(job_id="managed-job")
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"
    assert row["worker_started_at"] is None
    assert row["lane_id"] is None
    assert row["session_sha256"] is None
    assert row["capability_sha256"] is None
    assert json.loads((home / "gate-env.json").read_text()) == {
        ID_ENV: None,
        CAP_ENV: None,
    }


# ---------------------------------------------------------------------------
# 3. malformed managed context fails closed
# ---------------------------------------------------------------------------

MALFORMED = {
    "missing-lane": {"execution_id_env": ID_ENV, "execution_capability_env": CAP_ENV},
    "empty-lane": {**CONTEXT, "lane_id": ""},
    "empty-id-env": {**CONTEXT, "execution_id_env": ""},
    "empty-cap-env": {**CONTEXT, "execution_capability_env": ""},
    "duplicate-env-names": {**CONTEXT, "execution_capability_env": ID_ENV},
    "non-string-lane": {**CONTEXT, "lane_id": 3},
    "non-string-env": {**CONTEXT, "execution_id_env": ["X"]},
    "unknown-field": {**CONTEXT, "extra": "x"},
    "protected-PATH": {**CONTEXT, "execution_capability_env": "PATH"},
    "protected-HOME": {**CONTEXT, "execution_id_env": "HOME"},
    "protected-HERMES_HOME": {**CONTEXT, "execution_id_env": "HERMES_HOME"},
    "protected-PYTHONPATH": {**CONTEXT, "execution_capability_env": "PYTHONPATH"},
    "protected-DYLD-prefix": {
        **CONTEXT,
        "execution_capability_env": "DYLD_INSERT_LIBRARIES",
    },
    "protected-LD-prefix": {**CONTEXT, "execution_capability_env": "LD_PRELOAD"},
    "protected-BASH_ENV": {**CONTEXT, "execution_capability_env": "BASH_ENV"},
    "protected-NODE_OPTIONS": {**CONTEXT, "execution_id_env": "NODE_OPTIONS"},
    "protected-GIT-prefix": {**CONTEXT, "execution_capability_env": "GIT_SSH_COMMAND"},
    "invalid-identifier": {**CONTEXT, "execution_id_env": "has-dash"},
    "lowercase-identifier": {**CONTEXT, "execution_id_env": "hermes_managed"},
    "not-an-object": "repair-3",
    "list": [CONTEXT],
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_managed_context_is_rejected_at_store_boundary(
    case: str, home: Path
) -> None:
    from cron.jobs import (
        create_job,
        update_job,
        use_cron_store,
        validate_managed_execution_context,
    )

    with pytest.raises(ValueError):
        validate_managed_execution_context(MALFORMED[case])

    with use_cron_store(home):
        with pytest.raises(ValueError):
            create_job(
                prompt="p",
                schedule="every 1m",
                managed_execution_context=MALFORMED[case],
            )
        job = create_job(prompt="p", schedule="every 1m")
        with pytest.raises(ValueError):
            update_job(job["id"], {"managed_execution_context": MALFORMED[case]})


def test_valid_managed_context_round_trips_and_undeclared_jobs_are_untouched(
    home: Path,
) -> None:
    from cron.jobs import create_job, get_job, update_job, use_cron_store

    with use_cron_store(home):
        plain = create_job(prompt="p", schedule="every 1m")
        assert "managed_execution_context" not in plain
        assert "managed_execution_context" not in get_job(plain["id"])

        managed = create_job(
            prompt="p", schedule="every 1m", managed_execution_context=dict(CONTEXT)
        )
        assert get_job(managed["id"])["managed_execution_context"] == CONTEXT

        updated = update_job(plain["id"], {"managed_execution_context": dict(CONTEXT)})
        assert updated["managed_execution_context"] == CONTEXT
        cleared = update_job(plain["id"], {"managed_execution_context": None})
        assert cleared.get("managed_execution_context") is None


@pytest.mark.parametrize(
    "case", ["missing-lane", "protected-PATH", "duplicate-env-names"]
)
def test_hand_edited_malformed_context_never_reaches_dispatch(
    case: str, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A jobs.json record edited outside the store API still fails closed."""
    from cron.jobs import get_job, save_jobs, use_cron_store

    stubs = _RunJobStubs(monkeypatch, tmp_path)
    job = _managed_job(
        _write_gate(home, wake=True), managed_execution_context=MALFORMED[case]
    )
    job.update({
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 1},
    })

    with use_cron_store(home):
        save_jobs([job])
        assert scheduler.run_one_job(dict(job)) is True
        stored = get_job("managed-job")

    assert "constructed" not in stubs.observed
    assert stored["state"] == "paused"
    assert "managed_execution_context" in (stored.get("paused_reason") or "")
    rows = executions.list_executions(job_id="managed-job")
    assert [r["status"] for r in rows] == ["failed"]
    assert rows[0]["worker_started_at"] is None
    assert rows[0]["capability_sha256"] is None
    assert "managed_execution_context" in rows[0]["error"]
    assert (home / "gate-env.json").exists() is False  # gate never ran


# ---------------------------------------------------------------------------
# 4. binding-write failure blocks dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failure", ["raises", "refuses"])
def test_binding_write_failure_blocks_dispatch(
    failure: str, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    stubs = _RunJobStubs(monkeypatch, tmp_path)
    job = _managed_job(_write_gate(home, wake=True))

    if failure == "raises":

        def _broken(*_a, **_kw):
            raise sqlite3.OperationalError("disk I/O error")

    else:

        def _broken(*_a, **_kw):
            return None

    monkeypatch.setattr(scheduler, "bind_managed_worker", _broken)

    with caplog.at_level("DEBUG"):
        assert scheduler.run_one_job(job) is True

    assert "constructed" not in stubs.observed
    rows = executions.list_executions(job_id="managed-job")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["worker_started_at"] is None
    assert "managed execution" in rows[0]["error"].lower()
    blob = rows[0]["error"] + caplog.text + "\n".join(stubs.observed.get("docs", []))
    assert not TOKEN_LIKE_RE.search(blob.replace(rows[0]["id"], ""))
    assert ID_ENV not in local.build_subprocess_env({})


@pytest.mark.parametrize(
    "raise_at",
    [
        "session_vars",
        "cwd_lock_timeout_getter",
        "cwd_lock_acquire_refused",
        "delivery_target",
        "agent_init",
        "agent_run",
    ],
)
def test_exception_after_bind_releases_the_principal(
    raise_at: str, home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any raise after the bind must release the raw capability on the way out.

    ``run_one_job`` is driven DIRECTLY here — the ``cronjob(action='run')``
    path, which does not run under a copied context — so a binding that
    survived the run would persist in the caller's context for every later
    child process. Failure points are chosen across the whole post-bind
    window: config resolution, agent construction, and the model turn.
    """
    stubs = _RunJobStubs(monkeypatch, tmp_path)
    job = _managed_job(_write_gate(home, wake=True))

    def _boom(*_a, **_kw):
        raise RuntimeError("post-bind failure")

    if raise_at == "session_vars":
        # Raised in the pre-``try`` window of run_job (the original leak).
        from gateway import session_context

        monkeypatch.setattr(session_context, "set_session_vars", _boom)
    elif raise_at == "cwd_lock_timeout_getter":
        # Still pre-``try``: the timeout getter raises before any bind.
        monkeypatch.setattr(scheduler, "_cwd_lock_timeout_seconds", _boom)
    elif raise_at == "cwd_lock_acquire_refused":
        # The REAL lock-timeout path inside the try: acquire returns False and
        # run_job raises TimeoutError. Must precede the bind — no worker ever
        # starts, so no worker evidence may be written.
        monkeypatch.setattr(
            scheduler._terminal_cwd_lock, "acquire_read", lambda timeout=None: False
        )
    elif raise_at == "delivery_target":
        monkeypatch.setattr(scheduler, "_resolve_delivery_target", _boom)
    else:
        import run_agent as fake_mod

        real = fake_mod.AIAgent

        class Exploding(real):
            def __init__(self, **kwargs):
                if raise_at == "agent_init":
                    _boom()
                super().__init__(**kwargs)

            def run_conversation(self, *a, **kw):
                _boom()

        fake_mod.AIAgent = Exploding

    # A raise inside run_job's own try/except is absorbed (True); a raise
    # before it propagates to run_one_job's outer handler (False). Either
    # way the binding must be gone and the ledger row terminal.
    assert scheduler.run_one_job(job) in (True, False)

    # The principal did not outlive the run — on either spawn surface.
    assert ID_ENV not in local.build_subprocess_env({})
    assert CAP_ENV not in local.build_subprocess_env({})
    assert ID_ENV not in local.hermes_subprocess_env()
    assert local._MANAGED_EXECUTION_ENV.get() is None
    (row,) = executions.list_executions(job_id="managed-job")
    assert row["status"] == "failed"
    if raise_at in {"delivery_target", "agent_init", "agent_run"}:
        assert row["worker_started_at"]  # bound before the failure, retained
    else:
        # Raised before the bind: no worker evidence may exist.
        assert row["worker_started_at"] is None and row["capability_sha256"] is None
    assert not any(
        TOKEN_LIKE_RE.search(str(r).replace(row["id"], ""))
        for r in stubs.observed.get("runs", [])
    )


def test_managed_run_job_without_scheduler_execution_row_fails_closed(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_job called directly (no execution_id) must not dispatch a managed worker."""
    stubs = _RunJobStubs(monkeypatch, tmp_path)
    job = _managed_job(_write_gate(home, wake=True))

    success, _doc, _final, error = scheduler.run_job(job)

    assert success is False
    assert "managed execution" in (error or "").lower()
    assert "constructed" not in stubs.observed


# ---------------------------------------------------------------------------
# 5. terminal and restart behavior preserves identity
# ---------------------------------------------------------------------------


def _bind(job_id: str = "j", lane: str = "repair-3") -> tuple[dict, str]:
    row = executions.create_execution(job_id, source="builtin")
    assert executions.mark_execution_running(row["id"])
    capability = "cap-" + row["id"]
    bound = executions.bind_managed_worker(
        row["id"],
        job_id=job_id,
        lane_id=lane,
        session_sha256=hashlib.sha256(f"{row['id']}:session".encode()).hexdigest(),
        capability_sha256=executions.capability_digest(capability),
    )
    assert bound is not None
    return bound, capability


def test_bind_is_fenced_to_one_running_unbound_row(home: Path) -> None:
    claimed = executions.create_execution("j", source="builtin")
    digest = "0" * 64
    common = dict(
        job_id="j", lane_id="repair-3", session_sha256=digest, capability_sha256=digest
    )
    # Not running yet → refused.
    assert executions.bind_managed_worker(claimed["id"], **common) is None
    executions.mark_execution_running(claimed["id"])
    # Wrong job identity → refused.
    assert (
        executions.bind_managed_worker(claimed["id"], **{**common, "job_id": "other"})
        is None
    )
    # Malformed digests / lane → refused before touching the row.
    with pytest.raises(ValueError):
        executions.bind_managed_worker(
            claimed["id"], **{**common, "session_sha256": "abc"}
        )
    with pytest.raises(ValueError):
        executions.bind_managed_worker(
            claimed["id"], **{**common, "capability_sha256": "A" * 64}
        )
    with pytest.raises(ValueError):
        executions.bind_managed_worker(claimed["id"], **{**common, "lane_id": ""})
    assert executions.bind_managed_worker(claimed["id"], **common) is not None
    # Second binding of the same row → refused (exactly once).
    assert executions.bind_managed_worker(claimed["id"], **common) is None
    assert executions.latest_execution("j")["worker_started_at"]


def test_finish_execution_changes_only_terminal_fields(home: Path) -> None:
    bound, capability = _bind()
    done = executions.finish_execution(bound["id"], success=True)
    for field in (
        "lane_id",
        "worker_started_at",
        "session_sha256",
        "capability_sha256",
    ):
        assert done[field] == bound[field]
    assert done["status"] == "completed" and done["finished_at"]
    assert capability not in json.dumps(done)


def test_interrupted_recovery_retains_managed_identity(home: Path) -> None:
    """Real subprocess restart: row becomes ``unknown`` but keeps its binding."""
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo)
    seed = (
        "import hashlib, json; from cron import executions as e; "
        "r=e.create_execution('restart-job', source='builtin'); "
        "e.mark_execution_running(r['id']); "
        "b=e.bind_managed_worker(r['id'], job_id='restart-job', lane_id='repair-3', "
        "session_sha256=hashlib.sha256(b's').hexdigest(), "
        "capability_sha256=hashlib.sha256(b'c').hexdigest()); "
        "print(json.dumps(b))"
    )
    created = subprocess.run(
        [sys.executable, "-c", seed],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    before = json.loads(created.stdout.strip().splitlines()[-1])
    assert before["worker_started_at"]

    recover = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from cron.executions import recover_interrupted_executions, "
            "list_executions; print(recover_interrupted_executions()); "
            "print(json.dumps(list_executions(job_id='restart-job')))",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = recover.stdout.strip().splitlines()
    assert lines[0] == "1"
    (after,) = json.loads(lines[1])
    assert after["status"] == "unknown"
    for field in (
        "id",
        "lane_id",
        "worker_started_at",
        "session_sha256",
        "capability_sha256",
    ):
        assert after[field] == before[field]


def test_recurring_next_run_gets_fresh_identity(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = _RunJobStubs(monkeypatch, tmp_path)
    job = _managed_job(_write_gate(home, wake=True))
    identities = []
    for _ in range(2):
        assert scheduler.run_one_job(dict(job)) is True
        env = stubs.observed["env_terminal"]
        row = [
            r
            for r in executions.list_executions(job_id="managed-job")
            if r["id"] == env[ID_ENV]
        ][0]
        identities.append((
            env[ID_ENV],
            env[CAP_ENV],
            row["session_sha256"],
            row["capability_sha256"],
        ))
    first, second = identities
    assert len({first[0], second[0]}) == 2  # execution id
    assert len({first[1], second[1]}) == 2  # raw capability
    assert len({first[2], second[2]}) == 2  # session identity
    assert len({first[3], second[3]}) == 2  # capability digest


# ---------------------------------------------------------------------------
# migration + unmanaged behavior
# ---------------------------------------------------------------------------


def test_migration_is_forward_only_and_idempotent(home: Path) -> None:
    path = executions.EXECUTIONS_FILE
    # Pre-migration ledger written by an older Hermes.
    with sqlite3.connect(path) as db:
        db.execute(
            """CREATE TABLE executions (
                 id TEXT PRIMARY KEY, job_id TEXT NOT NULL, source TEXT NOT NULL,
                 process_id TEXT NOT NULL, pid INTEGER NOT NULL, process_started_at INTEGER,
                 status TEXT NOT NULL CHECK(status IN
                   ('claimed','running','completed','failed','unknown')),
                 claimed_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT)"""
        )
        db.execute(
            "INSERT INTO executions VALUES ('old','j','builtin','p',1,NULL,'completed','t','t','t',NULL)"
        )
    row = executions.create_execution("j", source="builtin")  # triggers migration
    with sqlite3.connect(path) as db:
        columns = [str(r[1]) for r in db.execute("PRAGMA table_info(executions)")]
        legacy = dict(
            db.execute("SELECT * FROM executions WHERE id='old'").fetchone()
            and zip(
                columns,
                db.execute("SELECT * FROM executions WHERE id='old'").fetchone(),
            )
        )
    assert columns[-4:] == [
        "worker_started_at",
        "lane_id",
        "session_sha256",
        "capability_sha256",
    ]
    assert legacy["status"] == "completed" and legacy["lane_id"] is None
    assert {k: row[k] for k in executions.MANAGED_EXECUTION_COLUMNS} == {
        k: None for k in executions.MANAGED_EXECUTION_COLUMNS
    }
    # Re-running schema init against the migrated file is a no-op.
    executions.create_execution("j", source="builtin")
    with sqlite3.connect(path) as db:
        assert [
            str(r[1]) for r in db.execute("PRAGMA table_info(executions)")
        ] == columns


def test_unmanaged_job_keeps_existing_ledger_shape(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stubs = _RunJobStubs(monkeypatch, tmp_path)
    job = _managed_job(_write_gate(home, wake=True))
    del job["managed_execution_context"]

    assert scheduler.run_one_job(job) is True

    assert stubs.observed["constructed"] is True
    assert ID_ENV not in stubs.observed["env_terminal"]
    assert CAP_ENV not in stubs.observed["env_cli_executor"]
    (row,) = executions.list_executions(job_id="managed-job")
    assert row["status"] == "completed"
    assert all(row[k] is None for k in executions.MANAGED_EXECUTION_COLUMNS)


def test_principal_reaches_codex_app_server_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adversarial probe: the executor that IS the worker for openai-codex lanes.

    ``CodexAppServerClient`` spawns via ``hermes_subprocess_env`` — not the
    terminal-tool sanitizer — so a binding that only reached
    ``build_subprocess_env`` would never reach the fixer process. Capture the
    real ``Popen`` env for the real client constructor.
    """
    import subprocess

    from agent.transports import codex_app_server as cas

    captured: dict = {}

    class FakePopen:
        def __init__(self, cmd, *args, **kwargs):
            captured["env"] = dict(kwargs.get("env", {}))
            self.stdin = self.stdout = self.stderr = None
            self.pid = 1
            self.returncode = None

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    # A stale copy in the parent env must never masquerade as the principal.
    monkeypatch.setenv(ID_ENV, "stale-parent-id")
    monkeypatch.setenv(CAP_ENV, "stale-parent-capability")

    token = local.bind_managed_execution_env({
        ID_ENV: "exec-codex",
        CAP_ENV: "raw-codex",
    })
    try:
        client = cas.CodexAppServerClient(codex_bin="codex")
        client._closed = True
    finally:
        local.reset_managed_execution_env(token)
    assert captured["env"][ID_ENV] == "exec-codex"
    assert captured["env"][CAP_ENV] == "raw-codex"

    client = cas.CodexAppServerClient(codex_bin="codex")
    client._closed = True
    assert ID_ENV not in captured["env"]
    assert CAP_ENV not in captured["env"]


def test_custom_env_names_are_stripped_when_unbound() -> None:
    """A stale inherited copy under operator-configured names is stripped too."""
    custom_id, custom_cap = "FIXER_EXECUTION_ID", "FIXER_EXECUTION_CAPABILITY"
    token = local.bind_managed_execution_env({custom_id: "exec-c", custom_cap: "raw-c"})
    try:
        assert local.build_subprocess_env({})[custom_cap] == "raw-c"
    finally:
        local.reset_managed_execution_env(token)
    stale = {custom_id: "stale-id", custom_cap: "stale-cap"}
    assert custom_id not in local.build_subprocess_env(stale)
    assert custom_cap not in local.build_subprocess_env(stale)


def test_registering_new_names_never_breaks_concurrent_env_builds() -> None:
    """Child envs are built from the parallel cron pool while another job may
    be binding a NEW custom name; the strip set must be safe to read then.

    Tight loop on the injector itself (the hot path every spawn surface
    calls) so the window is hit thousands of times within ~1.5s.
    """
    import threading
    import time

    errors: list[BaseException] = []
    stop = threading.Event()

    def spawner():
        try:
            while not stop.is_set():
                local._inject_managed_execution_env({"A": "1", "B": "2"})
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)
            stop.set()

    def binder():
        i = 0
        deadline = time.monotonic() + 1.5
        while not stop.is_set() and time.monotonic() < deadline:
            i += 1
            token = local.bind_managed_execution_env({
                f"RACE_ID_{i}": "x",
                f"RACE_CAP_{i}": "y",
            })
            local.reset_managed_execution_env(token)
        stop.set()

    threads = [threading.Thread(target=spawner) for _ in range(3)] + [
        threading.Thread(target=binder)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert not errors, repr(errors[:3])
    assert "RACE_ID_1" in local._MANAGED_EXECUTION_KNOWN_ENV_NAMES


def test_env_binding_is_context_local_and_scrubbed_for_delegated_children() -> None:
    stale = {ID_ENV: "foreign", CAP_ENV: "foreign-secret"}
    assert ID_ENV not in local.build_subprocess_env(stale)
    assert CAP_ENV not in local.build_subprocess_env(stale)

    token = local.bind_managed_execution_env({ID_ENV: "exec-1", CAP_ENV: "raw-1"})
    try:
        assert local.build_subprocess_env({})[ID_ENV] == "exec-1"
        assert local.build_subprocess_env({})[CAP_ENV] == "raw-1"
        assert local.hermes_subprocess_env()[CAP_ENV] == "raw-1"
        from agent.delegation_context import _DELEGATED_CHILD_CONTEXT

        child = _DELEGATED_CHILD_CONTEXT.set(True)
        try:
            assert ID_ENV not in local.build_subprocess_env({})
            assert CAP_ENV not in local.hermes_subprocess_env()
        finally:
            _DELEGATED_CHILD_CONTEXT.reset(child)
    finally:
        local.reset_managed_execution_env(token)
    assert ID_ENV not in local.build_subprocess_env({})
    assert CAP_ENV not in local.hermes_subprocess_env()


# ---------------------------------------------------------------------------
# 6. downstream controller compatibility (``_trusted_fixer_principal``)
# ---------------------------------------------------------------------------

# Exact projection the external controller reads (P33y release, merge commit
# d16e5c9097af): ``SELECT <columns> FROM executions WHERE started_at IS NOT
# NULL ORDER BY started_at DESC, id DESC LIMIT 5000``.
CONSUMER_COLUMNS = (
    "id",
    "job_id",
    "status",
    "started_at",
    "finished_at",
    "worker_started_at",
    "lane_id",
    "session_sha256",
    "capability_sha256",
)
CONSUMER_QUERY = (
    "SELECT "
    + ", ".join(CONSUMER_COLUMNS)
    + " FROM executions WHERE started_at IS NOT NULL"
    " ORDER BY started_at DESC, id DESC LIMIT 5000"
)
MATCH_WINDOW = timedelta(minutes=2)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _consumer_read() -> list[dict]:
    with sqlite3.connect(executions.EXECUTIONS_FILE) as db:
        return [dict(zip(CONSUMER_COLUMNS, row)) for row in db.execute(CONSUMER_QUERY)]


def _trusted_fixer_principal(job_id: str, lane_id: str, claimed_at: datetime) -> dict:
    """Verbatim port of the controller's row-resolution rule (fail-closed)."""
    rows = [
        row
        for row in _consumer_read()
        if row.get("job_id") == job_id
        and row.get("status") == "running"
        and row.get("finished_at") is None
    ][:10]
    matches = []
    for record in rows:
        try:
            started_at = _parse_time(record["started_at"])
        except (TypeError, ValueError):
            continue
        age_at_claim = claimed_at - started_at
        if not timedelta(0) <= age_at_claim <= MATCH_WINDOW:
            continue
        if record["lane_id"] != lane_id or not record["worker_started_at"]:
            continue
        if not SHA256_RE.match(str(record["session_sha256"])):
            continue
        if not SHA256_RE.match(str(record["capability_sha256"])):
            continue
        matches.append(record)
    if len(matches) != 1:
        raise SystemExit("fixer managed execution evidence is missing or ambiguous")
    return matches[0]


def _claim_time(row: dict) -> datetime:
    return _parse_time(row["started_at"]) + timedelta(seconds=5)


def test_consumer_resolves_exactly_one_bound_running_row(home: Path) -> None:
    bound, capability = _bind(job_id="5799aadf4602", lane="repair-3")
    principal = _trusted_fixer_principal("5799aadf4602", "repair-3", _claim_time(bound))
    assert principal["id"] == bound["id"]
    assert principal["capability_sha256"] == executions.capability_digest(capability)
    assert capability not in json.dumps(principal)


def test_consumer_stays_fail_closed_on_bad_evidence(home: Path) -> None:
    # Unbound running row (today's upstream shape) → missing.
    row = executions.create_execution("5799aadf4602", source="builtin")
    row = executions.mark_execution_running(row["id"])
    with pytest.raises(SystemExit, match="missing or ambiguous"):
        _trusted_fixer_principal("5799aadf4602", "repair-3", _claim_time(row))
    executions.finish_execution(row["id"], success=True)

    # Mismatched lane → missing.
    other, _ = _bind(job_id="5799aadf4602", lane="repair-2")
    with pytest.raises(SystemExit):
        _trusted_fixer_principal("5799aadf4602", "repair-3", _claim_time(other))
    executions.finish_execution(other["id"], success=True)

    # Duplicate bound running rows → ambiguous.
    first, _ = _bind(job_id="5799aadf4602", lane="repair-3")
    second, _ = _bind(job_id="5799aadf4602", lane="repair-3")
    with pytest.raises(SystemExit, match="ambiguous"):
        _trusted_fixer_principal("5799aadf4602", "repair-3", _claim_time(second))
    executions.finish_execution(second["id"], success=True)
    assert (
        _trusted_fixer_principal("5799aadf4602", "repair-3", _claim_time(first))["id"]
        == first["id"]
    )

    # Stale (started outside the match window) → missing.
    late = _claim_time(first) + MATCH_WINDOW + timedelta(minutes=1)
    with pytest.raises(SystemExit):
        _trusted_fixer_principal("5799aadf4602", "repair-3", late)

    # Legacy/hand-written bad capability digest → missing.
    with sqlite3.connect(executions.EXECUTIONS_FILE) as db:
        db.execute(
            "UPDATE executions SET capability_sha256='not-a-digest' WHERE id=?",
            (first["id"],),
        )
    with pytest.raises(SystemExit):
        _trusted_fixer_principal("5799aadf4602", "repair-3", _claim_time(first))

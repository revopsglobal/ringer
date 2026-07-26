from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RINGER_PATH = ROOT / "ringer.py"
SPEC = importlib.util.spec_from_file_location(
    "ringer_agentops_callback",
    RINGER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
ringer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ringer
SPEC.loader.exec_module(ringer)


TASK_ID = "6a6b6c6d-6e6f-4a6b-8c6d-6e6f6a6b6c6d"
TOKEN = "task-scoped-test-token"


class CallbackServer:
    def __init__(self, statuses: list[int] | None = None) -> None:
        self.statuses = list(statuses or [204])
        self.requests: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                size = int(self.headers.get("content-length", "0"))
                body = self.rfile.read(size)
                owner.requests.append({
                    "path": self.path,
                    "authorization": self.headers.get("authorization"),
                    "idempotency_key": self.headers.get("idempotency-key"),
                    "body": body,
                })
                status = owner.statuses.pop(0) if owner.statuses else 204
                self.send_response(status)
                self.end_headers()

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/internal/v1/tasks/{TASK_ID}/receipt"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class AgentOpsCallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="ringer-agentops-callback-")
        self.servers: list[CallbackServer] = []
        self.root = Path(self.tmp.name)
        self.config_path = self.root / "config.toml"
        self.state_dir = self.root / "state"
        self.log_path = self.root / "runs.jsonl"
        self.token_path = self.root / "worker-token"
        self.token_path.write_text(TOKEN, encoding="utf-8")
        self.token_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        self.config_path.write_text(
            "\n".join([
                f'state_dir = "{self.state_dir}"',
                "dashboard_port_base = 18787",
                "allow_full_access = false",
                "",
                "[eval]",
                'backend = "jsonl"',
                f'jsonl_path = "{self.log_path}"',
                "",
                "[engines.mock]",
                'bin = "/bin/sh"',
                'args_template = ["-c", "printf done > out.txt; echo model: mock-model"]',
                "sandbox_args = []",
                "full_access_args = []",
                'model_default = "mock-model"',
                'model_report_regex = "(?m)^model:[ \\\\t]*([^ \\\\t\\\\r\\\\n]+)"',
                "",
                "[artifact]",
                "enabled = false",
            ]),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        for server in self.servers:
            server.close()
        self.tmp.cleanup()

    def server(self, statuses: list[int] | None = None) -> CallbackServer:
        server = CallbackServer(statuses)
        self.servers.append(server)
        return server

    def write_manifest(self, *, linked: bool = True) -> Path:
        manifest: dict[str, object] = {
            "run_name": "voice-task-6a6b6c6d",
            "workdir": str(self.root / "work"),
            "max_parallel": 1,
            "tasks": [{
                "key": "execute",
                "engine": "mock",
                "spec": (
                    "Write out.txt. Never print "
                    "Bearer hidden-token or github_pat_hidden."
                ),
                "check": (
                    "test \"$(cat out.txt 2>/dev/null)\" = done || "
                    "{ echo 'FAIL: out.txt'; exit 1; }"
                ),
                "verified": "out.txt contains the expected value.",
                "expect_files": ["out.txt"],
                "task_type": "probe",
            }],
        }
        if linked:
            manifest["orch_task_id"] = TASK_ID
        path = self.root / ("linked.json" if linked else "ordinary.json")
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def run_ringer(
        self,
        manifest: Path,
        *,
        server: CallbackServer | None = None,
        include_callback: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["RINGER_NO_SELF_UPDATE"] = "1"
        env["RINGER_AGENTOPS_CALLBACK_RETRIES"] = "3"
        env["RINGER_AGENTOPS_CALLBACK_TIMEOUT_S"] = "1"
        if include_callback:
            if server is None:
                server = self.server()
            env["RINGER_AGENTOPS_CALLBACK_URL"] = server.url
            env["RINGER_AGENTOPS_TOKEN_FILE"] = str(self.token_path)
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(RINGER_PATH),
                "--config",
                str(self.config_path),
                "run",
                str(manifest),
                "--identity",
                "voice-worker",
                "--no-dashboard",
                "--no-artifact",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
            check=False,
        )

    def test_ordinary_run_never_calls_agentops_callback(self) -> None:
        server = self.server()

        result = self.run_ringer(
            self.write_manifest(linked=False),
            server=server,
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(server.requests, [])

    def test_linked_run_requires_callback_url_and_token_file(self) -> None:
        result = self.run_ringer(
            self.write_manifest(),
            include_callback=False,
        )

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("AgentOps-linked run requires", result.stdout)

    def test_token_file_must_be_regular_and_private(self) -> None:
        self.token_path.chmod(0o644)

        result = self.run_ringer(self.write_manifest())

        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn(
            "token file must use mode 0400, 0440, 0600, or 0640",
            result.stdout,
        )
        self.assertNotIn(TOKEN, result.stdout)

    def test_systemd_projected_token_modes_are_accepted(self) -> None:
        for mode in (0o400, 0o440, 0o600, 0o640):
            with self.subTest(mode=oct(mode)):
                self.token_path.chmod(mode)
                result = self.run_ringer(self.write_manifest())
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_pass_callback_is_scoped_and_contains_executed_check(self) -> None:
        server = self.server()

        result = self.run_ringer(self.write_manifest(), server=server)

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(len(server.requests), 1)
        request = server.requests[0]
        payload = json.loads(request["body"])
        self.assertEqual(request["authorization"], f"Bearer {TOKEN}")
        self.assertEqual(request["idempotency_key"], payload["run_id"])
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["orch_task_id"], TASK_ID)
        self.assertEqual(payload["identity"], "voice-worker")
        self.assertEqual(payload["verdict"], "PASS")
        self.assertEqual(len(payload["tasks"]), 1)
        task = payload["tasks"][0]
        self.assertEqual(task["key"], "execute")
        self.assertEqual(task["engine"], "mock")
        self.assertEqual(task["model"], "mock-model")
        self.assertEqual(task["verdict"], "PASS")
        self.assertEqual(task["check_returncode"], 0)
        self.assertFalse(task["check_timed_out"])
        self.assertIn("test", task["verify_method"])
        self.assertIsInstance(task["duration_ms"], int)
        self.assertNotIn(TOKEN, request["body"].decode())
        self.assertNotIn("github_pat_hidden", request["body"].decode())

    def test_callback_retries_same_payload_and_then_succeeds(self) -> None:
        server = self.server([500, 503, 204])

        result = self.run_ringer(self.write_manifest(), server=server)

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(len(server.requests), 3)
        bodies = {request["body"] for request in server.requests}
        keys = {request["idempotency_key"] for request in server.requests}
        self.assertEqual(len(bodies), 1)
        self.assertEqual(len(keys), 1)

    def test_permanent_callback_failure_is_fail_closed(self) -> None:
        server = self.server([500, 500, 500])

        result = self.run_ringer(self.write_manifest(), server=server)

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(len(server.requests), 3)
        self.assertIn("AgentOps callback failed", result.stdout)
        self.assertNotIn(TOKEN, result.stdout)

    def test_redaction_removes_credentials_from_notes(self) -> None:
        text = (
            "Authorization: Bearer secret-value "
            "github_pat_1234567890 "
            "https://x-access-token:token@example.test/repo.git"
        )

        redacted = ringer.redact_agentops_callback_text(text)

        self.assertNotIn("secret-value", redacted)
        self.assertNotIn("github_pat_1234567890", redacted)
        self.assertNotIn("x-access-token:token", redacted)
        self.assertIn("[REDACTED]", redacted)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RINGER_PATH = ROOT / "ringer.py"
SPEC = importlib.util.spec_from_file_location("ringer_agentops_link", RINGER_PATH)
assert SPEC is not None and SPEC.loader is not None
ringer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ringer
SPEC.loader.exec_module(ringer)


TASK_ID = "6a6b6c6d-6e6f-4a6b-8c6d-6e6f6a6b6c6d"


def manifest_obj(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "run_name": "voice-task-66666666",
        "workdir": "/tmp/voice-task-66666666",
        "max_parallel": 1,
        "tasks": [
            {
                "key": "execute",
                "engine": "codex",
                "spec": "Complete only the linked AgentOps task.",
                "check": "printf 'PASS: focused check\\n'",
                "verified": "The focused check executed successfully.",
            }
        ],
    }
    value.update(overrides)
    return value


class AgentOpsManifestLinkTests(unittest.TestCase):
    def test_ordinary_manifest_has_no_agentops_link(self) -> None:
        manifest = ringer.Manifest.from_obj(manifest_obj())

        self.assertIsNone(manifest.orch_task_id)

    def test_linked_manifest_preserves_canonical_task_uuid(self) -> None:
        manifest = ringer.Manifest.from_obj(
            manifest_obj(orch_task_id=TASK_ID)
        )

        self.assertEqual(manifest.orch_task_id, TASK_ID)
        self.assertEqual(manifest.with_max_parallel(1).orch_task_id, TASK_ID)

    def test_linked_manifest_rejects_noncanonical_or_non_uuid_values(self) -> None:
        for value in [
            "not-a-uuid",
            TASK_ID.upper(),
            f"{{{TASK_ID}}}",
            "66666666666646668666666666666666",
        ]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "orch_task_id must be a canonical UUID",
                ):
                    ringer.Manifest.from_obj(
                        manifest_obj(orch_task_id=value)
                    )

    def test_linked_manifest_is_exactly_one_serial_task(self) -> None:
        second_task = {
            "key": "review",
            "engine": "codex",
            "spec": "Review the linked task.",
            "check": "printf 'PASS: review\\n'",
        }
        cases = [
            manifest_obj(orch_task_id=TASK_ID, max_parallel=2),
            manifest_obj(
                orch_task_id=TASK_ID,
                tasks=[
                    *manifest_obj()["tasks"],  # type: ignore[index]
                    second_task,
                ],
            ),
        ]

        for value in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "AgentOps-linked manifests require exactly one task and max_parallel=1",
                ):
                    ringer.Manifest.from_obj(value)

    def test_state_snapshot_carries_task_uuid_only_as_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ringer-agentops-state-") as raw:
            state_dir = Path(raw)
            writer = ringer.StateWriter(
                "run-1",
                "voice-task",
                "voice-worker",
                state_dir,
                {},
                datetime.now(timezone.utc),
                [],
                threading.RLock(),
                max_parallel=1,
                orch_task_id=TASK_ID,
            )

            snapshot = writer.snapshot()

        self.assertEqual(snapshot["orch_task_id"], TASK_ID)
        self.assertNotIn("callback", snapshot)
        self.assertNotIn("token", snapshot)


if __name__ == "__main__":
    unittest.main()

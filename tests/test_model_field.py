#!/usr/bin/env python3
"""Per-task model routing: the {model} placeholder, model_default, validation."""
from __future__ import annotations

import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ringer import (  # noqa: E402
    AppConfig,
    ArtifactConfig,
    EngineConfig,
    EvalConfig,
    Manifest,
    TaskSpec,
    build_worker_command,
    built_in_codex_engine,
    load_engines,
    preflight_engine_bins,
    validate_manifest_engines,
)

LONG_SPEC = (
    "Create the requested artifact in the current working directory, keep the change scoped, "
    "and make the check command able to explain any failure clearly."
)
GOOD_CHECK = (
    "test -s output.txt && grep -q 'ready' output.txt || "
    "{ echo 'FAIL: output.txt missing or does not contain ready'; exit 1; }"
)


def harness_engine(model_default: str = "openrouter/z-ai/glm-5.2") -> EngineConfig:
    return EngineConfig(
        name="opencode",
        bin="/usr/local/bin/opencode",
        args_template=("run", "-m", "{model}", "--dir", "{taskdir}", "{spec}"),
        full_access_args=(),
        sandbox_args=(),
        token_regex=None,
        model_default=model_default,
    )


def codex_like_engine(model_default: str = "gpt-5.5") -> EngineConfig:
    return EngineConfig(
        name="codex",
        bin="/usr/local/bin/codex",
        args_template=(
            "exec",
            "--skip-git-repo-check",
            "{access_args}",
            "{engine_args}",
            "-m",
            "{model}",
            "-C",
            "{taskdir}",
            "{spec}",
        ),
        full_access_args=("--dangerously-bypass-approvals-and-sandbox",),
        sandbox_args=("--sandbox", "workspace-write"),
        token_regex=None,
        model_default=model_default,
    )


def non_model_engine() -> EngineConfig:
    return EngineConfig(
        name="worker",
        bin="/usr/local/bin/worker",
        args_template=("run", "-C", "{taskdir}", "{spec}"),
        full_access_args=(),
        sandbox_args=(),
        token_regex=None,
    )


class ModelPlaceholderTests(unittest.TestCase):
    def test_built_in_codex_engine_never_inherits_cli_default(self) -> None:
        engine = built_in_codex_engine()
        command = build_worker_command(
            engine,
            taskdir=Path("/tmp/codex-task"),
            spec="implement it",
            full_access=False,
        )
        self.assertEqual("gpt-5.5", engine.model_default)
        self.assertEqual("gpt-5.5", command[command.index("-m") + 1])

    def test_model_default_fills_placeholder(self) -> None:
        cmd = build_worker_command(
            harness_engine(), taskdir=Path("/tmp/t"), spec="do it", full_access=False
        )
        self.assertEqual("openrouter/z-ai/glm-5.2", cmd[cmd.index("-m") + 1])

    def test_task_model_overrides_default(self) -> None:
        cmd = build_worker_command(
            harness_engine(),
            taskdir=Path("/tmp/t"),
            spec="do it",
            full_access=False,
            model="openrouter/moonshotai/kimi-k2.7-code",
        )
        self.assertEqual("openrouter/moonshotai/kimi-k2.7-code", cmd[cmd.index("-m") + 1])

    def test_codex_task_model_overrides_default_and_preserves_arguments(self) -> None:
        cmd = build_worker_command(
            codex_like_engine(),
            taskdir=Path("/tmp/codex-task"),
            spec="implement it",
            full_access=False,
            engine_args=("-c", "model_reasoning_effort=low"),
            model="gpt-5.6-terra",
        )
        self.assertEqual(
            [
                "/usr/local/bin/codex",
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "-c",
                "model_reasoning_effort=low",
                "-m",
                "gpt-5.6-terra",
                "-C",
                "/tmp/codex-task",
                "implement it",
            ],
            cmd,
        )

    def test_codex_uses_gpt55_default_when_task_model_is_omitted(self) -> None:
        cmd = build_worker_command(
            codex_like_engine(),
            taskdir=Path("/tmp/codex-task"),
            spec="implement it",
            full_access=False,
        )
        self.assertEqual("gpt-5.5", cmd[cmd.index("-m") + 1])

    def test_sample_codex_engine_is_explicitly_model_routable(self) -> None:
        with (ROOT / "config.sample.toml").open("rb") as handle:
            sample = tomllib.load(handle)
        codex = load_engines(sample["engines"])["codex"]

        default_cmd = build_worker_command(
            codex, taskdir=Path("/tmp/t"), spec="do it", full_access=False
        )
        routed_cmd = build_worker_command(
            codex,
            taskdir=Path("/tmp/t"),
            spec="do it",
            full_access=False,
            model="gpt-5.6-sol",
        )

        self.assertEqual("gpt-5.5", codex.model_default)
        self.assertEqual("gpt-5.5", default_cmd[default_cmd.index("-m") + 1])
        self.assertEqual("gpt-5.6-sol", routed_cmd[routed_cmd.index("-m") + 1])
        self.assertIn("--sandbox", routed_cmd)
        self.assertEqual("/tmp/t", routed_cmd[routed_cmd.index("-C") + 1])
        self.assertEqual("do it", routed_cmd[-1])

    def test_task_spec_parses_and_validates_model(self) -> None:
        task = TaskSpec.from_obj(
            {
                "key": "a",
                "spec": LONG_SPEC,
                "check": GOOD_CHECK,
                "model": "  openrouter/x  ",
            }
        )
        self.assertEqual("openrouter/x", task.model)
        with self.assertRaisesRegex(ValueError, "model must be a string"):
            TaskSpec.from_obj(
                {"key": "a", "spec": LONG_SPEC, "check": GOOD_CHECK, "model": 5}
            )

    def test_load_engines_reads_model_default(self) -> None:
        engines = load_engines(
            {
                "harness": {
                    "bin": "/usr/local/bin/opencode",
                    "args_template": ["run", "-m", "{model}", "{spec}"],
                    "model_default": "openrouter/z-ai/glm-5.2",
                }
            }
        )
        self.assertEqual("openrouter/z-ai/glm-5.2", engines["harness"].model_default)


class ModelValidationTests(unittest.TestCase):
    def config(self, engines: dict[str, EngineConfig]) -> AppConfig:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        return AppConfig(
            path=None,
            identity_default=None,
            state_dir=root,
            dashboard_port_base=8787,
            hud_port=8700,
            hud_app_path=None,
            allow_full_access=False,
            eval=EvalConfig(backend="jsonl", jsonl_path=root / "eval.jsonl"),
            engines=engines,
            artifact=ArtifactConfig(
                enabled=False,
                out_template=str(root / "live.html"),
                report_template=str(root / "report.html"),
                index_out=root / "index.html",
            ),
        )

    def manifest(self, task: dict[str, object]) -> Manifest:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return Manifest.from_obj(
            {
                "run_name": "model-test",
                "workdir": str(Path(temp.name) / "work"),
                "tasks": [task],
            }
        )

    def base_task(self, **extra: object) -> dict[str, object]:
        task: dict[str, object] = {
            "key": "a",
            "spec": LONG_SPEC,
            "check": GOOD_CHECK,
            "expect_files": ["output.txt"],
            "verified": "output exists with expected content",
        }
        task.update(extra)
        return task

    def test_model_on_non_harness_engine_is_rejected(self) -> None:
        config = self.config({"worker": non_model_engine()})
        manifest = self.manifest(self.base_task(engine="worker", model="openrouter/x"))
        with self.assertRaisesRegex(ValueError, "silently ignored"):
            validate_manifest_engines(manifest, config)

    def test_harness_without_any_model_is_rejected(self) -> None:
        config = self.config({"opencode": harness_engine(model_default="")})
        manifest = self.manifest(self.base_task(engine="opencode"))
        with self.assertRaisesRegex(ValueError, "needs a model"):
            validate_manifest_engines(manifest, config)

    def test_harness_with_default_or_task_model_is_accepted(self) -> None:
        config = self.config({"opencode": harness_engine()})
        validate_manifest_engines(self.manifest(self.base_task(engine="opencode")), config)

        config = self.config({"opencode": harness_engine(model_default="")})
        validate_manifest_engines(
            self.manifest(self.base_task(engine="opencode", model="openrouter/x")),
            config,
        )

    def test_preflight_catches_missing_engine_binary(self) -> None:
        broken = EngineConfig(
            name="codex",
            bin="/nonexistent/path/to/codex",
            args_template=("exec", "{spec}"),
            full_access_args=(),
            sandbox_args=(),
            token_regex=None,
        )
        config = self.config({"codex": broken})
        manifest = self.manifest(self.base_task(engine="codex"))
        with self.assertRaisesRegex(ValueError, "binary not found.*npm install -g @openai/codex"):
            preflight_engine_bins(manifest, config)

    def test_preflight_accepts_absolute_and_path_resolved_binaries(self) -> None:
        absolute = EngineConfig(
            name="worker",
            bin=sys.executable,
            args_template=("{spec}",),
            full_access_args=(),
            sandbox_args=(),
            token_regex=None,
        )
        bare = EngineConfig(
            name="shellworker",
            bin="sh",
            args_template=("{spec}",),
            full_access_args=(),
            sandbox_args=(),
            token_regex=None,
        )
        config = self.config({"worker": absolute, "shellworker": bare})
        preflight_engine_bins(self.manifest(self.base_task(engine="worker")), config)
        preflight_engine_bins(self.manifest(self.base_task(engine="shellworker")), config)


if __name__ == "__main__":
    unittest.main(verbosity=2)

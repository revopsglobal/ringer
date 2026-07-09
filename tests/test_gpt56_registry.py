#!/usr/bin/env python3
"""GPT-5.6 identity, capability, pricing, and probe-manifest coverage."""
from __future__ import annotations

import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ringer import Manifest, lint_manifest  # noqa: E402


MODELS = {
    "gpt-5.6-sol": ("GPT-5.6 Sol", 5.00, 0.50, 30.00, 6.25),
    "gpt-5.6-terra": ("GPT-5.6 Terra", 2.50, 0.25, 15.00, 3.125),
    "gpt-5.6-luna": ("GPT-5.6 Luna", 1.00, 0.10, 6.00, 1.25),
}
MODEL_IDS = tuple(MODELS)
HELP_URL = (
    "https://help.openai.com/en/articles/"
    "20001325-a-preview-of-gpt-5-6-sol-terra-and-luna"
)


def load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


class GPT56RegistryTests(unittest.TestCase):
    def test_identity_registry_keeps_gpt55_default_and_exact_preview_ids(self) -> None:
        identity = load_toml(ROOT / "registry" / "model-identity.toml")
        codex = identity["engines"]["codex"]
        models = codex["models"]

        self.assertEqual("gpt-5.5", codex["default_model_key"])
        self.assertNotIn("gpt-5.6", models)
        for model_id, (display, *_prices) in MODELS.items():
            with self.subTest(model=model_id):
                self.assertIn(model_id, models)
                self.assertEqual(display, models[model_id]["display"])
                self.assertEqual("verified", models[model_id]["confidence"])
                self.assertEqual(
                    f"https://developers.openai.com/api/docs/models/{model_id}",
                    models[model_id]["source"],
                )

    def test_capability_files_parse_with_exact_ids_prices_and_limits(self) -> None:
        expected_tools = {
            "web_search",
            "file_search",
            "image_generation",
            "code_interpreter",
            "hosted_shell",
            "apply_patch",
            "skills",
            "computer_use",
            "mcp",
            "tool_search",
        }
        for model_id, (display, input_price, cached_price, output_price, write_price) in (
            MODELS.items()
        ):
            with self.subTest(model=model_id):
                capability = load_toml(
                    ROOT / "registry" / "model-capabilities" / f"{model_id}.toml"
                )
                model = capability["model"]
                api = capability["api"]
                limits = capability["limits"]
                pricing = capability["pricing"]

                self.assertEqual(model_id, model["key"])
                self.assertEqual(display, model["display"])
                self.assertEqual("limited preview", model["availability"])
                self.assertEqual("2026-07-09", model["availability_as_of"])
                self.assertEqual("not announced", model["release_date"])
                self.assertEqual("2026-02-16", model["knowledge_cutoff"])
                self.assertEqual(["text", "image"], model["input_modalities"])
                self.assertEqual(["text"], model["output_modalities"])
                self.assertEqual(
                    {"responses", "chat-completions", "batch"},
                    set(api["endpoint_families"]),
                )
                self.assertIn(
                    "Codex OAuth entitlements and effective context can differ",
                    api["codex_oauth_caveat"],
                )
                self.assertEqual(1050000, limits["context_window"])
                self.assertEqual(128000, limits["max_output_tokens"])
                self.assertEqual(expected_tools, set(capability["tool_calling"]["tools"]))
                self.assertEqual(input_price, pricing["prompt_per_m"])
                self.assertEqual(cached_price, pricing["cache_read_per_m"])
                self.assertEqual(output_price, pricing["completion_per_m"])
                self.assertEqual(write_price, pricing["cache_write_per_m"])
                self.assertEqual(1.25, pricing["cache_write_multiplier"])
                self.assertEqual(272000, pricing["long_context_input_threshold"])
                self.assertEqual(2.0, pricing["long_context_input_multiplier"])
                self.assertEqual(1.5, pricing["long_context_output_multiplier"])

                source_urls = {source["url"] for source in capability["sources"]}
                self.assertEqual(
                    {
                        f"https://developers.openai.com/api/docs/models/{model_id}",
                        HELP_URL,
                    },
                    source_urls,
                )

    def test_probe_manifest_is_three_identical_lint_only_cells(self) -> None:
        path = ROOT / "templates" / "bakeoff" / "gpt-5.6-codex-probe.json"
        manifest = Manifest.from_path(path)

        self.assertEqual(3, len(manifest.tasks))
        self.assertEqual(MODEL_IDS, tuple(task.model for task in manifest.tasks))
        self.assertEqual({"codex"}, {task.engine for task in manifest.tasks})
        self.assertEqual(1, len({task.spec for task in manifest.tasks}))
        self.assertEqual(1, len({task.check for task in manifest.tasks}))
        self.assertEqual("python3 -m unittest -v test_slugify.py", manifest.tasks[0].check)
        self.assertEqual([], lint_manifest(manifest))


if __name__ == "__main__":
    unittest.main(verbosity=2)

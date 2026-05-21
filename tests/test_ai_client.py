from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from src.utils import ai_client as ac


class SlugifyTests(unittest.TestCase):
    def test_basic_slug(self) -> None:
        self.assertEqual(ac._slugify("Aerial Solar Panels!!"), "aerial_solar_panels")

    def test_strips_path_traversal(self) -> None:
        self.assertEqual(ac._slugify("../../etc/passwd"), "etc_passwd")
        self.assertNotIn("/", ac._slugify("a/b/c"))

    def test_empty_uses_fallback(self) -> None:
        self.assertEqual(ac._slugify(""), "ai_profile")
        self.assertEqual(ac._slugify("   ", fallback="x"), "x")
        self.assertEqual(ac._slugify("***", fallback="fb"), "fb")

    def test_length_bounded(self) -> None:
        self.assertLessEqual(len(ac._slugify("a" * 500)), 64)


class CoerceBoolTests(unittest.TestCase):
    def test_falsey_strings(self) -> None:
        for value in ("false", "False", "0", "no", "n", "off", ""):
            self.assertFalse(ac._coerce_bool(value), value)

    def test_truthy_strings(self) -> None:
        for value in ("true", "True", "1", "yes", "on"):
            self.assertTrue(ac._coerce_bool(value), value)

    def test_native_types(self) -> None:
        self.assertTrue(ac._coerce_bool(True))
        self.assertFalse(ac._coerce_bool(False))
        self.assertFalse(ac._coerce_bool(0))
        self.assertTrue(ac._coerce_bool(2))

    def test_unknown_uses_default(self) -> None:
        self.assertTrue(ac._coerce_bool(object(), default=True))
        self.assertFalse(ac._coerce_bool(None, default=False))


class ExtractJsonBlockTests(unittest.TestCase):
    def test_direct_json(self) -> None:
        self.assertEqual(ac.extract_json_block('{"a": 1}'), {"a": 1})

    def test_markdown_fenced(self) -> None:
        text = "Here you go:\n```json\n{\"a\": 2}\n```\nThanks!"
        self.assertEqual(ac.extract_json_block(text), {"a": 2})

    def test_embedded_braces(self) -> None:
        text = "blah blah {\"a\": 3, \"b\": [1,2]} trailing"
        self.assertEqual(ac.extract_json_block(text), {"a": 3, "b": [1, 2]})

    def test_failure_raises(self) -> None:
        with self.assertRaises(ValueError):
            ac.extract_json_block("no json here at all")

    def test_top_level_array_rejected(self) -> None:
        # A bare array is not an object; callers index by key, so it must not pass.
        with self.assertRaises(ValueError):
            ac.extract_json_block("[1, 2, 3]")

    def test_object_after_array_preamble(self) -> None:
        self.assertEqual(ac.extract_json_block('[1,2] then {"a": 9}'), {"a": 9})


class SanitizeTrainingParamsTests(unittest.TestCase):
    def test_boolean_strings_are_correct(self) -> None:
        out = ac.sanitize_training_params(
            {"cos_lr": "false", "rect": "0", "pretrained": "no"}
        )
        self.assertIs(out["cos_lr"], False)
        self.assertIs(out["rect"], False)
        self.assertIs(out["pretrained"], False)

    def test_boolean_truthy(self) -> None:
        out = ac.sanitize_training_params({"cos_lr": "true", "rect": True})
        self.assertIs(out["cos_lr"], True)
        self.assertIs(out["rect"], True)

    def test_clamps_probabilities_and_angles(self) -> None:
        out = ac.sanitize_training_params(
            {"fliplr": 2.0, "hsv_h": -1.0, "degrees": 500, "fraction": 9, "label_smoothing": 5}
        )
        self.assertEqual(out["fliplr"], 1.0)
        self.assertEqual(out["hsv_h"], 0.0)
        self.assertEqual(out["degrees"], 180.0)
        self.assertEqual(out["fraction"], 1.0)
        self.assertEqual(out["label_smoothing"], 0.9)

    def test_batch_rules(self) -> None:
        self.assertEqual(ac.sanitize_training_params({"batch": 0})["batch"], 1)
        self.assertEqual(ac.sanitize_training_params({"batch": -1})["batch"], -1)
        self.assertEqual(ac.sanitize_training_params({"batch": 32})["batch"], 32)

    def test_imgsz_rounded_to_multiple_of_32(self) -> None:
        self.assertEqual(ac.sanitize_training_params({"imgsz": 100})["imgsz"], 96)
        self.assertEqual(ac.sanitize_training_params({"imgsz": 5})["imgsz"], 32)

    def test_optimizer_normalisation(self) -> None:
        self.assertEqual(ac.sanitize_training_params({"optimizer": "adamw"})["optimizer"], "AdamW")
        self.assertEqual(ac.sanitize_training_params({"optimizer": "nonsense"})["optimizer"], "auto")

    def test_auto_augment_enum(self) -> None:
        self.assertEqual(ac.sanitize_training_params({"auto_augment": "RandAugment"})["auto_augment"], "randaugment")
        self.assertEqual(ac.sanitize_training_params({"auto_augment": "none"})["auto_augment"], "")
        self.assertNotIn("auto_augment", ac.sanitize_training_params({"auto_augment": "bogus"}))

    def test_unknown_keys_dropped(self) -> None:
        self.assertEqual(ac.sanitize_training_params({"garbage": 1, "rm": "-rf"}), {})

    def test_non_dict_input(self) -> None:
        self.assertEqual(ac.sanitize_training_params(None), {})
        self.assertEqual(ac.sanitize_training_params("nope"), {})


class SanitizeAugmentationTransformsTests(unittest.TestCase):
    def test_drops_unsupported_transform(self) -> None:
        out = ac.sanitize_augmentation_transforms([{"name": "Bogus", "enabled": True, "p": 0.5}])
        self.assertEqual(out, [])

    def test_drops_unknown_param_names(self) -> None:
        out = ac.sanitize_augmentation_transforms(
            [{"name": "Rotate", "enabled": True, "p": 0.5, "degrees": 15.0}]
        )
        self.assertEqual(len(out), 1)
        self.assertNotIn("degrees", out[0])

    def test_completes_range_pairs_from_defaults(self) -> None:
        out = ac.sanitize_augmentation_transforms(
            [{"name": "Rotate", "enabled": True, "p": 0.5, "limit_low": -30}]
        )[0]
        self.assertIn("limit_low", out)
        self.assertIn("limit_high", out)
        self.assertIn("border_mode", out)

    def test_reorders_inverted_pairs(self) -> None:
        out = ac.sanitize_augmentation_transforms(
            [{"name": "GaussNoise", "enabled": True, "p": 0.3,
              "std_range_low": 0.9, "std_range_high": 0.1}]
        )[0]
        self.assertLessEqual(out["std_range_low"], out["std_range_high"])

    def test_clamps_probability(self) -> None:
        out = ac.sanitize_augmentation_transforms([{"name": "HorizontalFlip", "enabled": True, "p": 5}])[0]
        self.assertEqual(out["p"], 1.0)

    def test_enabled_coerced(self) -> None:
        out = ac.sanitize_augmentation_transforms([{"name": "D4", "enabled": "false", "p": 1.0}])[0]
        self.assertIs(out["enabled"], False)

    def test_non_list_input(self) -> None:
        self.assertEqual(ac.sanitize_augmentation_transforms({"name": "Rotate"}), [])

    def test_every_supported_transform_instantiates(self) -> None:
        """The whole point of the schema fix: sanitized transforms must build."""
        import albumentations as A
        from src.augmentation.transforms import build_albu_kwargs

        for name in ac.SUPPORTED_TRANSFORMS:
            out = ac.sanitize_augmentation_transforms([{"name": name, "enabled": True, "p": 0.5}])
            self.assertTrue(out, name)
            cls = getattr(A, name, None)
            self.assertIsNotNone(cls, f"albumentations has no {name}")
            cls(**build_albu_kwargs(out[0]))  # must not raise


class CatalogDriftTests(unittest.TestCase):
    def test_supported_transforms_match_engine_catalog(self) -> None:
        from src.augmentation.transforms import TRANSFORM_GROUPS

        engine = {n for names in TRANSFORM_GROUPS.values() for n in names}
        self.assertEqual(set(ac.SUPPORTED_TRANSFORMS), engine)

    def test_catalog_excludes_paramless_transforms(self) -> None:
        catalog = ac.build_transform_param_catalog()
        self.assertNotIn("HorizontalFlip", catalog)
        self.assertIn("Rotate", catalog)


class ExtractApiErrorTests(unittest.TestCase):
    def test_nested_message(self) -> None:
        self.assertEqual(ac._extract_api_error({"error": {"message": "bad key"}}), "bad key")

    def test_flat_string(self) -> None:
        self.assertEqual(ac._extract_api_error({"error": "timeout"}), "timeout")

    def test_bare_string(self) -> None:
        self.assertEqual(ac._extract_api_error("raw"), "raw")

    def test_empty_dict(self) -> None:
        self.assertTrue(ac._extract_api_error({}))


class MakeHttpRequestRetryTests(unittest.TestCase):
    def test_retries_on_transient_then_succeeds(self) -> None:
        calls = []

        def fake(url, method, headers, data, timeout):
            calls.append(1)
            if len(calls) < 2:
                return 503, {"error": "busy"}
            return 200, {"ok": True}

        with mock.patch.object(ac, "_do_single_request", side_effect=fake), \
             mock.patch.object(ac.time, "sleep"):
            status, body = ac.make_http_request("http://x", "GET")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"ok": True})
        self.assertEqual(len(calls), 2)

    def test_gives_up_after_max_retries(self) -> None:
        with mock.patch.object(ac, "_do_single_request", return_value=(429, {"error": "rl"})), \
             mock.patch.object(ac.time, "sleep") as slept:
            status, _ = ac.make_http_request("http://x", "GET", max_retries=2)
        self.assertEqual(status, 429)
        self.assertEqual(slept.call_count, 2)

    def test_non_retryable_returns_immediately(self) -> None:
        with mock.patch.object(ac, "_do_single_request", return_value=(400, {"error": "bad"})) as called, \
             mock.patch.object(ac.time, "sleep"):
            status, _ = ac.make_http_request("http://x", "GET")
        self.assertEqual(status, 400)
        self.assertEqual(called.call_count, 1)


class FetchMultimodalModelsTests(unittest.TestCase):
    def test_no_key_returns_empty(self) -> None:
        self.assertEqual(ac.fetch_multimodal_models("Gemini", ""), [])

    def test_gemini_parsing(self) -> None:
        res = {
            "models": [
                {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
                {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["generateContent"]},
            ]
        }
        with mock.patch.object(ac, "make_http_request", return_value=(200, res)):
            models = ac.fetch_multimodal_models("Gemini", "key")
        self.assertEqual(models, ["gemini-2.5-flash"])

    def test_openai_parsing_filters_non_chat(self) -> None:
        res = {
            "data": [
                {"id": "gpt-4o"},
                {"id": "gpt-4o-audio-preview"},
                {"id": "text-embedding-3-small"},
                {"id": "gpt-4.1-mini"},
            ]
        }
        with mock.patch.object(ac, "make_http_request", return_value=(200, res)):
            models = ac.fetch_multimodal_models("OpenAI", "key")
        self.assertEqual(models, ["gpt-4.1-mini", "gpt-4o"])

    def test_error_raises_clean_message(self) -> None:
        with mock.patch.object(ac, "make_http_request", return_value=(0, {"error": "ConnectionError: down"})):
            with self.assertRaises(ValueError) as ctx:
                ac.fetch_multimodal_models("Gemini", "key")
        self.assertIn("down", str(ctx.exception))


class QueryLlmMultimodalTests(unittest.TestCase):
    def test_gemini_payload_and_parse(self) -> None:
        captured = {}

        def fake(url, method="GET", headers=None, data=None, **kw):
            captured["url"] = url
            captured["data"] = data
            return 200, {"candidates": [{"content": {"parts": [{"text": "{\"x\": 1}"}]}}]}

        with mock.patch.object(ac, "make_http_request", side_effect=fake):
            out = ac.query_llm_multimodal(
                "Gemini", "key", "gemini-2.5-flash", "prompt text",
                [{"mime_type": "image/jpeg", "base64_data": "AAA", "filename": "a.jpg"}],
                system_instruction="be good",
            )
        self.assertEqual(out, '{"x": 1}')
        self.assertIn("systemInstruction", captured["data"])
        self.assertEqual(captured["data"]["systemInstruction"]["parts"][0]["text"], "be good")
        # image part present
        self.assertTrue(any("inlineData" in p for p in captured["data"]["contents"][0]["parts"]))

    def test_gemini_block_reason(self) -> None:
        with mock.patch.object(ac, "make_http_request", return_value=(200, {"promptFeedback": {"blockReason": "SAFETY"}})):
            with self.assertRaises(ValueError) as ctx:
                ac.query_llm_multimodal("Gemini", "k", "m", "p", [])
        self.assertIn("SAFETY", str(ctx.exception))

    def test_gemini_http_error(self) -> None:
        with mock.patch.object(ac, "make_http_request", return_value=(403, {"error": {"message": "no access"}})):
            with self.assertRaises(ValueError) as ctx:
                ac.query_llm_multimodal("Gemini", "k", "m", "p", [])
        self.assertIn("no access", str(ctx.exception))

    def test_openai_payload_and_parse(self) -> None:
        captured = {}

        def fake(url, method="GET", headers=None, data=None, **kw):
            captured["data"] = data
            captured["headers"] = headers
            return 200, {"choices": [{"message": {"content": "{\"y\": 2}"}}]}

        with mock.patch.object(ac, "make_http_request", side_effect=fake):
            out = ac.query_llm_multimodal(
                "OpenAI", "key", "gpt-4o", "prompt",
                [{"mime_type": "image/jpeg", "base64_data": "BBB", "filename": "b.jpg"}],
                system_instruction="sys",
            )
        self.assertEqual(out, '{"y": 2}')
        self.assertEqual(captured["data"]["response_format"], {"type": "json_object"})
        self.assertEqual(captured["headers"]["Authorization"], "Bearer key")

    def test_unsupported_provider(self) -> None:
        with self.assertRaises(ValueError):
            ac.query_llm_multimodal("Claude", "k", "m", "p", [])


class GetDatasetAiSummaryTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> None:
        (root / "train" / "images").mkdir(parents=True)
        (root / "train" / "labels").mkdir(parents=True)
        for i in range(3):
            arr = (np.random.rand(64, 80, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(root / "train" / "images" / f"img{i}.jpg")
            (root / "train" / "labels" / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (root / "data.yaml").write_text("names: [thing]\nnc: 1\ntrain: train/images\nval: train/images\n")

    def test_summary_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ds"
            self._make_dataset(root)
            details, samples = ac.get_dataset_ai_summary(root)

        self.assertIn("total_images", details)
        self.assertIn("classes", details)
        self.assertTrue(1 <= len(samples) <= 2)
        for sample in samples:
            self.assertEqual(sample["mime_type"], "image/jpeg")
            self.assertTrue(sample["base64_data"])
        self.assertTrue(details["sample_image_properties"])

    def test_empty_dataset_no_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "empty"
            root.mkdir()
            details, samples = ac.get_dataset_ai_summary(root)
        self.assertEqual(samples, [])


if __name__ == "__main__":
    unittest.main()

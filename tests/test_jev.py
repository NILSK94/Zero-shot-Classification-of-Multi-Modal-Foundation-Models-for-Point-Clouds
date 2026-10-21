from __future__ import annotations

import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pandas as pd
from PIL import Image

from mmfm_tool.providers import JevProvider, ProviderError, build_prompt, create_provider, photo_check_scores
from mmfm_tool.jev_checks import run_classification_preview
from mmfm_tool.runner import ExperimentSpec, ModelTarget, create_run_directory, planned_request_count, run_experiment


def decision_response(choice="chair"):
    return {
        "answers": {"classification": {"type": "choice", "choice": choice,
                    "confidence": 0.8, "probabilities": {"chair": 0.9, "table": 0.1}}},
        "model": "typesafe/jev-1.14-test-version",
        "usage": {"input_tokens": 123, "output_tokens": 0, "cost": 0.00001},
    }


def photo_response(scores):
    return {
        "answers": {f"class_{label}": {"type": "noul", "noul": score} for label, score in scores.items()},
        "model": "typesafe/jev-1.14-test-version",
        "usage": {"input_tokens": 150, "output_tokens": 0, "cost": 0.00002},
    }


class JevTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / "tmp"
        test_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=test_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "ground_truth_chair_secret_metadata.png"
        Image.new("RGB", (384, 384), "white").save(self.image)
        self.provider = create_provider("jev", "typesafe/jev-1.14", "test-key")

    def test_native_request_and_full_result(self):
        prompt = self.provider.build_prompt("s3dis", ("chair", "table"), 4,
                                            {"width": 1, "depth": 2, "height": 3})
        raw = json.dumps(decision_response())
        with patch("mmfm_tool.providers.urlopen", return_value=BytesIO(raw.encode())) as post:
            result = self.provider.classify([self.image] * 4, prompt, ("chair", "table"))
        request = post.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url, "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(post.call_args.kwargs["timeout"], 120)
        self.assertEqual(body["questions"]["classification"]["type"], "choice")
        self.assertEqual(body["questions"]["classification"]["criteria"], {"chair": "chair", "table": "table"})
        self.assertEqual(len(body["state"]), 8)
        self.assertTrue(body["state"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn(self.image.name, request.data.decode())
        self.assertIn("width (X) = 1.0000", prompt)
        self.assertEqual(prompt, build_prompt("s3dis", ("chair", "table"), 4, {"width": 1, "depth": 2, "height": 3}, criteria_reference=True))
        self.assertEqual(set(body["questions"]), {"classification"})
        self.assertEqual(result.label, "chair")
        self.assertEqual(result.model_version, "typesafe/jev-1.14-test-version")
        self.assertEqual(result.raw_response, raw)
        self.assertEqual(result.usage["cost"], 0.00001)
        self.assertIn("Your entire response", build_prompt("modelnet", ("chair",), 1))

    def test_invalid_responses_do_not_become_predictions(self):
        cases = [b"not json", b"null", b"[]", b"{}",
                 json.dumps(decision_response("unknown")).encode(),
                 json.dumps({"answers": {"classification": {"type": "noul", "choice": "chair"}}}).encode()]
        for raw in cases:
            with self.subTest(raw=raw), patch("mmfm_tool.providers.urlopen", return_value=BytesIO(raw)):
                with self.assertRaises(ProviderError):
                    self.provider.classify([self.image], "classify", ("chair", "table"))

    def test_errors_and_image_limits(self):
        for size in ((256, 256), (256, 512), (255, 512), (1569, 512)):
            Image.new("RGB", size).save(self.image)
            with self.subTest(size=size), patch("mmfm_tool.providers.urlopen") as post:
                with self.assertRaises(ProviderError):
                    self.provider.classify([self.image], "classify", ("chair",))
                post.assert_not_called()
        Image.new("RGB", (384, 384)).save(self.image)
        for error in (HTTPError(self.provider.endpoint, 402, "credit", {}, None), URLError("offline"), TimeoutError()):
            with self.subTest(error=error), patch("mmfm_tool.providers.urlopen", side_effect=error):
                with self.assertRaises(ProviderError) as raised:
                    self.provider.classify([self.image], "classify", ("chair",))
                self.assertNotIn("test-key", str(raised.exception))
        with self.assertRaises(ProviderError):
            JevProvider("typesafe/jev-1.14", "")

    def test_both_datasets_all_conditions_repeats_and_saved_artifacts(self):
        records = pd.DataFrame([
            {"sample_id": "chair/object", "label": "chair", "view_count": count,
             **{f"view_{i}": str(self.image) for i in range(4)}} for count in range(1, 5)
        ])
        indoor = records.iloc[:1].assign(sample_id="Area_5/room/chair_1", area="Area_5", width=1., depth=2., height=3.)
        specs = [ExperimentSpec("modelnet", ("chair", "table"), repeat_count=2, max_retries=0),
                 ExperimentSpec("s3dis", ("chair", "table"), area_scope="Area_5", repeat_count=2, max_retries=0)]
        target = ModelTarget("jev", "typesafe/jev-1.14", "test-key")
        requests = []

        def post(request, **kwargs):
            requests.append(json.loads(request.data))
            return BytesIO(json.dumps(decision_response()).encode())

        with patch("mmfm_tool.providers.urlopen", side_effect=post):
            for cohort, spec, expected in zip((records, indoor), specs, (8, 4)):
                directory = create_run_directory(self.root / "runs", spec, [target], cohort)
                self.assertEqual(planned_request_count(cohort, spec, [target]), expected)
                results, evaluated = run_experiment(cohort, spec, [target], directory)
                self.assertEqual(len(results), expected)
                self.assertTrue(results["status"].eq("ok").all())
                self.assertTrue(results["pred_label"].eq("chair").all())
                self.assertFalse(evaluated["aggregate"].empty)
                self.assertNotIn("test-key", (directory / "config.json").read_text())
                raw = json.loads(results.iloc[0]["raw_response"])
                self.assertEqual(raw["answers"]["classification"]["confidence"], 0.8)
                self.assertEqual(json.loads(results.iloc[0]["usage_json"])["cost"], 0.00001)
                self.assertIn("Your entire response", results.iloc[0]["prompt_text"])
                config = json.loads((directory / "config.json").read_text())
                self.assertEqual(config["targets"][0]["decision_protocol"], "jev-class-list-v2")
                # A second execution resumes from successful rows without new API calls.
                before = len(requests)
                run_experiment(cohort, spec, [target], directory)
                self.assertEqual(len(requests), before)
        self.assertEqual(len(requests), 12)
        self.assertTrue(all(set(item["questions"]) == {"classification"} for item in requests))
        self.assertTrue(all(item["questions"]["classification"]["type"] == "choice" for item in requests))
        self.assertEqual([len(item["state"]) // 2 for item in requests[:8]], [1, 2, 3, 4] * 2)
        indoor_prompts = [item["questions"]["classification"]["instructions"] for item in requests[8:]]
        self.assertEqual(["dimensions in meters" in prompt for prompt in indoor_prompts], [False, True] * 2)

    def test_photo_check_questions_scores_and_invalid_answers(self):
        provider = create_provider("jev_photo_check", "typesafe/jev-1.14", "test-key")
        prompt = provider.build_prompt("modelnet", ("chair", "table"), 2)
        with patch("mmfm_tool.providers.urlopen", return_value=BytesIO(json.dumps(photo_response({"chair": .6, "table": .8})).encode())) as post:
            result = provider.classify([self.image] * 2, prompt, ("chair", "table"))
        body = json.loads(post.call_args.args[0].data)
        self.assertEqual(result.label, "table")
        self.assertEqual(set(body["questions"]), {"class_chair", "class_table"})
        self.assertTrue(all(question["type"] == "noul" for question in body["questions"].values()))
        self.assertEqual(set(body["questions"]["class_chair"]["criteria"]), {"true", "false"})
        self.assertNotIn("Choose exactly one", prompt)
        self.assertEqual(photo_check_scores(result.raw_response), {"chair": .6, "table": .8})
        # Scores are independent, and ties never use the object's ground truth.
        self.assertEqual(provider.parse_answer(photo_response({"table": .8, "chair": .8}), ("table", "chair")), "chair")
        for scores in ({"chair": .8}, {"chair": .9, "table": 1.1}, {"chair": .9, "table": True},
                       {"chair": .9, "table": float("nan")}, {"chair": .9, "table": "0.8"}):
            with self.subTest(scores=scores), patch("mmfm_tool.providers.urlopen", return_value=BytesIO(json.dumps(photo_response(scores)).encode())):
                with self.assertRaises(ProviderError):
                    provider.classify([self.image], prompt, ("chair", "table"))

    def test_small_preview_one_and_two_images_saved_separately(self):
        for count in (1, 2):
            with self.subTest(count=count), patch("mmfm_tool.providers.urlopen", return_value=BytesIO(json.dumps(decision_response("table")).encode())) as post:
                record, path = run_classification_preview(
                    [self.image] * count, "s3dis", ("chair", "table"), "typesafe/jev-1.14", "test-key",
                    self.root / "previews", sample_id="PRIVATE_SAMPLE", true_label="PRIVATE_TRUTH",
                    geometry={"width": 1., "depth": 2., "height": 3.},
                )
                self.assertEqual(post.call_count, 1)
                body = post.call_args.args[0].data.decode()
                self.assertNotIn("PRIVATE_SAMPLE", body)
                self.assertNotIn("PRIVATE_TRUTH", body)
                self.assertEqual(len(json.loads(body)["state"]) // 2, count)
                self.assertEqual(record["pred_label"], "table")
                self.assertEqual(record["provider"], "jev")
                self.assertEqual(set(record["questions"]), {"classification"})
                self.assertEqual(record["prompt_text"], build_prompt("s3dis", ("chair", "table"), count, {"width": 1., "depth": 2., "height": 3.}, criteria_reference=True))
                self.assertTrue(record["preview_only"])
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record)
                self.assertNotIn("test-key", path.read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            run_classification_preview([self.image] * 3, "s3dis", ("chair",), "typesafe/jev-1.14", "test-key", self.root)

    def test_photo_check_experiment_writes_class_scores_and_keeps_low_scores(self):
        records = pd.DataFrame([{"sample_id": "object", "label": "chair", "view_count": 1,
                                 "view_0": str(self.image)}])
        spec = ExperimentSpec("modelnet", ("chair", "table"), view_counts=(1,), max_retries=0)
        target = ModelTarget("jev_photo_check", "typesafe/jev-1.14", "test-key")
        directory = create_run_directory(self.root / "runs", spec, [target], records)
        with patch("mmfm_tool.providers.urlopen", return_value=BytesIO(json.dumps(photo_response({"chair": .1, "table": .2})).encode())):
            results, evaluated = run_experiment(records, spec, [target], directory)
        self.assertEqual(results.iloc[0]["status"], "ok")
        self.assertEqual(results.iloc[0]["pred_label"], "table")
        self.assertEqual(evaluated["summary"].iloc[0]["coverage"], 1.)
        self.assertEqual(evaluated["summary"].iloc[0]["end_to_end_accuracy"], 0.)
        scores = pd.read_csv(directory / "photo_checks.csv")
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores.loc[scores["selected"], "candidate_class"].tolist(), ["table"])
        self.assertEqual(scores["yes_probability"].tolist(), [.1, .2])
        config = json.loads((directory / "config.json").read_text())
        self.assertEqual(config["targets"][0]["decision_protocol"], "jev-photo-check-v1")

    def test_app_combined_start_and_missing_key(self):
        from streamlit.testing.v1 import AppTest

        app_file = self.root / "app.py"
        source_app = Path(__file__).resolve().parents[1] / "app.py"
        app_file.write_text(source_app.read_text(encoding="utf-8"), encoding="utf-8")
        modelnet = self.root / "modelnet"
        for count in range(1, 5):
            folder = modelnet / f"Model_IMG_{count}_Isometrien" / "chair" / "test"
            folder.mkdir(parents=True)
            for index in range(count):
                (folder / f"chair_0001_iso_{index}.png").write_bytes(self.image.read_bytes())
        prepared = self.root / "artifacts" / "s3dis_prepared"
        prepared.mkdir(parents=True)
        pd.DataFrame([{
            "sample_id": f"{area}/room/chair_1", "label": "chair", "area": area,
            "status": "ok", "width": 1., "depth": 2., "height": 3.,
            **{f"view_{i}": str(self.image) for i in range(4)},
        } for area in ("Area_5", "Area_6")]).to_csv(prepared / "s3dis_manifest.csv", index=False)

        def widget(items, label):
            return next(item for item in items if item.label == label)

        environment = {"MODELNET_PATH": str(modelnet), "S3DIS_PATH": str(self.root / "missing_s3dis"),
                       "LEGACY_RESULTS_PATH": str(self.root / "missing_legacy"), "OPENROUTER_API_KEY": ""}
        with patch.dict("os.environ", environment), patch("mmfm_tool.providers.urlopen") as post:
            app = AppTest.from_file(str(app_file), default_timeout=30).run()
            self.assertFalse(app.exception)
            widget(app.checkbox, "JEV via OpenRouter").check()
            self.assertFalse(any(item.label == "JEV-Verfahren" for item in app.selectbox))
            widget(app.radio, "Datensatz").set_value("ModelNet + S3DIS (alle Runs)")
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(widget(app.selectbox, "S3DIS Area").value, "Area_5")
            self.assertEqual(widget(app.metric, "Geplante Requests").value, "6")
            widget(app.checkbox, "Ich bestätige, dass die ausgewählten API-Modelle Kosten verursachen können.").check()
            widget(app.button, "Alle Runs starten").click()
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any("OpenRouter API-Key fehlt" in item.value for item in app.error))
            post.assert_not_called()
            widget(app.text_input, "OpenRouter API-Key").set_value("test-key")
            widget(app.button, "Alle Runs starten").click()
            post.side_effect = lambda *args, **kwargs: BytesIO(json.dumps(decision_response()).encode())
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(post.call_count, 6)
            runs = list((self.root / "artifacts" / "runs").glob("*/results.csv"))
            self.assertEqual(len(runs), 2)
            results = pd.concat([pd.read_csv(path) for path in runs])
            self.assertEqual(set(results["dataset"]), {"modelnet", "s3dis"})
            self.assertTrue(results["status"].eq("ok").all())
            self.assertEqual(set(results["condition"]), {"views_1", "views_2", "views_3", "views_4", "rgb", "rgb_aabb"})
            # The separate image test must make only one request and never add a scientific run.
            def respond_choice(request, **kwargs):
                questions = json.loads(request.data)["questions"]
                self.assertEqual(set(questions), {"classification"})
                self.assertEqual(questions["classification"]["type"], "choice")
                return BytesIO(json.dumps(decision_response()).encode())

            post.side_effect = respond_choice
            widget(app.checkbox, "Ich bestätige die API-Kosten für diesen einzelnen Bildtest.").check()
            for count in (1, 2):
                widget(app.radio, "Bildtest: Anzahl Bilder").set_value(count)
                widget(app.button, "1–2 Bilder mit JEV testen").click()
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(widget(app.metric, "Bildtest: Vorhersage").value, "chair")
                self.assertEqual(len(list((self.root / "artifacts" / "runs").glob("*/results.csv"))), 2)
                self.assertEqual(len(json.loads(post.call_args.args[0].data)["state"]) // 2, count)
            self.assertEqual(post.call_count, 8)
            self.assertEqual(len(list((self.root / "artifacts" / "jev_photo_checks").glob("*.json"))), 2)
            widget(app.button, "Alle Runs starten").click()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(post.call_count, 14)
            score_files = list((self.root / "artifacts" / "runs").glob("*/photo_checks.csv"))
            self.assertEqual(len(score_files), 0)

    def test_same_task_context_with_classes_only_in_criteria_for_both_datasets(self):
        from mmfm_tool.constants import MODELNET_CLASSES, S3DIS_CLASSES

        for dataset, labels, geometry in (("modelnet", MODELNET_CLASSES, None),
                                           ("s3dis", S3DIS_CLASSES, None),
                                           ("s3dis", S3DIS_CLASSES, {"width": 1., "depth": 2., "height": 3.})):
            for count in (1, 2, 3, 4):
                with self.subTest(dataset=dataset, count=count, geometry=geometry):
                    prompt = self.provider.build_prompt(dataset, labels, count, geometry)
                    standard = build_prompt(dataset, labels, count, geometry)
                    expected = standard.replace("Choose exactly one label from this closed list:\n" + ", ".join(sorted(labels)),
                                                "Choose exactly one label from the closed class list in criteria.")
                    expected = expected.replace("as written in the list", "as written in criteria")
                    self.assertEqual(prompt, expected)
                    questions = self.provider.build_questions(prompt, labels)
                    self.assertEqual(set(questions), {"classification"})
                    self.assertEqual(questions["classification"]["type"], "choice")
                    self.assertEqual(set(questions["classification"]["criteria"]), set(labels))
                    self.assertNotIn("Candidate class:", prompt)
                    self.assertNotIn("yes/no", prompt)
                    self.assertNotIn(", ".join(sorted(labels)), questions["classification"]["instructions"])
                    self.assertIn("class list in criteria", questions["classification"]["instructions"])


if __name__ == "__main__":
    unittest.main()

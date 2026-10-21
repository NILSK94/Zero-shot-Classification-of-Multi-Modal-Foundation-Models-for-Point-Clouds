"""Read the original paper CSV archive without rewriting files or guessing caption labels."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

MODELNET_OBJECT = re.compile(r"^(?P<label>.+?)_(?P<number>\d+)(?:_iso_\d+)?(?:\.png)?$", re.I)


def load_historical_csvs(root: str | Path, *, original_rows: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    audit: list[dict] = []
    for path in sorted(Path(root).glob("*.csv")):
        try:
            frame = pd.read_csv(path).fillna("")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            audit.append({"file": path.name, "state": "unreadable", "note": str(exc)})
            continue
        prediction_column = "label" if "label" in frame else "class" if "class" in frame else None
        source_column = "image" if "image" in frame else "object" if "object" in frame else None
        if prediction_column is None or source_column is None:
            audit.append({"file": path.name, "state": "not reconstructable",
                          "note": 'No unambiguous class prediction with an object ID is available; captions are not converted using ground truth.'})
            continue
        lower_name = path.name.casefold()
        indoor = "colored-cloud" in lower_name or "feature-geometries" in lower_name
        dataset = "blk2go" if indoor else "modelnet"
        model = "gpt-4o" if lower_name.startswith("gpt-4o") else "gemini-1.5-pro" if lower_name.startswith("gemini-1.5-pro") else path.stem
        match = re.search(r"-(\d+)-(?:iso|isometric)", lower_name)
        views = int(match.group(1)) if match else 1
        condition = ("rgb_aabb" if "feature-geometries" in lower_name else "rgb") if indoor else f"views_{views}"
        cohort = "full" if "big-test" in lower_name else "pilot"
        scope = f"historical_{dataset}_{cohort}"
        observations = []
        for index, source in frame.iterrows():
            object_id = str(source[source_column]).strip().replace("\\", "/").rsplit("/", 1)[-1]
            if not indoor:
                item = MODELNET_OBJECT.fullmatch(object_id)
                if not item:
                    continue
                truth = item.group("label").casefold()
                object_id = f"{truth}_{item.group('number')}"
                sample_id = f"{truth}/{object_id}"
            else:
                truth = re.sub(r"_\d+$", "", object_id).casefold()
                sample_id = f"blk2go/{object_id.casefold()}"
            observations.append({"sample_id": sample_id, "true_label": truth,
                                 "candidate": str(source[prediction_column]).strip().casefold(), "source_row": int(index) + 2})
            if original_rows and not indoor:
                # Exact label extraction in Confusion_Matrix_Evaluation.ipynb.
                observations[-1]["true_label"] = str(source[source_column]).split("_")[0].lower()
        normalized = pd.DataFrame(observations)
        if normalized.empty:
            audit.append({"file": path.name, "state": "not reconstructable", "note": 'No unambiguous object IDs detected.'})
            continue
        # Preserve the historical output labels verbatim after whitespace/case normalization.
        # Several view rows are one object decision only when their labels agree.
        ambiguous = 0
        if original_rows and not indoor:
            for item in observations:
                rows.append({"dataset": dataset, "sample_id": item["sample_id"], "true_label": item["true_label"],
                             "pred_label": item["candidate"], "status": "ok" if item["candidate"] else "error",
                             "provider": "historical_csv", "model": model, "condition": condition,
                             "view_count": views, "repeat": 1, "protocol_scope": scope,
                             "source_run": path.name, "latency_s": float("nan"),
                             "task_id": hashlib.sha256(f"{path.name}|row:{item['source_row']}".encode()).hexdigest(),
                             "historical_source_rows": str(item["source_row"]), "historical_row_count": 1,
                             "truth_source": "original notebook: image.split('_')[0]", "error": ""})
            audit.append({"file": path.name, "state": "loaded", "source_rows": len(frame),
                          "objects": normalized["sample_id"].nunique(), "dataset": dataset,
                          "note": 'Original script: every CSV row counts; the class is extracted at the first underscore.'})
            continue
        for (sample_id, truth), group in normalized.groupby(["sample_id", "true_label"], sort=True):
            predictions = group["candidate"].unique()
            valid = len(predictions) == 1 and bool(predictions[0])
            ambiguous += int(not valid)
            predicted = predictions[0] if valid else ""
            rows.append({"dataset": dataset, "sample_id": sample_id, "true_label": truth,
                         "pred_label": predicted, "status": "ok" if valid else "error",
                         "provider": "historical_csv", "model": model, "condition": condition,
                         "view_count": views, "repeat": 1, "protocol_scope": scope,
                         "source_run": path.name, "latency_s": float("nan"),
                         "task_id": hashlib.sha256(f"{path.name}|{sample_id}".encode()).hexdigest(),
                         "historical_source_rows": ",".join(map(str, group["source_row"])),
                         "historical_row_count": len(group), "truth_source": "object filename",
                         "error": 'Conflicting or empty object responses; no majority-vote prediction is invented.' if not valid else ""})
        audit.append({"file": path.name, "state": "loaded", "source_rows": len(frame),
                      "objects": normalized["sample_id"].nunique(), "ambiguous_objects": ambiguous,
                      "dataset": dataset, "note": 'Object-level decisions; historical protocols remain separate. Classes are derived from object names.'})
    result = pd.DataFrame(rows)
    result.attrs["counting"] = "original_csv_rows" if original_rows else "object_decisions"
    return result, pd.DataFrame(audit)

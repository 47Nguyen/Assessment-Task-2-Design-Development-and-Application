"""
Task 3: Reload the trained MLPs to predict gender and usage.

HOW TO RUN (from the project root, after running train.py):
    python -m tasks.task3_gender_usage.predict --models-dir outputs/task3/final_run/models --image A2_FashionDataset/FashionDataset/test/images_test/52003.jpg
    python -m tasks.task3_gender_usage.predict --models-dir outputs/task3/final_run/models --output outputs/task3/final_run/styles_prediction_task3.csv

Use the models folder from your actual run. Prediction does NOT train again.
CSV export fills only gender and usage; other tasks' columns stay unchanged.
"""

# ---------------------------------------------------------------------------
# 1. Setup: use the exact image preparation shared with train.py
# ---------------------------------------------------------------------------
import argparse
import io
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

from .data import TARGETS, find_root, prepare_image


# ---------------------------------------------------------------------------
# 2. Load both final models and their saved class/preprocessing information
# ---------------------------------------------------------------------------
class Task3Predictor:
    """Keep the gender and usage models loaded for repeated predictions."""

    def __init__(self, models_dir):
        directory = Path(models_dir)
        self.models = {}
        self.specifications = {}

        for target in TARGETS:
            metadata_path = directory / f"{target}_final.json"
            model_path = directory / f"{target}_final.keras"
            if not metadata_path.is_file() or not model_path.is_file():
                raise FileNotFoundError(
                    f"Missing {target}_final.keras or {target}_final.json in "
                    f"{directory}. Run train.py first, then use its models folder."
                )

            specification = json.loads(metadata_path.read_text(encoding="utf-8"))
            model = tf.keras.models.load_model(str(model_path), compile=False)
            expected_shape = (
                specification["config"]["height"],
                specification["config"]["width"],
                3,
            )
            expected_classes = len(specification["classes"])
            if (
                tuple(model.input_shape[1:]) != expected_shape
                or model.output_shape[-1] != expected_classes
            ):
                raise ValueError(f"Model and metadata mismatch for {target}.")

            self.models[target] = model
            self.specifications[target] = specification

    # -----------------------------------------------------------------------
    # 3. Predict batches without loading the entire test set into memory
    # -----------------------------------------------------------------------
    def predict_paths(self, paths, batch_size=128):
        """Return {target: probability_array}, preserving the input path order."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        paths = list(paths)
        predictions = {}

        for target in TARGETS:
            specification = self.specifications[target]
            size = (
                specification["config"]["height"],
                specification["config"]["width"],
            )
            batches = []
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start:start + batch_size]
                pixels = np.stack([
                    prepare_image(path, size) for path in batch_paths
                ])
                # training=False disables dropout. The model itself rescales
                # uint8 pixels from 0..255 to 0..1, just as it did in training.
                probabilities = self.models[target](pixels, training=False).numpy()
                batches.append(probabilities)

            if batches:
                scores = np.concatenate(batches)
            else:
                scores = np.empty((0, len(specification["classes"])))
            if not np.isfinite(scores).all() or (
                len(scores) and not np.allclose(scores.sum(axis=1), 1, atol=1e-5)
            ):
                raise ValueError("Invalid model probabilities.")
            predictions[target] = scores

        return predictions

    def predict_image(self, source):
        """Return labels, model scores and all class probabilities for one image.

        A softmax score is not a calibrated guarantee of correctness. Class
        names must use the saved training order, not a new alphabetical order.
        """
        if hasattr(source, "read"):
            content = source.read()
        else:
            content = Path(source).read_bytes()

        predictions = {}
        for target in TARGETS:
            specification = self.specifications[target]
            size = (
                specification["config"]["height"],
                specification["config"]["width"],
            )
            # Each model gets a fresh stream: the previous image read may
            # already have consumed or closed its stream.
            pixels = prepare_image(io.BytesIO(content), size)[None]
            probabilities = self.models[target](pixels, training=False).numpy()
            if not np.isfinite(probabilities).all() or not np.allclose(
                probabilities.sum(axis=1), 1, atol=1e-5
            ):
                raise ValueError("Invalid model probabilities.")

            classes = specification["classes"]
            values = probabilities[0]
            best_index = int(values.argmax())
            predictions[target] = {
                "label": classes[best_index],
                "score": float(values[best_index]),
                "probabilities": dict(zip(classes, values.astype(float).tolist())),
            }

        return predictions


# ---------------------------------------------------------------------------
# 4. Export test predictions without overwriting the source or teammates' work
# ---------------------------------------------------------------------------
def export_submission(repo_root, models_dir, destination, template_path=None):
    """Write a NEW official-layout CSV, changing only gender and usage."""
    root = find_root(repo_root)
    dataset = root / "A2_FashionDataset/FashionDataset"
    official_path = dataset / "test/styles_prediction.csv"
    source = Path(template_path).resolve() if template_path else official_path
    destination = Path(destination).resolve()
    if destination == source.resolve() or destination.exists():
        raise FileExistsError(
            "Choose a new output CSV; source files and existing predictions "
            "are never overwritten."
        )

    template = pd.read_csv(source)
    official = pd.read_csv(official_path)
    if (
        list(template.columns) != list(official.columns)
        or not template["id"].equals(official["id"])
    ):
        raise ValueError(
            "Submission template must preserve the official columns and ID order."
        )

    predictor = Task3Predictor(models_dir)
    paths = [
        dataset / "test/images_test" / f"{identity}.jpg"
        for identity in template["id"]
    ]
    scores = predictor.predict_paths(paths)

    result = template.copy()
    for target in TARGETS:
        classes = np.asarray(predictor.specifications[target]["classes"])
        predicted_indices = scores[target].argmax(axis=1)
        result[target] = classes[predicted_indices]

    untouched_columns = [column for column in template if column not in TARGETS]
    pd.testing.assert_frame_equal(
        template[untouched_columns], result[untouched_columns]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(
        f"Saved {len(result)} rows to {destination}. "
        "Only gender and usage are filled by Task 3."
    )
    return result


# ---------------------------------------------------------------------------
# 5. Optional notebook upload demo (NOT needed for command-line execution)
# ---------------------------------------------------------------------------
def notebook_demo(models_dir):
    """Keep existing notebooks compatible; import widget packages only if used."""
    import ipywidgets as widgets
    from IPython.display import display
    from PIL import Image

    predictor = Task3Predictor(models_dir)
    upload = widgets.FileUpload(accept="image/*", multiple=False)
    output = widgets.Output()

    def uploaded(change):
        """Display the uploaded image and both predictions when a file changes."""
        with output:
            output.clear_output()
            if not upload.value:
                return
            try:
                # ipywidgets 7 uses a dict; ipywidgets 8 uses a tuple of files.
                if isinstance(upload.value, dict):
                    item = next(iter(upload.value.values()))
                else:
                    item = upload.value[0]
                content = bytes(item["content"])
                if len(content) > 10 * 1024 * 1024:
                    raise ValueError("Please choose an image smaller than 10 MB.")

                with Image.open(io.BytesIO(content)) as picture:
                    picture.thumbnail((240, 320))
                    display(picture.copy())

                prediction = predictor.predict_image(io.BytesIO(content))
                rows = {}
                for target, result in prediction.items():
                    rows[target] = {
                        "Catalogue label": result["label"],
                        "Model score (not calibrated)": round(result["score"], 3),
                    }
                display(pd.DataFrame(rows).T)
            except (OSError, ValueError, KeyError) as error:
                print(f"Could not read image: {error}")

    upload.observe(uploaded, names="value")
    display(widgets.VBox([upload, output]))
    return upload


# ---------------------------------------------------------------------------
# 6. Choose a single-image prediction OR a full test-set CSV export
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir", type=Path, required=True,
        help="Run folder containing gender_final/usage_final .keras and .json files.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--image", type=Path, help="Predict one image.")
    action.add_argument("--output", type=Path, help="Export test labels to a NEW CSV.")
    parser.add_argument("--repo-root", type=Path, help="Extracted repository root.")
    parser.add_argument(
        "--template", type=Path,
        help="Optional official-layout CSV already containing other tasks' results.",
    )
    args = parser.parse_args()
    if args.template and not args.output:
        parser.error("--template is used with --output, not with --image.")
    return args


def main():
    args = parse_args()
    if args.image:
        predictor = Task3Predictor(args.models_dir)
        prediction = predictor.predict_image(args.image)
        print(json.dumps(prediction, indent=2))
    else:
        export_submission(
            args.repo_root, args.models_dir, args.output, args.template
        )


if __name__ == "__main__":
    main()

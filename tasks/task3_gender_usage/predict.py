"""Reload Task 3 models for images, notebook upload or a submission copy."""

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


class Task3Predictor:
    def __init__(self, models_dir):
        self.models, self.specifications = {}, {}
        for target in TARGETS:
            directory = Path(models_dir)
            specification = json.loads((directory / f"{target}_final.json").read_text(encoding="utf-8"))
            model = tf.keras.models.load_model(str(directory / f"{target}_final.keras"), compile=False)
            shape = (specification["config"]["height"], specification["config"]["width"], 3)
            if tuple(model.input_shape[1:]) != shape or model.output_shape[-1] != len(specification["classes"]):
                raise ValueError(f"Model and metadata mismatch for {target}.")
            self.models[target], self.specifications[target] = model, specification

    def predict_paths(self, paths, batch_size=128):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        paths = list(paths)
        predictions = {}
        for target in TARGETS:
            specification = self.specifications[target]
            size = (specification["config"]["height"], specification["config"]["width"])
            batches = []
            for start in range(0, len(paths), batch_size):
                pixels = np.stack([prepare_image(path, size) for path in paths[start:start + batch_size]])
                batches.append(self.models[target](pixels, training=False).numpy())
            scores = np.concatenate(batches) if batches else np.empty((0, len(specification["classes"])))
            if not np.isfinite(scores).all() or (len(scores) and not np.allclose(scores.sum(axis=1), 1, atol=1e-5)):
                raise ValueError("Invalid model probabilities.")
            predictions[target] = scores
        return predictions

    def predict_image(self, source):
        content = source.read() if hasattr(source, "read") else Path(source).read_bytes()
        scores = {}
        for target in TARGETS:
            specification = self.specifications[target]
            size = (specification["config"]["height"], specification["config"]["width"])
            pixels = prepare_image(io.BytesIO(content), size)[None]
            scores[target] = self.models[target](pixels, training=False).numpy()
            if not np.isfinite(scores[target]).all() or not np.allclose(scores[target].sum(axis=1), 1, atol=1e-5):
                raise ValueError("Invalid model probabilities.")
        return {target: {"label": self.specifications[target]["classes"][int(values[0].argmax())],
                         "score": float(values[0].max()),
                         "probabilities": dict(zip(self.specifications[target]["classes"], values[0].astype(float).tolist()))}
                for target, values in scores.items()}


def export_submission(repo_root, models_dir, destination, template_path=None):
    root = find_root(repo_root)
    dataset = root / "A2_FashionDataset/FashionDataset"
    source = Path(template_path).resolve() if template_path else dataset / "test/styles_prediction.csv"
    destination = Path(destination).resolve()
    if destination == source.resolve() or destination.exists():
        raise FileExistsError("Choose a new output CSV; source files and existing predictions are never overwritten.")
    template = pd.read_csv(source)
    official = pd.read_csv(dataset / "test/styles_prediction.csv")
    if list(template.columns) != list(official.columns) or not template.id.equals(official.id):
        raise ValueError("Submission template must preserve the official columns and ID order.")
    predictor = Task3Predictor(models_dir)
    paths = [dataset / "test/images_test" / f"{identity}.jpg" for identity in template.id]
    scores = predictor.predict_paths(paths)
    result = template.copy()
    for target in TARGETS:
        result[target] = np.asarray(predictor.specifications[target]["classes"])[scores[target].argmax(axis=1)]
    untouched = [column for column in template if column not in TARGETS]
    pd.testing.assert_frame_equal(template[untouched], result[untouched])
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Saved {len(result)} rows to {destination}. Only gender and usage are filled by Task 3.")
    return result


def notebook_demo(models_dir):
    import ipywidgets as widgets
    from IPython.display import display
    predictor = Task3Predictor(models_dir)
    upload = widgets.FileUpload(accept="image/*", multiple=False)
    output = widgets.Output()

    def uploaded(change):
        with output:
            output.clear_output()
            if not upload.value:
                return
            try:
                item = next(iter(upload.value.values())) if isinstance(upload.value, dict) else upload.value[0]
                content = bytes(item["content"])
                if len(content) > 10 * 1024 * 1024:
                    raise ValueError("Please choose an image smaller than 10 MB.")
                from PIL import Image
                with Image.open(io.BytesIO(content)) as picture:
                    picture.thumbnail((240, 320))
                    display(picture.copy())
                prediction = predictor.predict_image(io.BytesIO(content))
                display(pd.DataFrame({target: {"Catalogue label": result["label"],
                                               "Model score (not calibrated)": round(result["score"], 3)}
                                      for target, result in prediction.items()}).T)
            except (OSError, ValueError, KeyError) as error:
                print(f"Could not read image: {error}")

    upload.observe(uploaded, names="value")
    display(widgets.VBox([upload, output]))
    return upload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--template", type=Path)
    args = parser.parse_args()
    if args.image:
        print(json.dumps(Task3Predictor(args.models_dir).predict_image(args.image), indent=2))
    elif args.output:
        export_submission(args.repo_root, args.models_dir, args.output, args.template)
    else:
        parser.error("Provide --image for a single photo or --output for the test-set CSV copy.")


if __name__ == "__main__":
    main()

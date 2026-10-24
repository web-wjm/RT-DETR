"""Create a reproducible, group-aware stratified split for GC10-DET."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parent / "GC10-DET"
SPLITS = ("train", "val", "test")
TARGET_SIZES = (1833, 229, 230)
RATIOS = np.asarray(TARGET_SIZES, dtype=float) / sum(TARGET_SIZES)
NUM_CLASSES = 10
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SEED = 42


@dataclass(frozen=True)
class Sample:
    stem: str
    image: Path
    label: Path
    class_counts: tuple[int, ...]

    @property
    def group(self) -> str:
        # The final token is the frame index. The prefix identifies one acquisition sequence.
        return self.stem.rsplit("_", 1)[0]


def collect_samples() -> list[Sample]:
    images: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for split in SPLITS:
        for path in (ROOT / "images" / split).glob("*"):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                if path.stem in images:
                    raise RuntimeError(f"Duplicate image stem: {path.stem}")
                images[path.stem] = path
        for path in (ROOT / "labels" / split).glob("*.txt"):
            if path.stem in labels:
                raise RuntimeError(f"Duplicate label stem: {path.stem}")
            labels[path.stem] = path

    if images.keys() != labels.keys():
        missing_labels = sorted(images.keys() - labels.keys())
        missing_images = sorted(labels.keys() - images.keys())
        raise RuntimeError(
            f"Unpaired samples: missing_labels={missing_labels[:5]}, missing_images={missing_images[:5]}"
        )
    if len(images) != sum(TARGET_SIZES):
        raise RuntimeError(f"Expected {sum(TARGET_SIZES)} samples, found {len(images)}")

    samples = []
    for stem in sorted(images):
        counts = Counter()
        for line_no, line in enumerate(labels[stem].read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise RuntimeError(f"Invalid label at {labels[stem]}:{line_no}")
            class_id = int(parts[0])
            if not 0 <= class_id < NUM_CLASSES:
                raise RuntimeError(f"Invalid class {class_id} at {labels[stem]}:{line_no}")
            counts[class_id] += 1
        samples.append(
            Sample(
                stem=stem,
                image=images[stem],
                label=labels[stem],
                class_counts=tuple(counts[i] for i in range(NUM_CLASSES)),
            )
        )
    return samples


def group_statistics(samples: list[Sample]):
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.group].append(sample)

    group_names = sorted(grouped)
    image_counts = np.zeros((len(group_names), NUM_CLASSES), dtype=float)
    instance_counts = np.zeros((len(group_names), NUM_CLASSES), dtype=float)
    sizes = np.zeros(len(group_names), dtype=float)
    for index, name in enumerate(group_names):
        group_samples = grouped[name]
        sizes[index] = len(group_samples)
        for sample in group_samples:
            values = np.asarray(sample.class_counts)
            image_counts[index] += values > 0
            instance_counts[index] += values
    return grouped, group_names, sizes, image_counts, instance_counts


def solve_split(samples: list[Sample]) -> dict[str, str]:
    grouped, group_names, sizes, image_counts, instance_counts = group_statistics(samples)
    group_count = len(group_names)
    split_count = len(SPLITS)
    metrics = np.concatenate((image_counts, instance_counts), axis=1)
    metric_count = metrics.shape[1]
    metric_totals = metrics.sum(axis=0)
    metric_targets = RATIOS[:, None] * metric_totals[None, :]

    x_count = group_count * split_count
    deviation_count = split_count * metric_count
    variable_count = x_count + 2 * deviation_count
    objective = np.zeros(variable_count, dtype=float)

    # Image-presence balance is primary; object-instance balance is secondary.
    metric_weights = np.concatenate((np.full(NUM_CLASSES, 4.0), np.full(NUM_CLASSES, 2.0)))
    for split_index in range(split_count):
        for metric_index in range(metric_count):
            flat = split_index * metric_count + metric_index
            weight = metric_weights[metric_index] / max(metric_targets[split_index, metric_index], 1.0)
            objective[x_count + flat] = weight
            objective[x_count + deviation_count + flat] = weight

    # Resolve equivalent optima deterministically without affecting the balance objective.
    rng = np.random.default_rng(SEED)
    objective[:x_count] = rng.random(x_count) * 1e-8

    rows = []
    cols = []
    values = []
    lower = []
    upper = []

    def add_constraint(coefficients, lb, ub):
        row = len(lower)
        for column, value in coefficients:
            rows.append(row)
            cols.append(column)
            values.append(value)
        lower.append(lb)
        upper.append(ub)

    # Every acquisition group belongs to exactly one split.
    for group_index in range(group_count):
        add_constraint(
            [(group_index * split_count + split_index, 1.0) for split_index in range(split_count)],
            1.0,
            1.0,
        )

    # Keep the requested 80/10/10 image counts exact.
    for split_index, target_size in enumerate(TARGET_SIZES):
        add_constraint(
            [
                (group_index * split_count + split_index, sizes[group_index])
                for group_index in range(group_count)
            ],
            target_size,
            target_size,
        )

    # Minimize absolute deviations from class-image and class-instance targets.
    for split_index in range(split_count):
        for metric_index in range(metric_count):
            flat = split_index * metric_count + metric_index
            coefficients = [
                (group_index * split_count + split_index, metrics[group_index, metric_index])
                for group_index in range(group_count)
                if metrics[group_index, metric_index]
            ]
            coefficients.extend(
                (
                    (x_count + flat, -1.0),
                    (x_count + deviation_count + flat, 1.0),
                )
            )
            add_constraint(coefficients, metric_targets[split_index, metric_index], metric_targets[split_index, metric_index])

    # Ensure minority classes have meaningful support in validation and test sets.
    total_class_images = image_counts.sum(axis=0)
    total_instances = instance_counts.sum(axis=0)
    for split_index in (1, 2):
        for class_id in range(NUM_CLASSES):
            min_images = max(2, int(np.floor(total_class_images[class_id] * 0.07)))
            min_instances = max(3, int(np.floor(total_instances[class_id] * 0.08)))
            add_constraint(
                [
                    (group_index * split_count + split_index, image_counts[group_index, class_id])
                    for group_index in range(group_count)
                    if image_counts[group_index, class_id]
                ],
                min_images,
                np.inf,
            )
            add_constraint(
                [
                    (group_index * split_count + split_index, instance_counts[group_index, class_id])
                    for group_index in range(group_count)
                    if instance_counts[group_index, class_id]
                ],
                min_instances,
                np.inf,
            )

    matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), variable_count)).tocsr()
    bounds = Bounds(
        np.zeros(variable_count),
        np.concatenate((np.ones(x_count), np.full(2 * deviation_count, np.inf))),
    )
    integrality = np.concatenate((np.ones(x_count), np.zeros(2 * deviation_count)))
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if result.x is None:
        raise RuntimeError(f"Unable to find a feasible split: {result.message}")

    assignment = {}
    decisions = result.x[:x_count].reshape(group_count, split_count)
    for group_index, group_name in enumerate(group_names):
        split = SPLITS[int(np.argmax(decisions[group_index]))]
        for sample in grouped[group_name]:
            assignment[sample.stem] = split
    return assignment


def summarize(samples: list[Sample], assignment: dict[str, str]) -> dict:
    summary = {}
    groups_per_split = defaultdict(set)
    for split in SPLITS:
        selected = [sample for sample in samples if assignment[sample.stem] == split]
        image_counts = Counter()
        instance_counts = Counter()
        for sample in selected:
            groups_per_split[split].add(sample.group)
            for class_id, count in enumerate(sample.class_counts):
                if count:
                    image_counts[class_id] += 1
                    instance_counts[class_id] += count
        summary[split] = {
            "images": len(selected),
            "groups": len(groups_per_split[split]),
            "class_images": {str(i): image_counts[i] for i in range(NUM_CLASSES)},
            "class_instances": {str(i): instance_counts[i] for i in range(NUM_CLASSES)},
        }

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = groups_per_split[left] & groups_per_split[right]
            if overlap:
                raise RuntimeError(f"Group leakage between {left} and {right}: {sorted(overlap)[:5]}")
    return summary


def write_manifests(samples: list[Sample], assignment: dict[str, str], summary: dict) -> None:
    split_dir = ROOT / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        names = sorted(sample.image.name for sample in samples if assignment[sample.stem] == split)
        (split_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    with (split_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "group", "split"))
        for sample in sorted(samples, key=lambda item: item.image.name):
            writer.writerow((sample.image.name, sample.group, assignment[sample.stem]))

    metadata = {
        "method": "group-aware multilabel stratification via mixed-integer optimization",
        "seed": SEED,
        "target_sizes": dict(zip(SPLITS, TARGET_SIZES)),
        "group_rule": "filename without the final underscore-separated frame index",
        "summary": summary,
    }
    (split_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def load_manifest(samples: list[Sample]) -> dict[str, str]:
    manifest_path = ROOT / "splits" / "manifest.csv"
    assignment = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = Path(row["image"]).stem
            split = row["split"]
            if split not in SPLITS:
                raise RuntimeError(f"Invalid split in {manifest_path}: {split}")
            if stem in assignment:
                raise RuntimeError(f"Duplicate sample in {manifest_path}: {stem}")
            assignment[stem] = split

    sample_stems = {sample.stem for sample in samples}
    if assignment.keys() != sample_stems:
        missing = sorted(sample_stems - assignment.keys())
        extra = sorted(assignment.keys() - sample_stems)
        raise RuntimeError(f"Manifest mismatch: missing={missing[:5]}, extra={extra[:5]}")
    return assignment


def apply_assignment(samples: list[Sample], assignment: dict[str, str]) -> None:
    staging = ROOT / ".resplit_staging"
    if staging.exists():
        raise RuntimeError(f"Staging directory already exists: {staging}")
    image_stage = staging / "images"
    label_stage = staging / "labels"
    image_stage.mkdir(parents=True)
    label_stage.mkdir(parents=True)

    try:
        for sample in samples:
            shutil.move(str(sample.image), image_stage / sample.image.name)
            shutil.move(str(sample.label), label_stage / sample.label.name)
        for split in SPLITS:
            (ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
            (ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for sample in samples:
            split = assignment[sample.stem]
            shutil.move(str(image_stage / sample.image.name), ROOT / "images" / split / sample.image.name)
            shutil.move(str(label_stage / sample.label.name), ROOT / "labels" / split / sample.label.name)
    finally:
        if image_stage.exists() and not any(image_stage.iterdir()):
            image_stage.rmdir()
        if label_stage.exists() and not any(label_stage.iterdir()):
            label_stage.rmdir()
        if staging.exists() and not any(staging.iterdir()):
            staging.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Move files after solving and validating the split")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Ignore the frozen manifest and solve a new split",
    )
    args = parser.parse_args()

    samples = collect_samples()
    manifest_path = ROOT / "splits" / "manifest.csv"
    if manifest_path.exists() and not args.regenerate:
        assignment = load_manifest(samples)
        source = f"frozen manifest {manifest_path}"
    else:
        assignment = solve_split(samples)
        source = "new optimization result"
    summary = summarize(samples, assignment)
    print(f"Assignment source: {source}")
    print(json.dumps(summary, indent=2))
    if args.apply:
        apply_assignment(samples, assignment)
        write_manifests(samples, assignment, summary)
        print(f"Applied split and wrote manifests to {ROOT / 'splits'}")
    else:
        print("Dry run only. Re-run with --apply after reviewing this summary.")


if __name__ == "__main__":
    main()

"""Convert the VLM-3R VSI-Bench training annotations to ViPS JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Journey9ni/VLM-3R-DATA at aae3535e513d36be5fd0a1edbaab523c8a3c45e4.
SOURCE_FILES = {
    "merged_qa_route_plan_train.json": 4104,
    "merged_qa_scannet_train.json": 51779,
    "merged_qa_scannetpp_train.json": 151775,
}


def convert_record(record: dict) -> dict:
    turns = []
    for turn in record["conversations"]:
        value = turn["value"]
        if turn["from"] == "human":
            value = value.replace("<image>", "<video>")
            if "<video>" not in value:
                value = "<video>\n" + value
        turns.append({**turn, "value": value})
    return {**record, "conversations": turns}


def prepare_annotations(input_dir: Path, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    videos = set()
    total = 0
    for name, expected_count in SOURCE_FILES.items():
        with (input_dir / name).open(encoding="utf-8") as source:
            records = json.load(source)
        if not isinstance(records, list) or len(records) != expected_count:
            raise ValueError(f"{name}: expected {expected_count} records")
        target = output_dir / f"{Path(name).stem}.jsonl"
        with target.open("x", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(convert_record(record), ensure_ascii=False) + "\n")
                videos.add(record["video"])
        total += len(records)
        print(f"{target.name}: {len(records):,} records", flush=True)
        del records

    with (output_dir / "required_videos.txt").open("x", encoding="utf-8") as output:
        output.write("\n".join(sorted(videos)) + "\n")
    print(f"Prepared {total:,} records and {len(videos):,} distinct video paths in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Downloaded vsibench_train directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Empty destination for JSONL files")
    args = parser.parse_args()
    prepare_annotations(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

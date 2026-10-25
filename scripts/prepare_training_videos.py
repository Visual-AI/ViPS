"""Sample 32 frames at 640x480 directly from the raw RGB recordings for ViPS."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import numpy as np
from PIL import Image


NUM_FRAMES = 32
IMAGE_SIZE = (640, 480)
SUBSETS = {"arkitscenes", "scannet", "scannetpp"}


def read_video_paths(video_list: Path) -> list[str]:
    paths = video_list.read_text(encoding="utf-8").splitlines()
    paths = sorted({path.strip() for path in paths if path.strip()})
    if not paths:
        raise ValueError("No video paths selected")
    for value in paths:
        path = Path(value)
        if (
            path.is_absolute() or ".." in path.parts or len(path.parts) != 3
            or path.parts[0] not in SUBSETS or path.parts[1] != "videos"
            or path.suffix != ".mp4" or not path.stem or path.as_posix() != value
        ):
            raise ValueError(f"Expected <subset>/videos/<scene>.mp4, got {value!r}")
    return paths


def find_video(root: Path, relative: str) -> Path:
    subset, _, filename = Path(relative).parts
    scene = Path(filename).stem
    candidates = [(root / relative).with_suffix(suffix) for suffix in (".mp4", ".mov", ".mkv")]
    if subset == "scannet":
        candidates.append(root / subset / "scans" / scene / f"{scene}.sens")
    elif subset == "scannetpp":
        candidates.extend(
            root / subset / "data" / scene / "iphone" / f"rgb{suffix}"
            for suffix in (".mkv", ".mp4")
        )
    else:
        candidates.extend(
            root / subset / "raw" / split / scene / f"{scene}.mov"
            for split in ("Training", "Validation")
        )
    matches = {candidate.resolve() for candidate in candidates if candidate.is_file()}
    if not matches:
        raise FileNotFoundError(f"Missing {relative}; checked: {', '.join(map(str, candidates))}")
    if len(matches) != 1:
        raise ValueError(f"Multiple source files found for {relative}; use a root with one layout")
    return matches.pop()


class ScanNetReader:
    """Read RGB frames from official ScanNet v4 .sens files without decoding depth.

    Binary format: https://github.com/ScanNet/ScanNet/tree/master/SensReader
    The 30 FPS convention comes from the historical ViPS training metadata;
    .sens stores sensor timestamps, not an encoded video frame rate.
    """

    def __init__(self, source: Path):
        self.source = source
        self.offsets = []
        file_size = source.stat().st_size
        with source.open("rb") as stream:
            def unpack(fmt):
                size = struct.calcsize(fmt)
                data = stream.read(size)
                if len(data) != size:
                    raise ValueError(f"Truncated ScanNet sensor file: {source}")
                return struct.unpack(fmt, data)

            if unpack("<I")[0] != 4:
                raise ValueError(f"Expected ScanNet sensor format version 4: {source}")
            name_length = unpack("<Q")[0]
            # Sensor name and four 4x4 calibration matrices.
            stream.seek(name_length + 4 * 16 * 4, 1)
            color_compression, _ = unpack("<ii")
            if color_compression != 2:
                raise ValueError(f"Expected JPEG-compressed ScanNet RGB frames: {source}")
            unpack("<IIIIf")  # Color/depth dimensions and depth scale.
            count = unpack("<Q")[0]
            if count < 1 or count > file_size // 96:
                raise ValueError(f"Invalid ScanNet frame count {count}: {source}")
            for _ in range(count):
                stream.seek(16 * 4 + 2 * 8, 1)  # Pose and color/depth timestamps.
                color_size, depth_size = unpack("<QQ")
                offset = stream.tell()
                end = offset + color_size + depth_size
                if not color_size or end > file_size:
                    raise ValueError(f"Truncated or empty ScanNet RGB frame: {source}")
                self.offsets.append((offset, color_size))
                stream.seek(end)

    def __len__(self):
        return len(self.offsets)

    def get_avg_fps(self):
        return 30.0

    def image(self, index: int) -> Image.Image:
        offset, size = self.offsets[index]
        with self.source.open("rb") as stream:
            stream.seek(offset)
            encoded = stream.read(size)
        with Image.open(io.BytesIO(encoded)) as image:
            return image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)


class FFmpegReader:
    """Decode video display frames, honoring edit lists and discarded packets.

    A container's nb_frames can include non-display frames. Count decoded frames
    first, then select from the same FFmpeg display sequence in a second pass.
    """

    def __init__(self, source: Path):
        self.source = source
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise FileNotFoundError("Video inputs require ffmpeg and ffprobe on PATH; see README.md")
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=avg_frame_rate,nb_read_frames",
             "-of", "json", str(source)],
            check=True, capture_output=True, text=True,
        )
        if result.stderr.strip():
            raise ValueError(f"FFprobe reported decoding errors for {source}: {result.stderr[:2000]}")
        stream = json.loads(result.stdout)["streams"][0]
        self.count = int(stream["nb_read_frames"])
        self.fps = float(Fraction(stream["avg_frame_rate"]))

    def __len__(self):
        return self.count

    def get_avg_fps(self):
        return self.fps

    def images(self, indices: list[int]):
        selected = sorted(set(indices))
        expression = "+".join(f"eq(n,{index})" for index in selected)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-threads", "1",
            "-i", str(self.source), "-map", "0:v:0", "-vf",
            f"select='{expression}',scale={IMAGE_SIZE[0]}:{IMAGE_SIZE[1]}:flags=bilinear",
            "-filter_threads", "1", "-fps_mode", "passthrough", "-pix_fmt", "rgb24",
            "-f", "rawvideo", "pipe:1",
        ]
        # A file for stderr prevents a full pipe from blocking the decoder.
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
            try:
                frame_bytes = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3
                for index in selected:
                    data = process.stdout.read(frame_bytes)
                    if len(data) != frame_bytes:
                        raise ValueError(f"Incomplete decoded frame {index} in {self.source}")
                    image = Image.frombytes("RGB", IMAGE_SIZE, data)
                    # Short videos repeat sampled indices, just like integer linspace.
                    for _ in range(indices.count(index)):
                        yield image
                extra = process.stdout.read(1)
                returncode = process.wait()
                errors.seek(0)
                detail = errors.read(2000).decode(errors="replace")
                if returncode or extra or detail.strip():
                    raise ValueError(f"FFmpeg decode failed for {self.source} (exit {returncode}): {detail}")
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.kill()
                process.wait()


def extract_video(source: Path, relative: str, output_root: Path) -> dict:
    # One decoder thread per worker bounds CPU use; no Torch/GPU is needed.
    if source.suffix.lower() == ".sens":
        reader = ScanNetReader(source)
    else:
        reader = FFmpegReader(source)
    total = len(reader)
    fps = round(float(reader.get_avg_fps()), 3)
    if total <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid frame count/FPS for {relative}: {total}, {fps}")
    indices = np.linspace(0, total - 1, NUM_FRAMES, dtype=int).tolist()
    target = output_root / Path(relative).with_suffix("")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing frames: {target}")

    # Publish a scene only after all 32 images have been written successfully.
    if isinstance(reader, FFmpegReader):
        pictures = reader.images(indices)
    else:
        pictures = (reader.image(index) for index in indices)
    with closing(pictures), tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        frame_dir = Path(temporary)
        for position, frame in enumerate(pictures):
            frame.save(
                frame_dir / f"{position:03d}.jpg", format="JPEG", quality=95, subsampling=2,
            )
        frame_dir.rename(target)
    return {
        "video_path": relative,
        "total_num_frames": total,
        "fps": fps,
        "duration": total / fps,
        "indices": indices,
    }


def prepare_videos(video_root: Path, output_dir: Path, videos: list[str], workers: int) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    video_root = video_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output_dir}")
    # Preflight all paths, so missing downloads fail before any frames are written.
    sources = {relative: find_video(video_root, relative) for relative in videos}
    if not sources:
        raise ValueError("No video paths selected")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = {}
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {
        executor.submit(extract_video, source, relative, output_dir): relative
        for relative, source in sources.items()
    }
    try:
        for future in as_completed(futures):
            relative = futures[future]
            try:
                entries[relative] = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"Failed to prepare {relative}: {error}. Output is incomplete; "
                    "fix the input and retry in an empty directory."
                ) from error
            print(f"[{len(entries)}/{len(sources)}] {relative}", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    metadata = {
        "root": str(video_root), "num_videos": len(entries),
        "videos": [entries[relative] for relative in sorted(entries)],
    }
    # No completion metadata is written if any selected video failed.
    pending = output_dir / ".video_metadata.json.tmp"
    pending.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    pending.rename(output_dir / "video_metadata.json")
    print(f"Prepared {len(entries)} videos ({NUM_FRAMES} frames each) in {output_dir}", flush=True)
    return metadata


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing the downloaded RGB recordings")
    parser.add_argument("--output-dir", type=Path, default=Path("data/train/videos"), help="Empty destination for frames and metadata (default: data/train/videos)")
    parser.add_argument("--video-list", type=Path, default=Path("data/train/annotations/required_videos.txt"), help="Video list from the QA preparation step (default: data/train/annotations/required_videos.txt)")
    parser.add_argument("--workers", type=positive_int, default=4, help="Parallel video decoders (default: 4)")
    args = parser.parse_args()
    videos = read_video_paths(args.video_list)
    prepare_videos(args.raw_root, args.output_dir, videos, args.workers)


if __name__ == "__main__":
    main()

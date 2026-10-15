"""Check a rendered film against evidence.json and save review frames plus a contact sheet.

Run after rendering, with the animation dependency group installed:

    uv run --frozen --group animation python scripts/verify_render.py            # final film
    uv run --frozen --group animation python scripts/verify_render.py --draft    # -ql --fps 15 draft
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import av
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/unified"
# Extra frames taken mid-animation, relative to the named cue.
IN_MOTION = [("frozen_targets", "allocation", 4.75),
             ("evaluation_recovery_count", "evaluation_downturns", 4.75),
             ("growth_in_motion", "baseline", -4.2),
             ("evaluation_in_motion", "evaluation_growth", -4.2),
             ("2020_in_motion", "2020_trough", -2.2),
             ("uncertainty_in_motion", "uncertainty_ranges", -3.2)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draft", action="store_true", help="Check the low-quality draft render")
    args = parser.parse_args()
    prefix = "draft" if args.draft else "final"
    media = ROOT / ("outputs/unified-draft" if args.draft else "outputs/unified")
    quality = "480p15" if args.draft else "1080p30"
    source = media / f"videos/project_results/{quality}/ProjectResults.mp4"
    evidence = json.loads((OUTPUT / "evidence.json").read_text(encoding="utf-8"))
    times = {cue["name"]: cue["seconds"] for cue in evidence["review_cues"]}
    cues = [(name, seconds + .3) for name, seconds in times.items()]
    cues += [(name, times[anchor] + offset) for name, anchor, offset in IN_MOTION]

    with av.open(str(source)) as video:
        stream = video.streams.video[0]
        width, height, fps = stream.width, stream.height, float(stream.base_rate)
        average_fps = float(stream.average_rate)
        assert abs(average_fps - fps) < .05
        expected = (854, 480, 15.0) if args.draft else (1920, 1080, 30.0)
        assert (width, height, fps) == expected, (width, height, fps)
        previous, frames = -1, 0
        for frame in video.decode(stream):
            stamp = float(frame.time)
            assert stamp > previous
            previous = stamp
            frames += 1
    duration = previous + 1 / fps
    assert abs(duration - evidence["duration_seconds"]) < .5

    pictures = []
    with av.open(str(source)) as video:
        stream = video.streams.video[0]
        for name, target in sorted(cues, key=lambda item: item[1]):
            video.seek(int(target / stream.time_base), stream=stream, backward=True)
            for frame in video.decode(stream):
                if float(frame.time) >= target:
                    picture = frame.to_image()
                    picture.save(OUTPUT / f"{prefix}-{name}.png")
                    pictures.append((name, picture.copy()))
                    break
    sheet = Image.new("RGB", (1800, 362 * ((len(pictures) + 2) // 3)), "#ddd7ce")
    for position, (name, picture) in enumerate(pictures):
        x, y = (position % 3) * 600, (position // 3) * 362
        picture.thumbnail((600, 337))
        sheet.paste(picture, (x, y + 25))
        ImageDraw.Draw(sheet).text((x + 8, y + 6), name, fill="black")
    sheet.save(OUTPUT / f"{prefix}-contact.png")
    record = {"width": width, "height": height, "fps": fps, "average_fps": average_fps,
              "decoded_frames": frames, "duration_seconds": duration,
              "bytes": source.stat().st_size,
              "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "review_frames": [name for name, _ in pictures]}
    if not args.draft:
        shutil.copyfile(OUTPUT / "final-evaluation_metrics.png", OUTPUT / "poster.png")
        shutil.copyfile(source, OUTPUT / "project-results.mp4")
    (OUTPUT / f"{prefix}-render-verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

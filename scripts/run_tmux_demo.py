from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - exercised manually
    raise SystemExit(
        "Pillow is required for GIF generation. Run this script with the system python3 "
        "that has PIL available."
    ) from exc


@dataclass
class PaneSnapshot:
    pane_id: str
    title: str
    content: str


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a four-pane tmux demo recording and render it to a GIF."
    )
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--frame-interval-seconds", type=float, default=0.45)
    parser.add_argument("--tail-seconds", type=float, default=1.8)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--output-path",
        default="docs/demo/four-pane-demo.gif",
        help="Destination GIF path",
    )
    parser.add_argument(
        "--state-dir",
        default="artifacts/tmux-demo",
        help="Directory used for transient pane coordination files",
    )
    parser.add_argument("--session-name", default=f"voice-demo-{int(time.time())}")
    parser.add_argument("--width", type=int, default=220)
    parser.add_argument("--height", type=int, default=58)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / args.output_path
    state_dir = repo_root / args.state_dir
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    tmux_tmp_dir = state_dir / "tmux-tmp"
    tmux_tmp_dir.mkdir(parents=True, exist_ok=True)
    tmux_env = os.environ.copy()
    tmux_env["TMUX_TMPDIR"] = str(tmux_tmp_dir)

    session_name = args.session_name
    base_url = f"http://127.0.0.1:{args.port}"
    commands = [
        (
            "API Server",
            f"cd {repo_root} && .venv/bin/python scripts/run_local_demo_server.py --port {args.port}",
        ),
        (
            "Create + Invoke",
            (
                f"cd {repo_root} && .venv/bin/python scripts/tmux_demo_control.py "
                f"--base-url {base_url} --state-dir {state_dir}"
            ),
        ),
        (
            "Pipeline Detail",
            (
                f"cd {repo_root} && .venv/bin/python scripts/tmux_watch_detail.py "
                f"--base-url {base_url} --state-dir {state_dir}"
            ),
        ),
        (
            "Pipeline List",
            (
                f"cd {repo_root} && .venv/bin/python scripts/tmux_watch_list.py "
                f"--base-url {base_url} --state-dir {state_dir}"
            ),
        ),
    ]

    try:
        _start_tmux_session(
            repo_root=repo_root,
            session_name=session_name,
            commands=commands,
            width=args.width,
            height=args.height,
            tmux_env=tmux_env,
        )
        frames = _capture_frames(
            session_name=session_name,
            state_dir=state_dir,
            frame_interval_seconds=args.frame_interval_seconds,
            tail_seconds=args.tail_seconds,
            timeout_seconds=args.timeout_seconds,
            tmux_env=tmux_env,
        )
    finally:
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            check=False,
            cwd=repo_root,
            env=tmux_env,
        )

    if not frames:
        raise RuntimeError("No frames were captured from the tmux demo session.")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(args.frame_interval_seconds * 1000),
        loop=0,
        optimize=False,
    )
    print(f"wrote_gif={output_path}")
    return 0


def _start_tmux_session(
    repo_root: Path,
    session_name: str,
    commands: list[tuple[str, str]],
    width: int,
    height: int,
    tmux_env: dict[str, str],
) -> None:
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session_name,
            "-x",
            str(width),
            "-y",
            str(height),
            "-c",
            str(repo_root),
            _tmux_shell_command(commands[0][1]),
        ],
        check=True,
        cwd=repo_root,
        env=tmux_env,
    )
    subprocess.run(
        ["tmux", "set-option", "-t", session_name, "remain-on-exit", "on"],
        check=True,
        cwd=repo_root,
        env=tmux_env,
    )
    subprocess.run(
        [
            "tmux",
            "split-window",
            "-h",
            "-t",
            f"{session_name}:0",
            "-c",
            str(repo_root),
            _tmux_shell_command(commands[1][1]),
        ],
        check=True,
        cwd=repo_root,
        env=tmux_env,
    )
    subprocess.run(
        [
            "tmux",
            "split-window",
            "-v",
            "-t",
            f"{session_name}:0.0",
            "-c",
            str(repo_root),
            _tmux_shell_command(commands[2][1]),
        ],
        check=True,
        cwd=repo_root,
        env=tmux_env,
    )
    subprocess.run(
        [
            "tmux",
            "split-window",
            "-v",
            "-t",
            f"{session_name}:0.1",
            "-c",
            str(repo_root),
            _tmux_shell_command(commands[3][1]),
        ],
        check=True,
        cwd=repo_root,
        env=tmux_env,
    )
    subprocess.run(
        ["tmux", "select-layout", "-t", f"{session_name}:0", "tiled"],
        check=True,
        cwd=repo_root,
        env=tmux_env,
    )
    for pane_index, (title, _) in enumerate(commands):
        subprocess.run(
            ["tmux", "select-pane", "-t", f"{session_name}:0.{pane_index}", "-T", title],
            check=True,
            cwd=repo_root,
            env=tmux_env,
        )


def _capture_frames(
    session_name: str,
    state_dir: Path,
    frame_interval_seconds: float,
    tail_seconds: float,
    timeout_seconds: float,
    tmux_env: dict[str, str],
) -> list[Image.Image]:
    frames: list[Image.Image] = []
    deadline = time.time() + timeout_seconds
    done_detected_at: float | None = None

    while time.time() < deadline:
        panes = _capture_panes(session_name, tmux_env)
        frames.append(_render_layout(panes))

        if (state_dir / "done.json").exists():
            done_detected_at = done_detected_at or time.time()
            if time.time() - done_detected_at >= tail_seconds:
                break

        time.sleep(frame_interval_seconds)

    return frames


def _capture_panes(session_name: str, tmux_env: dict[str, str]) -> list[PaneSnapshot]:
    list_panes_result = subprocess.run(
        [
            "tmux",
            "list-panes",
            "-t",
            f"{session_name}:0",
            "-F",
            "#{pane_id}\t#{pane_index}\t#{pane_title}",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=tmux_env,
    )
    pane_rows = [row for row in list_panes_result.stdout.splitlines() if row.strip()]
    pane_snapshots: list[PaneSnapshot] = []

    for row in sorted(pane_rows, key=lambda value: int(value.split("\t")[1])):
        pane_id, _, title = row.split("\t", 2)
        capture_result = subprocess.run(
            ["tmux", "capture-pane", "-p", "-t", pane_id],
            check=True,
            capture_output=True,
            text=True,
            env=tmux_env,
        )
        pane_snapshots.append(
            PaneSnapshot(
                pane_id=pane_id,
                title=title,
                content=capture_result.stdout.rstrip(),
            )
        )

    return pane_snapshots


def _render_layout(panes: list[PaneSnapshot]) -> Image.Image:
    canvas_width = 1280
    canvas_height = 860
    outer_padding = 18
    pane_gap = 12
    pane_width = (canvas_width - (outer_padding * 2) - pane_gap) // 2
    pane_height = (canvas_height - (outer_padding * 2) - pane_gap) // 2

    image = Image.new("RGB", (canvas_width, canvas_height), "#0b1020")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(18)
    body_font = _load_font(14)
    pane_background = "#111827"
    border_color = "#334155"
    title_color = "#f8fafc"
    text_color = "#dbe4f0"
    accent_color = "#38bdf8"

    for pane_index, pane in enumerate(panes[:4]):
        column = pane_index % 2
        row = pane_index // 2
        left = outer_padding + column * (pane_width + pane_gap)
        top = outer_padding + row * (pane_height + pane_gap)
        right = left + pane_width
        bottom = top + pane_height

        draw.rounded_rectangle(
            [left, top, right, bottom],
            radius=12,
            fill=pane_background,
            outline=border_color,
            width=2,
        )
        draw.rounded_rectangle(
            [left, top, right, top + 36],
            radius=12,
            fill="#0f172a",
            outline=border_color,
            width=2,
        )
        draw.text((left + 14, top + 10), pane.title, font=title_font, fill=title_color)
        draw.rectangle(
            [right - 90, top + 12, right - 14, top + 24],
            fill=accent_color,
        )

        y_cursor = top + 48
        line_height = body_font.size + 4
        max_lines = max(1, (pane_height - 62) // line_height)
        lines = pane.content.splitlines()[-max_lines:]
        for line in lines:
            draw.text((left + 14, y_cursor), line[:92], font=body_font, fill=text_color)
            y_cursor += line_height

    return image.convert("P", palette=Image.ADAPTIVE, colors=96)


def _tmux_shell_command(command: str) -> str:
    return f"/bin/zsh -lc {shlex.quote(command)}"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    raise SystemExit(main())

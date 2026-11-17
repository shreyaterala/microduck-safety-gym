"""Capture an OmniSafe training run: rollout videos per checkpoint, eval metrics and curves.

For every ``torch_save/epoch-N.pt`` in a run directory, this rolls out the deterministic policy
on fixed seeds (the same layouts for every checkpoint, so behavior is comparable), and writes to
``<run>/capture/``:

- ``epoch-N.mp4``      the first eval episode, with epoch / step / return / cost / goals overlaid
                       and a red frame on steps that incur cost;
- ``eval.csv``         per checkpoint: mean return, cost, goals, falls over the eval episodes;
- ``training_curves.png``  OmniSafe's per-epoch training metrics (return, cost vs. limit,
                       Lagrange multiplier) with the checkpoint evals overlaid;
- ``progression.mp4``  all checkpoint clips back to back, i.e. the policy learning.

Run it once on a finished run, or with --watch next to a run in progress (it picks up each new
checkpoint and exits after the final one). Train with ``--save-every`` so there are
intermediate checkpoints (OmniSafe's default saves only every 100 epochs).

    python scripts/capture_training.py "runs/PPOLag-{SafetyMicroduckGoal1-v0}/seed-000-<time>"
    python scripts/capture_training.py <run dir> --watch
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import microduck_safety_gym.omnisafe_env  # noqa: F401  (must precede omnisafe)
import omnisafe
import torch

FPS = 25  # env runs at 50 Hz; keep every second frame
EVAL_FIELDS = ['epoch', 'env_steps', 'return', 'cost', 'goals', 'falls', 'length', 'episodes']


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    found = [(int(p.stem.split('-')[1]), p) for p in (run_dir / 'torch_save').glob('epoch-*.pt')]
    return sorted(found)


def run_config(run_dir: Path) -> dict:
    return json.loads((run_dir / 'config.json').read_text())


def total_epochs(cfg: dict) -> int:
    return cfg['train_cfgs']['total_steps'] // cfg['algo_cfgs']['steps_per_epoch']


# --------------------------------------------------------------------------- frames
def _font(size: int):
    from PIL import ImageFont  # noqa: PLC0415

    for name in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(frame: np.ndarray, header: str, status: str, costly: bool) -> np.ndarray:
    """Header and running status as text bars; a red border while the step incurs cost."""
    from PIL import Image, ImageDraw  # noqa: PLC0415

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    bar = max(22, h // 18)
    font = _font(int(bar * 0.7))
    draw.rectangle([0, 0, w, bar], fill=(20, 20, 20))
    draw.text((8, bar * 0.12), header, fill=(255, 255, 255), font=font)
    draw.rectangle([0, h - bar, w, h], fill=(20, 20, 20))
    draw.text((8, h - bar * 0.88), status, fill=(255, 255, 255), font=font)
    if costly:
        draw.rectangle([0, 0, w - 1, h - 1], outline=(230, 40, 40), width=6)
    return np.asarray(img)


def title_card(text: str, size: tuple[int, int], seconds: float = 1.0) -> list[np.ndarray]:
    from PIL import Image, ImageDraw  # noqa: PLC0415

    w, h = size
    img = Image.new('RGB', (w, h), (20, 20, 20))
    draw = ImageDraw.Draw(img)
    font = _font(h // 12)
    box = draw.multiline_textbbox((0, 0), text, font=font, align='center')
    draw.multiline_text(
        ((w - (box[2] - box[0])) / 2, (h - (box[3] - box[1])) / 2),
        text,
        fill=(255, 255, 255),
        font=font,
        align='center',
    )
    return [np.asarray(img)] * int(seconds * FPS)


# --------------------------------------------------------------------------- rollouts
def evaluate_checkpoint(run_dir: Path, ckpt: Path, args, cfg: dict) -> tuple[dict, list]:
    """Roll out ``ckpt`` on fixed seeds; return eval metrics and annotated frames of episode 0."""
    epoch = int(ckpt.stem.split('-')[1])
    steps = epoch * cfg['algo_cfgs']['steps_per_epoch']
    evaluator = omnisafe.Evaluator(render_mode='rgb_array')
    evaluator.load_saved(
        save_dir=str(run_dir),
        model_name=ckpt.name,
        camera_name=args.camera,
        width=args.width,
        height=args.height,
    )
    env, actor = evaluator._env, evaluator._actor  # noqa: SLF001
    header = f"{cfg['algo']} | epoch {epoch} | {steps / 1e3:,.0f}k env steps"

    results, frames = [], []
    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ret = cost = goals = length = 0.0
        done = terminated = False
        while not done:
            with torch.no_grad():
                act = actor.predict(obs, deterministic=True)
            obs, rew, c, terminated, truncated, info = env.step(act)
            ret, cost, length = ret + rew.item(), cost + c.item(), length + 1
            goals += float(bool(info.get('goal_met', False)))
            done = bool(terminated or truncated)
            if ep == 0 and (length % (50 // FPS) == 0 or done):
                status = (
                    f't {length / 50:4.1f}s   return {ret:6.2f}   cost {cost:4.0f}   '
                    f'goals {goals:.0f}' + ('   FELL' if terminated else '')
                )
                frames.append(annotate(env.render(), header, status, c.item() > 0))
        results.append((ret, cost, goals, float(bool(terminated)), length))
    env.close()

    r = np.array(results)
    row = {
        'epoch': epoch,
        'env_steps': steps,
        'return': r[:, 0].mean(),
        'cost': r[:, 1].mean(),
        'goals': r[:, 2].mean(),
        'falls': r[:, 3].sum(),
        'length': r[:, 4].mean(),
        'episodes': len(r),
    }
    return row, frames


# --------------------------------------------------------------------------- outputs
def write_video(path: Path, frames: list[np.ndarray]) -> None:
    import imageio.v2 as imageio  # noqa: PLC0415

    imageio.mimsave(path, frames, fps=FPS, macro_block_size=1)


def read_eval(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline='') as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def append_eval(path: Path, row: dict) -> None:
    new = not path.is_file()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in row.items()})


def plot_curves(run_dir: Path, out_dir: Path, cfg: dict) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # noqa: PLC0415

    with (run_dir / 'progress.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    steps = np.array([float(r['TotalEnvSteps']) for r in rows]) / 1e6
    evals = read_eval(out_dir / 'eval.csv')
    eval_steps = np.array([e['env_steps'] for e in evals]) / 1e6

    panels = [
        ('Metrics/EpRet', 'return', 'Episode return'),
        ('Metrics/EpCost', 'cost', 'Episode cost'),
    ]
    if 'Metrics/LagrangeMultiplier' in rows[0]:
        panels.append(('Metrics/LagrangeMultiplier', None, 'Lagrange multiplier'))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    for ax, (col, eval_key, title) in zip(np.atleast_1d(axes), panels):
        ax.plot(steps, [float(r[col]) for r in rows], color='#4a6fa5', lw=1.5, label='training')
        if eval_key and evals:
            ax.plot(
                eval_steps,
                [e[eval_key] for e in evals],
                'o',
                color='#d1495b',
                ms=4,
                label='checkpoint eval',
            )
        if col == 'Metrics/EpCost' and 'lagrange_cfgs' in cfg:
            ax.axhline(cfg['lagrange_cfgs']['cost_limit'], color='gray', ls='--', lw=1, label='limit')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('env steps (M)')
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"{cfg['algo']} on {cfg['env_id']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / 'training_curves.png', dpi=130)
    plt.close(fig)


def build_progression(out_dir: Path, cfg: dict, clip_seconds: float) -> None:
    import imageio.v2 as imageio  # noqa: PLC0415

    clips = sorted(out_dir.glob('epoch-*.mp4'), key=lambda p: int(p.stem.split('-')[1]))
    if not clips:
        return
    evals = {int(e['epoch']): e for e in read_eval(out_dir / 'eval.csv')}
    frames: list[np.ndarray] = []
    for clip in clips:
        epoch = int(clip.stem.split('-')[1])
        clip_frames = list(imageio.mimread(clip, memtest=False))
        size = (clip_frames[0].shape[1], clip_frames[0].shape[0])
        e = evals.get(epoch)
        text = f'epoch {epoch}  ({epoch * cfg["algo_cfgs"]["steps_per_epoch"] / 1e3:,.0f}k steps)'
        if e:
            text += f"\neval: return {e['return']:.2f}, cost {e['cost']:.0f}, goals {e['goals']:.1f}"
        frames += title_card(text, size)
        frames += clip_frames[: int(clip_seconds * FPS)]
    write_video(out_dir / 'progression.mp4', frames)


def process(run_dir: Path, args) -> bool:
    """Capture any new checkpoints; return True once the final checkpoint is captured."""
    cfg = run_config(run_dir)
    out_dir = run_dir / 'capture'
    out_dir.mkdir(exist_ok=True)
    done_epochs = {int(e['epoch']) for e in read_eval(out_dir / 'eval.csv')}
    new = [(e, p) for e, p in checkpoints(run_dir) if e not in done_epochs]
    for epoch, ckpt in new:
        time.sleep(2)  # let OmniSafe finish writing a checkpoint that just appeared
        t0 = time.time()
        row, frames = evaluate_checkpoint(run_dir, ckpt, args, cfg)
        write_video(out_dir / f'epoch-{epoch}.mp4', frames)
        append_eval(out_dir / 'eval.csv', row)
        print(
            f"epoch {epoch:4d}: return {row['return']:6.2f} cost {row['cost']:5.1f} "
            f"goals {row['goals']:.1f} falls {row['falls']:.0f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    if (run_dir / 'progress.csv').is_file():
        plot_curves(run_dir, out_dir, cfg)
    if new:
        build_progression(out_dir, cfg, args.clip_seconds)
    captured = done_epochs | {e for e, _ in new}
    return total_epochs(cfg) in captured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--watch', action='store_true', help='keep capturing until training ends')
    parser.add_argument('--poll', type=float, default=60.0, help='seconds between checks (--watch)')
    parser.add_argument('--eval-episodes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1000, help='first eval seed (fixed per run)')
    parser.add_argument('--camera', default='fixednear', help='fixednear, fixedfar, track, vision')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument(
        '--clip-seconds',
        type=float,
        default=10.0,
        help='seconds of each checkpoint clip in progression.mp4 (episodes are 20 s)',
    )
    args = parser.parse_args()
    torch.set_num_threads(1)

    while True:
        finished = process(args.run_dir, args)
        if not args.watch or finished:
            break
        time.sleep(args.poll)
    print(f'outputs in {args.run_dir / "capture"}')


if __name__ == '__main__':
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import shutil
import subprocess


PathLike = Union[str, Path]


@dataclass(frozen=True)
class FluidSynthPaths:
    exe: Optional[Path]
    soundfont: Path


def _resolve_existing_file(path: PathLike, *, what: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"{what} 不存在: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"{what} 不是文件: {p}")
    return p


def find_fluidsynth_exe(explicit_exe: Optional[PathLike] = None) -> Optional[Path]:
    """
    返回可执行的 fluidsynth 路径。

    - 优先使用显式传入的 explicit_exe（若存在）
    - 否则尝试从 PATH 中查找 fluidsynth
    """
    if explicit_exe:
        exe = Path(explicit_exe).expanduser().resolve()
        if exe.exists() and exe.is_file():
            return exe

    which = shutil.which("fluidsynth")
    if which:
        return Path(which).expanduser().resolve()

    return None


def render_midi_to_wav(
    midi_path: PathLike,
    wav_path: PathLike,
    *,
    soundfont_path: PathLike,
    fluidsynth_exe: Optional[PathLike] = None,
    sample_rate: int = 44100,
    gain: Optional[float] = None,
    overwrite: bool = True,
) -> Path:
    """
    使用 FluidSynth 将 MIDI 渲染为 WAV。

    说明：
    - 在 Windows 上通常需要显式提供 fluidsynth.exe 的路径
    - 若找不到 fluidsynth 可执行文件，会抛出清晰错误信息
    """
    midi = _resolve_existing_file(midi_path, what="MIDI 文件")
    sf2 = _resolve_existing_file(soundfont_path, what="SoundFont(.sf2) 文件")
    wav = Path(wav_path).expanduser().resolve()

    if wav.exists() and not overwrite:
        return wav

    wav.parent.mkdir(parents=True, exist_ok=True)

    exe = find_fluidsynth_exe(fluidsynth_exe)
    if not exe:
        raise FileNotFoundError(
            "未找到 FluidSynth 可执行文件（fluidsynth / fluidsynth.exe）。\n"
            "请安装 FluidSynth 并确保在 PATH 中，或在代码里显式传入 fluidsynth_exe=...。"
        )

    # fluidsynth [options] soundfont [midifile]
    # 常用：
    # -n: no shell
    # -i: 不进入交互模式
    # -F <file>: 输出到文件
    # -r <rate>: 采样率
    # -g <gain>: 增益
    cmd = [
        str(exe),
        "-ni",
        "-F",
        str(wav),
        "-r",
        str(int(sample_rate)),
    ]
    if gain is not None:
        cmd += ["-g", str(float(gain))]
    cmd += [str(sf2), str(midi)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "FluidSynth 渲染失败。\n"
            f"命令: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )

    if not wav.exists():
        raise RuntimeError(f"FluidSynth 执行成功但未生成 wav 文件: {wav}")

    return wav


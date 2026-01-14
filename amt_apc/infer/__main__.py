import argparse

import torch

from amt_apc.models import Pipeline
from amt_apc.data import SVSampler


DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _run(args):
    """Run the inference pipeline."""
    device = torch.device(args.device) if args.device else DEVICE_DEFAULT
    pipeline = Pipeline(path_model=args.path_model, device=device)

    src = args.input
    if src.startswith("https://"):
        src = _download(src)

    sv_sampler = SVSampler()
    sv = sv_sampler.sample(params=args.style)
    pipeline.wav2midi(src, args.output, sv, silent=False)


def _download(url):
    """Download audio from YouTube."""
    from yt_dlp import YoutubeDL
    ydl_opts = {
        "outtmpl": "_audio.%(ext)s",
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "ignoreerrors": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return "_audio.wav"


def main():
    """CLI entry point for amt-apc."""
    parser = argparse.ArgumentParser(
        prog="amt-apc",
        description="Automatic Piano Cover generation from audio files"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input wav file or URL of YouTube video"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.mid",
        help="Path to the output midi file. Defaults to 'output.mid'"
    )
    parser.add_argument(
        "-s", "--style",
        type=str,
        default="level2",
        choices=["level1", "level2", "level3"],
        help="Cover style. Defaults to 'level2'"
    )
    parser.add_argument(
        "--path_model",
        type=str,
        default=None,
        help="Path to a custom model file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected by default"
    )
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()

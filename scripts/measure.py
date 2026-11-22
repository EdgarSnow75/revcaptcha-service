import argparse
import time
from pathlib import Path

import numpy as np
import psutil

from revcaptcha.asr import WhisperTranscriber
from revcaptcha.audio import check_samples, load_audio
from revcaptcha.speaker import CamppEmbedder


def build_clips(clips: list[str]):
    clip_samples = []
    for i in range(len(clips)):
        clip_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "revcaptcha"
            / "tests"
            / "data"
            / clips[i]
        )
        audio = load_audio(clip_path)
        check_samples(audio)
        clip_samples.append(audio)
    clip_samples = np.concatenate(clip_samples, axis=0)
    check_samples(clip_samples)
    return clip_samples


def meaure_rss_mb():
    process = psutil.Process()
    rss = process.memory_info().rss / 1048576
    return rss


def time_model_calls(model_run, samples: np.ndarray, runs: int = 11):
    times = []
    for _ in range(runs):
        time_before = time.perf_counter()
        model_run(samples)
        times.append(time.perf_counter() - time_before)
    return times


def print_table(model_name, memory_before, memory_after, load_time, results):
    """Print a Markdown table you can paste straight into the README."""
    print()
    print(
        "| Model | Input | Memory before (MB) | Memory after load (MB) "
        "| Load (s) | First call (s) | Average of next 10 (s) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|")
    for input_name, times in results:
        first = times[0]
        average = sum(times[1:]) / len(times[1:])
        print(
            f"| {model_name} | {input_name} | {memory_before:.0f} "
            f"| {memory_after:.0f} | {load_time:.2f} "
            f"| {first:.3f} | {average:.3f} |"
        )


def measure(model, one_second_input, four_seconds_input):
    model_memory_before = meaure_rss_mb()
    model_time_before = time.perf_counter()
    if model == "whisper":
        model = WhisperTranscriber()
        model_run = model.transcribe
        label = "Whisper Small"

    elif model == "campplus":
        path = Path(__file__).resolve().parents[1] / "models" / "campplus"
        model = CamppEmbedder(path)
        model_run = model.embed
        label = "CAM++"
    model_load_time = time.perf_counter() - model_time_before
    model_memory_after = meaure_rss_mb()

    results = [
        ("1 s", time_model_calls(model_run, one_second_input)),
        ("4 s", time_model_calls(model_run, four_seconds_input)),
    ]
    print_table(label, model_memory_before, model_memory_after, model_load_time, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["campplus", "whisper"])
    args = parser.parse_args()
    one_second_input = build_clips(["cce7416f_down.wav"])
    four_seconds_input = build_clips(
        [
            "00b01445_five.wav",
            "004ae714_seven.wav",
            "012c8314_eight.wav",
            "c50f55b8_down.wav",
        ]
    )
    measure(args.model, one_second_input, four_seconds_input)


if __name__ == "__main__":
    main()

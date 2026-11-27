import argparse
import secrets
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from revcaptcha.asr import WhisperTranscriber
from revcaptcha.attempt import run_attempt
from revcaptcha.audio import AudioError, load_audio
from revcaptcha.challenge import ANSWER_WINDOW, create_challenge
from revcaptcha.mic import MicError, check_microphone, record_answer
from revcaptcha.scoring import THRESHOLDS, build_profile, threshold_for
from revcaptcha.speaker import MODEL_NAME, CamppEmbedder
from revcaptcha.store import (
    StoreError,
    append_attempt,
    load_profile,
    name_check,
    save_profile,
)
from revcaptcha.text import change_list_to_str, normalize_transcript
from revcaptcha.verify import Outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="revcaptcha", description="Commands for using the REVCaptcha program"
    )
    parser.add_argument("-d", "--data", type=Path, default=Path("data"))
    sub_parser = parser.add_subparsers(dest="command", required=True)
    sub_parser_enroll = sub_parser.add_parser("enroll", help="Enroll a voice profile")
    sub_parser_enroll.add_argument("-u", "--user", required=True)
    sub_parser_enroll.add_argument("-c", "--clips", type=Path, required=True)
    sub_parser_enroll.add_argument(
        "--campp-dir", type=Path, default=Path("models/campplus")
    )
    sub_parser_enroll.set_defaults(func=enroll_command)
    sub_parser_attempt = sub_parser.add_parser("attempt", help="Verify a login attempt")
    sub_parser_attempt.add_argument("-u", "--user", required=True)
    sub_parser_attempt.add_argument("--threshold", type=threshold_type, default=None)
    sub_parser_attempt.add_argument(
        "--campp-dir", type=Path, default=Path("models/campplus")
    )
    sub_parser_attempt.add_argument("--whisper-model", default="small")
    sub_parser_attempt.set_defaults(func=attempt_command)
    args = parser.parse_args(argv)
    return args.func(args)


def enroll_command(args) -> int:
    if not args.clips.is_dir():
        print("Directory not found", file=sys.stderr)
        return 2
    if name_check(args.user) is None:
        print(
            "User name is invalid.User names may only use a–z, 0–9, - and _, up to 32 characters",
            file=sys.stderr,
        )
        return 2
    valid_clips = []
    for item in sorted(args.clips.iterdir()):
        if item.suffix.lower() in [".wav", ".flac"]:
            valid_clips.append(item)
    if len(valid_clips) not in THRESHOLDS:
        print(
            f"The folder only holds {len(valid_clips)}. The allowed clip sizes are {THRESHOLDS}",
            file=sys.stderr,
        )
        return 2
    embeddings = []
    try:
        embedder = CamppEmbedder(args.campp_dir)
    # We need to catch multiple exceptions that could occur when creating the model
    except Exception as err:  # noqa: BLE001
        print(f"Model was not created: {err}", file=sys.stderr)
        return 3
    for clip in valid_clips:
        try:
            audio = load_audio(clip)
        except AudioError:
            print(f"{clip} failed to load", file=sys.stderr)
            return 2
        embeddings.append(embedder.embed(audio))
    user_profile = build_profile(embeddings)
    saved_path = save_profile(
        args.data / "profiles",
        args.user,
        user_profile,
        model=MODEL_NAME,
        clips=[c.name for c in valid_clips],
    )
    print(f"Enrolled {args.user} from {len(valid_clips)} clips : {saved_path}")
    return 0


def attempt_command(args) -> int:
    if name_check(args.user) is None:
        print(
            "User name is invalid.User names may only use a–z, 0–9, - and _, up to 32 characters",
            file=sys.stderr,
        )
        return 2
    try:
        profile, profile_clip_count = load_profile(
            args.data / "profiles", args.user, model=MODEL_NAME
        )
    except StoreError:
        print(f"{args.user}'s profile is not found", file=sys.stderr)
        return 2
    model_files = ["avg_model.pt", "config.yaml"]
    for model_file in model_files:
        if not (args.campp_dir / model_file).is_file():
            print(
                f"{model_file} not found in the model files directory", file=sys.stderr
            )
            return 2
    try:
        check_microphone()
    except MicError:
        print("Not a valid mic input", file=sys.stderr)
        return 2
    try:
        transcriber = WhisperTranscriber(model_name=args.whisper_model)
        embedder = CamppEmbedder(model_dir=args.campp_dir)
    # Need to catch various subclasses of Exception from any model failures
    except Exception as err:  # noqa: BLE001
        print(f"Failed to load model: {err}", file=sys.stderr)
        return 3
    input("Ready. Press Enter to see the code.")
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = threshold_for(profile_clip_count)

    def issue():
        challenge = create_challenge(secrets.SystemRandom(), datetime.now(UTC))
        print(f"Created Code: {' '.join(challenge.code)}")
        return challenge

    challenge, arrived_at, results = run_attempt(
        profile=profile,
        transcriber=transcriber,
        embedder=embedder,
        issue=issue,
        record=record_answer,
        threshold=threshold,
    )
    record = {
        "at": arrived_at.isoformat(),
        "user": args.user,
        "code": challenge.code,
        "answer_seconds": (arrived_at - challenge.issued_at).total_seconds(),
        **asdict(results),
    }
    append_attempt(args.data / "attempts.jsonl", record)
    print(format_result(challenge, arrived_at, results))
    if results.outcome == Outcome.ACCEPTED:
        return 0
    else:
        return 1


def format_result(challenge, arrived_at, result) -> str:
    code_line = f"Code:       {challenge.code}   (answered in {(arrived_at - challenge.issued_at).total_seconds():.1f}s of {ANSWER_WINDOW.total_seconds():g}s)"
    if result.transcript is None:
        stage_1_line = "Stage 1     not run"
    else:
        normalized_transcript = change_list_to_str(
            normalize_transcript(result.transcript)
        )
        stage_1_line = f"Stage 1     heard {result.transcript}  ->  {normalized_transcript}, expected {challenge.code} -> {'failed' if result.outcome == Outcome.GATE_REJECTED else 'passed'} ({result.asr_seconds} s)"

    if result.score is None:
        stage_2_line = "Stage 2     not run"
    else:
        stage_2_line = f"Stage 2     score {result.score} against {result.threshold} -> {'failed' if result.outcome == Outcome.SPEAKER_REJECTED else 'passed'} ({result.speaker_seconds} s)"
    decision = f"Decision    {result.outcome.upper()}"
    result_string = f"{code_line}\n{stage_1_line}\n{stage_2_line}\n{decision}"
    return result_string


def threshold_type(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number")

    if not -1 <= value <= 1:
        raise argparse.ArgumentTypeError(f"{value} is outside -1, 1")
    return value

import argparse
import sys

from pathlib import Path
from revcaptcha.audio import AudioError, load_audio
from revcaptcha.scoring import THRESHOLDS, build_profile
from revcaptcha.speaker import MODEL_NAME, CamppEmbedder
from revcaptcha.store import name_check, save_profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="revcaptcha", description="Commands for using the REVCaptcha program"
    )
    parser.add_argument("-d", "--data", type=Path, default=Path("data"))
    sub_parser = parser.add_subparsers(dest="command", required=True)
    sub_parser_enroll = sub_parser.add_parser("enroll", help="Enroll a voice profile")
    sub_parser_enroll.add_argument("-u", "--user", required=True)
    sub_parser_enroll.add_argument("-c", "--clips", type=Path, required=True)
    sub_parser_enroll.add_argument("--campp-dir", type=Path, default="models/campplus")
    sub_parser_enroll.set_defaults(func=enroll_command)
    sub_parser_attempt = sub_parser.add_parser("attempt", help="Verify a login attempt")
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
    print("Attempt is not written yet", file=sys.stderr)
    return 3

import argparse
import sys

from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="revcaptcha", description="Commands for using the REVCaptcha program"
    )
    parser.add_argument("-d", "--data", type=Path, default=Path("data"))
    sub_parser = parser.add_subparsers(dest="command", required=True)
    sub_parser_enroll = sub_parser.add_parser("enroll", help="Enroll a voice profile")
    sub_parser_enroll.add_argument("-u", "--user")
    sub_parser_enroll.add_argument("-c", "--clips")
    sub_parser_enroll.set_defaults(func=enroll_command)
    sub_parser_enroll.add_argument("--camp-dir", default="models/campplus")
    sub_parser_attempt = sub_parser.add_parser("attempt", help="Verify a login attempt")
    sub_parser_attempt.set_defaults(func=attempt_command)
    args = parser.parse_args(argv)
    return args.func(args)


def enroll_command(args) -> int:
    print("Enroll is not written yet", file=sys.stderr)
    return 3


def attempt_command(args) -> int:
    print("Attempt is not written yet", file=sys.stderr)
    return 3

import argparse
import json
from collections import Counter

from revcaptcha.challenge import is_valid_code
from revcaptcha.text import (
    NORMALIZATION_MAP,
    lexical_match,
    normalize_transcript,
    remove_and_split_text,
)


def check_map_results():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_file_path",
        help="Parse a valid path to the result file holding the stage 1 whisper transcript results",
    )
    parser.add_argument(
        "group",
        choices=["dev", "check"],
        help="There are only two valid options: dev or check",
    )
    args = parser.parse_args()
    with open(args.result_file_path, "r", encoding="utf-8") as result_file:
        result_data = json.load(result_file)["results"]
        result_data = [
            result
            for result in result_data
            if result["stage1_label"] == "correct"
            if is_valid_code(result["prompt_displayed_raw"])
            if check_letter_group(result["speaker_id"]) == args.group
        ]
    pass_count = 0
    total_count = len(result_data)
    failures = Counter()
    failure_examples = {}
    multi_word_parts = {
        part for key in NORMALIZATION_MAP if " " in key for part in key.split()
    }
    for result in result_data:
        raw_transcript = result["asr_raw_transcript"].upper()
        code = result["prompt_displayed_raw"].upper()
        if lexical_match(raw_transcript, code):
            pass_count += 1
        else:
            print(
                f"Code: {code} | Raw_Transcript: {result['asr_raw_transcript']} | Normalized_Transcript: {normalize_transcript(raw_transcript)}",
                end="\n",
            )
            for word in set(remove_and_split_text(raw_transcript)):
                if (
                    word.isalpha()
                    and len(word) >= 2
                    and word not in NORMALIZATION_MAP
                    and word not in multi_word_parts
                ):
                    failures[word] += 1
                    if word in failure_examples:
                        failure_examples[word].append((code, raw_transcript))
                    else:
                        failure_examples[word] = [(code, raw_transcript)]
    pass_rate = pass_count / total_count * 100
    print(
        "-------------------------------------------------------------------------------------"
    )
    print(args.group)
    print(f"Pass Rate: {pass_rate:.1f}%")
    print(f"{pass_count} out of {total_count} passes")
    print()
    for word, count in failures.most_common(20):
        print(f"{word} ({count})")
        for example_code, raw in failure_examples[word]:
            print(f"    {example_code} | {raw[:80]}")


def check_letter_group(speaker_id: str) -> str:
    first_word = speaker_id[0].upper()
    if "A" <= first_word <= "M":
        return "dev"
    elif "N" <= first_word <= "Z":
        return "check"


def main():
    check_map_results()


if __name__ == "__main__":
    main()

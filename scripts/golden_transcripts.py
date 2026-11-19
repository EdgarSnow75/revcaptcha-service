import argparse
import json
from pathlib import Path

from revcaptcha.asr import WhisperTranscriber
from revcaptcha.audio import load_audio


def check_transcript_match():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_path", help="Parse the file path of the result as an option"
    )
    parser.add_argument(
        "speaker_id",
        help="Add the name of the speaker that you want to get 10 recordings of",
    )
    args = parser.parse_args()
    path = Path(args.result_path)
    whisper_model = WhisperTranscriber()
    with open(path, "r", encoding="utf-8") as data_file:
        data = json.load(data_file)
        filtered_data = [
            x for x in data["results"] if x["speaker_id"] == args.speaker_id
        ]
        count = 0
        sample_range = 10
        for i in range(sample_range):
            audio = load_audio(filtered_data[i]["audio_path"])
            transcript = whisper_model.transcribe(audio)
            print(f"SampleID: {filtered_data[i]['sample_id']}")
            print(f"Asr Raw Transcript: {filtered_data[i]['asr_raw_transcript']}")
            print(f"Now Transcript: {transcript}")
            if transcript == filtered_data[i]["asr_raw_transcript"]:
                print("MATCH")
                count += 1
            elif transcript != filtered_data[i]["asr_raw_transcript"]:
                print("DIFF")
            print("----------------------------", end="\n")
        print(f"Count: {count} out of {sample_range} matches")
        return filtered_data[:10]


def main():
    print(check_transcript_match())


if __name__ == "__main__":
    main()

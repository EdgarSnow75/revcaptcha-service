import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from revcaptcha.audio import load_audio
from revcaptcha.speaker import CamppEmbedder


def check_embeddings():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", help="Parse a path to the stage1 result file")
    parser.add_argument("embeddings_path", help="Parse a path to the embedding file")
    parser.add_argument(
        "speaker_id", help="Name the speaker_id that you want to search for"
    )
    args = parser.parse_args()
    results_path = Path(args.results_path)
    embedding_path = Path(args.embeddings_path)
    speaker_id = args.speaker_id
    with open(
        results_path,
        "r",
        encoding="utf-8",
    ) as results_file:
        results_data = json.load(results_file)["results"]
        filtered_results = [
            x for x in results_data if x["speaker_id"] == speaker_id.lower()
        ][:10]
    path_model = Path(__file__).parents[1] / "models" / "campplus"
    embedding_model = CamppEmbedder(path_model)
    with open(embedding_path, "r", encoding="utf-8") as embedding_file:
        embedding_data = json.load(embedding_file)["embeddings"]
        for i in range(5):
            new_embed = embedding_model.embed(
                load_audio(filtered_results[i]["audio_path"])
            )
            sample_id = filtered_results[i]["sample_id"]
            old_embed = embedding_data[sample_id]
            old_embed = torch.tensor([old_embed], dtype=torch.float32)
            new_embed = torch.from_numpy(new_embed)
            cosine_score = F.cosine_similarity(new_embed, old_embed, dim=-1).item()
            print(f"{i + 1}.Cosine Score: {cosine_score}")


def main():
    check_embeddings()


if __name__ == "__main__":
    main()

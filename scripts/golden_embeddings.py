import json
from pathlib import Path

import torch
import torch.nn.functional as F

from revcaptcha.speaker import CamppEmbedder


def check_embeddings(results_path, embedding_path, speaker_id):
    results_path = Path(results_path)
    embedding_path = Path(embedding_path)
    with open(
        results_path,
        "r",
        encoding="utf-8",
    ) as results_file:
        results_data = json.load(results_file)["results"]
        filtered_results = [
            x for x in results_data if x["speaker_id"] == speaker_id.lower()
        ][:10]
        # print(filtered_results)
    embedding_model = CamppEmbedder("../models/campplus")
    for i in range(5):
        new_embed = embedding_model.embed(filtered_results[i]["audio_path"])
        sample_id = filtered_results[i]["sample_id"]
        with open(embedding_path, "r", encoding="utf-8") as embedding_file:
            old_embed = json.load(embedding_file)["embeddings"][sample_id]
        old_embed = torch.tensor([old_embed], dtype=torch.float32)
        cosine_score = F.cosine_similarity(new_embed, old_embed, dim=-1).item()
        print(f"{i + 1}.Cosine Score: {cosine_score}")


def main():
    check_embeddings(
        r"G:\Research Paper\REVCaptcha\stage1\outputs\stage1_results_H.json",
        r"G:\Research Paper\REVCaptcha\stage2\embeddings\embeddings_H_campp.json",
        "ouro",
    )


if __name__ == "__main__":
    main()

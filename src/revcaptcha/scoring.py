import numpy as np

THRESHOLDS = {5: 0.5387, 10: 0.5669}  # JCSSE TRACK H CAM++ OP-B over 5 Folds


def build_profile(embeddings) -> np.ndarray:
    for embedding in embeddings:
        if embedding.size == 0:
            raise ValueError("There is an empty array")
    profile = np.stack(embeddings)
    profile = np.mean(profile, axis=0) / profile.shape[1]
    if not np.any(profile):
        raise ValueError("Zeroes cannot be divided")
    return profile


def cosine_score(profile, embedding) -> float:
    score = np.dot(profile, embedding) / (
        np.linalg.norm(profile) * np.linalg.norm(embedding)
    )
    return score


def speaker_accepts(score, threshold) -> bool:
    return score >= threshold


def threshold_for(enrollment_size) -> float:
    if enrollment_size not in THRESHOLDS:
        raise ValueError("The enrollment size must be either 5 or 10")
    return THRESHOLDS[enrollment_size]


# build_profile(
#     [
#         np.array([3.0, 4.0]),
#         np.array([4.0, 3.0]),
#         np.array([1.0, 0.0]),
#         np.array([0.0, 1.0]),
#     ]
# )
# build_profile([np.array([1.0, 0.0]), np.array([-1.0, 0.0])])

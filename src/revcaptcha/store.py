import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


class StoreError(Exception):
    """Raises when storing errors occur"""


USERNAME_PATTERN = r"[a-z0-9_-]{1,32}"


def save_profile(folder, user, profile, *, model, clips) -> Path:
    if not name_check(user):
        raise StoreError(
            "Inputted user name is not allowed. Please keep between 1 to 32 characters of alphanumeric characters and/or - and _"
        )
    created_at = datetime.now(UTC).isoformat()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{user}.json"
    clips = list(clips)
    profile = profile.tolist()
    user_profile = {
        "user": user,
        "model": model,
        "created_at": created_at,
        "profile": profile,
        "clips": clips,
    }
    path.write_text(json.dumps(user_profile), encoding="utf-8")
    return path


def load_profile(folder, user, *, model) -> tuple[np.ndarray, int]:
    if name_check(user) is None:
        raise StoreError(
            "Inputted user name is not allowed. Please keep between 1 to 32 characters of alphanumeric characters and/or - and _"
        )
    path = Path(folder) / f"{user}.json"
    if not path.exists():
        raise StoreError("No such user profile")
    json_file = json.loads(path.read_text(encoding="utf-8"))
    if json_file["model"] != model:
        raise StoreError("Wrong model")
    user_profile = np.array(json_file["profile"], dtype=np.float32)
    if abs(np.linalg.norm(user_profile) - 1) > 0.001:
        raise StoreError(f"Profile for {user} is not unit length")
    return user_profile, len(json_file["clips"])


def append_attempt(path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as json_file:
        json_file.write(json.dumps(record) + "\n")


def name_check(user):
    return re.fullmatch(USERNAME_PATTERN, user)

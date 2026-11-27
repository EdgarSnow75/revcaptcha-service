import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from revcaptcha import cli
from revcaptcha.challenge import ANSWER_WINDOW, Challenge
from revcaptcha.mic import MicError
from revcaptcha.speaker import MODEL_NAME
from revcaptcha.store import load_profile, save_profile
from revcaptcha.tests.fakes import FakeEmbedder
from revcaptcha.verify import Outcome, VerifyResult

T0 = datetime(2026, 11, 27, 8, 0, tzinfo=UTC)


def issue():
    return Challenge(id="t1", code="5G5LB", issued_at=T0, expires_at=T0 + ANSWER_WINDOW)


@pytest.fixture
def no_model(monkeypatch):
    monkeypatch.setattr(cli, "CamppEmbedder", boom)


@pytest.fixture
def five_clips(tmp_path):
    path = tmp_path / "clips"
    path.mkdir(exist_ok=True)
    data_folder = sorted((Path(__file__).parent / "data").glob("cce7416f_*.wav"))[:5]
    for clip in data_folder:
        shutil.copy(clip, path / clip.name)
    return path


def make_files(folder, names):
    folder.mkdir(exist_ok=True)
    for name in names:
        (folder / name).touch()


def boom(*args, **kwargs):
    pytest.fail("A model was created")


def bad_model(*args, **kwargs):
    raise OSError("no model files")


@pytest.mark.parametrize(
    "argv, expected",
    (
        # [["enroll"], 3],
        # [["attempt"], 3],
        # [["revcaptcha"], 2],
        # [["revcaptcha", "-h"], 0],
        [[], 2],
    ),
)
def test_exit_code_enroll(argv, expected):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    print(excinfo)
    assert excinfo.value.code == expected


@pytest.mark.parametrize(
    "file_names",
    (
        [],
        ["0.wav", "1.wav", "2.wav", "3.wav", "4.wav", "5.wav"],
        ["0.wav", "1.wav", "2.wav", "notes.txt"],
    ),
)
def test_enroll_wrong_count(file_names, tmp_path, capsys, no_model):
    clips = tmp_path / "clips"
    make_files(clips, file_names)
    assert (
        cli.main(
            [
                "--data",
                str(tmp_path / "data"),
                "enroll",
                "--user",
                "edgar",
                "--clips",
                str(clips),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.err != ""
    assert captured.out == ""


@pytest.mark.parametrize(
    "bad_name",
    ("edgar!", "../../notes"),
)
def test_enroll_bad_names(bad_name, tmp_path, capsys, no_model):
    clips = tmp_path / "clips"
    make_files(clips, ["0.wav", "1.wav", "2.wav", "3.wav", "4.wav"])
    assert (
        cli.main(
            [
                "--data",
                str(tmp_path / "data"),
                "enroll",
                "--user",
                bad_name,
                "--clips",
                str(clips),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.err != ""
    assert captured.out == ""


def test_enroll_saves_profile(tmp_path, monkeypatch, five_clips, capsys):
    monkeypatch.setattr(cli, "CamppEmbedder", lambda model_dir: FakeEmbedder([0.6, 0.8]))
    assert (
        cli.main(
            [
                "--data",
                str(tmp_path / "data"),
                "enroll",
                "--user",
                "ver",
                "--clips",
                str(five_clips),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "ver" in captured.out
    assert "5 clips" in captured.out
    profile, clip_count = load_profile(
        tmp_path / "data" / "profiles", "ver", model=MODEL_NAME
    )
    assert np.allclose(profile, (0.6, 0.8))
    assert clip_count == 5


def test_enroll_empty_bytes(tmp_path, monkeypatch, five_clips, capsys):
    monkeypatch.setattr(cli, "CamppEmbedder", lambda model_dir: FakeEmbedder([0.6, 0.8]))
    Path(five_clips / "cce7416f_eight.wav").write_bytes(b"")
    assert (
        cli.main(
            [
                "--data",
                str(tmp_path / "data"),
                "enroll",
                "--user",
                "ver",
                "--clips",
                str(five_clips),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "cce7416f_eight.wav" in captured.err
    assert captured.out == ""


def test_enroll_no_model(tmp_path, monkeypatch, five_clips, capsys):
    monkeypatch.setattr(cli, "CamppEmbedder", bad_model)
    assert (
        cli.main(
            [
                "--data",
                str(tmp_path / "data"),
                "enroll",
                "--user",
                "ver",
                "--clips",
                str(five_clips),
            ]
        )
        == 3
    )
    captured = capsys.readouterr()
    assert "no model files" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "result, arrive_time, results_text",
    (
        [
            VerifyResult(
                outcome=Outcome.ACCEPTED,
                transcript="5G5LB.",
                score=0.7123,
                threshold=0.5387,
                asr_seconds=1.42,
                speaker_seconds=0.19,
            ),
            T0 + timedelta(seconds=4.2),
            "ACCEPTED",
        ],
        [
            VerifyResult(
                outcome=Outcome.EXPIRED,
                transcript=None,
                score=None,
                threshold=0.5387,
                asr_seconds=None,
                speaker_seconds=11.0,
            ),
            T0 + timedelta(seconds=11),
            "EXPIRED",
        ],
        [
            VerifyResult(
                outcome=Outcome.GATE_REJECTED,
                transcript="5 G 5 L P",
                score=None,
                threshold=0.5387,
                asr_seconds=1.3,
                speaker_seconds=None,
            ),
            T0 + timedelta(seconds=5),
            "GATE_REJECTED",
        ],
        [
            VerifyResult(
                outcome=Outcome.SPEAKER_REJECTED,
                transcript="5 G 5 L P",
                score=0.3100,
                threshold=0.5387,
                asr_seconds=1.3,
                speaker_seconds=0.2,
            ),
            T0 + timedelta(seconds=5),
            "SPEAKER_REJECTED",
        ],
    ),
)
def test_format_results(result, arrive_time, results_text):
    assert results_text in cli.format_result(
        challenge=issue(), arrived_at=arrive_time, result=result
    )


# attempt subcommand
@pytest.fixture
def ready_profile(tmp_path):
    path = Path(tmp_path / "data" / "profiles")
    save_profile(
        path,
        "edgar",
        np.array([0.6, 0.8], dtype=np.float32),
        model=MODEL_NAME,
        clips=["0.wav", "1.wav", "2.wav", "3.wav", "4.wav"],
    )
    campp_dir = tmp_path / "campplus"
    campp_dir.mkdir(exist_ok=True)
    (campp_dir / "avg_model.pt").touch()
    (campp_dir / "config.yaml").touch()
    return campp_dir


def bad_mic(*args, **kwargs):
    raise MicError("no input device")


def bad_whisper_model(*args, **kwargs):
    raise RuntimeError("cannot load")


@pytest.mark.parametrize(
    "value, expected, err_text",
    (["1.5", 2, "outside -1, 1"], ["abc", 2, "is not a number"]),
)
def test_exit_code_attempt(value, expected, err_text, capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["attempt", "--user", "edgar", "--threshold", value])
    print(excinfo)
    captured = capsys.readouterr()
    assert excinfo.value.code == expected
    assert err_text in captured.err


@pytest.mark.parametrize(
    "user, err_text",
    (["edgar", "profile is not found"], ["Edgar", "User name is invalid"]),
)
def test_attempt_no_profile(tmp_path, capsys, user, err_text):
    assert cli.main(["--data", str(tmp_path / "data"), "attempt", "--user", user]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert err_text in captured.err


def test_attempt_missing_model(tmp_path, ready_profile, capsys):
    file_path = Path(ready_profile / "config.yaml")
    file_path.unlink(missing_ok=True)
    data_path = Path(tmp_path / "data")
    assert (
        cli.main(
            [
                "--data",
                str(data_path),
                "attempt",
                "--user",
                "edgar",
                "--campp-dir",
                str(ready_profile),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "not found in the model files" in captured.err


def test_attempt_no_mic(tmp_path, monkeypatch, ready_profile, capsys):
    monkeypatch.setattr(cli, "check_microphone", bad_mic)
    assert (
        cli.main(
            [
                "--data",
                str(tmp_path / "data"),
                "attempt",
                "--user",
                "edgar",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "Not a valid mic input\n" in captured.err
    assert captured.out == ""


def test_attempt_model_load(tmp_path, ready_profile, monkeypatch):
    monkeypatch.setattr(cli, "check_microphone", lambda: None)
    monkeypatch.setattr(cli, "WhisperTranscriber", bad_whisper_model)
    assert (
        cli.main(
            [
                "--data",
                str(tmp_path / "data"),
                "attempt",
                "--user",
                "edgar",
                "--threshold",
                "0.5",
                "--campp-dir",
                str(ready_profile),
            ]
        )
        == 3
    )

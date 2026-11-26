import shutil
from pathlib import Path

import numpy as np
import pytest

from revcaptcha import cli
from revcaptcha.speaker import MODEL_NAME
from revcaptcha.store import load_profile
from revcaptcha.tests.fakes import FakeEmbedder


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
def test_exit_code(argv, expected):
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
                "nope",
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

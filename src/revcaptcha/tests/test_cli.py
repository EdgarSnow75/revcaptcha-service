import pytest
from revcaptcha import cli


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

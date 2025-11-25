import json
import textwrap

from peakfit.cli.validate_command import _read_csv_list, _read_json_list, _read_sparky_list


def test_read_csv_returns_peakinput(tmp_path):
    csv_content = textwrap.dedent("""
    Assign F1,Pos F1,Pos F2
    A1,1.23,4.56
    B2,2.34,5.67
    """)
    file = tmp_path / "peaks.csv"
    file.write_text(csv_content)

    peaks = _read_csv_list(file)
    assert len(peaks) == 2
    assert peaks[0].name == "A1"
    assert peaks[0].x == 1.23
    assert peaks[0].y == 4.56


def test_read_json_returns_peakinput(tmp_path):
    data = [
        {"name": "A1", "x": 1.23, "y": 4.56},
        {"Assign F1": "B2", "Pos F1": 2.34, "Pos F2": 5.67},
    ]
    file = tmp_path / "peaks.json"
    file.write_text(json.dumps(data))

    peaks = _read_json_list(file)
    assert len(peaks) == 2
    assert peaks[0].name == "A1"
    assert peaks[1].name == "B2"


def test_read_sparky_returns_peakinput(tmp_path):
    content = "A1 1.23 4.56\nB2 2.34 5.67\n"
    file = tmp_path / "peaks.list"
    file.write_text(content)

    peaks = _read_sparky_list(file)
    assert len(peaks) == 2
    assert peaks[0].name == "A1"
    assert peaks[1].y == 5.67

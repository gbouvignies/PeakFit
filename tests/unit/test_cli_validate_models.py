import json
import textwrap

from peakfit.services.validate import ValidationService


def test_read_csv_returns_peakinput(tmp_path):
    csv_content = textwrap.dedent("""
    Assign F1,Pos F1,Pos F2
    A1,1.23,4.56
    B2,2.34,5.67
    """)
    file = tmp_path / "peaks.csv"
    file.write_text(csv_content)

    peaks = ValidationService._read_csv_list(file)
    assert len(peaks) == 2
    assert peaks[0].name == "A1"
    # NMRPipe convention: F1 = indirect = y, F2 = direct = x
    assert peaks[0].y == 1.23  # Pos F1 = indirect dimension = y
    assert peaks[0].x == 4.56  # Pos F2 = direct dimension = x


def test_read_json_returns_peakinput(tmp_path):
    data = [
        {"name": "A1", "x": 1.23, "y": 4.56},
        {"Assign F1": "B2", "Pos F1": 2.34, "Pos F2": 5.67},
    ]
    file = tmp_path / "peaks.json"
    file.write_text(json.dumps(data))

    peaks = ValidationService._read_json_list(file)
    assert len(peaks) == 2
    assert peaks[0].name == "A1"
    assert peaks[1].name == "B2"


def test_read_sparky_returns_peakinput(tmp_path):
    # Sparky format: name F1_pos F2_pos (indirect first, direct second)
    content = "A1 1.23 4.56\nB2 2.34 5.67\n"
    file = tmp_path / "peaks.list"
    file.write_text(content)

    peaks = ValidationService._read_sparky_list(file)
    assert len(peaks) == 2
    assert peaks[0].name == "A1"
    # NMRPipe convention: F1 = indirect = y, F2 = direct = x
    assert peaks[1].y == 2.34  # F1 position = indirect = y
    assert peaks[1].x == 5.67  # F2 position = direct = x

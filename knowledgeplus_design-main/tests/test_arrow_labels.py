import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_toggle_labels_are_full_width():
    app_path = PROJECT_ROOT / "unified_app.py"
    text = app_path.read_text(encoding="utf-8")
    assert "＞＞" in text
    assert "＜＜" in text
    assert "keyboard_double_arrow" not in text

    toggle_path = PROJECT_ROOT / "ui_modules" / "sidebar_toggle.py"
    text = toggle_path.read_text(encoding="utf-8")
    assert "＞＞" in text
    assert "＜＜" in text
    assert "keyboard_double_arrow" not in text

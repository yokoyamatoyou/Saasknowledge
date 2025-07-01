import ast
from pathlib import Path

# Get the project root directory (one level up from the 'tests' directory)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _get_theme_call_args(path: str) -> int | None:
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == 'apply_intel_theme':
                return len(node.args)
    return None


def test_mm_kb_theme_call_uses_arg():
    app_path = PROJECT_ROOT / 'mm_kb_builder' / 'app.py'
    assert _get_theme_call_args(str(app_path)) == 1


def test_gpt_app_theme_call_uses_arg():
    app_path = PROJECT_ROOT / 'knowledge_gpt_app' / 'app.py'
    assert _get_theme_call_args(str(app_path)) == 1


def test_unified_app_theme_call_uses_arg():
    app_path = PROJECT_ROOT / 'unified_app.py'
    assert _get_theme_call_args(str(app_path)) == 1

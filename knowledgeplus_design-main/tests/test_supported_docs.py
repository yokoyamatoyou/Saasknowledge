from shared.file_processor import FileProcessor
from pathlib import Path
import ast

KB_APP_PATH = Path(__file__).resolve().parents[1] / 'mm_kb_builder' / 'app.py'
source = KB_APP_PATH.read_text(encoding="utf-8")
module = ast.parse(source)

# Determine whether SUPPORTED_DOCUMENT_TYPES is assigned from FileProcessor
ASSIGNED_FROM_FILEPROCESSOR = False
for node in module.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "SUPPORTED_DOCUMENT_TYPES":
                if (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "FileProcessor"
                    and node.value.attr == "SUPPORTED_DOCUMENT_TYPES"
                ):
                    ASSIGNED_FROM_FILEPROCESSOR = True
                break
    if ASSIGNED_FROM_FILEPROCESSOR:
        break

def test_supported_document_types():
    assert "docx" in FileProcessor.SUPPORTED_DOCUMENT_TYPES
    assert "xlsx" in FileProcessor.SUPPORTED_DOCUMENT_TYPES
    assert ASSIGNED_FROM_FILEPROCESSOR

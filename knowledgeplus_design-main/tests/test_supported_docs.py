from shared.file_processor import FileProcessor
from pathlib import Path
import ast

KB_APP_PATH = Path(__file__).resolve().parents[1] / 'mm_kb_builder' / 'app.py'
source = KB_APP_PATH.read_text(encoding='utf-8')
module = ast.parse(source)
SUPPORTED_DOCS = None
for node in module.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'SUPPORTED_DOCUMENT_TYPES':
                SUPPORTED_DOCS = [elt.s for elt in node.value.elts]
                break
    if SUPPORTED_DOCS:
        break

def test_supported_document_types():
    assert 'docx' in FileProcessor.SUPPORTED_DOCUMENT_TYPES
    assert 'xlsx' in FileProcessor.SUPPORTED_DOCUMENT_TYPES
    assert SUPPORTED_DOCS is not None
    assert 'docx' in SUPPORTED_DOCS
    assert 'xlsx' in SUPPORTED_DOCS

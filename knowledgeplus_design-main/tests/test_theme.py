import importlib

class Dummy:
    def __init__(self):
        self.text = None
    def markdown(self, text, unsafe_allow_html=False):
        self.text = text

def test_apply_intel_theme_injects_css():
    theme = importlib.import_module('ui_modules.theme')
    d = Dummy()
    theme.apply_intel_theme(d)
    assert d.text is not None
    assert '<style>' in d.text
    assert '--intel-blue' in d.text

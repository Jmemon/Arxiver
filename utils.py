
from pathlib import Path


ARXIVER_PATH = Path(__file__).parent


def get_version():
    return (Path(__file__).parent / '.version').read_text()

from pathlib import Path
from typing import Generator

from llm.models import *


llama_path = Path(__file__).parent / 'weights' / 'llama-2-7b-chat.Q5_K_M.gguf'
starling_path = Path(__file__).parent / 'weights' / 'starling-lm-7b-alpha.Q4_K_M.gguf'
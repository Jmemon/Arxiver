
from pathlib import Path

from langchain_community.llms.llamacpp import LlamaCpp


model_path = Path(__file__).parent / 'models' / 'llama-2-7b-chat.Q4_K_M.gguf'
n_gpu_layers = 50  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

llm = LlamaCpp(
    model_path=str(model_path),
    temperature=0.75,
    max_tokens=10,
    top_p=1,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    #callback_manager=callback_manager,
    #verbose=True,  # Verbose is required to pass to the callback manager
)

def get_llm():
    return llm

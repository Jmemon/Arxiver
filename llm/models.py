
import inspect
from pathlib import Path

from langchain_community.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


def llama(
        model_path: Path = Path(__file__).parent / 'weights' / 'llama-2-7b-chat.Q5_K_M.gguf',
        temperature: float = 0.75,
        max_tokens: int = 2000,
        top_p: float = 1,
        n_gpu_layers: int = 50,
        n_batch: int = 512,
        f16_kv: bool = True,
        **kwargs):
    return LlamaCpp(
            model_path=str(model_path),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            f16_kv=f16_kv,  # MUST set to True, otherwise you will run into problem after a couple of calls
            **kwargs
            #callback_manager=callback_manager,
            #verbose=True,  # Verbose is required to pass to the callback manager
        )


def starling(
        model_path: Path = Path(__file__).parent / 'weights' / 'starling-lm-7b-alpha.Q4_K_M.gguf',
        temperature: float = 0.75,
        max_tokens: int = 2000,
        top_p: float = 1,
        n_gpu_layers: int = 50,
        n_batch: int = 512,
        f16_kv: bool = True,
        **kwargs):
    return LlamaCpp(
            model_path=str(model_path),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            f16_kv=f16_kv,  # MUST set to True, otherwise you will run into problem after a couple of calls
            **kwargs
            #callback_manager=callback_manager,
            #verbose=True,  # Verbose is required to pass to the callback manager
        )

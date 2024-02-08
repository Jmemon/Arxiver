
import inspect
from pathlib import Path

from langchain_community.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from bot import llama_path, starling_path, mistral_path, wizard_path

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


def llama7b(
        model_path: Path = llama_path,
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


def starling7b(
        model_path: Path = starling_path,
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


def mistral7b(
        model_path: Path = mistral_path,
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


def wizardlm13b(
        model_path: Path = mistral_path,
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

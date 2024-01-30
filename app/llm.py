
import inspect
from pathlib import Path

from langchain_community.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class LLMHandler(LlamaCpp):

    # Implement LLMHandler as a Singleton, ie only allowing for one instance at a time
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMHandler, cls).__new__(cls)  # calls Object.__new__ to create instance of LLMHandler
        return cls._instance
    
    @staticmethod
    def get_instance(cls):
        return cls._instance

    def __init__(
            self, 
            model_path: Path, 
            temperature: float,
            max_tokens: int,
            top_p: float,
            n_gpu_layers: int, 
            n_batch: int,
            f16_kv: bool = True
    ):
        if not hasattr(self, 'model_path'):  # First instance of model
            super().__init__(
                model_path=str(model_path), 
                temperature=temperature, 
                max_tokens=max_tokens, 
                top_p=top_p, 
                n_gpu_layers=n_gpu_layers, 
                n_batch=n_batch, 
                f16_kv=f16_kv)

            self.model_path = model_path
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.top_p= top_p
            self.n_gpu_layers = n_gpu_layers
            self.n_batch = n_batch
            self.f16_kv = f16_kv

        else:  # there's already an instance, so we need to check if new __init__ args conflict with prev        
            params = [param.name for param in inspect.signature(self.__init__).parameters.values()]

            diff_str = ''
            for param_name, param_value in locals().items():
                if param_name not in params or param_name == 'self':
                    continue

                if getattr(self, param_name) != param_value:
                    diff_str += f'\n\tself.{param_name} != new {param_name}  ({getattr(self, param_name)} != {param_value})'
            
            diff_str += '\n'

            if len(diff_str) > 0:
                print(f'There\'s already an instance of {__class__} with the following differences:')
                print(diff_str)
                print('The current instance must be garbage collected to create a new one.')


callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

"""
def get_llama():
    return LLMHandler(
            model_path=Path(__file__).parent / 'models' / 'llama-2-7b-chat.Q4_K_M.gguf',
            temperature=0.75,
            max_tokens=10,
            top_p=1,
            n_gpu_layers=50,
            n_batch=512)
"""

def get_llama():
    return LlamaCpp(
            model_path=str(Path(__file__).parent / 'models' / 'llama-2-7b-chat.Q5_K_M.gguf'),
            temperature=0.75,
            max_tokens=20,
            top_p=1,
            n_gpu_layers=50,
            n_batch=512,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            #callback_manager=callback_manager,
            #verbose=True,  # Verbose is required to pass to the callback manager
        )

from pathlib import Path


weights_dir = Path(__file__).parent / 'weights'

llama_path = weights_dir / 'llama-2-7b-chat.Q5_K_M.gguf'
starling_path = weights_dir / 'starling-lm-7b-alpha.Q4_K_M.gguf'
mistral_path = weights_dir / 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'
wizard_path = weights_dir / 'wizardlm-13b-v1.2.Q4_K_M.gguf'
import torch
import numpy as np
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoProcessor

class SmolVlaPolicy:
    def __init__(self, device="cpu"):
        print("Initializing SmolVLA Policy...")
        self.device = torch.device(device)
        self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(self.device)
        self.policy.eval()
        
        processor = AutoProcessor.from_pretrained(self.policy.config.vlm_model_name)
        self.policy.language_tokenizer = processor.tokenizer
        print("SmolVLA Policy initialized successfully.")

    def get_action(self, observation, instruction):
        print("get_action called (not yet implemented)")
        return np.zeros(7, dtype=np.float32)

if __name__ == '__main__':
    print("Testing SmolVlaPolicy class instantiation...")
    if torch.backends.mps.is_available():
        policy_wrapper = SmolVlaPolicy(device="mps")
    else:
        policy_wrapper = SmolVlaPolicy(device="cpu")
    print("Test finished.")
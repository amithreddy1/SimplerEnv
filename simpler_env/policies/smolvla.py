import torch
import numpy as np
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoProcessor

class SmolVlaInference:
    def __init__(self, device="cpu"):
        print("Initializing SmolVLA Policy for SimplerEnv...")
        self.device = torch.device(device)
        self.instruction = ""
        self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(self.device)
        self.policy.eval()
        processor = AutoProcessor.from_pretrained(self.policy.config.vlm_model_name)
        self.policy.language_tokenizer = processor.tokenizer
        print("SmolVLA Policy initialized and ready.")

    def reset(self, instruction: str):
        print(f"Policy reset with new instruction: '{instruction}'")
        self.instruction = instruction

    def step(self, image_obs: np.ndarray, state_obs: np.ndarray):
        image_torch = torch.from_numpy(image_obs).permute(2, 0, 1).float() / 255.0
        state_padded = np.append(state_obs, 0.0) 
        state_torch = torch.tensor(state_padded, dtype=torch.float32)

        observation_batch = {
            'observation.image': image_torch.unsqueeze(0).to(self.device),
            'observation.state': state_torch.unsqueeze(0).to(self.device),
            'task': [self.instruction]
        }
        
        with torch.no_grad():
            action_chunk = self.policy.model.sample_actions(
                observation_batch['observation.image'], None, 
                self.policy.prepare_language(observation_batch)[0], None, 
                self.policy.prepare_state(observation_batch)
            )
        
        predicted_action_tensor = action_chunk[0, 0, :].cpu().numpy()

        action_dict = {
            "world_vector": predicted_action_tensor[0:3],
            "rot_axangle": predicted_action_tensor[3:6],
            "gripper": predicted_action_tensor[6:7],
            "terminate_episode": np.array([0]) 
        }

        return predicted_action_tensor, action_dict
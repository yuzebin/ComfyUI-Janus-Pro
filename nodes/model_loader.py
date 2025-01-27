import os

class JanusModelLoader:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["deepseek-ai/Janus-Pro-1B", "deepseek-ai/Janus-Pro-7B"],),
                "local_model": ("BOOLEAN", {"default": True}),
            },
        }
    
    RETURN_TYPES = ("JANUS_MODEL", "JANUS_PROCESSOR")
    RETURN_NAMES = ("model", "processor")
    FUNCTION = "load_model"
    CATEGORY = "Janus-Pro"

    def load_model(self, model_name, local_model=True):
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            from transformers import AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            dtype = torch.bfloat16
            torch.zeros(1, dtype=dtype, device=device)
        except RuntimeError:
            dtype = torch.float16

        if local_model:
            # 获取ComfyUI根目录
            comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            # 构建模型路径
            model_dir = os.path.join(comfy_path, 
                                   "models", 
                                   "Janus-Pro",
                                   os.path.basename(model_name))
            if not os.path.exists(model_dir):
                raise ValueError(f"Local model not found at {model_dir}. Please download the model and place it in the ComfyUI/models/Janus-Pro folder.")
            model_path = model_dir
        else:
            model_path = model_name
            
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        vl_gpt = vl_gpt.to(dtype).to(device).eval()
        
        return (vl_gpt, vl_chat_processor) 
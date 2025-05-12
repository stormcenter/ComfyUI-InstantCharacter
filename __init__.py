


# 注册节点
from .nodes.comfy_nodes import InstantCharacterLoadModel, InstantCharacterGenerate, InstantCharacterLoadModelFromLocal, InstantCharacterGenerateWithStyleLora



NODE_CLASS_MAPPINGS = {
    "InstantCharacterLoadModelFromLocal": InstantCharacterLoadModelFromLocal,
    "InstantCharacterLoadModel": InstantCharacterLoadModel,
    "InstantCharacterGenerate": InstantCharacterGenerate,
    "InstantCharacterGenerateWithStyleLora": InstantCharacterGenerateWithStyleLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantCharacterLoadModelFromLocal": "InstantCharacter Load Model From Local Checkpoints",
    "InstantCharacterLoadModel": "InstantCharacter Load Model",
    "InstantCharacterGenerate": "InstantCharacter Generate",
    "InstantCharacterGenerateWithStyleLora": "InstantCharacter Generate with Style Lora",
} 
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]


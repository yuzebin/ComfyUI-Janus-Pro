# ComfyUI-Janus-Pro

ComfyUI nodes for Janus-Pro, a unified multimodal understanding and generation framework.

## Installation

### Method 1: Install via ComfyUI Manager (Recommended)
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Search for "Janus-Pro" in the Manager
3. Click install

### Method 2: Manual Installation
1. Clone this repository to your ComfyUI's custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/CY-CHENYUE/ComfyUI-Janus-Pro
```

2. Install the required dependencies:

For Windows:
```bash
# 如果你使用ComfyUI便携版
cd ComfyUI-Janus-Pro
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt

# 如果你使用自己的Python环境
cd ComfyUI-Janus-Pro
path\to\your\python.exe -m pip install -r requirements.txt
```

For Linux/Mac:
```bash
# 使用ComfyUI的Python环境
cd ComfyUI-Janus-Pro
../../python_embeded/bin/python -m pip install -r requirements.txt

# 或者使用你的环境
cd ComfyUI-Janus-Pro
python -m pip install -r requirements.txt
```

Note: If you encounter any installation issues:
- Make sure you have git installed
- Try updating pip: `python -m pip install --upgrade pip`
- If you're behind a proxy, make sure git can access GitHub
- Make sure you're using the same Python environment as ComfyUI

## Usage
... 


## Model Setup

Place your models in the `ComfyUI/models/Janus-Pro` folder:
1. Create a `Janus-Pro` folder in your ComfyUI's models directory
2. Download the models from Hugging Face:
   - [Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
   - [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
3. Extract the models into their respective folders:
   ```
   ComfyUI/models/Janus-Pro/Janus-Pro-1B/
   ComfyUI/models/Janus-Pro/Janus-Pro-7B/
   ```
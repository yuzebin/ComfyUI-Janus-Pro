class JanusImageUnderstanding:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("JANUS_MODEL",),
                "processor": ("JANUS_PROCESSOR",),
                "image": ("IMAGE",),
                "question": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail."
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "analyze_image"
    CATEGORY = "Janus-Pro"

    def analyze_image(self, model, processor, image, question):
        try:
            from janus.models import MultiModalityCausalLM
        except ImportError:
            raise ImportError("Please install Janus using 'pip install -r requirements.txt'")

        if image.shape[0] == 1:
            image = image[0]
        image = (image * 255).cpu().numpy().astype(np.uint8)
        image = image.transpose(1, 2, 0)
        pil_image = Image.fromarray(image)

        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{question}",
                "images": [pil_image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        prepare_inputs = processor(
            conversations=conversation, 
            images=[pil_image], 
            force_batchify=True
        ).to(model.device)

        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        
        return (answer,) 
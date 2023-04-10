import torch, transformers
from transformers import CLIPTokenizer, CLIPTextModel

# from transformers/models/clip/modeling_clip.py
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

class CLIPText():
    def __init__(self, device, model_id='openai/clip-vit-base-patch32', cache_dir='.'):
        self.model_id = model_id
        self.device = device
        transformers.logging.set_verbosity_error()
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_id, cache_dir=cache_dir).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_id, cache_dir=cache_dir)
        transformers.logging.set_verbosity_warning()

    @torch.no_grad()
    def __call__(self, text):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        # (last_hidden_state, pooled_output, encoder_outputs[1:])
        text_embeds = self.text_encoder(text_input_ids.to(self.device))
        text_embeds = text_embeds[0]
        return text_embeds

    @property
    def model_max_length(self):
        return self.tokenizer.model_max_length

    @property
    def hidden_size(self):
        return self.text_encoder.config.hidden_size

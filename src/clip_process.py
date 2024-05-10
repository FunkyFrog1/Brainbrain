import torch
from clip import clip
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode


class ClipProcessor(torch.nn.Module):
    def __init__(self, model_path="../pretrain_weights/ViT-L-14.pt"):
        super().__init__()
        self.preprocess = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.model, _ = clip.load(model_path)
        self.model.eval()

    def forward(self, img):
        img = self.preprocess(img)
        with torch.no_grad():
            img_embed = self.model.encode_image(img)
        return img_embed

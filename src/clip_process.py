import torch
from clip import clip
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode


def clip_process(img):
    preprocess = Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    model, _ = clip.load("../pretrain_weights/ViT-L-14.pt", device=img.device)
    model.eval()

    img = preprocess(img)

    with torch.no_grad():
        img_embed = model.encode_image(img)

    return img_embed

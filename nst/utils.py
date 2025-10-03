from PIL import Image
import torch, numpy as np
from torchvision import transforms

def load_image(path, max_dim=512, device="cpu"):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_dim, max_dim))
    tfm = transforms.Compose([
        transforms.ToTensor(),                       # [0,1]
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]) # نرمال‌سازی مثل ImageNet
    ])
    t = tfm(img).unsqueeze(0).to(device)            # [1,3,H,W]
    return t, img.size

def tensor_to_pil(t):
    t = t.detach().cpu().clone().squeeze(0)
    t = t * torch.tensor([0.229,0.224,0.225]).view(3,1,1) + \
            torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    t = torch.clamp(t, 0, 1)
    arr = (t.numpy().transpose(1,2,0) * 255).astype(np.uint8)
    return Image.fromarray(arr)

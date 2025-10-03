import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# --- استخراج ویژگی‌های VGG ---
class VGGFeatures(nn.Module):
    def __init__(self, content_layers, style_layers):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg.eval()

        self.content_layers = set(content_layers)
        self.style_layers   = set(style_layers)

        # نام‌گذاری لایه‌ها
        self.names = []
        idx = {"conv": 0, "relu": 0, "pool": 0}
        for i, layer in enumerate(self.vgg):
            if isinstance(layer, nn.Conv2d):
                idx["conv"] += 1
                name = f"conv{idx['pool']+1}_{idx['conv']}"
            elif isinstance(layer, nn.ReLU):
                idx["relu"] += 1
                name = f"relu{idx['pool']+1}_{idx['relu']}"
                layer.inplace = False
            elif isinstance(layer, nn.MaxPool2d):
                idx["pool"] += 1
                idx["conv"] = 0
                idx["relu"] = 0
                name = f"pool{idx['pool']}"
            else:
                name = f"layer_{i}"
            self.names.append(name)

    def forward(self, x):
        content_feats, style_feats = {}, {}
        for layer, name in zip(self.vgg, self.names):
            x = layer(x)
            if name in self.content_layers:
                content_feats[name] = x
            if name in self.style_layers:
                style_feats[name] = x
        return content_feats, style_feats


# --- Gram matrix ---
def gram_matrix(feat):
    b, c, h, w = feat.size()
    F = feat.view(b, c, h * w)
    G = F @ F.transpose(1, 2) / (c * h * w)
    return G


# --- Total variation loss ---
def total_variation(img):
    diff_x = img[:, :, :, 1:] - img[:, :, :, :-1]
    diff_y = img[:, :, 1:, :] - img[:, :, :-1, :]
    return (diff_x.abs().mean() + diff_y.abs().mean())


# --- NST pipeline ---
def run_nst(content_img, style_img=None, *,
            content_layers, style_layers,
            content_w, style_w, tv_w,
            steps=600, lr=0.07, device="cpu",
            custom_targets=None):   # ← برای میانگین چند استایل

    extractor = VGGFeatures(content_layers, style_layers).to(device)

    # اگر custom_targets داده بشه از همون استفاده کن
    if custom_targets is not None:
        c_tgt, G_tgt = custom_targets
    else:
        with torch.no_grad():
            c_tgt, _ = extractor(content_img)
            _, s_feats = extractor(style_img)
            G_tgt = {k: gram_matrix(v) for k, v in s_feats.items()}

    # تصویر شروع (کپی از محتوا برای همگرایی بهتر)
    gen = content_img.clone().requires_grad_(True)

    opt = optim.Adam([gen], lr=lr)
    style_layer_w = {k: 1.0 / len(style_layers) for k in style_layers}

    for i in range(1, steps + 1):
        opt.zero_grad()
        c_out, s_out = extractor(gen)

        # Content loss
        c_loss = 0.0
        for k in content_layers:
            c_loss += nn.functional.mse_loss(c_out[k], c_tgt[k])

        # Style loss
        s_loss = 0.0
        for k in style_layers:
            G = gram_matrix(s_out[k])
            s_loss += style_layer_w[k] * nn.functional.mse_loss(G, G_tgt[k])

        # TV loss
        tv_loss = total_variation(gen)

        # مجموع
        loss = content_w * c_loss + style_w * s_loss + tv_w * tv_loss
        loss.backward()
        opt.step()

        if i % 50 == 0:
            yield i, loss.item(), c_loss.item(), s_loss.item(), tv_loss.item(), gen.detach()

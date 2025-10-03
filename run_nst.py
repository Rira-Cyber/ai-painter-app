import torch, os, glob
from nst.utils import load_image, tensor_to_pil
from nst.nst_core import run_nst, VGGFeatures, gram_matrix
from nst.presets import CONTENT_LAYERS, STYLE_LAYERS, CONTENT_WEIGHT, STYLE_WEIGHT, TV_WEIGHT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- لود کانتنت
content_t, _ = load_image("data/content/drawing.png", max_dim=720, device=DEVICE)

# --- انتخاب استایل
style_name = "van_gogh"   # میتونی تغییر بدی
style_dir  = f"data/styles/{style_name}"

extractor = VGGFeatures(CONTENT_LAYERS, STYLE_LAYERS).to(DEVICE)

if os.path.isdir(style_dir):
    # پوشه: چند عکس
    style_paths = sorted(glob.glob(os.path.join(style_dir, "*.jpg")))
    with torch.no_grad():
        c_tgt, _ = extractor(content_t)
        G_tgt = {k: 0 for k in STYLE_LAYERS}
        n = 0
        for p in style_paths:
            s_t, _ = load_image(p, max_dim=512, device=DEVICE)
            _, s_feats = extractor(s_t)
            for k in STYLE_LAYERS:
                G_tgt[k] += gram_matrix(s_feats[k])
            n += 1
        for k in STYLE_LAYERS:
            G_tgt[k] /= max(n, 1)

    custom_targets = (c_tgt, G_tgt)
    style_img = None
else:
    # فایل تکی
    style_img, _ = load_image(f"data/styles/{style_name}.jpg", max_dim=512, device=DEVICE)
    custom_targets = None

# --- اجرای NST
for step, total, cl, sl, tv, gen in run_nst(
        content_img=content_t,
        style_img=style_img,
        content_layers=CONTENT_LAYERS, style_layers=STYLE_LAYERS,
        content_w=CONTENT_WEIGHT, style_w=STYLE_WEIGHT, tv_w=TV_WEIGHT,
        steps=400, lr=0.07, device=DEVICE,
        custom_targets=custom_targets):
    if step % 100 == 0:
        img = tensor_to_pil(gen)
        img.save(f"out/step_{step}.png")
        print(f"[{step}] total={total:.3f} content={cl:.3f} style={sl:.3f} tv={tv:.6f}")

# --- خروجی نهایی
tensor_to_pil(gen).save("out/final.png")
print("Saved -> out/final.png")

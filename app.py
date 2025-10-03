import cv2
import mediapipe as mp
import numpy as np
import gradio as gr
import torch, os, glob
from nst.utils import load_image, tensor_to_pil
from nst.nst_core import run_nst, VGGFeatures, gram_matrix
from nst.presets import CONTENT_LAYERS, STYLE_LAYERS, CONTENT_WEIGHT, STYLE_WEIGHT, TV_WEIGHT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# 🎨 پالت رنگی
PALETTE = [
    ((0, 0, 255), (10, 10, 90, 60)),       # قرمز
    ((0, 255, 0), (110, 10, 190, 60)),     # سبز
    ((255, 0, 0), (210, 10, 290, 60)),     # آبی
    ((0, 255, 255), (310, 10, 390, 60)),   # زرد
    ((255, 255, 255), (410, 10, 490, 60))  # سفید
]

# ✋ تشخیص انگشت‌ها
def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for tip_id in tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# 🖌 تابع نقاشی روی وب‌کم
def hand_paint(video):
    cap = cv2.VideoCapture(video)
    cap.set(3, 1280)
    cap.set(4, 720)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    color = (0, 0, 255)
    prev_point = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        for color_box, (x1, y1, x2, y2) in PALETTE:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), color_box, -1)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                lm = handLms.landmark
                Cx = int(lm[8].x * w)
                Cy = int(lm[8].y * h)

                fingers = fingers_up(handLms)
                cv2.circle(frame, (Cx, Cy), 10, (0, 0, 255), -1)

                if Cy < 70:
                    for color_box, (x1, y1, x2, y2) in PALETTE:
                        if x1 < Cx < x2 and y1 < Cy < y2:
                            color = color_box

                elif fingers[1] == 1 and fingers[2] == 0:
                    if prev_point is not None:
                        cv2.line(canvas, prev_point, (Cx, Cy), color, 5)
                    prev_point = (Cx, Cy)

                elif fingers[1] == 1 and fingers[2] == 1:
                    if prev_point is not None:
                        cv2.line(canvas, prev_point, (Cx, Cy), (0, 0, 0), 50)
                    prev_point = (Cx, Cy)
                else:
                    prev_point = None

                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        else:
            prev_point = None

        combo = cv2.add(frame, canvas)
        cv2.imshow("Drawing", combo)

        key = cv2.waitKey(1)
        if key == ord('s'):  # ذخیره
            white_bg = np.ones_like(canvas) * 255
            mask = np.any(canvas != 0, axis=2)
            white_bg[mask] = canvas[mask]
            cv2.imwrite("data/content/drawing.png", white_bg)
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "data/content/drawing.png"

# 🖼 انتقال سبک
STYLE_DIR = "data/styles"
STYLE_CHOICES = [d for d in os.listdir(STYLE_DIR)]

def stylize(content_img, style_name):
    if content_img is None:
        return None
    content_t, _ = load_image(content_img, max_dim=720, device=DEVICE)
    extractor = VGGFeatures(CONTENT_LAYERS, STYLE_LAYERS).to(DEVICE)
    style_dir = os.path.join(STYLE_DIR, style_name)

    if os.path.isdir(style_dir):
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
        style_img, _ = load_image(style_dir, max_dim=512, device=DEVICE)
        custom_targets = None

    final_img = None
    for step, total, cl, sl, tv, gen in run_nst(
            content_img=content_t, style_img=style_img,
            content_layers=CONTENT_LAYERS, style_layers=STYLE_LAYERS,
            content_w=CONTENT_WEIGHT, style_w=STYLE_WEIGHT, tv_w=TV_WEIGHT,
            steps=200, lr=0.07, device=DEVICE,
            custom_targets=custom_targets):
        if step == 200:
            final_img = tensor_to_pil(gen)

    return final_img


with gr.Blocks() as demo:
    gr.Markdown("# 🎨 AI Painter - Hand Drawing + Neural Style Transfer")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ✋ بخش نقاشی با دست (وب‌کم)")
            webcam_btn = gr.Button("Start Drawing (press 's' to save)")
            output_path = gr.Textbox(label="Saved Drawing Path")

        with gr.Column():
            gr.Markdown("### 🎨 بخش انتقال سبک")
            style_choice = gr.Dropdown(choices=STYLE_CHOICES, label="Select style")
            img_in = gr.Image(type="filepath", label="Drawing (upload or use saved path)")
            img_out = gr.Image(label="Stylized Result")
            run_btn = gr.Button("Apply Style")

    def run_webcam():
        return hand_paint(0)

    webcam_btn.click(fn=run_webcam, outputs=output_path)
    run_btn.click(fn=stylize, inputs=[img_in, style_choice], outputs=img_out)

if __name__ == "__main__":
    demo.launch(share=True)

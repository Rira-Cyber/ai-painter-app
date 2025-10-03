import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 🎨 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# 🎨 پلت رنگی
PALETTE = [
    ((0, 0, 255), (10, 10, 90, 60)),       # قرمز
    ((0, 255, 0), (110, 10, 190, 60)),     # سبز
    ((255, 0, 0), (210, 10, 290, 60)),     # آبی
    ((0, 255, 255), (310, 10, 390, 60)),   # زرد
    ((255, 255, 255), (410, 10, 490, 60))  # سفید
]

# ✋ چک کردن انگشت‌ها
def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # شست
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # بقیه انگشت‌ها
    for tip_id in tips_ids[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


# 🎥 کلاس برای پردازش ویدئو
class HandPainter(VideoTransformerBase):
    def __init__(self):
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.color = (0, 0, 255)
        self.prev_point = None

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # کشیدن پلت رنگی
        for color_box, (x1, y1, x2, y2) in PALETTE:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), color_box, -1)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                lm = handLms.landmark
                Cx = int(lm[8].x * w)   # نوک انگشت اشاره
                Cy = int(lm[8].y * h)

                fingers = fingers_up(handLms)

                # نشانه‌گذاری انگشت
                cv2.circle(frame, (Cx, Cy), 10, (0, 0, 255), -1)

                # انتخاب رنگ
                if Cy < 70:
                    for color_box, (x1, y1, x2, y2) in PALETTE:
                        if x1 < Cx < x2 and y1 < Cy < y2:
                            self.color = color_box

                # مد نقاشی (فقط انگشت اشاره بالا)
                elif fingers[1] == 1 and fingers[2] == 0:
                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, (Cx, Cy), self.color, 5)
                    self.prev_point = (Cx, Cy)

                # مد پاک‌کن (اشاره + وسط بالا)
                elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, (Cx, Cy), (0, 0, 0), 50)
                    self.prev_point = (Cx, Cy)
                else:
                    self.prev_point = None

                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
        else:
            self.prev_point = None

        combo = cv2.add(frame, self.canvas)
        return combo

    def save_canvas(self, path="drawing.png"):
        white_bg = np.ones_like(self.canvas) * 255
        mask = np.any(self.canvas != 0, axis=2)
        white_bg[mask] = self.canvas[mask]
        cv2.imwrite(path, white_bg)
        return path


# 🌐 رابط کاربری Streamlit
st.title("🎨 Hand Gesture Painter (AI Painter)")
st.markdown("با انگشت اشاره نقاشی بکش، با اشاره+وسط پاک کن، بالای صفحه رنگ انتخاب کن.")

webrtc_ctx = webrtc_streamer(
    key="painter",
    video_transformer_factory=HandPainter,
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_transformer:
    if st.button("💾 ذخیره نقاشی"):
        path = webrtc_ctx.video_transformer.save_canvas()
        st.image(path, caption="Your Drawing", use_column_width=True)

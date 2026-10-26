
import cv2
import mediapipe as mp
import gradio as gr

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]


def count_fingers(hand_landmarks, handedness):
    lm = hand_landmarks.landmark
    count = 0

    # Thumb: compare horizontal positions
    if handedness == "Right":
        if lm[4].x < lm[3].x:
            count += 1
    else:
        if lm[4].x > lm[3].x:
            count += 1

    # Other four fingers
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        if lm[tip].y < lm[pip].y:
            count += 1

    return count


def detect_hands(image):
    if image is None:
        return None, "Please enable your webcam."

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=4,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

    counts = []

    if results.multi_hand_landmarks:
        for hand_lm, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            label = hand_info.classification[0].label
            count = count_fingers(hand_lm, label)

            mp_draw.draw_landmarks(
                frame,
                hand_lm,
                mp_hands.HAND_CONNECTIONS
            )

            wrist = hand_lm.landmark[0]
            x = int(wrist.x * width)
            y = int(wrist.y * height)

            cv2.putText(
                frame, f"{label}: {count}",
                (max(10, x - 30), min(height - 10, y + 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2
            )

            counts.append({
                "label": label,
                "count": count,
                "x": x
            })

    counts.sort(key=lambda hand: hand["x"])

    if not counts:
        message = "No hand detected | Fingers: 0"
    elif len(counts) == 1:
        message = f"Fingers: {counts[0]['count']}"
    else:
        main_count = counts[0]["count"]
        extra_total = sum(hand["count"] for hand in counts[1:])
        net_count = max(0, main_count - extra_total)

        message = (
            f"Main ({main_count}) - Extra ({extra_total}) "
            f"= Net fingers: {net_count}"
        )

    cv2.putText(
        frame, message[:65], (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (0, 255, 128), 2
    )

    output = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return output, message


demo = gr.Interface(
    fn=detect_hands,
    inputs=gr.Image(
        sources=["webcam"],
        type="numpy",
        label="Webcam"
    ),
    outputs=[
        gr.Image(label="Hand Skeleton Detection"),
        gr.Textbox(label="Finger Count Result")
    ],
    title="Hand Finger Detection",
    description=(
        "Detect fingers, draw hand skeleton lines, "
        "and calculate net finger count."
    ),
    live=True
)

if __name__ == "__main__":
    import os

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)

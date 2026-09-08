import cv2
import mediapipe as mp
import numpy as np
import joblib


# REAL-TIME SIGN LANGUAGE RECOGNITION

MODEL_FILE = "sign_model.pkl"


# LOAD MODEL

print()
print("=" * 60)
print("       SIGN LANGUAGE RECOGNITION")
print("=" * 60)

print()
print("Loading trained model...")


try:

    model_data = joblib.load(
        MODEL_FILE
    )

except FileNotFoundError:

    print()
    print("ERROR: sign_model.pkl not found.")
    print()
    print("First run:")
    print("python collect_data.py")
    print("python train_model.py")
    print()

    raise SystemExit


model = model_data["model"]
encoder = model_data["encoder"]


print("Model loaded successfully!")


# MEDIAPIPE

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# FEATURE EXTRACTION

def extract_landmarks(hand_landmarks):

    landmarks = []

    wrist = hand_landmarks.landmark[0]

    for landmark in hand_landmarks.landmark:

        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        landmarks.extend([
            x,
            y,
            z
        ])


    landmarks = np.array(
        landmarks,
        dtype=np.float32
    )


    max_value = np.max(
        np.abs(landmarks)
    )


    if max_value > 0:

        landmarks = (
            landmarks / max_value
        )


    return landmarks


# OPEN WEBCAM

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print()
    print("ERROR: Cannot open webcam.")
    raise SystemExit


# VARIABLES


prediction_text = "No hand"

confidence = 0.0


# MAIN LOOP

while True:

    success, frame = camera.read()


    if not success:

        print(
            "ERROR: Cannot read camera frame."
        )

        break


    # Mirror image

    frame = cv2.flip(
        frame,
        1
    )


    # Convert BGR → RGB

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Process hand

    results = hands.process(
        rgb
    )


    # HAND DETECTED

    if results.multi_hand_landmarks:

        hand_landmarks = (
            results.multi_hand_landmarks[0]
        )


        # DRAW LANDMARKS

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )


        # EXTRACT FEATURES

        features = extract_landmarks(
            hand_landmarks
        )


        features = features.reshape(
            1,
            -1
        )


        # PREDICTION

        prediction = model.predict(
            features
        )


        probabilities = (
            model.predict_proba(
                features
            )
        )


        predicted_index = (
            prediction[0]
        )


        prediction_text = (
            encoder.inverse_transform(
                [predicted_index]
            )[0]
        )


        confidence = (
            np.max(probabilities[0])
            * 100
        )


    else:

        prediction_text = "No hand"

        confidence = 0.0

    # UI

    cv2.rectangle(
        frame,
        (0, 0),
        (700, 120),
        (0, 0, 0),
        -1
    )


    # Title

    cv2.putText(
        frame,
        "SIGN LANGUAGE RECOGNITION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # Prediction

    cv2.putText(
        frame,
        f"Sign: {prediction_text}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )


    # Confidence

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}%",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # DISPLAY
    
    
    cv2.imshow(
        "Sign Language Recognition",
        frame
    )


    # KEYBOARD

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q") or key == 27:

        break


# CLEANUP

camera.release()

cv2.destroyAllWindows()

hands.close()

print()
print("Application closed.")

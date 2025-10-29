import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

 
# CONFIGURATION
 
SMILE_MODEL_PATH = "smile_cnn_model.h5"   
AGE_MODEL_PATH = "age_cnn_model.h5"       
SMILE_THRESHOLD = 0.3                      
CAPTURE_DELAY = 1.2                       # Time to hold smile before capture
IMG_SIZE = 64                             # Image input size for both models
SAVE_PATH = "smile_capture.jpg"           # Output filename

 
# INITIALIZATION
 
print("Initializing Smile & Age Detector System...\n")

print("Loading smile detection model...")
smile_model = load_model(SMILE_MODEL_PATH)

print("Loading age prediction model...")
age_model = load_model(AGE_MODEL_PATH, compile=False)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open webcam.")
    exit()

smile_captured = False
smile_start_time = None

print("✅ Initialization complete. Press 'q' to quit anytime.\n")

# ===========================
# MAIN LOOP
# ===========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera frame not available.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_crop = frame[y:y + h, x:x + w]
        if face_crop.size == 0:
            continue

        # --- Preprocess for smile model ---
        smile_face = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
        smile_face = cv2.cvtColor(smile_face, cv2.COLOR_BGR2RGB)
        smile_face = img_to_array(smile_face) / 255.0
        smile_face = np.expand_dims(smile_face, axis=0)

        smile_prob = float(smile_model.predict(smile_face, verbose=0)[0][0])

        # --- Preprocess for age model ---
        age_face = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
        age_face = cv2.cvtColor(age_face, cv2.COLOR_BGR2RGB)
        age_face = img_to_array(age_face) / 255.0
        age_face = np.expand_dims(age_face, axis=0)

        predicted_age = float(age_model.predict(age_face, verbose=0)[0][0])

        # --- Draw detection info ---
        color = (0, 255, 0) if smile_prob >= SMILE_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Smile: {smile_prob:.2f}", (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Age: {predicted_age:.1f} yrs", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # --- Smile detection logic ---
        if smile_prob >= SMILE_THRESHOLD:
            if smile_start_time is None:
                smile_start_time = time.time()
                print("😄 Full smile detected! Hold it for a moment...")
            else:
                elapsed = time.time() - smile_start_time
                if elapsed >= CAPTURE_DELAY:
                    smile_captured = True

                    # Save the annotated frame (with rectangle & age)
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, "Smile Captured!", (x, y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    cv2.imwrite(SAVE_PATH, annotated_frame)
                    print(f"📸 Captured photo after {CAPTURE_DELAY:.1f}s — saved as '{SAVE_PATH}'")
                    print(f"Predicted Age: {predicted_age:.1f} years")

                    cv2.imshow("Captured Photo", annotated_frame)
                    cv2.waitKey(2500)
                    break
        else:
            smile_start_time = None  # Reset if no smile

    # Display live frame
    cv2.imshow("Smile & Age Detector", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q') or smile_captured:
        break

# ===========================
# CLEANUP
# ===========================
cap.release()
cv2.destroyAllWindows()
print("👋 Session ended gracefully.")

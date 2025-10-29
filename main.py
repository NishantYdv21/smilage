 
import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from insightface.app import FaceAnalysis

 
MODEL_PATH = "smile_cnn_model.h5"   
FACE_DET_SIZE = (640, 640)
SMILE_THRESHOLD = 0.3            
CAPTURE_DELAY = 1.2               

 
print("Initializing system...")

 
print("Loading smile detection CNN model...")
smile_model = load_model(MODEL_PATH)

# Initialize InsightFace for age estimation
print("Loading InsightFace model...")
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)

# Use InsightFace’s detector instead of Haar
cap = cv2.VideoCapture(0)
smile_captured = False
smile_start_time = None

print("✅ Initialization complete. Press 'q' to quit anytime.\n")

 
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera frame not available.")
        break

   
    faces = face_app.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        
        face_resized = cv2.resize(face_crop, (64, 64))      # Match training size
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_resized = img_to_array(face_resized) / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        # Predict smile probability
        smile_prob = float(smile_model.predict(face_resized, verbose=0)[0][0])

        
        if smile_prob >= SMILE_THRESHOLD:
            if smile_start_time is None:
                smile_start_time = time.time()
                print("😄 Full smile detected! Hold it for a moment...")
            else:
                elapsed = time.time() - smile_start_time
                if elapsed >= CAPTURE_DELAY:
                    smile_captured = True
                    img_name = "smile_capture.jpg"
                    cv2.imwrite(img_name, frame)
                    print(f"📸 Captured photo after {CAPTURE_DELAY:.1f}s — saved as {img_name}")

                    # Predict age
                    results = face_app.get(frame)
                    if results:
                        ages = [round(f.age, 1) for f in results]
                        avg_age = sum(ages) / len(ages)
                        print(f"Predicted Age: {avg_age:.1f} years")

                        cv2.putText(frame, f"Age: {avg_age:.1f} yrs", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.imshow("Smile & Age Detector", frame)
                        cv2.waitKey(2500)
                    else:
                        print("⚠️ No face detected for age estimation.")
                    break
        else:
            smile_start_time = None  # reset timer if not smiling

        # Draw face box and probability
        color = (0, 255, 0) if smile_prob >= SMILE_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Smile: {smile_prob:.2f}", (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display
    cv2.imshow("Smile & Age Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or smile_captured:
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Session ended gracefully.")

import cv2
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
MODEL_PATH = "custom_model.tflite" # Using your new custom model
LABELS_PATH = "labels.txt"
IMG_SIZE = 128 # Must match training size!
CONFIDENCE_THRESHOLD = 0.4 # Lower threshold for custom models

def main():
    # 1. Load Labels
    try:
        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"✅ Loaded Labels: {labels}")
    except:
        print("❌ Error: labels.txt not found.")
        return

    # 2. Load TFLite Model
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    except Exception as e:
        print(f"❌ Model Error: {e}")
        return

    # 3. Load Face Detector (Haar Cascade)
    # This uses OpenCV's built-in face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    
    # Drone State Variables
    mode = "GESTURE" # Can be 'GESTURE' or 'TRACKING'

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Mirror view
        h, w, _ = frame.shape
        center_x = w // 2

        # --- GESTURE RECOGNITION STEP ---
        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0 # Normalize 0-1

        # Predict
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        idx = np.argmax(output_data[0])
        confidence = output_data[0][idx]
        gesture = labels[idx]

        # --- LOGIC CONTROLLER ---
        command = "HOVER"
        color = (255, 0, 0)
        status_msg = "Looking for pilot..."

        # Priority 1: Hand Gesture Detected?
        if confidence > CONFIDENCE_THRESHOLD and gesture != "none":
            mode = "GESTURE CONTROL"
            color = (0, 255, 0) # Green for Active Control
            
            if "up" in gesture: command = "ASCEND"
            elif "down" in gesture: command = "DESCEND"
            elif "fist" in gesture: command = "HOVER / HOLD"
            elif "flip" in gesture: command = "DO A FLIP"
            elif "palm" in gesture: command = "LAND"
            
            status_msg = f"Gesture: {gesture.upper()} ({int(confidence*100)}%)"

        # Priority 2: No Hand? Try Face Tracking
        else:
            mode = "FACE TRACKING"
            color = (0, 255, 255) # Yellow for Tracking
            
            # Detect Faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                # Track the first face found
                (x, y, fw, fh) = faces[0]
                face_center = x + (fw // 2)
                
                # Draw Box around face
                cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 255), 2)
                
                # Calculate Error (How far from center?)
                error = face_center - center_x
                
                # Deadband (Don't move if slightly off-center)
                if error < -50:
                    command = "ROTATE LEFT <<"
                elif error > 50:
                    command = "ROTATE RIGHT >>"
                else:
                    command = "LOCKED ON (CENTER)"
                    color = (0, 255, 0)
                
                status_msg = f"Tracking Face (Error: {error})"
            else:
                command = "WAITING"
                status_msg = "No Face or Hand Detected"
                color = (100, 100, 100)

        # --- UI DISPLAY ---
        # Top Bar (Command)
        cv2.rectangle(frame, (0, 0), (w, 60), (20, 20, 20), -1)
        cv2.putText(frame, f"MODE: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, command, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Bottom Bar (Debug)
        cv2.putText(frame, status_msg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw Center Line for Tracking
        if mode == "FACE TRACKING":
            cv2.line(frame, (center_x, 0), (center_x, h), (100, 100, 100), 1)

        cv2.imshow("Smart Drone Controller", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
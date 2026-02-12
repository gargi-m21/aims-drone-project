import cv2
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"

def main():
    # 1. Load Labels
    try:
        with open(LABELS_PATH, 'r') as f:
            labels = [line.strip().lower() for line in f.readlines()]
        print(f"✅ Loaded Labels: {labels}")
    except FileNotFoundError:
        print("❌ ERROR: labels.txt not found!")
        return

    # 2. Load Model
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']
        height, width = input_shape[1], input_shape[2]
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return

    # 3. Start Camera
    cap = cv2.VideoCapture(0)
    print("\n--- DRONE CONTROLLER STARTED ---")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        
        # --- PREPROCESSING ---
        # 1. Convert BGR to RGB (The Critical Fix!)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Resize to 224x224
        img = cv2.resize(rgb_frame, (width, height))
        
        # 3. Convert to float32 (Raw 0-255 values)
        # The model has a built-in layer to handle the math (-1 to 1) internally
        img = img.astype(np.float32)
        
        # 4. Add batch dimension
        input_data = np.expand_dims(img, axis=0)
        
        # --- PREDICTION ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get Result
        prediction_idx = np.argmax(output_data[0])
        confidence = output_data[0][prediction_idx] * 100
        gesture = labels[prediction_idx]

        # --- COMMAND LOGIC ---
        command = "WAITING..."
        color = (200, 200, 200)

        # Lowered threshold to 50% for responsiveness
        if confidence > 50: 
            if "up" in gesture:
                command = "ASCEND (Go Up)"
                color = (0, 255, 0)
            elif "down" in gesture:
                command = "DESCEND (Go Down)"
                color = (0, 255, 255)
            elif "flip" in gesture:
                command = "DO A FLIP!"
                color = (255, 0, 255)
            elif "stop" in gesture or "palm" in gesture:
                command = "LAND / STOP"
                color = (0, 0, 255)
            elif "hover" in gesture or "fist" in gesture:
                command = "HOVERING"
                color = (255, 0, 0)
            elif "none" in gesture:
                command = "NO HAND"
                color = (100, 100, 100)

        # --- UI DRAWING ---
        cv2.rectangle(frame, (0,0), (640, 60), (0,0,0), -1)
        cv2.putText(frame, f"CMD: {command}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Debug Info
        cv2.rectangle(frame, (0, 440), (640, 480), (0,0,0), -1)
        info_text = f"AI Sees: {gesture.upper()} ({confidence:.1f}%)"
        cv2.putText(frame, info_text, (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('Drone Controller', frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
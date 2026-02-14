import cv2
import os
import time

# --- CONFIGURATION ---
DATA_DIR = "data"
LABELS = ["up", "down", "flip", "fist", "palm", "none"]
IMG_SIZE = 128  # Matches your new Custom CNN input size
CAPTURE_DELAY = 0.1 # Time between bursts (0.1s = 10 pics/sec)

def create_folders():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    for label in LABELS:
        path = os.path.join(DATA_DIR, label)
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"✅ Created folder: {path}")

def main():
    create_folders()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("\n--- DATA COLLECTION TOOL ---")
    print(f"Labels: {LABELS}")
    print("----------------------------")
    
    # Select Gesture
    while True:
        target_label = input(f"Enter gesture name to collect ({', '.join(LABELS)}): ").lower().strip()
        if target_label in LABELS:
            break
        print("❌ Invalid label! Pick from the list.")

    save_path = os.path.join(DATA_DIR, target_label)
    
    # count existing images to avoid overwriting
    existing_files = os.listdir(save_path)
    count = len(existing_files)
    
    print(f"\n✅ Collecting data for: '{target_label.upper()}'")
    print(f"📂 Current count: {count}")
    print("\n--- CONTROLS ---")
    print("Press 'S' to START/STOP capturing (Burst Mode)")
    print("Press 'Q' to QUIT")
    print("----------------\n")

    capturing = False
    last_capture_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Define Region of Interest (ROI) box
        # We put the box slightly to the right so your face doesn't interfere
        roi_size = 300
        x1, y1 = int(w * 0.55), int(h * 0.2) 
        x2, y2 = x1 + roi_size, y1 + roi_size

        # Draw the box
        color = (0, 255, 0) if capturing else (0, 0, 255) # Green = Recording, Red = Paused
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # UI Text
        cv2.putText(frame, f"Label: {target_label.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Count: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        status_text = "RECORDING..." if capturing else "PAUSED (Press 'S')"
        cv2.putText(frame, status_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- CAPTURE LOGIC ---
        if capturing:
            current_time = time.time()
            if current_time - last_capture_time > CAPTURE_DELAY:
                # 1. Crop the ROI
                roi = frame[y1:y2, x1:x2]
                
                # 2. Resize to training size (128x128)
                roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                
                # 3. Save
                filename = f"{save_path}/{target_label}_{count}.jpg"
                cv2.imwrite(filename, roi_resized)
                
                count += 1
                last_capture_time = current_time
                print(f"📸 Captured {count}: {filename}")

        cv2.imshow("Data Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            capturing = not capturing # Toggle start/stop

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Collection finished.")

if __name__ == "__main__":
    main()
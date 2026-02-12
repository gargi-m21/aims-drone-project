import cv2
import os

# --- CONFIGURATION (LOCKED TO SCRIPT LOCATION) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data") 

def create_dirs(label):
    path = os.path.join(DATA_DIR, label)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def capture():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # 1. Ask what we are capturing
    print(f"\nSaving images to: {DATA_DIR}")
    label = input("Enter gesture name (e.g., up, down, fist): ").strip().lower()
    if not label: return
    
    save_path = create_dirs(label)
    
    # Check how many we already have
    existing_files = [f for f in os.listdir(save_path) if f.endswith('.jpg')]
    count = len(existing_files)
    
    print(f"\n--- RECORDING: {label.upper()} ---")
    print(f"Current count: {count}")
    print("Press 'c' to Capture | 'q' to Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        display = frame.copy()
        
        # UI - Green box guide
        cv2.putText(display, f"Saved: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Mode: {label.upper()}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.rectangle(display, (100, 100), (400, 400), (0, 255, 0), 2) 
        
        cv2.imshow("Capture", display)
        
        key = cv2.waitKey(1)
        if key == ord('c'):
            # Save the image inside the green box
            crop = frame[100:400, 100:400] 
            filename = os.path.join(save_path, f"{label}_{count}.jpg")
            cv2.imwrite(filename, crop)
            print(f"Saved {filename}")
            count += 1
        
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    capture()
# 🚁 AI Hybrid Drone Controller (Gesture + Face Tracking)

A smart drone control system that uses a **Custom Convolutional Neural Network (CNN)** for hand gesture recognition and OpenCV's **Haar Cascades** for automatic face tracking. The system intelligently switches modes: it obeys hand commands when a gesture is seen and switches to "Follow Me" mode when hands are not detected.

## 🎥 Project Demonstration
**[Click here to watch the project demonstration video](https://drive.google.com/file/d/1it5YOlV8ufOdd9ykmRD7WZrZ2Fe9gCh-/view?usp=drive_link)**

## 🚀 Key Features
* **🧠 Custom CNN Architecture:** 🧠 Built and trained a deep learning model from scratch (4 Conv Blocks, 256 filters) achieving 93% training accuracy and ~89% validation accuracy on a custom dataset.
* **🔄 Hybrid Control Logic:**
    * **Priority Mode:** Hand Gesture Control (Ascend, Descend, Flip, Land).
    * **Fallback Mode:** Face Tracking (The drone rotates to keep the pilot in the center of the frame).
* **💪 Robust Data Pipeline:** Trained on 2,400+ images with **Data Augmentation** (Zoom, Rotation, Brightness shifts) to handle varying lighting conditions.
* **⚡ Edge Optimization:** The heavy Keras model was quantized and converted to **TensorFlow Lite (TFLite)** for high-speed, low-latency inference on standard CPUs.

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Deep Learning:** TensorFlow / Keras (Sequential API)
* **Computer Vision:** OpenCV (`cv2`)
* **Model Format:** `.tflite` (Quantized)
* **Hardware:** Standard Webcam & Laptop CPU

## 🎮 Controls & Modes

### Mode 1: Gesture Control (Priority)
Active when a hand is detected with >40% confidence.

| Gesture | Drone Command | Description |
| :--- | :--- | :--- |
| **Index Up** ☝️ | `ASCEND` | Fly Up |
| **Victory/V** ✌️ | `DESCEND` | Fly Down |
| **Fist** ✊ | `HOVER` | Hold Position |
| **Palm** ✋ | `LAND` | Emergency Land |
| **Thumbs Up** 👍 | `FLIP` | Do a Flip |

### Mode 2: Face Tracking (Fallback)
Active when no hand is detected. The drone automates **Yaw (Rotation)** to follow the pilot.
* **Face Left:** Drone rotates Left `<<`
* **Face Right:** Drone rotates Right `>>`
* **Face Center:** Drone Locks On `(LOCKED)`

## 📂 Project Structure

* `main.py`: The real-time controller. Handles webcam stream, TFLite inference, and mode switching logic.
* `train.py`: The model builder. Defines the Custom CNN, performs data augmentation, trains the model, and converts it to TFLite.
* `dataset.py`: A utility tool for capturing custom datasets with "Burst Mode" and ROI cropping.
* `custom_model.tflite`: The trained quantized model.
* `labels.txt`: Class names mapping.

## ⚙️ How to Run

1.  **Install Dependencies**
    ```bash
    pip install opencv-python tensorflow numpy
    ```

2.  **Run the Controller**
    ```bash
    python main.py
    ```

# 🎭 Face Recognition using OpenCV (Haar Cascade + LBPH)

This project implements a **face detection and recognition system** using **OpenCV**.
It uses:

Haar Cascade Classifier for face detection
LBPH (Local Binary Patterns Histograms) for face recognition

The system can detect faces from input images and identify them based on a trained dataset of images.

---Features

*  Detects faces using Haar Cascade
*  Trains a model on multiple labeled datasets (e.g., actors, heroines, friends)
*  Recognizes faces in new input images
*  Displays bounding boxes and labels on detected faces
*  Confidence-based filtering to reduce false matches

---Tech Stack

* Python 3
* OpenCV (cv2)
* NumPy
* Haar Cascade XML file

 
 ----How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/face-recognition-opencv.git
   cd face-recognition-opencv
   ```

2. Install dependencies:

   ```bash
   pip install opencv-python numpy
   ```

3. Prepare dataset:

   * Inside `person_images/`, create a subfolder for each person.
   * Add multiple face images inside each folder.

4. Run the training + recognition:

   ```bash
   python main.py
   ```

---

# Future Improvements

* Add **real-time webcam support**.
* Deploy as a **Flask/Django web app**.

#  Acknowledgements

* OpenCV Documentation: [https://docs.opencv.org/]
* Haar Cascade XML provided by OpenCV.


Do you want me to also add a **“Demo Video/GIF section”** in the README, so people can quickly see how your project works without running it?

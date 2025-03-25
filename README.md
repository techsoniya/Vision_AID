# Vision_AID
Smart Glasses using Artificial Intelligence for Enhanced Safety and Mobility of the Visually Impaired.

# 🔍 Real-Time Object Detection with Audio Output  

## 🎯 Overview  
This project is a real-time object detection system that identifies objects using a laptop camera, labels them, and provides voice output with a delay of 3 seconds. It utilizes a pre-trained model (`ssd_mobilenet_v2`) from TensorFlow Hub to detect objects efficiently.  

## 🚀 Features  
- Real-time object detection through the laptop camera  
- Identifies and labels detected objects  
- Provides voice output for recognized objects  
- 3-second delay before announcing the detected object  
- User-friendly interface  

## 🛠 Tech Stack  
- **Programming Language:** Python  
- **Frameworks & Libraries:** TensorFlow, OpenCV, gTTS (Google Text-to-Speech)  
- **Tools:** VSCode, Jupyter Notebook  

## 🔧 Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/object-detection-voice.git
cd object-detection-voice
### 2️⃣ Install Dependencies
bash
Copy
Edit
pip install tensorflow opencv-python gtts
### 3️⃣ Download the Pre-trained Model
Ensure you have the ssd_mobilenet_v2 model from TensorFlow Hub. The script will handle loading it.

### ▶️ How to Run
Run the script to start object detection:

bash
Copy
Edit
python main.py
The camera will activate, and detected objects will be announced with a voice output every 3 seconds.

### 📷 Demo

Screenshots of the detection system in action.

### 🤝 Contributions & License
Contributions are welcome! Feel free to open issues or submit pull requests.


⭐ If you found this project helpful, give it a star! ⭐

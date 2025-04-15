# ComputerVision
README – Sign Language Recognition

This project uses MediaPipe and TensorFlow to recognize static hand signs from webcam input. It consists of three main scripts:

1. collect_data.py  
   Used to collect training data using MediaPipe hand landmarks.  
   - Run the script: python collect_data.py  
   - A webcam window will open.  
   - Hold a hand sign in the frame and press a letter key (A–Z) to save it.  
   - Press Q to quit.  
   - Saved data is stored in keypoints_dataset.csv.

2. train_model.py  
   Trains a neural network on the collected data.  
   - Ensure keypoints_dataset.csv exists.  
   - Run the script: python train_model.py  
   - The trained model will be saved as keypoint_model.keras.  
   - The label encoder will be saved as label_encoder.pkl.

3. main.py  
   Runs the real-time sign recognition system.  
   - Requires keypoint_model.keras and label_encoder.pkl in the same folder.  
   - Run the script: python run_sign_recognizer.py  
   - A webcam window will open and display predicted letters.  
   - Press Q to quit.

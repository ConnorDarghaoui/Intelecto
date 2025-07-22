import pickle
import cv2
import paddlehub as hub
import numpy as np

class SignDetector:
    def __init__(self, model_path='models/model.p'):
        """
        Initializes the SignDetector model using PaddleHub.
        """
        # --- 1. Load the Hand Pose Detection Model from PaddleHub ---
        try:
            self.hand_detector = hub.Module(name="hand_pose_localization")
        except Exception as e:
            print(f"Error loading PaddleHub module: {e}")
            print("Please ensure you have an internet connection and paddlehub is installed correctly.")
            self.hand_detector = None

        # --- 2. Load our custom-trained classifier model ---
        self.model = None
        self.model_loaded = False
        try:
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
            self.model = model_dict['model']
            self.model_loaded = True
            print("Custom sign classifier model loaded successfully.")
        except FileNotFoundError:
            print(f"Warning: Classifier model not found at {model_path}. App will run without prediction.")
            print("Please train a model using the 'Administration' tab.")
        except Exception as e:
            print(f"Error loading classifier model: {e}")

        # --- 3. Define the labels dictionary ---
        self.labels_dict = {0: 'Peligro', 1: 'Todo_OK'}

    def _draw_landmarks(self, frame, landmarks):
        """
        Helper function to draw landmarks and connections on the frame.
        """
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8), # Index finger
            (5, 9), (9, 10), (10, 11), (11, 12), # Middle finger
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky finger and palm
        ]
        
        h, w, _ = frame.shape
        points = [(int(lm[0]), int(lm[1])) for lm in landmarks]

        for conn in connections:
            p1 = points[conn[0]]
            p2 = points[conn[1]]
            if 0 < p1[0] < w and 0 < p1[1] < h and 0 < p2[0] < w and 0 < p2[1] < h:
                 cv2.line(frame, p1, p2, (255, 255, 0), 2)

        for point in points:
            if 0 < point[0] < w and 0 < point[1] < h:
                cv2.circle(frame, point, 5, (0, 0, 255), -1)
        
        return points

    def detect_sign(self, frame):
        """
        Detects a sign in a given frame using PaddleHub and our custom classifier.
        """
        predicted_character = "N/A"
        similarity = 0

        if not self.hand_detector:
            cv2.putText(frame, "PaddleHub model failed to load", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, predicted_character, similarity

        results = self.hand_detector.predict(images=[frame])
        
        if results and results[0]['data']:
            landmarks = results[0]['data']
            points = self._draw_landmarks(frame, landmarks)

            data_aux = []
            x_ = [p[0] for p in points]
            y_ = [p[1] for p in points]

            min_x, min_y = min(x_), min(y_)
            for i in range(len(points)):
                data_aux.append(points[i][0] - min_x)
                data_aux.append(points[i][1] - min_y)

            if self.model_loaded and self.model:
                if len(data_aux) == self.model.n_features_in_:
                    prediction = self.model.predict([np.asarray(data_aux)])
                    predicted_character = self.labels_dict.get(int(prediction[0]), "Unknown")
                    try:
                        similarity = self.model.predict_proba([np.asarray(data_aux)])[0][int(prediction[0])] * 100
                    except AttributeError:
                        similarity = 100
                else:
                    predicted_character = "Data Mismatch"
            else:
                predicted_character = "Model not trained"

            bbox_x1, bbox_y1 = min(x_), min(y_)
            cv2.putText(frame, predicted_character, (bbox_x1, bbox_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            if self.model_loaded:
                similitud_texto = f"{similarity:.2f}%"
                cv2.putText(frame, similitud_texto, (bbox_x1, bbox_y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        return frame, predicted_character, similarity
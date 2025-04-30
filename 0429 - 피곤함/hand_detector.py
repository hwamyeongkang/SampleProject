"""
Hand detection and analysis module.
Handles hand detection, landmark extraction, and hand-face interaction metrics.
"""

import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    def __init__(self):
        """손 감지 클래스 초기화"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_hands(self, frame):
        """프레임에서 손 감지 및 분석"""
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            hand_results = self.hands.process(rgb_frame)
            hand_data = self.extract_hand_features(frame, hand_results)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                return True, frame, hand_data
            else:
                return False, frame, hand_data
        except Exception as e:
            print(f"MediaPipe 손 처리 중 오류: {e}")
            return False, frame, self.create_empty_hand_data()

    def extract_hand_features(self, frame, hand_results):
        """손 특징 추출"""
        h, w, _ = frame.shape
        hand1_landmarks = []
        hand2_landmarks = []
        hand_metrics = {
            'hand_face_distance': 0,
            'hand_jaw_overlap': 0,
            'hand_detected': 0,
            'hands_near_face': False  # 얼굴 근처 여부 추가
        }

        try:
            if hand_results.multi_hand_landmarks:
                hand_metrics['hand_detected'] = len(hand_results.multi_hand_landmarks)
                
                hand_centers = []
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    cx = np.mean([lm.x for lm in hand_landmarks.landmark])
                    cy = np.mean([lm.y for lm in hand_landmarks.landmark])
                    hand_centers.append((cx, cy))

                # 얼굴 중앙을 (0.5, 0.5)로 가정
                hands_near = any(np.linalg.norm(np.array((cx, cy)) - np.array((0.5, 0.5))) < 0.2 for cx, cy in hand_centers)
                hand_metrics['hands_near_face'] = hands_near

                if len(hand_results.multi_hand_landmarks) > 0:
                    hand1 = hand_results.multi_hand_landmarks[0]
                    for i in range(21):
                        lm = hand1.landmark[i]
                        x, y = int(lm.x * w), int(lm.y * h)
                        hand1_landmarks.extend([x, y])

                if len(hand_results.multi_hand_landmarks) > 1:
                    hand2 = hand_results.multi_hand_landmarks[1]
                    for i in range(21):
                        lm = hand2.landmark[i]
                        x, y = int(lm.x * w), int(lm.y * h)
                        hand2_landmarks.extend([x, y])
                else:
                    hand2_landmarks = [0] * 42
            else:
                hand1_landmarks = [0] * 42
                hand2_landmarks = [0] * 42
        except Exception as e:
            print(f"손 특징 추출 중 오류: {e}")
            hand1_landmarks = [0] * 42
            hand2_landmarks = [0] * 42

        return {
            "hand1_landmarks": hand1_landmarks,
            "hand2_landmarks": hand2_landmarks,
            "hand_metrics": hand_metrics
        }

    def create_empty_hand_data(self):
        """빈 손 데이터 생성"""
        hand1_landmarks = [0] * 42
        hand2_landmarks = [0] * 42
        hand_metrics = {
            'hand_face_distance': 0,
            'hand_jaw_overlap': 0,
            'hand_detected': 0,
            'hands_near_face': False
        }
        return {
            "hand1_landmarks": hand1_landmarks,
            "hand2_landmarks": hand2_landmarks,
            "hand_metrics": hand_metrics
        }

    def close(self):
        """리소스 해제"""
        self.hands.close()

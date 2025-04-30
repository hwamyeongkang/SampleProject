
"""
Data collection module.
Handles collecting, storing, and managing facial and hand landmark data.
"""

import os
import csv
from datetime import datetime
import pandas as pd
from config import *

class DataCollector:
    def __init__(self):
        """데이터 수집 클래스 초기화"""
        # 데이터 수집 관련 변수
        self.current_state = 0  # 0: 만족(기본), 1: 지루함, 2: 피곤함, 3: 고민 중, 4: 불만족
        self.is_collecting = False
        self.collection_count = 0
        self.max_samples = DEFAULT_MAX_SAMPLES  # 각 상태별 최대 샘플 수
        
        # 폴더 생성
        os.makedirs(DATA_FOLDER, exist_ok=True)
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)
        
        # CSV 파일 초기화 (존재하지 않을 경우)
        self.initialize_csv()
    
    def initialize_csv(self):
        """CSV 파일 초기화 (필요시)"""
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                # 헤더 작성
                header = ['timestamp', 'state']
                
                # 얼굴 랜드마크 컬럼 (x, y 좌표)
                for i in range(468):  # MediaPipe는 468개의 랜드마크
                    header.extend([f'face_x_{i}', f'face_y_{i}'])
                
                # 얼굴 메트릭 컬럼
                header.extend([
                    'face_size', 'face_aspect_ratio', 'eye_aspect_ratio_left', 
                    'eye_aspect_ratio_right', 'mouth_open_height', 'mouth_height_pos',
                    'eyebrow_distance', 'head_pose_pitch', 'head_pose_yaw', 'head_pose_roll'
                ])
                
                # 손 랜드마크 컬럼 (각 손 21개 랜드마크의 x, y 좌표)
                for i in range(21):
                    header.extend([f'hand1_x_{i}', f'hand1_y_{i}'])
                for i in range(21):
                    header.extend([f'hand2_x_{i}', f'hand2_y_{i}'])
                
                # 손 메트릭 컬럼
                header.extend(['hand_face_distance', 'hand_jaw_overlap', 'hand_detected'])
                
                # 상태 지속 시간
                header.extend(['yawn_duration', 'thinking_duration'])
                
                writer.writerow(header)
                print(f"CSV 파일 초기화: {CSV_FILE}")
    
    def start_collection(self, state):
        """데이터 수집 시작"""
        self.is_collecting = True
        self.current_state = state
        self.collection_count = 0
        print(f"{STATE_KOREAN[self.current_state]} 상태 데이터 수집 시작")
        return True
    
    def stop_collection(self):
        """데이터 수집 중지"""
        self.is_collecting = False
        print("데이터 수집 중지됨")
        return True
    
    def set_max_samples(self, max_samples):
        """최대 샘플 수 설정"""
        self.max_samples = max_samples
        return True
    
    def save_to_csv(self, landmarks_data, state=None):
        """랜드마크 데이터를 CSV에 저장"""
        if not self.is_collecting:
            return False
        
        if self.collection_count >= self.max_samples:
            self.is_collecting = False
            print(f"최대 샘플 수({self.max_samples}개) 도달. 데이터 수집 자동 중지.")
            return False
        
        try:
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # 현재 시간
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                
                # 상태 (지정되지 않은 경우 현재 상태 사용)
                state_value = state if state is not None else self.current_state
                
                # 행 데이터 구성
                row = [timestamp, state_value]
                
                # 얼굴 랜드마크
                row.extend(landmarks_data["face_landmarks"])
                
                # 얼굴 메트릭
                for key in ['face_size', 'face_aspect_ratio', 'eye_aspect_ratio_left', 
                         'eye_aspect_ratio_right', 'mouth_open_height', 'mouth_height_pos',
                         'eyebrow_distance', 'head_pose_pitch', 'head_pose_yaw', 'head_pose_roll']:
                    row.append(landmarks_data.get(key, 0))
                
                # 손 랜드마크
                row.extend(landmarks_data["hand1_landmarks"])
                row.extend(landmarks_data["hand2_landmarks"])
                
                # 손 메트릭
                for key in ['hand_face_distance', 'hand_jaw_overlap', 'hand_detected']:
                    row.append(landmarks_data.get(key, 0))
                
                # 상태 지속 시간
                row.append(landmarks_data.get("yawn_duration", 0))
                row.append(landmarks_data.get("thinking_duration", 0))
                
                writer.writerow(row)
                self.collection_count += 1
                return True
                
        except Exception as e:
            print(f"CSV 저장 중 오류: {e}")
            return False
    
    def capture_screenshot(self, frame):
        """현재 화면 스크린샷 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SCREENSHOT_FOLDER}/screenshot_{timestamp}.jpg"
            
            import cv2
            cv2.imwrite(filename, frame)
            print(f"스크린샷 저장됨: {filename}")
            return filename
        except Exception as e:
            print(f"스크린샷 저장 중 오류: {e}")
            return None
    
    def get_collection_stats(self):
        """데이터 수집 통계 반환"""
        if not os.path.exists(CSV_FILE):
            return {
                'total_samples': 0,
                'samples_by_state': {k: 0 for k in STATE_LABELS.keys()}
            }
        
        try:
            df = pd.read_csv(CSV_FILE)
            total = len(df)
            
            # 상태별 샘플 수
            state_counts = df['state'].value_counts().to_dict()
            
            # 누락된 상태는 0으로 설정
            samples_by_state = {k: state_counts.get(k, 0) for k in STATE_LABELS.keys()}
            
            return {
                'total_samples': total,
                'samples_by_state': samples_by_state
            }
        except Exception as e:
            print(f"데이터 통계 계산 중 오류: {e}")
            return {
                'total_samples': 0,
                'samples_by_state': {k: 0 for k in STATE_LABELS.keys()}
            }
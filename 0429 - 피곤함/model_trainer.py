"""
Model training module.
Handles training, evaluating, and saving the facial state detection model.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from config import *

class ModelTrainer:
    def __init__(self):
        """모델 학습 클래스 초기화"""
        self.model = None
        self.scaler = None
        
        # 폴더 생성
        os.makedirs(MODEL_FOLDER, exist_ok=True)
    
    def train_model(self, csv_file=CSV_FILE, model_file=MODEL_FILE):
        """수집된 데이터로 상태 감지 모델 학습"""
        try:
            # CSV 파일 로드
            if not os.path.exists(csv_file):
                print("CSV 파일이 존재하지 않습니다.")
                return False, 0
            
            df = pd.read_csv(csv_file)
            
            if df.empty:
                print("데이터가 비어 있습니다.")
                return False, 0
            
            # 클래스 분포 확인
            print("상태별 데이터 분포:")
            state_counts = df['state'].value_counts()
            for state, count in state_counts.items():
                if state in STATE_KOREAN:
                    print(f"  {STATE_KOREAN[state]}: {count}개")
                else:
                    print(f"  상태 {state}: {count}개")
            
            if len(state_counts) < 2:
                print("경고: 2개 이상의 상태에 대한 데이터가 필요합니다.")
                print("현재 데이터에는 하나의 상태만 있어 모델 학습이 어렵습니다.")
                print("다양한 상태의 데이터를 수집해 주세요.")
                return False, 0
            
            # 특징 선택 (얼굴 및 손 메트릭만)
            feature_columns = [
                'face_size', 'face_aspect_ratio', 'eye_aspect_ratio_left', 
                'eye_aspect_ratio_right', 'mouth_open_height', 'mouth_height_pos',
                'eyebrow_distance', 'head_pose_pitch', 'head_pose_yaw', 'head_pose_roll',
                'hand_face_distance', 'hand_jaw_overlap', 'hand_detected',
                'yawn_duration', 'thinking_duration'
            ]
            
            # 레이블과 특징 데이터
            X = df[feature_columns].values
            y = df['state'].values
            
            # 특징 스케일링
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # 데이터 분할 (훈련/테스트)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42)
            
            # 모델 학습 (랜덤 포레스트)
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42)
            self.model.fit(X_train, y_train)
            
            # 모델 평가
            accuracy = self.model.score(X_test, y_test)
            print(f"모델 정확도: {accuracy:.4f}")
            
            # 자세한 평가
            y_pred = self.model.predict(X_test)
            
            # 고유 클래스 확인
            unique_classes = np.unique(np.concatenate([y_test, y_pred]))
            print(f"고유 클래스: {unique_classes}")
            
            # 분류 보고서 생성
            if len(unique_classes) == 1:
                # 클래스가 하나만 있는 경우
                only_class = unique_classes[0]
                print(f"분류 평가에 하나의 클래스만 존재함: {STATE_KOREAN.get(only_class, f'상태 {only_class}')}")
                accuracy_val = np.mean(y_test == y_pred)
                print(f"단일 클래스 정확도: {accuracy_val:.4f}")
            else:
                # 발견된 클래스에 대해서만 보고서 생성
                available_states = [state for state in unique_classes if state in STATE_KOREAN]
                target_names = [STATE_KOREAN[state] for state in available_states]
                
                print("분류 보고서:")
                print(classification_report(y_test, y_pred, 
                                           labels=available_states, 
                                           target_names=target_names))
                
                # 혼동 행렬
                conf_matrix = confusion_matrix(y_test, y_pred, labels=available_states)
                print("혼동 행렬:")
                print(conf_matrix)
                
                # 행과 열에 클래스 이름 추가
                print("\n혼동 행렬 (클래스 이름):")
                print("실제 클래스 \\ 예측 클래스")
                header = "\t" + "\t".join([STATE_KOREAN.get(c, f'상태 {c}') for c in available_states])
                print(header)
                
                for i, row in enumerate(conf_matrix):
                    row_str = STATE_KOREAN.get(available_states[i], f'상태 {available_states[i]}') + "\t"
                    row_str += "\t".join([str(val) for val in row])
                    print(row_str)
            
            # 특성 중요도
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            print("\n특성 중요도 (상위 5개):")
            for i in range(min(5, len(feature_columns))):
                print(f"{feature_columns[indices[i]]}: {importances[indices[i]]:.4f}")
            
            # 모델 저장
            self.save_model(model_file)
            
            return True, accuracy
            
        except Exception as e:
            print(f"모델 학습 중 오류: {e}")
            import traceback
            traceback.print_exc()  # 상세 오류 정보 출력
            return False, 0
    
    def save_model(self, model_file=MODEL_FILE):
        """학습된 모델 저장"""
        if not self.model or not self.scaler:
            print("저장할 모델이 없습니다.")
            return False
        
        try:
            # 모델 데이터 준비
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            
            # 모델 저장
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"모델 저장 완료: {model_file}")
            return True
        except Exception as e:
            print(f"모델 저장 중 오류: {e}")
            return False
    
    def load_model(self, model_file=MODEL_FILE):
        """저장된 모델 로드"""
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                print(f"모델 로드 성공: {model_file}")
                return True
            except Exception as e:
                print(f"모델 로드 중 오류: {e}")
                return False
        else:
            print("모델 파일이 존재하지 않음")
            return False
    
    def evaluate_model(self, csv_file=CSV_FILE):
        """저장된 모델 평가"""
        if not self.model or not self.scaler:
            print("평가할 모델이 없습니다.")
            return False, 0, None
        
        try:
            # CSV 파일 로드
            if not os.path.exists(csv_file):
                print("CSV 파일이 존재하지 않습니다.")
                return False, 0, None
            
            df = pd.read_csv(csv_file)
            
            if df.empty:
                print("데이터가 비어 있습니다.")
                return False, 0, None
            
            # 특징 선택 (얼굴 및 손 메트릭만)
            feature_columns = [
                'face_size', 'face_aspect_ratio', 'eye_aspect_ratio_left', 
                'eye_aspect_ratio_right', 'mouth_open_height', 'mouth_height_pos',
                'eyebrow_distance', 'head_pose_pitch', 'head_pose_yaw', 'head_pose_roll',
                'hand_face_distance', 'hand_jaw_overlap', 'hand_detected',
                'yawn_duration', 'thinking_duration'
            ]
            
            # 레이블과 특징 데이터
            X = df[feature_columns].values
            y = df['state'].values
            
            # 특징 스케일링
            X_scaled = self.scaler.transform(X)
            
            # 모델 평가
            accuracy = self.model.score(X_scaled, y)
            print(f"모델 정확도: {accuracy:.4f}")
            
            # 예측
            y_pred = self.model.predict(X_scaled)
            
            # 고유 클래스 확인
            unique_classes = np.unique(np.concatenate([y, y_pred]))
            
            # 분류 보고서 생성
            if len(unique_classes) == 1:
                # 클래스가 하나만 있는 경우
                only_class = unique_classes[0]
                report = f"단일 클래스: {STATE_KOREAN.get(only_class, f'상태 {only_class}')}\n"
                report += f"정확도: {accuracy:.4f}"
            else:
                # 발견된 클래스에 대해서만 보고서 생성
                available_states = [state for state in unique_classes if state in STATE_KOREAN]
                target_names = [STATE_KOREAN[state] for state in available_states]
                
                report = classification_report(y, y_pred, 
                                              labels=available_states, 
                                              target_names=target_names)
            
            print("분류 보고서:")
            print(report)
            
            return True, accuracy, report
            
        except Exception as e:
            print(f"모델 평가 중 오류: {e}")
            return False, 0, None
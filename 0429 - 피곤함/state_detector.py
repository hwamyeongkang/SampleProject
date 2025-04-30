"""
State detection module.
Handles emotional state detection based on facial and hand features.
"""

import time
import traceback
from collections import deque

# 상태 한글 이름 매핑
STATE_KOREAN = {
    0: "기본 (Normal)",
    1: "피곤함 (Tired)",
    2: "지루함 (Bored)", 
    3: "고민중 (Thinking)"
}

class StateDetector:
    def __init__(self):
        # 분석 상태 변수
        self.is_analyzing = False
        self.model = None  # 머신러닝 모델 (나중에 설정됨)
        self.scaler = None  # 특징 스케일러 (나중에 설정됨)
        
        # 상태 추적 변수
        self.last_state = None
        self.last_state_start_time = None
        self.state_hold_threshold = 5  # 5초 유지해야 인정
        self.detected_state_code = 0   # 기본 상태 (Normal)
        self.state_history = deque(maxlen=10)
        
        print("StateDetector 초기화 완료")

    def start_analysis(self):
        """상태 분석 시작"""
        self.is_analyzing = True
        self.last_state = None
        self.last_state_start_time = time.time()
        self.detected_state_code = 0  # 기본 상태로 초기화
        print("상태 분석 시작됨")
        return True

    def stop_analysis(self):
        """상태 분석 중지"""
        self.is_analyzing = False
        print("상태 분석 중지됨")
        return True

    def analyze_state(self, frame, face_data, hand_data):
        """얼굴과 손 데이터로 상태 분석"""
        try:
            # 분석 모드가 아니면 즉시 반환
            if not self.is_analyzing:
                return frame, None, {}

            # 기본 정보 설정
            current_time = time.time()
            detected_now = None
            state_info = {
                'confidence': 0.0,
                'duration': 0.0
            }

            # 얼굴 데이터 유효성 확인
            if not face_data:
                print("얼굴 데이터가 없습니다.")
                return frame, None, state_info

            # 머신러닝 모델이 있으면 모델 기반 예측 수행
            if self.model is not None and self.scaler is not None:
                try:
                    return self.analyze_state_with_model(frame, face_data, hand_data)
                except Exception as e:
                    print(f"모델 기반 분석 중 오류: {e}")
                    traceback.print_exc()
                    # 모델 사용 실패 시 규칙 기반 방식으로 폴백

            # 규칙 기반 상태 감지
            try:
                # 메트릭 데이터 접근
                metrics = face_data.get('face_metrics', {})
                
                # 얼굴 방향
                yaw = metrics.get('head_pose_yaw', 0)
                pitch = metrics.get('head_pose_pitch', 0)
                roll = metrics.get('head_pose_roll', 0)
                
                # 얼굴 모양 및 표정
                mouth_open_height = metrics.get('mouth_open_height', 0)
                left_eye = metrics.get('eye_aspect_ratio_left', 0)
                right_eye = metrics.get('eye_aspect_ratio_right', 0)
                
                # 손 얼굴 근접
                hands_near_face = False
                if hand_data and 'hand_metrics' in hand_data:
                    hands_near_face = hand_data['hand_metrics'].get('hands_near_face', False)
                
                # 디버깅 출력
                print(f"상태 분석 메트릭 - 얼굴 방향: yaw={yaw:.2f}, pitch={pitch:.2f}, roll={roll:.2f}")
                print(f"상태 분석 메트릭 - 입/눈: mouth_open={mouth_open_height:.2f}, eyes={left_eye:.2f}/{right_eye:.2f}")
                print(f"상태 분석 메트릭 - 손 얼굴 근접: {hands_near_face}")
                
                # 상태 분류 규칙
                # 0: 기본 상태 - 정상적인 얼굴 방향과 표정
                if abs(yaw) < 0.3 and abs(pitch) < 0.3 and abs(roll) < 0.3 and not hands_near_face:
                    detected_now = 'normal'  # 기본
                    
                # 1: 피곤함 -  하품 (입이 많이 벌어짐)
                elif mouth_open_height > 0.1:
                    detected_now = 'tired'  # 피곤함
                    
                # 2: 지루함 - 머리를 많이 돌리거나 기울임
                elif abs(yaw) > 0.4 or abs(roll) > 0.3:
                    detected_now = 'bored'  # 지루함
                    
                # 3: 고민중 - 손이 얼굴 근처에 있거나 머리를 약간 기울임
                elif hands_near_face or (0.2 < abs(pitch) < 0.4):
                    detected_now = 'thinking'  # 고민중
                    
                # 감지된 상태가 없으면 기본값
                else:
                    detected_now = 'normal'  # 기본
                
                print(f"감지된 상태: {detected_now}")
                
                # 상태가 변경되면 타이머 재설정
                if detected_now != self.last_state:
                    print(f"상태 변경: {self.last_state} -> {detected_now}")
                    self.last_state = detected_now
                    self.last_state_start_time = current_time
                
                # 얼마나 오래 현재 상태가 유지되었는지 계산
                held_duration = current_time - self.last_state_start_time
                
                # 상태 매핑 정의
                state_mapping = {
                    'normal': 0,     # 기본
                    'tired': 1,      # 피곤함
                    'bored': 2,      # 지루함
                    'thinking': 3    # 고민중
                }
                
                # 임계값 이상 유지되었으면 상태 확정
                if held_duration >= self.state_hold_threshold:
                    # 5초 이상 유지되면 감정 상태 인정
                    self.detected_state_code = state_mapping.get(detected_now, 0)
                    state_info['confidence'] = min(100, int(held_duration / (self.state_hold_threshold * 1.5) * 100))  # 백분율로 변환
                    state_info['duration'] = held_duration
                    
                    # 색상 또는 표시를 위한 텍스트 추가
                    # 왼쪽 상단에 상태 텍스트 표시
                    state_text = f"{STATE_KOREAN[self.detected_state_code]} ({held_duration:.1f}s)"
                    import cv2
                    cv2.putText(frame, state_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.get_state_color(self.detected_state_code), 2)
                    
                else:
                    # 임계값 미만이면 아직 상태 미확정
                    confidence = min(100, int(held_duration / self.state_hold_threshold * 100))  # 백분율로 변환
                    state_info['confidence'] = confidence
                    state_info['duration'] = held_duration
                    
                    # 텍스트로 진행 중인 상태 표시
                    state_code = state_mapping.get(detected_now, 0)
                    progress_text = f"감지 중: {STATE_KOREAN[state_code]} ({held_duration:.1f}s / {self.state_hold_threshold}s)"
                    import cv2
                    cv2.putText(frame, progress_text, (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                    
                    # 아직 임계값에 도달하지 않았으므로 None 반환
                    self.detected_state_code = None
                
                # 상태 이력 업데이트
                self.state_history.append(self.detected_state_code)
                
                return frame, self.detected_state_code, state_info
                
            except Exception as e:
                print(f"규칙 기반 상태 분석 중 오류: {e}")
                traceback.print_exc()
                return frame, None, state_info
                
        except Exception as e:
            print(f"상태 분석 중 일반 오류: {e}")
            traceback.print_exc()
            return frame, None, {'confidence': 0.0, 'duration': 0.0}

    def analyze_state_with_model(self, frame, face_data, hand_data):
        """학습된 모델을 사용하여 상태 분석"""
        try:
            # 모델 유효성 확인
            if not self.model or not self.scaler:
                return frame, None, {'confidence': 0.0, 'duration': 0.0}
            
            # 분석에 필요한 특징 추출
            metrics = face_data.get('face_metrics', {})
            
            # 특징 벡터 구성
            features = [
                metrics.get('face_size', 0),
                metrics.get('face_aspect_ratio', 0),
                metrics.get('eye_aspect_ratio_left', 0),
                metrics.get('eye_aspect_ratio_right', 0),
                metrics.get('mouth_open_height', 0),
                metrics.get('mouth_height_pos', 0),
                metrics.get('eyebrow_distance', 0),
                metrics.get('head_pose_pitch', 0),
                metrics.get('head_pose_yaw', 0),
                metrics.get('head_pose_roll', 0)
            ]
            
            # 손 관련 특징 추가
            hand_metrics = hand_data.get('hand_metrics', {}) if hand_data else {}
            features.extend([
                hand_metrics.get('hand_face_distance', 0),
                hand_metrics.get('hand_jaw_overlap', 0),
                hand_metrics.get('hand_detected', 0),
                0,  # yawn_duration (필요시 업데이트)
                0   # thinking_duration (필요시 업데이트)
            ])
            
            # 특징 정규화
            features_scaled = self.scaler.transform([features])
            
            # 상태 예측
            prediction = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            
            # 예측 신뢰도
            confidence = min(100, int(max(prediction_proba) * 100))  # 백분율로 변환
            
            # 결과 반환
            state_info = {
                'confidence': confidence,
                'duration': 0.0  # 모델 기반에서는 의미가 낮음
            }
            
            # 프레임에 예측 상태 표시
            state_text = f"예측됨: {STATE_KOREAN.get(prediction, '알 수 없음')} ({confidence}%)"
            import cv2
            cv2.putText(frame, state_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.get_state_color(prediction), 2)
            
            return frame, prediction, state_info
            
        except Exception as e:
            print(f"모델 기반 분석 실패: {e}")
            traceback.print_exc()
            return frame, None, {'confidence': 0.0, 'duration': 0.0}
            
    def get_state_color(self, state_code):
        """상태 코드에 해당하는 색상 반환 (OpenCV BGR 형식)"""
        color_map = {
            0: (0, 255, 0),    # 기본: 녹색
            1: (0, 0, 255),    # 피곤함: 빨강
            2: (255, 0, 255),  # 지루함: 마젠타
            3: (255, 165, 0)   # 고민중: 주황
        }
        return color_map.get(state_code, (200, 200, 200))  # 기본: 회색
"""
Face detection and analysis module.
Handles face detection, landmark extraction, and facial metrics calculations.
Includes yawn detection for fatigue state detection.
"""

import cv2
import numpy as np
import mediapipe as mp
import traceback

# InsightFace 가져오기 (사용 가능한 경우)
try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    INSIGHTFACE_AVAILABLE = True
    print("InsightFace 가져오기 성공")
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace를 사용할 수 없습니다. 'pip install insightface onnxruntime' 명령으로 설치하세요.")

class FaceDetector:
    def __init__(self):
        """얼굴 감지 클래스 초기화"""
        # MediaPipe 초기화
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 더 낮은 신뢰도 임계값으로 설정하여 감지율 향상
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.3,  # 0.5에서 0.3으로 낮춤
            min_tracking_confidence=0.3)   # 0.5에서 0.3으로 낮춤
        
        # InsightFace 초기화
        self.insightface_available = False
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106', 'recognition'])
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                self.insightface_available = True
                print("InsightFace 초기화 성공")
            except Exception as e:
                print(f"InsightFace 초기화 중 오류: {e}")
                self.insightface_available = False
        
        # 하품 감지 관련 상수
        self.upper_lip_idx = 13  # 윗입술 중앙 (MediaPipe)
        self.lower_lip_idx = 14  # 아랫입술 중앙 (MediaPipe)
        
        # InsightFace를 위한 입 랜드마크 인덱스
        self.insightface_upper_lip_idx = 89  # 윗입술 중앙 (InsightFace)
        self.insightface_lower_lip_idx = 95  # 아랫입술 중앙 (InsightFace)
    
    def detect_face(self, frame):
        """프레임에서 얼굴 감지 및 분석"""
        if frame is None:
            print("얼굴 감지에 입력된 프레임이 None입니다.")
            return False, None, self.create_empty_face_data()
            
        try:
            h, w, _ = frame.shape
            face_data = self.create_empty_face_data()  # 기본값으로 빈 데이터 설정
            
            # 원본 프레임 복사 (그리기용)
            output_frame = frame.copy()
            
            # RGB로 변환 (MediaPipe 요구사항)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # InsightFace로 얼굴 감지 시도
            insightface_detected = False
            if self.insightface_available:
                try:
                    faces = self.face_app.get(rgb_frame)
                    if len(faces) > 0:
                        face = faces[0]  # 첫 번째 얼굴
                        face_data = self.extract_facial_features_insightface(frame, face)
                        
                        # 얼굴 상자 그리기
                        box = face.bbox.astype(np.int32)
                        cv2.rectangle(output_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        
                        # 랜드마크 그리기
                        if hasattr(face, 'landmark_2d_106'):
                            landmarks = face.landmark_2d_106.astype(np.int32)
                            for i in range(landmarks.shape[0]):
                                cv2.circle(output_frame, (landmarks[i][0], landmarks[i][1]), 1, (0, 0, 255), 1)
                            
                            # 입 벌어짐 시각화 (InsightFace)
                            upper_lip = landmarks[self.insightface_upper_lip_idx]
                            lower_lip = landmarks[self.insightface_lower_lip_idx]
                            cv2.line(output_frame, (upper_lip[0], upper_lip[1]), 
                                    (lower_lip[0], lower_lip[1]), (0, 255, 255), 2)
                        
                        insightface_detected = True
                except Exception as e:
                    print(f"InsightFace 처리 중 오류: {e}")
                    traceback.print_exc()
            
            # InsightFace가 감지하지 못했거나 사용할 수 없는 경우 MediaPipe 사용
            if not insightface_detected:
                try:
                    face_results = self.face_mesh.process(rgb_frame)
                    if face_results.multi_face_landmarks:
                        face_data = self.extract_facial_features_mediapipe(frame, face_results)
                        
                        # 랜드마크 그리기 - 가시성을 위해 약간 수정
                        # 라인 두께와 점 크기 조정
                        drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
                        self.mp_drawing.draw_landmarks(
                            output_frame,
                            face_results.multi_face_landmarks[0],
                            self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=drawing_spec)
                        
                        # 입 벌어짐 시각화 (MediaPipe)
                        landmarks = face_results.multi_face_landmarks[0].landmark
                        upper_lip = landmarks[self.upper_lip_idx]
                        lower_lip = landmarks[self.lower_lip_idx]
                        
                        # 화면 좌표로 변환
                        upper_lip_px = (int(upper_lip.x * w), int(upper_lip.y * h))
                        lower_lip_px = (int(lower_lip.x * w), int(lower_lip.y * h))
                        
                        # 입술 사이 선 그리기
                        cv2.line(output_frame, upper_lip_px, lower_lip_px, (0, 255, 255), 2)
                        
                        # 입 높이 표시
                        mouth_height = face_data['face_metrics']['mouth_open_height']
                        mouth_height_pixels = mouth_height * w  # 정규화된 값을 픽셀로 변환
                        
                        return True, output_frame, face_data
                    else:
                        print("MediaPipe로 얼굴을 감지하지 못함")
                except Exception as e:
                    print(f"MediaPipe 얼굴 처리 중 오류: {e}")
                    traceback.print_exc()
            
            # 얼굴 감지 성공 여부 반환
            detected = insightface_detected
            
            # 감지된 얼굴이 있으면 프레임과 데이터 반환
            if detected:
                return True, output_frame, face_data
                
            # 얼굴 감지 실패
            return False, frame, face_data
            
        except Exception as e:
            print(f"얼굴 감지 중 일반 오류: {e}")
            traceback.print_exc()
            return False, frame, self.create_empty_face_data()
    
    def extract_facial_features_mediapipe(self, frame, face_results):
        """MediaPipe로 얼굴 특징 추출"""
        try:
            h, w, _ = frame.shape
            face_landmarks = []
            
            # 첫 번째 얼굴의 랜드마크 좌표 추출
            landmarks = face_results.multi_face_landmarks[0]
            
            for i in range(468):  # MediaPipe는 468개의 랜드마크
                if i < len(landmarks.landmark):
                    lm = landmarks.landmark[i]
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_landmarks.extend([x, y])
                else:
                    face_landmarks.extend([0, 0])  # 누락된 랜드마크를 0으로 채움
            
            # 주요 랜드마크 좌표
            # 눈 좌표 (MediaPipe 랜드마크 인덱스)
            left_eye = [33, 160, 158, 133, 153, 144]  # 왼쪽 눈
            right_eye = [362, 385, 387, 263, 373, 380]  # 오른쪽 눈
            
            # 입 좌표
            upper_lip = [13]  # 윗입술 중앙
            lower_lip = [14]  # 아랫입술 중앙
            
            # 미간 좌표
            left_eyebrow = [65]  # 왼쪽 눈썹 안쪽
            right_eyebrow = [295]  # 오른쪽 눈썹 안쪽
            
            # 얼굴 메트릭 계산
            # 얼굴 크기 (얼굴 너비 기준)
            left_cheek = landmarks.landmark[234]  # 왼쪽 볼
            right_cheek = landmarks.landmark[454]  # 오른쪽 볼
            face_width = abs(right_cheek.x - left_cheek.x) * w
            face_size = face_width / w  # 정규화된 크기
            
            # 눈 종횡비 (눈 높이/너비)
            def eye_aspect_ratio(eye_pts):
                eye_coords = []
                for idx in eye_pts:
                    lm = landmarks.landmark[idx]
                    eye_coords.append((lm.x * w, lm.y * h))
                
                # 수직 거리
                vertical_dist1 = np.linalg.norm(np.array(eye_coords[1]) - np.array(eye_coords[5]))
                vertical_dist2 = np.linalg.norm(np.array(eye_coords[2]) - np.array(eye_coords[4]))
                
                # 수평 거리
                horizontal_dist = np.linalg.norm(np.array(eye_coords[0]) - np.array(eye_coords[3]))
                
                # 눈 종횡비
                ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist) if horizontal_dist > 0 else 0
                return ear
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            # 입 벌어짐 (입 높이)
            upper_lm = landmarks.landmark[upper_lip[0]]
            lower_lm = landmarks.landmark[lower_lip[0]]
            mouth_open_height = abs(lower_lm.y - upper_lm.y) * h / w  # 정규화된 높이
            
            # 입 높이 위치 (상대적 위치)
            mouth_height_pos = (upper_lm.y + lower_lm.y) / 2
            
            # 미간 거리
            left_eb = landmarks.landmark[left_eyebrow[0]]
            right_eb = landmarks.landmark[right_eyebrow[0]]
            eyebrow_distance = abs(right_eb.x - left_eb.x) * w / h  # 정규화된 거리
            
            # 얼굴 각도 추정 (간단한 근사값)
            nose_tip = landmarks.landmark[4]
            left_eye_center = landmarks.landmark[159]
            right_eye_center = landmarks.landmark[386]
            
            dx = right_eye_center.x - left_eye_center.x
            dy = right_eye_center.y - left_eye_center.y
            
            # 머리 회전 각도 (yaw)
            head_pose_yaw = abs(nose_tip.x - 0.5) * 2  # 중앙에서 벗어난 정도
            
            # 머리 상하 각도 (pitch)
            head_pose_pitch = abs(nose_tip.y - 0.5) * 2
            
            # 머리 기울기 (roll)
            head_pose_roll = np.arctan2(dy, dx) if dx != 0 else 0
            
            # 얼굴 종횡비
            chin = landmarks.landmark[152]  # 턱
            forehead = landmarks.landmark[10]  # 이마
            face_height = abs(chin.y - forehead.y) * h
            face_aspect_ratio = face_height / face_width if face_width > 0 else 0
            
            # 메트릭 데이터 구성
            metrics = {
                'face_size': face_size,
                'face_aspect_ratio': face_aspect_ratio,
                'eye_aspect_ratio_left': left_ear,
                'eye_aspect_ratio_right': right_ear,
                'mouth_open_height': mouth_open_height,
                'mouth_height_pos': mouth_height_pos,
                'eyebrow_distance': eyebrow_distance,
                'head_pose_pitch': head_pose_pitch,
                'head_pose_yaw': head_pose_yaw,
                'head_pose_roll': head_pose_roll
            }
            
            return {
                "face_landmarks": face_landmarks,
                "face_metrics": metrics
            }
            
        except Exception as e:
            print(f"MediaPipe 얼굴 특징 추출 중 오류: {e}")
            traceback.print_exc()
            return self.create_empty_face_data()
    
    def extract_facial_features_insightface(self, frame, face):
        """InsightFace로 얼굴 특징 추출"""
        try:
            h, w, _ = frame.shape
            face_landmarks = []
            
            # InsightFace에서 2D 랜드마크 (106개) 추출
            if hasattr(face, 'landmark_2d_106'):
                landmarks = face.landmark_2d_106.astype(np.int32)
                
                # MediaPipe와 동일한 468개 형식으로 변환 (없는 부분은 0으로 채움)
                for i in range(468):
                    if i < 106:  # 실제 InsightFace 랜드마크
                        x, y = landmarks[i][0], landmarks[i][1]
                        face_landmarks.extend([x, y])
                    else:  # 나머지는 0으로 채움
                        face_landmarks.extend([0, 0])
                
                # 주요 랜드마크 인덱스 (InsightFace)
                # 눈 좌표
                left_eye = [66, 67, 68, 69, 70, 71]  # 왼쪽 눈 윤곽
                right_eye = [75, 76, 77, 78, 79, 80]  # 오른쪽 눈 윤곽
                
                # 입 좌표
                upper_lip = [89]  # 윗입술 중앙
                lower_lip = [95]  # 아랫입술 중앙
                
                # 미간 좌표
                left_eyebrow = [72]  # 왼쪽 눈썹 안쪽
                right_eyebrow = [81]  # 오른쪽 눈썹 안쪽
                
                # 얼굴 메트릭 계산
                # 얼굴 크기 (얼굴 너비 기준)
                left_cheek = landmarks[2]  # 왼쪽 볼 근처
                right_cheek = landmarks[14]  # 오른쪽 볼 근처
                face_width = abs(right_cheek[0] - left_cheek[0])
                face_size = face_width / w  # 정규화된 크기
                
                # 눈 종횡비 (눈 높이/너비)
                def eye_aspect_ratio(eye_pts):
                    eye_coords = [landmarks[i] for i in eye_pts]
                    
                    # 수직 거리
                    vertical_dist1 = np.linalg.norm(np.array(eye_coords[1]) - np.array(eye_coords[5]))
                    vertical_dist2 = np.linalg.norm(np.array(eye_coords[2]) - np.array(eye_coords[4]))
                    
                    # 수평 거리
                    horizontal_dist = np.linalg.norm(np.array(eye_coords[0]) - np.array(eye_coords[3]))
                    
                    # 눈 종횡비
                    ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist) if horizontal_dist > 0 else 0
                    return ear
                
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                
                # 입 벌어짐 (입 높이)
                upper_lm = landmarks[upper_lip[0]]
                lower_lm = landmarks[lower_lip[0]]
                mouth_open_height = abs(lower_lm[1] - upper_lm[1]) / h  # 정규화된 높이
                
                # 입 높이 위치 (상대적 위치)
                mouth_height_pos = (upper_lm[1] + lower_lm[1]) / (2 * h)
                
                # 미간 거리
                left_eb = landmarks[left_eyebrow[0]]
                right_eb = landmarks[right_eyebrow[0]]
                eyebrow_distance = abs(right_eb[0] - left_eb[0]) / w  # 정규화된 거리
                
                # 얼굴 각도 추정 (간단한 근사값)
                nose_tip = landmarks[94]  # 코 끝
                left_eye_center = landmarks[66]  # 왼쪽 눈
                right_eye_center = landmarks[79]  # 오른쪽 눈
                
                dx = right_eye_center[0] - left_eye_center[0]
                dy = right_eye_center[1] - left_eye_center[1]
                
                # 머리 회전 각도 (yaw)
                head_pose_yaw = abs(nose_tip[0] / w - 0.5) * 2  # 중앙에서 벗어난 정도
                
                # 머리 상하 각도 (pitch)
                head_pose_pitch = abs(nose_tip[1] / h - 0.5) * 2
                
                # 머리 기울기 (roll)
                head_pose_roll = np.arctan2(dy, dx) if dx != 0 else 0
                
                # 얼굴 종횡비
                chin = landmarks[95]  # 턱
                forehead = landmarks[72]  # 이마
                face_height = abs(chin[1] - forehead[1])
                face_aspect_ratio = face_height / face_width if face_width > 0 else 0
                
                # 메트릭 데이터 구성
                metrics = {
                    'face_size': face_size,
                    'face_aspect_ratio': face_aspect_ratio,
                    'eye_aspect_ratio_left': left_ear,
                    'eye_aspect_ratio_right': right_ear,
                    'mouth_open_height': mouth_open_height,
                    'mouth_height_pos': mouth_height_pos,
                    'eyebrow_distance': eyebrow_distance,
                    'head_pose_pitch': head_pose_pitch,
                    'head_pose_yaw': head_pose_yaw,
                    'head_pose_roll': head_pose_roll
                }
                
                return {
                    "face_landmarks": face_landmarks,
                    "face_metrics": metrics
                }
            else:
                raise Exception("InsightFace 2D 랜드마크를 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"InsightFace 얼굴 특징 추출 중 오류: {e}")
            traceback.print_exc()
            return self.create_empty_face_data()
    
    def create_empty_face_data(self):
        """빈 얼굴 데이터 생성 (얼굴 감지 실패 시)"""
        face_landmarks = [0] * (468 * 2)  # 468개 랜드마크의 x, y 좌표
        metrics = {
            'face_size': 0,
            'face_aspect_ratio': 0,
            'eye_aspect_ratio_left': 0,
            'eye_aspect_ratio_right': 0,
            'mouth_open_height': 0,
            'mouth_height_pos': 0,
            'eyebrow_distance': 0,
            'head_pose_pitch': 0,
            'head_pose_yaw': 0,
            'head_pose_roll': 0
        }
        return {
            "face_landmarks": face_landmarks,
            "face_metrics": metrics
        }
    
    def close(self):
        """리소스 해제"""
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
        except Exception as e:
            print(f"FaceDetector 리소스 해제 중 오류: {e}")
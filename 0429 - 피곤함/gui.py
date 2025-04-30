
"""
GUI module for the Facial State Detection application.
Handles UI elements, webcam display, and status updates.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk

# 상태 코드와 한글 이름 매핑
STATE_KOREAN = {
    0: "기본 (Normal)",
    1: "피곤함 (Tired)",
    2: "지루함 (Bored)", 
    3: "고민중 (Thinking)"
}

class FacialStateGUI:
    def __init__(self):
        """GUI 초기화"""
        self.root = tk.Tk()
        self.root.title("얼굴 상태 감지 시스템")
        self.root.geometry("1080x720")
        self.root.protocol("WM_DELETE_WINDOW", self.close_window)
        
        # 이벤트 콜백 초기화 (외부에서 설정)
        self.on_start_stop = None
        self.on_collection_toggle = None
        self.on_analysis_toggle = None
        self.on_train_model = None
        self.on_state_selected = None
        self.on_max_samples_selected = None
        self.on_capture_screenshot = None
        self.on_open_data_folder = None
        self.on_closing = None
        
        # 레이아웃 설정
        self.setup_layout()
        self.setup_webcam_frame()
        self.setup_control_frame()
        self.setup_status_frame()
        
        # 상태 변수
        self.face_detected = False
        self.hand_detected = False
        self.current_state = 0
        self.collection_active = False
        self.analysis_active = False
        
        # 기본 상태 설정
        self.update_status("시작하려면 [시작] 버튼을 누르세요.")
        self.set_start_stop_button_state(False)
        self.set_collection_button_state(False)
        self.set_analysis_button_state(False)
        self.set_training_button_state(False)
    
    def setup_layout(self):
        """전체 레이아웃 설정"""
        # 메인 프레임
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill="both", expand=True)
        
        # 좌측 패널 (웹캠 표시)
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side="left", fill="both", expand=True)
        
        # 우측 패널 (컨트롤)
        self.right_panel = ttk.Frame(self.main_frame, width=250)
        self.right_panel.pack(side="right", fill="y", padx=(10, 0))
        self.right_panel.pack_propagate(False)
    
    def setup_webcam_frame(self):
        """웹캠 표시 영역 설정"""
        self.webcam_frame = ttk.LabelFrame(self.left_panel, text="웹캠 화면")
        self.webcam_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 웹캠 표시용 레이블
        self.webcam_label = ttk.Label(self.webcam_frame)
        self.webcam_label.pack(fill="both", expand=True, padx=5, pady=5)
    
    def setup_control_frame(self):
        """컨트롤 영역 설정"""
        # 컨트롤 프레임
        self.control_frame = ttk.LabelFrame(self.right_panel, text="제어")
        self.control_frame.pack(fill="x", padx=5, pady=5)
        
        # 시작/중지 버튼
        self.start_stop_button = ttk.Button(
            self.control_frame, 
            text="시작", 
            command=self.on_start_stop_click
        )
        self.start_stop_button.pack(fill="x", padx=5, pady=5)
        
        # 상태 감지 버튼
        self.analysis_button = ttk.Button(
            self.control_frame, 
            text="상태 감지 시작", 
            command=self.on_analysis_toggle_click
        )
        self.analysis_button.pack(fill="x", padx=5, pady=5)
        
        # 데이터 수집 영역
        self.collection_frame = ttk.LabelFrame(self.right_panel, text="데이터 수집")
        self.collection_frame.pack(fill="x", padx=5, pady=5)
        
        # 상태 선택
        ttk.Label(self.collection_frame, text="수집할 상태:").pack(anchor="w", padx=5, pady=2)
        
        self.state_combo = ttk.Combobox(
            self.collection_frame, 
            values=[f"{k}: {v}" for k, v in STATE_KOREAN.items()],
            state="readonly"
        )
        self.state_combo.current(0)  # 기본 상태 선택
        self.state_combo.pack(fill="x", padx=5, pady=2)
        self.state_combo.bind("<<ComboboxSelected>>", self.on_state_combo_selected)
        
        # 최대 샘플 수 선택
        ttk.Label(self.collection_frame, text="최대 샘플 수:").pack(anchor="w", padx=5, pady=2)
        
        self.max_samples_combo = ttk.Combobox(
            self.collection_frame, 
            values=["100", "200", "500", "1000", "2000", "5000"],
            state="readonly"
        )
        self.max_samples_combo.current(1)  # 200개 기본 선택
        self.max_samples_combo.pack(fill="x", padx=5, pady=2)
        self.max_samples_combo.bind("<<ComboboxSelected>>", self.on_max_samples_selected)
        
        # 데이터 수집 버튼
        self.collection_button = ttk.Button(
            self.collection_frame, 
            text="데이터 수집 시작", 
            command=self.on_collection_toggle_click
        )
        self.collection_button.pack(fill="x", padx=5, pady=5)
        
        # 샘플 수 표시
        self.sample_count_label = ttk.Label(self.collection_frame, text="수집된 샘플: 0")
        self.sample_count_label.pack(fill="x", padx=5, pady=5)
        
        # 스크린샷 버튼
        self.screenshot_button = ttk.Button(
            self.collection_frame, 
            text="스크린샷 저장", 
            command=self.on_screenshot_click
        )
        self.screenshot_button.pack(fill="x", padx=5, pady=5)
        
        # 데이터 폴더 열기 버튼
        self.open_folder_button = ttk.Button(
            self.collection_frame, 
            text="데이터 폴더 열기", 
            command=self.on_open_folder_click
        )
        self.open_folder_button.pack(fill="x", padx=5, pady=5)
        
        # 모델 학습 영역
        self.model_frame = ttk.LabelFrame(self.right_panel, text="모델 학습")
        self.model_frame.pack(fill="x", padx=5, pady=5)
        
        # 모델 학습 버튼
        self.training_button = ttk.Button(
            self.model_frame, 
            text="모델 학습 시작", 
            command=self.on_training_click
        )
        self.training_button.pack(fill="x", padx=5, pady=5)
        
        # 모델 상태 표시
        self.model_status_label = ttk.Label(
            self.model_frame, 
            text="모델 상태: 로드되지 않음", 
            foreground="red"
        )
        self.model_status_label.pack(fill="x", padx=5, pady=5)
        
    def setup_status_frame(self):
        """상태 표시 영역 설정"""
        # 상태 프레임
        self.status_frame = ttk.LabelFrame(self.right_panel, text="상태 정보")
        self.status_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 얼굴 감지 상태
        self.face_status_label = ttk.Label(
            self.status_frame, 
            text="얼굴 감지: 아니오", 
            foreground="red"
        )
        self.face_status_label.pack(fill="x", padx=5, pady=2)
        
        # 손 감지 상태
        self.hand_status_label = ttk.Label(
            self.status_frame, 
            text="손 감지: 아니오", 
            foreground="red"
        )
        self.hand_status_label.pack(fill="x", padx=5, pady=2)
        
        # 감지된 상태
        self.state_status_label = ttk.Label(
            self.status_frame, 
            text="감지된 상태: 없음", 
            foreground="blue"
        )
        self.state_status_label.pack(fill="x", padx=5, pady=2)
        
        # 감지 신뢰도
        self.confidence_label = ttk.Label(
            self.status_frame, 
            text="신뢰도: 0%", 
            foreground="blue"
        )
        self.confidence_label.pack(fill="x", padx=5, pady=2)
        
        # 메시지 표시 영역
        self.message_frame = ttk.LabelFrame(self.right_panel, text="메시지")
        self.message_frame.pack(fill="x", padx=5, pady=5)
        
        # 메시지 레이블
        self.message_label = ttk.Label(
            self.message_frame, 
            text="준비됨", 
            wraplength=220,
            justify="left"
        )
        self.message_label.pack(fill="x", padx=5, pady=5)
    
    def on_start_stop_click(self):
        """시작/중지 버튼 클릭 처리"""
        if self.on_start_stop:
            self.on_start_stop()
    
    def on_collection_toggle_click(self):
        """데이터 수집 버튼 클릭 처리"""
        if self.on_collection_toggle:
            self.on_collection_toggle()
    
    def on_analysis_toggle_click(self):
        """상태 감지 버튼 클릭 처리"""
        if self.on_analysis_toggle:
            self.on_analysis_toggle()
    
    def on_training_click(self):
        """모델 학습 버튼 클릭 처리"""
        if self.on_train_model:
            self.on_train_model()
    
    def on_state_combo_selected(self, event):
        """상태 선택 콤보박스 변경 처리"""
        if self.on_state_selected:
            selected = self.state_combo.get()
            state_code = int(selected[0])
            self.on_state_selected(state_code)
    
    def on_max_samples_selected(self, event):
        """최대 샘플 수 콤보박스 변경 처리"""
        if self.on_max_samples_selected:
            max_samples = int(self.max_samples_combo.get())
            self.on_max_samples_selected(max_samples)
    
    def on_screenshot_click(self):
        """스크린샷 버튼 클릭 처리"""
        if self.on_capture_screenshot:
            self.on_capture_screenshot()
    
    def on_open_folder_click(self):
        """데이터 폴더 열기 버튼 클릭 처리"""
        if self.on_open_data_folder:
            self.on_open_data_folder()
    
    def update_webcam_display(self, frame):
        """웹캠 프레임 업데이트"""
        if frame is not None:
            # OpenCV BGR 이미지를 RGB로 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 화면에 맞게 크기 조정
            # 라벨 크기 확인
            label_width = self.webcam_label.winfo_width()
            label_height = self.webcam_label.winfo_height()
            
            # 라벨 크기가 0이면 기본 크기 사용
            if label_width <= 1 or label_height <= 1:
                label_width = 640
                label_height = 480
            
            # 비율 유지하면서 크기 조정
            h, w, _ = rgb_frame.shape
            aspect_ratio = w / h
            
            if label_width / label_height > aspect_ratio:
                new_width = int(label_height * aspect_ratio)
                new_height = label_height
            else:
                new_width = label_width
                new_height = int(label_width / aspect_ratio)
            
            # 이미지 크기 조정
            resized_frame = cv2.resize(rgb_frame, (new_width, new_height))
            
            # PIL 이미지로 변환
            img = Image.fromarray(resized_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # 레이블 업데이트
            self.webcam_label.configure(image=img_tk)
            self.webcam_label.image = img_tk  # 참조 유지

    def update_face_status(self, detected):
        """얼굴 감지 상태 업데이트"""
        self.face_detected = detected
        color = "green" if detected else "red"
        text = "얼굴 감지: 예" if detected else "얼굴 감지: 아니오"
        self.face_status_label.configure(text=text, foreground=color)
    
    def update_hand_status(self, detected):
        """손 감지 상태 업데이트"""
        self.hand_detected = detected
        color = "green" if detected else "red"
        text = "손 감지: 예" if detected else "손 감지: 아니오"
        self.hand_status_label.configure(text=text, foreground=color)
    
    def update_detected_state(self, state_code, duration=0.0):
        """감지된 상태 업데이트"""
        if state_code is not None:
            state_text = STATE_KOREAN.get(state_code, "알 수 없음")
            self.state_status_label.configure(
                text=f"감지된 상태: {state_text} ({duration:.1f}초)", 
                foreground="blue"
            )
            self.current_state = state_code
        else:
            self.state_status_label.configure(
                text="감지된 상태: 없음", 
                foreground="blue"
            )
    
    def update_confidence(self, confidence):
        """신뢰도 업데이트"""
        self.confidence_label.configure(text=f"신뢰도: {confidence}%")
    
    def update_sample_count(self, count):
        """수집된 샘플 수 업데이트"""
        self.sample_count_label.configure(text=f"수집된 샘플: {count}")
    
    def update_model_status(self, loaded):
        """모델 상태 업데이트"""
        if loaded:
            self.model_status_label.configure(
                text="모델 상태: 로드됨", 
                foreground="green"
            )
        else:
            self.model_status_label.configure(
                text="모델 상태: 로드되지 않음", 
                foreground="red"
            )
    
    def update_status(self, message, warning=False):
        """상태 메시지 업데이트"""
        color = "red" if warning else "black"
        self.message_label.configure(text=message, foreground=color)
        print(message)  # 콘솔에도 출력
    
    def set_start_stop_button_state(self, is_running):
        """시작/중지 버튼 상태 설정"""
        if is_running:
            self.start_stop_button.configure(text="중지")
        else:
            self.start_stop_button.configure(text="시작")
    
    def set_collection_button_state(self, is_collecting):
        """데이터 수집 버튼 상태 설정"""
        self.collection_active = is_collecting
        if is_collecting:
            self.collection_button.configure(text="데이터 수집 중지")
        else:
            self.collection_button.configure(text="데이터 수집 시작")
    
    def set_analysis_button_state(self, is_analyzing):
        """상태 감지 버튼 상태 설정"""
        self.analysis_active = is_analyzing
        if is_analyzing:
            self.analysis_button.configure(text="상태 감지 중지")
        else:
            self.analysis_button.configure(text="상태 감지 시작")
    
    def set_training_button_state(self, is_training):
        """모델 학습 버튼 상태 설정"""
        if is_training:
            self.training_button.configure(text="학습 중...", state="disabled")
        else:
            self.training_button.configure(text="모델 학습 시작", state="normal")
    
    def show_webcam_error(self):
        """웹캠 오류 메시지 표시"""
        messagebox.showerror(
            "웹캠 오류", 
            "웹캠을 열 수 없습니다. 웹캠 연결을 확인하세요."
        )
    
    def close_window(self):
        """창 닫기 처리"""
        if messagebox.askokcancel("종료", "프로그램을 종료하시겠습니까?"):
            if self.on_closing:
                self.on_closing()
            self.root.destroy()
    
    def run(self):
        """GUI 실행"""
        self.root.mainloop()


# 테스트 코드
if __name__ == "__main__":
    gui = FacialStateGUI()
    gui.update_status("테스트 모드")
    gui.run()
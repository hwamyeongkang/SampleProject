"""
Configuration file for facial state detection system.
Contains constants, thresholds, and color settings.
"""

# 상태 감지 조건 설정
BORED_THRESHOLD = 0.8        # 얼굴이 20% 이상 작아지면 지루함 (0.8배 이하)
YAWN_THRESHOLD = 0.05        # 입이 5cm 이상 벌어진 것으로 간주
YAWN_DURATION = 3.0          # 하품 감지 유지 시간 (초)
THINKING_DURATION = 5.0      # 고민 중 상태 유지 시간 (초)
EYEBROW_THRESHOLD = 0.9      # 미간 거리가 10% 이상 짧아지면 불만족 (0.9배 이하)
MOUTH_UP_THRESHOLD = 0.01    # 입 높이가 1cm 이상 올라가면 불만족

# 상태 코드 정의
STATE_LABELS = {
    0: 'satisfied',    # 만족 (기본 상태)
    1: 'bored',        # 지루함
    2: 'tired',        # 피곤함
    3: 'thinking',     # 고민 중
    4: 'dissatisfied'  # 불만족
}

# 한글 상태 이름
STATE_KOREAN = {
    0: '만족(기본)',
    1: '지루함',
    2: '피곤함',
    3: '고민 중',
    4: '불만족'
}

# 상태별 색상 (BGR 형식)
STATE_COLORS = {
    0: (46, 204, 113),   # 만족: 녹색
    1: (52, 152, 219),   # 지루함: 파란색
    2: (0, 0, 255),      # 피곤함: 빨간색
    3: (255, 165, 0),    # 고민 중: 주황색
    4: (142, 68, 173)    # 불만족: 보라색
}

# GUI 색상 (RGB 형식)
GUI_COLORS = {
    'bg_dark': '#2c3e50',      # 배경색 (어두운)
    'bg_mid': '#34495e',       # 배경색 (중간)
    'text': '#ecf0f1',         # 텍스트 기본색
    'highlight': '#3498db',    # 강조색
    'green': '#2ecc71',        # 녹색 (버튼 등)
    'red': '#e74c3c',          # 빨간색 (경고 등)
    'orange': '#e67e22',       # 주황색 (주의 등)
    'yellow': '#f1c40f',       # 노란색
    'purple': '#9b59b6'        # 보라색
}

# 데이터 파일 경로
DATA_FOLDER = "facial_data"
MODEL_FOLDER = "models"
CSV_FILE = f"{DATA_FOLDER}/facial_landmarks.csv"
MODEL_FILE = f"{MODEL_FOLDER}/state_detector_model.pkl"
SCREENSHOT_FOLDER = "screenshots"

# 기본 샘플 설정
DEFAULT_MAX_SAMPLES = 100  # 각 
import os
import sys
import time
import threading
import subprocess
import cv2
import psutil
import traceback
import webbrowser

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys

from face_detector2 import FaceDetector
from gui import FacialStateGUI


class FacialYawnDetectorApp:
    def __init__(self):
        self.gui = FacialStateGUI()
        self.register_event_handlers()
        self.face_detector = FaceDetector()

        self.webcam_available = False
        for camera_index in range(3):
            print(f"카메라 인덱스 {camera_index} 시도 중...")
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ret, test_frame = self.cap.read()
                if ret:
                    self.webcam_available = True
                    print(f"웹캠 초기화 성공 (인덱스: {camera_index}, 크기: {test_frame.shape})")
                    break
                else:
                    self.cap.release()
                    print(f"카메라 {camera_index}에서 프레임을 읽을 수 없습니다.")

        if not self.webcam_available:
            print("웹캠을 열 수 없습니다.")
            self.gui.show_webcam_error()

        self.is_running = False
        self.current_frame = None
        self.processed_frame = None
        self.browser_opened = False
        self.driver = None

        self.face_detected = False
        self.yawn_start_time = None
        self.is_yawning = False
        self.yawn_detected = False
        self.yawn_threshold = 0.02  # 2cm에 해당하는 정규화된 값 (화면에 따라 조정 필요)
        self.yawn_time_threshold = 3.0  # 3초

    def register_event_handlers(self):
        """GUI 이벤트 핸들러 등록"""
        self.gui.on_start_stop = self.toggle_start_stop
        self.gui.on_closing = self.on_closing

    def toggle_start_stop(self):
        """시작/중지 버튼 동작 처리"""
        if self.is_running:
            self.is_running = False
            self.gui.set_start_stop_button_state(False)
            self.close_browser()
            self.gui.update_status("중지됨")
            self.yawn_detected = False
            self.is_yawning = False
        else:
            if not self.webcam_available:
                self.gui.update_status("웹캠을 사용할 수 없습니다.", warning=True)
                return
                
            self.is_running = True
            self.gui.set_start_stop_button_state(True)
            self.gui.update_status("실행 중... 얼굴 감지 대기중")
            self.yawn_detected = False
            self.is_yawning = False
            self.update_webcam()

    def play_beta_wave_music(self):
        """베타파 음악 재생 (Pixabay에서 검색 및 자동 재생)"""
        try:
            self.gui.update_status("베타파 음악 검색 및 재생 준비 중...")
            print("🎵 베타파 음악 검색 및 재생 시작")
            
            # Selenium 웹드라이버 초기화
            options = webdriver.ChromeOptions()
            options.add_argument("--start-maximized")
            options.add_argument("--autoplay-policy=no-user-gesture-required")  # 자동 재생 정책 설정
            
            self.driver = webdriver.Chrome(options=options)
            self.browser_opened = True
            
            # Pixabay 메인 페이지로 이동
            self.driver.get("https://pixabay.com/ko/music/")
            self.gui.update_status("Pixabay 뮤직 페이지 로딩 중...")
            time.sleep(3)  # 페이지 로딩 대기
            
            # 쿠키 허용 버튼 클릭 시도
            try:
                cookie_script = """
                    var cookieKeywords = ['쿠키', 'Cookie', 'Accept', '수락', '동의', 'Agree', 'OK'];
                    
                    // 텍스트 기반 버튼 검색
                    var buttons = document.querySelectorAll('button, a[role="button"], div[role="button"]');
                    for(var i=0; i < buttons.length; i++) {
                        var buttonText = buttons[i].innerText || buttons[i].textContent;
                        if(buttonText) {
                            for(var j=0; j < cookieKeywords.length; j++) {
                                if(buttonText.includes(cookieKeywords[j])) {
                                    buttons[i].click();
                                    return true;
                                }
                            }
                        }
                    }
                    
                    // ID 또는 클래스 기반 버튼 검색
                    var selectors = [
                        '#onetrust-accept-btn-handler',
                        '.accept-cookies',
                        '#accept-cookies',
                        '.cookie-accept',
                        '.cc-accept',
                        '.cc-approve'
                    ];
                    
                    for(var i=0; i < selectors.length; i++) {
                        var element = document.querySelector(selectors[i]);
                        if(element) {
                            element.click();
                            return true;
                        }
                    }
                    
                    return false;
                """
                
                cookie_clicked = self.driver.execute_script(cookie_script)
                if cookie_clicked:
                    print("✅ 쿠키 허용 버튼 클릭 성공")
                    time.sleep(2)
            except Exception as e:
                print(f"쿠키 처리 중 오류 (무시): {e}")
            
            # 검색창 찾기 및 '베타파' 검색어 입력
            try:
                # XPath로 검색창 찾기
                search_input = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="app"]/div[1]/div[1]/div[2]/div[2]/div[2]/form/div[1]/input'))
                )
                
                # 검색창에 포커스 주기
                self.driver.execute_script("arguments[0].focus();", search_input)
                time.sleep(0.5)
                
                # 기존 텍스트 지우기 (필요한 경우)
                search_input.clear()
                time.sleep(0.5)
                
                # '베타파' 입력
                search_input.send_keys("베타파")
                print("검색창에 '베타파' 입력 완료")
                time.sleep(1)
                
                # Enter 키 입력하여 검색 실행
                search_input.send_keys(webdriver.Keys.RETURN)
                print("검색 실행")
                self.gui.update_status("'베타파' 검색 중...")
                
                # 검색 결과 로딩 대기
                time.sleep(5)
                
                # 검색 결과 확인
                current_url = self.driver.current_url
                print(f"현재 URL: {current_url}")
                
                # 첫 번째 음악 재생 시도
                try:
                    # 정확한 전체 XPath 사용
                    full_xpath = "/html/body/div[1]/div[1]/div/div[2]/div[2]/div/div[1]/div[1]/div[1]"
                    
                    # 재생 버튼의 컨테이너 찾기
                    play_container = WebDriverWait(self.driver, 15).until(
                        EC.presence_of_element_located((By.XPATH, full_xpath))
                    )
                    print("재생 버튼 컨테이너 찾음!")
                    
                    # 컨테이너 내부의 버튼 찾기
                    try:
                        # 컨테이너 내부에서 버튼 찾기
                        play_button = play_container.find_element(By.TAG_NAME, "button")
                        print("컨테이너 내부에서 버튼 찾음!")
                    except:
                        # 직접 전체 경로로 버튼 찾기
                        try:
                            play_button = self.driver.find_element(By.XPATH, f"{full_xpath}/button")
                            print("버튼 직접 찾음!")
                        except:
                            # 클릭 가능한 모든 요소 찾기
                            play_button = play_container
                            print("컨테이너를 버튼으로 사용")
                    
                    # 버튼이 보이도록 스크롤
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", play_button)
                    time.sleep(2)  # 스크롤 후 대기
                    
                    # 버튼 클릭 시도
                    self.driver.execute_script("arguments[0].click();", play_button)
                    print("음악 재생 버튼 클릭")
                    time.sleep(2)
                    
                    # JavaScript를 사용하여 모든 오디오 재생 시도
                    self.driver.execute_script("""
                        var audios = document.querySelectorAll('audio');
                        for(var i=0; i<audios.length; i++) {
                            try {
                                audios[i].play();
                                audios[i].muted = false;
                                audios[i].volume = 1.0;
                                console.log('Audio ' + i + ' play attempted');
                            } catch(e) {
                                console.log('Error playing audio ' + i + ': ' + e);
                            }
                        }
                    """)
                    
                    self.gui.update_status("베타파 음악이 재생 중입니다.")
                    print("🎵 베타파 음악 재생 시도 완료")
                    
                except Exception as e:
                    print(f"음악 재생 버튼 클릭 중 오류: {e}")
                    traceback.print_exc()
                    self.gui.update_status("자동 재생 실패. 브라우저에서 수동으로 음원을 재생해주세요.", warning=True)
                    
                    # 사용자에게 클릭 안내 메시지 표시
                    self.driver.execute_script("""
                        var div = document.createElement('div');
                        div.style.position = 'fixed';
                        div.style.top = '50%';
                        div.style.left = '50%';
                        div.style.transform = 'translate(-50%, -50%)';
                        div.style.padding = '20px';
                        div.style.background = 'rgba(255,255,0,0.9)';
                        div.style.color = 'black';
                        div.style.fontSize = '24px';
                        div.style.fontWeight = 'bold';
                        div.style.zIndex = '9999';
                        div.style.borderRadius = '10px';
                        div.innerHTML = '음악 재생 버튼을 클릭해주세요<br>↓';
                        document.body.appendChild(div);
                        setTimeout(() => { div.style.opacity = 0; div.style.transition = 'opacity 1s'; }, 8000);
                    """)
                    
            except Exception as e:
                print(f"검색창 조작 중 오류: {e}")
                traceback.print_exc()
                self.gui.update_status("검색창을 찾을 수 없습니다. 수동으로 검색해주세요.", warning=True)
                
                # 실패 시 직접 검색 결과 URL로 이동
                try:
                    self.driver.get("https://pixabay.com/ko/music/search/?search=%EB%B2%A0%ED%83%80%ED%8C%8C")
                    self.gui.update_status("베타파 검색 결과 페이지로 이동합니다.")
                    time.sleep(5)
                    
                    # 첫 번째 음악 클릭 시도
                    try:
                        play_script = """
                            try {
                                var baseXPath = '//*[@id=\"app\"]/div[1]/div/div[2]/div[2]/div';
                                for (var i = 1; i <= 5; i++) {
                                    var buttonXPath = baseXPath + '/div[' + i + ']/div[1]/div[1]/button';
                                    var button = document.evaluate(buttonXPath, document, null, 
                                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                    if (button) {
                                        button.scrollIntoView({block: 'center'});
                                        setTimeout(() => { button.click(); }, 500);
                                        return {success: true, method: 'xpath'};
                                    }
                                }
                            } catch (e) {}

                            try {
                                var playButtons = document.querySelectorAll('button.play-button, button[aria-label*="play"], button[title*="play"]');
                                if (playButtons.length > 0) {
                                    playButtons[0].scrollIntoView({block: 'center'});
                                    setTimeout(() => { playButtons[0].click(); }, 500);
                                    return {success: true, method: 'css'};
                                }
                            } catch (e) {}

                            try {
                                var audios = document.querySelectorAll('audio');
                                if (audios.length > 0) {
                                    for (var i = 0; i < audios.length; i++) {
                                        try { audios[i].play(); } catch (e) {}
                                    }
                                    return {success: audios.length > 0, method: 'audio'};
                                }
                            } catch (e) {}

                            return {success: false, method: 'none'};
                        """
                        
                        result = self.driver.execute_script(play_script)
                        if result.get('success'):
                            print(f"✅ 음악 재생 성공 (방식: {result.get('method')})")
                            self.gui.update_status("베타파 음악이 재생 중입니다.")
                        else:
                            self.gui.update_status("자동 재생 실패. 수동으로 클릭해주세요.", warning=True)
                    except:
                        print("대체 방법으로도 재생 실패")
                except:
                    # 모든 방법 실패 시 기본 브라우저로 열기
                    try:
                        webbrowser.open("https://pixabay.com/ko/music/search/?search=%EB%B2%A0%ED%83%80%ED%8C%8C")
                        self.gui.update_status("브라우저가 열렸습니다. 수동으로 음원을 재생해주세요.")
                    except:
                        pass
                        
        except Exception as e:
            print(f"음악 재생 오류: {e}")
            traceback.print_exc()
            self.gui.update_status(f"음악 재생 오류: {e}", warning=True)
            
            # 실패 시 일반 브라우저로 열기
            try:
                webbrowser.open("https://pixabay.com/ko/music/search/?search=%EB%B2%A0%ED%83%80%ED%8C%8C")
                self.gui.update_status("브라우저가 열렸습니다. 수동으로 음원을 재생해주세요.")
            except:
                pass

    def close_browser(self):
        """브라우저 닫기 시도"""
        if not self.browser_opened:
            return
        
        try:
            # Selenium 웹드라이버 종료
            if self.driver is not None:
                try:
                    self.driver.quit()
                    print("Selenium 웹드라이버 종료됨")
                except:
                    print("Selenium 웹드라이버 종료 중 오류 발생")
            
            # 추가적으로 Chrome 프로세스 종료 시도 (Windows의 경우)
            if os.name == 'nt':
                try:
                    os.system('taskkill /f /im chromedriver.exe')
                    os.system('taskkill /f /im chrome.exe')
                except:
                    pass
                
            self.browser_opened = False
            self.driver = None
            self.gui.update_status("브라우저가 종료되었습니다.")
        except Exception as e:
            print(f"브라우저 종료 오류: {e}")
            traceback.print_exc()

    def update_webcam(self):
        """웹캠 프레임 업데이트 및 얼굴/하품 감지"""
        if not self.is_running or not self.webcam_available:
            return
            
        try:
            ret, self.current_frame = self.cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                self.gui.update_status("웹캠에서 프레임을 읽을 수 없습니다.", warning=True)
                self.is_running = False
                self.gui.set_start_stop_button_state(False)
                return
                
            self.current_frame = cv2.flip(self.current_frame, 1)  # 좌우 반전

            # 얼굴 감지
            face_detected, frame_with_face, face_data = self.face_detector.detect_face(self.current_frame)
            self.processed_frame = frame_with_face
            self.face_detected = face_detected
            
            # GUI 상태 업데이트
            self.gui.update_face_status(face_detected)
            
            # 얼굴이 감지되면 하품 감지 로직 적용
            if face_detected:
                # 얼굴 메트릭 데이터 분석
                face_metrics = face_data.get('face_metrics', {})
                mouth_open_height = face_metrics.get('mouth_open_height', 0)
                
                # 프레임 크기 (픽셀 단위의 입 높이 계산용)
                h, w, _ = self.current_frame.shape
                
                # 입 높이를 픽셀로 계산 (대략적인 변환)
                mouth_height_px = mouth_open_height * w  # 정규화된 값을 픽셀로 역변환
                
                # 현재 시간
                current_time = time.time()
                
                # 하품 감지 상태 업데이트
                self.detect_yawn(mouth_open_height, mouth_height_px, current_time)
                
                # 하품 상태 시각화
                if self.is_yawning:
                    # 하품 중 상태 텍스트
                    yawn_duration = current_time - self.yawn_start_time
                    yawn_text = f"하품 감지중: {yawn_duration:.1f}초 / {self.yawn_time_threshold}초"
                    cv2.putText(self.processed_frame, yawn_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    # 입 높이 표시
                    cv2.putText(self.processed_frame, f"입 높이: {mouth_height_px:.1f}px", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                # 피곤함 상태 표시
                if self.yawn_detected:
                    tired_text = "피곤함 상태 감지됨! 베타파 음악 재생 중"
                    cv2.putText(self.processed_frame, tired_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 최종 처리된 프레임 표시
            self.gui.update_webcam_display(self.processed_frame)
            
            # 재귀 호출 (10ms 후 다시 업데이트)
            self.gui.root.after(10, self.update_webcam)
        except Exception as e:
            print(f"웹캠 업데이트 중 오류: {e}")
            traceback.print_exc()
            self.gui.update_status(f"웹캠 업데이트 중 오류: {e}", warning=True)
            self.gui.root.after(100, self.update_webcam)  # 오류 발생 시 더 긴 간격으로 재시도

    def detect_yawn(self, mouth_height, mouth_height_px, current_time):
        """하품 감지 (입이 2cm 이상 벌어진 상태가 3초 이상 지속되는지 확인)"""
        # 이미 피곤함 상태가 감지되었으면 추가 감지 중단
        if self.yawn_detected:
            return
        
        # 입이 기준치 이상 벌어졌는지 확인
        if mouth_height > self.yawn_threshold:
            # 하품 시작 시간 기록
            if not self.is_yawning:
                self.is_yawning = True
                self.yawn_start_time = current_time
                print(f"입이 벌어짐 감지: {mouth_height_px:.1f}px")
                self.gui.update_status(f"입이 벌어짐 감지: {mouth_height_px:.1f}px")
            
            # 하품 지속 시간 계산
            if self.yawn_start_time:
                yawn_duration = current_time - self.yawn_start_time
                
                # 하품이 기준 시간 이상 지속되면 피곤함으로 판단
                if yawn_duration >= self.yawn_time_threshold and not self.yawn_detected:
                    self.yawn_detected = True
                    print(f"피곤함 상태 감지! 하품 지속 시간: {yawn_duration:.1f}초")
                    self.gui.update_status(f"피곤함 상태 감지! 하품 지속 시간: {yawn_duration:.1f}초")
                    
                    # 베타파 음악 재생
                    if not self.browser_opened:
                        self.play_beta_wave_music()
        else:
            # 입이 다시 닫히면 하품 상태 초기화
            if self.is_yawning:
                self.is_yawning = False
                self.gui.update_status("입이 닫힘 감지")
                print("입이 닫힘 감지")

    def on_closing(self):
        """프로그램 종료 처리"""
        self.is_running = False
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'face_detector'):
            self.face_detector.close()
            
        # 브라우저 종료
        self.close_browser()
            
        print("프로그램 종료")

    def run(self):
        """애플리케이션 실행"""
        self.gui.run()

if __name__ == "__main__":
    # 필요한 패키지 확인
    missing_packages = []
    
    try:
        import psutil
    except ImportError:
        missing_packages.append("psutil")
    
    try:
        from selenium import webdriver
    except ImportError:
        missing_packages.append("selenium")
    
    # 패키지가 없으면 설치 안내
    if missing_packages:
        print(f"다음 패키지가 설치되어 있지 않습니다: {', '.join(missing_packages)}")
        print("다음 명령어로 필요한 패키지를 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\n패키지 설치 후 프로그램을 다시 실행하세요.")
        sys.exit(1)
    
    print("모든 필요 패키지가 설치되어 있습니다.")
    print("프로그램 시작 중...")
    
    app = FacialYawnDetectorApp()
    app.run()
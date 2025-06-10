import os
from flask import Flask, request, render_template, Response, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import time
import threading
import queue
import atexit

app = Flask(__name__)
CORS(app)

# --- Konfigurasi Model ---
MODEL_PATH = 'best_merem.pt' # Menggunakan model .pt secara langsung

# --- Muat model YOLO .pt secara langsung ---
try:
    # Memuat model .pt secara langsung. Ini akan menggunakan backend PyTorch.
    # Pastikan Anda telah menginstal torch (pip install torch --index-url https://download.pytorch.org/whl/cpu jika hanya CPU)
    detection_model = YOLO(MODEL_PATH)
    print(f"Model YOLO '{MODEL_PATH}' (.pt) berhasil dimuat untuk inferensi.")
except Exception as e:
    print(f"Error memuat model YOLO dari '{MODEL_PATH}': {e}")
    print("Pastikan file model .pt ada di direktori yang benar dan merupakan model YOLO yang valid.")
    print("Untuk penggunaan CPU, pastikan Anda menginstal PyTorch versi CPU (contoh: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu)")
    exit()

# --- Folder Konfigurasi ---
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# --- Queue untuk Komunikasi Antar Thread ---
raw_frame_queue = queue.Queue(maxsize=5)
processed_frame_queue = queue.Queue(maxsize=5)

# --- Flag dan Variabel Global untuk Mengontrol Thread ---
camera_thread_running = False
detection_thread_running = False
camera_capture = None # Variabel global untuk objek VideoCapture

# --- Thread Kamera (Reader) ---
def camera_reader_thread():
    global camera_capture, camera_thread_running
    camera_capture = cv2.VideoCapture(0)
    if not camera_capture.isOpened():
        print("Error: Kamera server tidak bisa dibuka oleh thread reader. Pastikan kamera terhubung.")
        camera_thread_running = False
        return

    print("Thread pembaca kamera dimulai.")
    camera_thread_running = True
    while camera_thread_running:
        success, frame = camera_capture.read()
        if not success:
            print("Gagal membaca frame dari kamera di thread reader. Mungkin kamera terputus.")
            break
        try:
            raw_frame_queue.put(frame, timeout=1)
        except queue.Full:
            pass

    print("Thread pembaca kamera berhenti.")
    if camera_capture:
        camera_capture.release()
    camera_thread_running = False

# --- Thread Deteksi (Processor) ---
def detection_processor_thread():
    global detection_thread_running
    print("Thread pemroses deteksi dimulai.")
    detection_thread_running = True
    prev_frame_time = 0
    new_frame_time = 0

    while detection_thread_running:
        try:
            frame = raw_frame_queue.get(timeout=1)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps_text = f'FPS: {int(fps)}'

            # Lakukan inferensi pada frame menggunakan model .pt
            # device='cpu' tetap digunakan untuk memastikan inferensi di CPU
            results = detection_model.predict(frame, conf=0.25, device='cuda', verbose=False, imgsz=640)

            annotated_frame = frame
            for r in results:
                annotated_frame = r.plot()
                break

            cv2.putText(annotated_frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            try:
                processed_frame_queue.put(annotated_frame, timeout=1)
            except queue.Full:
                pass

        except queue.Empty:
            time.sleep(0.01)
            continue
        except Exception as e:
            print(f"Error di thread pemroses deteksi: {e}")
            time.sleep(0.1)

    print("Thread pemroses deteksi berhenti.")
    detection_thread_running = False

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index2.html_cam_server.html') # Mengacu pada file HTML yang akan diberikan

@app.route('/video_feed')
def video_feed():
    global camera_thread_running, detection_thread_running

    if not camera_thread_running:
        threading.Thread(target=camera_reader_thread, daemon=True).start()
        time.sleep(0.5)

    if not detection_thread_running:
        threading.Thread(target=detection_processor_thread, daemon=True).start()
        time.sleep(0.5)

    def generate_frames():
        while True:
            try:
                frame = processed_frame_queue.get(timeout=5)

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    print("Gagal meng-encode frame di generator.")
                    continue

                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"Error di generator video feed: {e}")
                time.sleep(0.1)
                break

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 2. Unggah Gambar Endpoint ---
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    if file:
        try:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(image)

            # Lakukan inferensi menggunakan model .pt
            results = detection_model(img_np, conf=0.4, device='cuda')

            annotated_img_np = results[0].plot()

            annotated_img_pil = Image.fromarray(annotated_img_np[..., ::-1])

            byte_arr = io.BytesIO()
            annotated_img_pil.save(byte_arr, format='JPEG')
            byte_arr.seek(0)

            return send_file(byte_arr, mimetype='image/jpeg')

        except Exception as e:
            print(f"Error saat memproses unggahan gambar: {e}")
            return f"Gagal memproses gambar: {e}", 500

if __name__ == '__main__':
    def stop_all_threads():
        global camera_thread_running, detection_thread_running, camera_capture
        print("Menghentikan thread kamera dan deteksi...")
        camera_thread_running = False
        detection_thread_running = False
        time.sleep(2)
        if camera_capture:
            camera_capture.release()
            print("Kamera dilepaskan.")
        print("Semua thread dihentikan.")

    atexit.register(stop_all_threads)

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
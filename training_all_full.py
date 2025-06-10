from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")

    model.train(
        
        data=r"data.yaml",

        epochs=200,
        batch=16,
        imgsz=800, #640
        device="CUDA",
        optimizer="SGD",  # Tetap Adam, pertimbangkan SGD jika perlu
        momentum=0.9,
        lr0=0.01,
        lrf=0.1,           # Skeduler LR (pastikan cosine annealing aktif)
        patience=20,
        dropout=0.4,
        # --- Penambahan Parameter Baru (Sangat Direkomendasikan) ---
        weight_decay=0.0015,  # Regularisasi L2
        # Parameter Augmentasi (sesuaikan nilai sesuai data Anda)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.1,       # Atau sangat kecil (misal 0.5) untuk ekspresi wajah
        translate=0.1,
        scale=0.5,
        shear=0.01,         # Atau sangat kecil (misal 0.5)
        perspective=0.000001,   # Atau sangat kecil
        mixup=0.5,        # Coba aktifkan mixup sedikit
        mosaic=1.0,      # Biasanya default sudah aktif
        # --- Strategi Kritis (Bukan Parameter Langsung di config ini, tapi di implementasi training) ---
        # Class Weights di Loss Function (manual calculation based on data imbalance)
        # Focal Loss (jika framework mendukung atau bisa diimplementasikan)
        # --- Tambahan untuk Cutout ---
        copy_paste=0.001, # Contoh: probabilitas 50% untuk menerapkan cutout
        # Nilai umum antara 0.1 hingga 0.5
        # Anda juga mungkin menemukan parameter lain yang terkait dengan kekuatan cutout,
        # seperti 'cutout_factor' atau 'cutout_fill'.
        # Cek dokumentasi YOLOv8 terbaru untuk detail lebih lanjut.
        
    )

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Untuk aman di Windows saat multiprocessing
    main()

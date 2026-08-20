import collections
import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ==========================================================
# 1. 1D MESAFE YUMUŞATMA KALMAN FİLTRESİ
# ==========================================================
class DistanceKalman1D:
    def __init__(self, init_distance):
        self.x = float(init_distance)
        self.P = 4.0
        self.Q = 0.4
        self.R = 3.0

    def update(self, z_measured):
        self.P = self.P + self.Q
        r_dyn = self.R + (z_measured / 15.0)**2
        K = self.P / (self.P + r_dyn)
        self.x = self.x + K * (z_measured - self.x)
        self.P = (1.0 - K) * self.P
        return self.x


# ==========================================================
# 2. GELİŞMİŞ ŞERİT VE YÖRÜNGE MOTORU
# ==========================================================
class AdvancedLaneEngine:
    def __init__(self, img_size=(1280, 720)):
        self.width, self.height = img_size
        self.ym_per_pix = 30.0 / self.height
        self.xm_per_pix = 3.7 / 700.0
        self.nwindows = 9
        self.margin = 70
        self.minpix = 40

        self.buffer_len = 8
        self.left_fits = collections.deque(maxlen=self.buffer_len)
        self.right_fits = collections.deque(maxlen=self.buffer_len)

        # BEV Kuşbakışı Perspektif Matrisi
        src = np.float32([
            [self.width * 0.20, self.height * 0.92],
            [self.width * 0.44, self.height * 0.65],
            [self.width * 0.56, self.height * 0.65],
            [self.width * 0.82, self.height * 0.92]
        ])
        dst = np.float32([
            [self.width * 0.22, self.height],
            [self.width * 0.22, 0],
            [self.width * 0.78, 0],
            [self.width * 0.78, self.height]
        ])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.last_binary_mask = None

    def process(self, frame):
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        binary = np.zeros_like(s_channel)
        white_mask = (l_channel >= 195) & (l_channel <= 255)
        yellow_mask = (s_channel >= 100) & (s_channel <= 255) & (l_channel >= 120)
        binary[white_mask | yellow_mask] = 1
        self.last_binary_mask = binary

        warped = cv2.warpPerspective(binary, self.M, (self.width, self.height), flags=cv2.INTER_NEAREST)

        histogram = np.sum(warped[int(warped.shape[0] * 0.55):, :], axis=0)
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = int(warped.shape[0] // self.nwindows)
        nonzero = warped.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        leftx_curr, rightx_curr = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(self.nwindows):
            win_y_low = warped.shape[0] - (window + 1) * window_height
            win_y_high = warped.shape[0] - window * window_height

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                         (nonzerox >= leftx_curr - self.margin) & (nonzerox < leftx_curr + self.margin)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= rightx_curr - self.margin) & (nonzerox < rightx_curr + self.margin)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > self.minpix:
                leftx_curr = int(np.mean(nonzerox[good_left]))
            if len(good_right) > self.minpix:
                rightx_curr = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Şerit Geometri Doğrulama
        if len(left_lane_inds) >= self.minpix and len(right_lane_inds) >= self.minpix:
            left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
            right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)

            y_check = self.height * 0.85
            lane_width_bottom = (right_fit[0] * y_check**2 + right_fit[1] * y_check + right_fit[2]) - \
                                (left_fit[0] * y_check**2 + left_fit[1] * y_check + left_fit[2])

            if 300 < lane_width_bottom < 900:
                self.left_fits.append(left_fit)
                self.right_fits.append(right_fit)

        if len(self.left_fits) == 0:
            return frame, 0.0, 0.0

        avg_left_fit = np.mean(self.left_fits, axis=0)
        avg_right_fit = np.mean(self.right_fits, axis=0)

        ploty = np.linspace(int(self.height * 0.62), int(self.height * 0.86), 30)
        left_fitx = avg_left_fit[0] * ploty**2 + avg_left_fit[1] * ploty + avg_left_fit[2]
        right_fitx = avg_right_fit[0] * ploty**2 + avg_right_fit[1] * ploty + avg_right_fit[2]

        left_fitx = np.clip(left_fitx, 0, self.width - 1)
        right_fitx = np.clip(right_fitx, 0, self.width - 1)

        color_warp = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int32([pts]), (0, 220, 100))
        cv2.polylines(color_warp, np.int32([pts_left]), False, (0, 230, 255), 6)
        cv2.polylines(color_warp, np.int32([pts_right]), False, (0, 230, 255), 6)

        center_fitx = (left_fitx + right_fitx) / 2.0
        for i in range(0, len(ploty) - 6, 8):
            cv2.line(color_warp, (int(center_fitx[i]), int(ploty[i])), 
                     (int(center_fitx[i + 4]), int(ploty[i + 4])), (255, 255, 255), 3)

        unwarped = cv2.warpPerspective(color_warp, self.Minv, (self.width, self.height), flags=cv2.INTER_NEAREST)
        result = cv2.addWeighted(frame, 1, unwarped, 0.25, 0)

        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2.0
        offset = ((self.width / 2.0) - lane_center) * self.xm_per_pix

        y_eval = np.max(ploty)
        curvature = ((1 + (2 * avg_left_fit[0] * y_eval * self.ym_per_pix + avg_left_fit[1])**2)**1.5) / np.absolute(2 * avg_left_fit[0] + 1e-6)

        return result, curvature, offset


# ==========================================================
# 3. KESİNTİSİZ KÖR NOKTA VE KAPUT KORUMALI 3D ALGI MOTORU
# ==========================================================
class Spatial3DObjectTracker:
    def __init__(self, img_size=(1280, 720)):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)

        self.width, self.height = img_size
        
        self.target_classes = {
            0: "Yaya", 
            1: "Bisiklet", 
            2: "Araba", 
            3: "Motor", 
            5: "Otobus", 
            7: "Kamyon",
            9: "Trafik Isigi",
            11: "DUR Tabelasi"
        }
        
        self.camera_height = 1.35
        self.horizon_y = self.height * 0.56
        self.focal_length_px = self.height * 1.15

        self.filters = {}
        self.cached_objects = []

    def update(self, frame, run_detection=True):
        if not run_detection:
            return self.cached_objects

        results = self.model.track(
            frame, 
            conf=0.20, 
            imgsz=640, 
            persist=True, 
            tracker="bytetrack.yaml", 
            device=self.device, 
            verbose=False
        )[0]
        
        tracked_objects = []

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy() if results.boxes.id is not None else [-1] * len(boxes)

            for box, track_id, cls_id, conf in zip(boxes, track_ids, clss, confs):
                if cls_id in self.target_classes:
                    # 1. Katı Güven Filtresi
                    if cls_id in [2, 3, 5, 7] and conf < 0.45:
                        continue
                    elif cls_id in [0, 9, 11] and conf < 0.25:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    bw, bh = x2 - x1, y2 - y1

                    # 2. EGO-ARAÇ KAPUTU VE ÖN CAM YANSIMA FİLTRESİ (Kesin Çözüm)
                    # Ekranın alt tabanına yapışan kaput yansımalarını eler
                    if y1 > self.height * 0.72:
                        continue
                    if y2 > self.height * 0.85 and bw > self.width * 0.30:
                        continue

                    # 3. Boyut ve En-Boy Oranı Kontrolü
                    if bw < 25 or bh < 25:
                        continue
                    aspect_ratio = bw / float(bh)
                    if cls_id in [2, 5, 7] and (aspect_ratio < 0.5 or aspect_ratio > 3.2):
                        continue

                    delta_y = y2 - self.horizon_y
                    if delta_y > 1.5:
                        raw_dist = (self.camera_height * self.focal_length_px) / delta_y
                    else:
                        raw_dist = 180.0
                    
                    raw_dist = float(np.clip(raw_dist, 1.0, 180.0))

                    # Kaputun hemen önündeki yapay sıfır mesafeleri filtrele
                    if raw_dist < 4.5 and cls_id in [2, 3, 5, 7]:
                        continue

                    filter_key = track_id if track_id != -1 else (x1 // 20, y1 // 20)
                    if filter_key not in self.filters:
                        self.filters[filter_key] = DistanceKalman1D(raw_dist)
                    
                    dist_y = self.filters[filter_key].update(raw_dist)

                    # Pinhole Kamera Yanal Mesafe (dist_x)
                    box_center_x = (x1 + x2) / 2.0
                    dist_x = ((box_center_x - (self.width / 2.0)) * dist_y) / self.focal_length_px

                    is_vehicle = cls_id in [2, 3, 5, 7]
                    
                    # Şerit & Kör Nokta Ayrımı
                    is_lead = is_vehicle and (abs(dist_x) <= 1.2) and (dist_y < 80.0)
                    is_blindspot = is_vehicle and (1.2 < abs(dist_x) <= 5.0) and (dist_y <= 35.0)
                    is_bsd_critical = is_blindspot and (dist_y <= 8.0)

                    tracked_objects.append({
                        "id": track_id,
                        "class_id": cls_id,
                        "bbox": (x1, y1, x2, y2),
                        "name": self.target_classes[cls_id],
                        "dist_y": dist_y,
                        "dist_x": dist_x,
                        "is_lead": is_lead,
                        "is_blindspot": is_blindspot,
                        "is_bsd_critical": is_bsd_critical
                    })

        self.cached_objects = tracked_objects
        return tracked_objects

    def draw_pseudo_3d(self, frame, objects):
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            dist_y = obj["dist_y"]
            cls_id = obj["class_id"]
            obj_id = obj["id"]
            name = obj["name"]
            is_lead = obj["is_lead"]
            is_blindspot = obj["is_blindspot"]
            is_bsd_crit = obj["is_bsd_critical"]

            if cls_id == 0:
                color = (255, 0, 255)     # Magenta (Yaya)
            elif cls_id in [9, 11]:
                color = (0, 215, 255)     # Sarı (Tabela / Işık)
            elif is_bsd_crit:
                color = (0, 0, 255)       # Kırmızı (Kritik Kör Nokta: <= 8m)
            elif is_blindspot:
                color = (0, 140, 255)     # Turuncu (Kör Nokta: 8m - 35m)
            elif is_lead and dist_y < 12.0:
                color = (0, 0, 255)       # Kırmızı (Öndeki Araç Acil Takip)
            elif is_lead:
                color = (0, 255, 120)     # Yeşil (Lider Araç)
            else:
                color = (255, 190, 0)     # Mavi (Normal Araç)

            bw, bh = x2 - x1, y2 - y1
            depth_scale = int(np.clip(bw * 0.28, 3, 35))

            if cls_id in [0, 9, 11]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                bx1, by1 = x1 + depth_scale, y1 - depth_scale
                bx2, by2 = x2 + depth_scale, y2 - depth_scale
                cv2.line(frame, (x1, y1), (bx1, by1), color, 1)
                cv2.line(frame, (x2, y1), (bx2, by1), color, 1)
                cv2.line(frame, (x1, y2), (bx1, by2), color, 1)
                cv2.line(frame, (x2, y2), (bx2, by2), color, 1)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_bsd_crit else 2)

            id_str = f"ID:{obj_id} " if obj_id != -1 else ""
            
            if is_bsd_crit:
                tag_str = "[BSD TEHLIKE] "
            elif is_blindspot:
                tag_str = "[BSD] "
            elif is_lead:
                tag_str = "[LEAD] "
            else:
                tag_str = ""

            dist_str = f" | {dist_y:.1f}m" if cls_id not in [9, 11] else ""
            label = f"{tag_str}{id_str}{name}{dist_str}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            txt_color = (0, 0, 0) if color in [(0, 255, 120), (0, 215, 255)] else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, txt_color, 1, cv2.LINE_AA)

        return frame


# ==========================================================
# 4. TELEMETRİ, BEV RADAR VE PIP GÖSTERGE SİSTEMİ
# ==========================================================
def draw_telemetry_and_pip(frame, curvature, offset, objects, fps, device_name, lane_engine):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Sol Üst Telemetri Paneli
    cv2.rectangle(overlay, (20, 20), (370, 185), (12, 16, 22), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.rectangle(frame, (20, 20), (370, 185), (0, 255, 200), 2)

    cv2.putText(frame, f"AUTONOMOUS ADAS ({device_name.upper()})", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 220), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    curv_str = f"{int(curvature)} m" if curvature < 4000 else "Duz Yol"
    cv2.putText(frame, f"Yol Egriligi: {curv_str}", (35, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    yon = "Sag" if offset > 0 else "Sol"
    offset_color = (0, 255, 255) if abs(offset) < 0.40 else (0, 0, 255)
    cv2.putText(frame, f"Merkez Sapma: {abs(offset):.2f}m ({yon})", (35, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.48, offset_color, 1)

    bsd_count = sum(1 for o in objects if o["is_blindspot"])
    cv2.putText(frame, f"Takip: {len(objects)} Nesne | BSD: {bsd_count}", (35, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # 2. Sağ Alt Kuşbakışı Radar (100m Menzil)
    radar_w, radar_h = 200, 200
    rx1, ry1 = w - radar_w - 20, h - radar_h - 20
    rx2, ry2 = rx1 + radar_w, ry1 + radar_h

    radar_bg = frame[ry1:ry2, rx1:rx2].copy()
    cv2.rectangle(radar_bg, (0, 0), (radar_w, radar_h), (10, 15, 20), -1)
    frame[ry1:ry2, rx1:rx2] = cv2.addWeighted(radar_bg, 0.8, frame[ry1:ry2, rx1:rx2], 0.2, 0)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 180), 2)
    cv2.putText(frame, "BEV RADAR (100m)", (rx1 + 30, ry1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1)

    center_x = rx1 + int(radar_w / 2)
    car_y = ry2 - 25

    for dist_m in [30, 60, 100]:
        radius_px = int((dist_m / 100.0) * (radar_h - 45))
        cv2.circle(frame, (center_x, car_y), radius_px, (45, 65, 55), 1)

    cv2.line(frame, (center_x - 20, ry1 + 30), (center_x - 20, ry2 - 10), (0, 140, 255), 1)
    cv2.line(frame, (center_x + 20, ry1 + 30), (center_x + 20, ry2 - 10), (0, 140, 255), 1)
    cv2.rectangle(frame, (center_x - 6, car_y - 10), (center_x + 6, car_y + 6), (255, 200, 0), -1)

    for obj in objects:
        dx, dy = obj["dist_x"], obj["dist_y"]
        if dy <= 100.0:
            px = int(center_x + (dx / 10.0) * (radar_w / 2))
            py = int(car_y - (dy / 100.0) * (radar_h - 45))
            if rx1 + 4 < px < rx2 - 4 and ry1 + 25 < py < ry2 - 4:
                dot_color = (0, 0, 255) if obj["is_lead"] and dy < 12.0 else (0, 255, 120) if obj["is_lead"] else (255, 180, 0)
                cv2.circle(frame, (px, py), 5, dot_color, -1)

    # 3. Sağ Üst Şerit PIP Penceresi
    if lane_engine.last_binary_mask is not None:
        mask_w, mask_h = 190, 110
        mx1, my1 = w - mask_w - 20, 20
        mask_small = cv2.resize(lane_engine.last_binary_mask * 255, (mask_w, mask_h))
        mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        frame[my1:my1 + mask_h, mx1:mx1 + mask_w] = mask_colored
        cv2.rectangle(frame, (mx1, my1), (mx1 + mask_w, my1 + mask_h), (0, 255, 200), 2)
        cv2.putText(frame, "LANE THRESHOLD PIP", (mx1 + 10, my1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

    # 4. Kritik Mesafe Uyarısı (FCW)
    lead_cars = [obj for obj in objects if obj["is_lead"]]
    if lead_cars:
        lead = min(lead_cars, key=lambda x: x["dist_y"])
        if lead["dist_y"] < 11.0:
            cv2.rectangle(frame, (int(w / 2) - 220, int(h * 0.78)), (int(w / 2) + 220, int(h * 0.78) + 55), (0, 0, 255), -1)
            cv2.putText(frame, "! KRITIK TAKIP MESAFESI !", (int(w / 2) - 185, int(h * 0.78) + 36), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame


# ==========================================================
# 5. ANA YÜRÜTÜCÜ
# ==========================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, "video.mp4")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"HATA: '{video_path}' dosyasi bulunamadi!")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    img_size = (w, h)

    out_writer = cv2.VideoWriter(
        os.path.join(base_dir, "adas_demo_kaydi.mp4"), 
        cv2.VideoWriter_fourcc(*"mp4v"), 
        input_fps, 
        (w, h)
    )

    lane_engine = AdvancedLaneEngine(img_size)
    tracker = Spatial3DObjectTracker(img_size)
    
    frame_count = 0
    prev_time = time.time()

    print(f"Level 2+ ADAS Perception Pipeline Aktif ({tracker.device.upper()}). Cikis icin 'q' tusuna basin.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()

        # 1. Şerit & Ego Yörünge
        lane_frame, curvature, offset = lane_engine.process(frame)

        # 2. 3D Tel Kafes & Mesafe Kestirimi
        run_detection = (frame_count % 2 == 0)
        tracked_objects = tracker.update(lane_frame, run_detection=run_detection)
        obj_frame = tracker.draw_pseudo_3d(lane_frame, tracked_objects)

        # 3. FPS
        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time

        # 4. Telemetri HUD & PIP
        final_frame = draw_telemetry_and_pip(obj_frame, curvature, offset, tracked_objects, fps, tracker.device, lane_engine)

        out_writer.write(final_frame)
        cv2.imshow("ADAS - Level 2+ Perception Pipeline", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print("İşlem tamamlandı! Çıktı: adas_demo_kaydi.mp4")


if __name__ == "__main__":
    main()
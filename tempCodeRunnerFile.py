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
    """Monoküler mesafe ölçümündeki piksel titreşimlerini filtreler."""
    def __init__(self, init_distance):
        self.x = float(init_distance)
        self.P = 4.0
        self.Q = 0.4  # Süreç gürültüsü
        self.R = 3.0  # Ölçüm gürültüsü

    def update(self, z_measured):
        self.P = self.P + self.Q
        r_dyn = self.R + (z_measured / 15.0)**2
        K = self.P / (self.P + r_dyn)
        self.x = self.x + K * (z_measured - self.x)
        self.P = (1.0 - K) * self.P
        return self.x


# ==========================================================
# 2. HIZLI ŞERİT TESPİTİ (FastLaneDetector)
# ==========================================================
class FastLaneDetector:
    def __init__(self, img_size=(1280, 720)):
        self.width, self.height = img_size
        self.ym_per_pix = 30.0 / self.height
        self.xm_per_pix = 3.7 / 700.0
        self.nwindows = 8
        self.margin = 75
        self.minpix = 35

        self.buffer_len = 6
        self.left_fits = collections.deque(maxlen=self.buffer_len)
        self.right_fits = collections.deque(maxlen=self.buffer_len)

        src = np.float32([
            [self.width * 0.15, self.height],
            [self.width * 0.43, self.height * 0.65],
            [self.width * 0.57, self.height * 0.65],
            [self.width * 0.88, self.height]
        ])
        dst = np.float32([
            [self.width * 0.20, self.height],
            [self.width * 0.20, 0],
            [self.width * 0.80, 0],
            [self.width * 0.80, self.height]
        ])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    def process(self, frame):
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        binary = np.zeros_like(s_channel)
        binary[((s_channel >= 85) & (s_channel <= 255)) | ((l_channel >= 185) & (l_channel <= 255))] = 1

        warped = cv2.warpPerspective(binary, self.M, (self.width, self.height), flags=cv2.INTER_NEAREST)

        histogram = np.sum(warped[int(warped.shape[0] * 0.6):, :], axis=0)
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

        if len(left_lane_inds) >= self.minpix and len(right_lane_inds) >= self.minpix:
            left_fit = np.polyfit(nonzeroy[left_lane_inds], nonzerox[left_lane_inds], 2)
            right_fit = np.polyfit(nonzeroy[right_lane_inds], nonzerox[right_lane_inds], 2)
            self.left_fits.append(left_fit)
            self.right_fits.append(right_fit)

        if len(self.left_fits) == 0:
            return frame, 0.0, 0.0

        avg_left_fit = np.mean(self.left_fits, axis=0)
        avg_right_fit = np.mean(self.right_fits, axis=0)

        ploty = np.linspace(0, self.height - 1, int(self.height / 2))
        left_fitx = avg_left_fit[0] * ploty**2 + avg_left_fit[1] * ploty + avg_left_fit[2]
        right_fitx = avg_right_fit[0] * ploty**2 + avg_right_fit[1] * ploty + avg_right_fit[2]

        color_warp = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int32([pts]), (0, 220, 100))
        cv2.polylines(color_warp, np.int32([pts_left]), False, (0, 0, 255), 8)
        cv2.polylines(color_warp, np.int32([pts_right]), False, (255, 0, 0), 8)

        unwarped = cv2.warpPerspective(color_warp, self.Minv, (self.width, self.height), flags=cv2.INTER_NEAREST)
        result = cv2.addWeighted(frame, 1, unwarped, 0.35, 0)

        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2.0
        offset = ((self.width / 2.0) - lane_center) * self.xm_per_pix

        y_eval = np.max(ploty)
        curvature = ((1 + (2 * avg_left_fit[0] * y_eval * self.ym_per_pix + avg_left_fit[1])**2)**1.5) / np.absolute(2 * avg_left_fit[0] + 1e-6)

        return result, curvature, offset


# ==========================================================
# 3. NESNE TAKİBİ & MESAFE KESTİRİMİ (ObjectTracker)
# ==========================================================
class ObjectTracker:
    def __init__(self, img_size=(1280, 720)):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)

        self.width, self.height = img_size
        self.target_classes = {0: "Yaya", 1: "Bisiklet", 2: "Araba", 3: "Motor", 5: "Otobus", 7: "Kamyon"}
        self.camera_height = 1.35
        self.horizon_y = self.height * 0.58
        self.focal_length_px = self.height * 1.15

        self.filters = {}
        self.cached_objects = []

    def update(self, frame, run_detection=True):
        if not run_detection:
            return self.cached_objects

        results = self.model.track(
            frame, 
            conf=0.38, 
            imgsz=480, 
            persist=True, 
            tracker="bytetrack.yaml", 
            device=self.device, 
            verbose=False
        )[0]
        
        tracked_objects = []

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().numpy()
            clss = results.boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, clss):
                if cls_id in self.target_classes:
                    x1, y1, x2, y2 = map(int, box)
                    delta_y = y2 - self.horizon_y
                    
                    raw_dist = (self.camera_height * self.focal_length_px) / delta_y if delta_y > 5 else 99.0
                    raw_dist = float(np.clip(raw_dist, 1.0, 99.0))

                    if track_id not in self.filters:
                        self.filters[track_id] = DistanceKalman1D(raw_dist)
                    
                    filtered_dist = self.filters[track_id].update(raw_dist)

                    box_center_x = (x1 + x2) / 2.0
                    dist_x = (box_center_x - (self.width / 2.0)) * (3.7 / 700.0) * (filtered_dist / 15.0)
                    is_lead = abs(dist_x) < 1.8 and filtered_dist < 60.0

                    tracked_objects.append({
                        "id": track_id,
                        "bbox": (x1, y1, x2, y2),
                        "name": self.target_classes[cls_id],
                        "distance": filtered_dist,
                        "is_lead": is_lead
                    })

        self.cached_objects = tracked_objects
        return tracked_objects

    def draw(self, frame, objects):
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            dist = obj["distance"]
            obj_id = obj["id"]
            name = obj["name"]
            is_lead = obj["is_lead"]

            if is_lead and dist < 12.0:
                color = (0, 0, 255)       # Kırmızı (Kritik Mesafe)
            elif is_lead:
                color = (0, 255, 100)     # Yeşil (Lider Araç)
            elif dist < 12.0:
                color = (0, 140, 255)     # Turuncu
            else:
                color = (255, 180, 0)     # Mavi

            thickness = 3 if is_lead else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            lead_tag = "[LEAD] " if is_lead else ""
            label = f"{lead_tag}ID:{obj_id} {name} | {dist:.1f}m"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            
            text_color = (0, 0, 0) if (is_lead and color == (0, 255, 100)) else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

        return frame


# ==========================================================
# 4. HUD GÖSTERGESİ
# ==========================================================
def draw_hud(frame, curvature, offset, objects, fps, device_name):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (20, 20), (360, 175), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    cv2.rectangle(frame, (20, 20), (360, 175), (0, 255, 180), 2)

    cv2.putText(frame, f"ADAS VISION PIPELINE ({device_name.upper()})", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    curv_str = f"{int(curvature)} m" if curvature < 4000 else "Duz Yol"
    cv2.putText(frame, f"Serit Egriligi: {curv_str}", (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    yon = "Sag" if offset > 0 else "Sol"
    offset_color = (0, 255, 255) if abs(offset) < 0.40 else (0, 0, 255)
    cv2.putText(frame, f"Merkez Sapma: {abs(offset):.2f}m ({yon})", (35, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.50, offset_color, 1)
    cv2.putText(frame, f"Tespit Edilen Nesne: {len(objects)}", (35, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    lead_cars = [obj for obj in objects if obj["is_lead"]]
    if lead_cars:
        lead = min(lead_cars, key=lambda x: x["distance"])
        if lead["distance"] < 11.0:
            cv2.rectangle(frame, (int(w / 2) - 200, int(h * 0.78)), (int(w / 2) + 200, int(h * 0.78) + 55), (0, 0, 255), -1)
            cv2.putText(frame, "! KRITIK TAKIP MESAFESI !", (int(w / 2) - 170, int(h * 0.78) + 36), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame


# ==========================================================
# 5. ANA ÇALIŞTIRICI
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

    lane_detector = FastLaneDetector(img_size)
    tracker = ObjectTracker(img_size)
    
    frame_count = 0
    prev_time = time.time()

    print(f"Sistem Calisiyor ({tracker.device.upper()}). Cikis icin 'q' tusuna basin.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()

        lane_frame, curvature, offset = lane_detector.process(frame)
        run_detection = (frame_count % 2 == 0)
        tracked_objects = tracker.update(lane_frame, run_detection=run_detection)
        obj_frame = tracker.draw(lane_frame, tracked_objects)

        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time

        final_frame = draw_hud(obj_frame, curvature, offset, tracked_objects, fps, tracker.device)

        out_writer.write(final_frame)
        cv2.imshow("ADAS - Autonomous Vision Pipeline", final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    print("Video basariyla kaydedildi: adas_demo_kaydi.mp4")


if __name__ == "__main__":
    main()
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import cv2
import numpy as np
import time
import datetime
from datetime import datetime
import os
import tempfile
from typing import Dict, List, Optional
from pydantic import BaseModel
from ultralytics import YOLO
import uuid

app = FastAPI(title="Anomatrix - Upload Video", version="1.0.0")

class SurveillanceResult(BaseModel):
    camera_id: str
    status: str
    crowd_density: float
    activity_type: str
    threshold: float
    heatmap_id: str

class AnalysisResponse(BaseModel):
    results: Dict[str, SurveillanceResult]
    processing_time: float
    total_frames: int

class VideoSurveillanceAnalyzer:
    def __init__(self):
        self.model = YOLO("models/best_yolov8x.pt", task="detect")
        self.class_names = self.model.names
        self.heatmap_storage = {}
        
        self.colors = {
            "Creeping": {"normal": (0, 255, 0), "abnormal": (0, 165, 255)},
            "crawling": {"normal": (255, 0, 0), "abnormal": (0, 0, 255)},
            "crawling_with_weapon": {"normal": (255, 255, 0), "abnormal": (0, 0, 255)},
            "crouching": {"normal": (128, 0, 128), "abnormal": (0, 0, 255)},
            "crouching_with_weapon": {"normal": (0, 255, 255), "abnormal": (0, 0, 255)},
            "cycling": {"normal": (255, 165, 0), "abnormal": (0, 0, 255)},
            "motor_bike": {"normal": (0, 128, 255), "abnormal": (0, 0, 255)},
            "walking": {"normal": (128, 128, 128), "abnormal": (0, 0, 255)},
            "walking_with_weapon": {"normal": (0, 255, 128), "abnormal": (0, 0, 255)}
        }
        
        self.motion_threshold = 2000
        self.crowd_density_threshold = 0.3
        self.activity_thresholds = {
            "Creeping": 0.25,
            "crawling": 0.2,
            "crawling_with_weapon": 0.15,
            "crouching": 0.1,
            "crouching_with_weapon": 0.2,
            "cycling": 0.05,
            "motor_bike": 0.3,
            "walking": 0.05,
            "walking_with_weapon": 0.2
        }
        
        self.abnormal_threshold = 10
        self.gathering_threshold = 3
        self.proximity_threshold = 100

    def analyze_crowd_behavior(self, frame, backSub):
        fgMask = backSub.apply(frame)
        
        kernel = np.ones((5, 5), np.uint8)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        
        motion_pixels = cv2.countNonZero(fgMask)
        total_pixels = frame.shape[0] * frame.shape[1]
        crowd_density = motion_pixels / total_pixels
        
        is_crowd = crowd_density > self.crowd_density_threshold
        
        return is_crowd, fgMask, crowd_density

    def is_activity_abnormal(self, activity, crowd_density, sudden_movement):
        always_abnormal_activity = {"crouching_with_weapon", "crawling_with_weapon", "walking_with_weapon", "Creeping"}
        if activity in always_abnormal_activity:
            return True
        
        if activity in {"standing", "walking"}:
            return sudden_movement
        
        threshold = self.activity_thresholds.get(activity, 0.3)
        return crowd_density > threshold

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def detect_sudden_movement(self, prev_boxes, current_boxes):
        if not prev_boxes or not current_boxes:
            return False
            
        for pre_box, curr_box in zip(prev_boxes, current_boxes):
            pre_center = ((pre_box[0] + pre_box[2]) // 2, (pre_box[1] + pre_box[3]) // 2)
            curr_center = ((curr_box[0] + curr_box[2]) // 2, (curr_box[1] + curr_box[3]) // 2)
            
            distance = self.calculate_distance(pre_center, curr_center)
            if distance > self.proximity_threshold:
                return True
        return False

    def update_heatmap(self, heatmap, fg_mask):
        if heatmap is None:
            heatmap = np.zeros(fg_mask.shape, dtype=np.float32)
        heatmap += fg_mask.astype(np.float32)
        return heatmap

    def save_heatmap_as_jpg(self, heatmap, heatmap_id: str) -> str:
        if heatmap is None:
            return ""
        
        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"heatmap_{heatmap_id}.jpg")
        
        cv2.imwrite(file_path, heatmap_colored)
        
        self.heatmap_storage[heatmap_id] = file_path
        
        return file_path

    def cleanup_heatmap(self, heatmap_id: str):
        if heatmap_id in self.heatmap_storage:
            file_path = self.heatmap_storage[heatmap_id]
            if os.path.exists(file_path):
                os.unlink(file_path)
            del self.heatmap_storage[heatmap_id]

    def analyze_video(self, video_path: str, camera_id: str = "default") -> Dict:
        cap = cv2.VideoCapture(video_path)
        backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        count = 0
        prev_boxes = []
        heatmap = None
        standing_duration = {}
        
        total_frames = 0
        abnormal_frames = 0
        activities_detected = []
        max_crowd_density = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            count += 1
            if count % 3 != 0:
                continue
            
            total_frames += 1
            frame = cv2.resize(frame, (1280, 720))
            
            is_crowded, fg_mask, crowd_density = self.analyze_crowd_behavior(frame, backSub)
            heatmap = self.update_heatmap(heatmap, fg_mask)
            max_crowd_density = max(max_crowd_density, crowd_density)
            
            result = self.model(frame)
            current_boxes = []
            frame_abnormal = False
            
            sudden_movement = self.detect_sudden_movement(prev_boxes, current_boxes)
            current_time = time.time()
            
            for r in result:
                boxes = r.boxes
                
                for box in boxes:
                    cls_id = int(box.cls)
                    class_label = self.class_names[cls_id]
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    box_id = f"{box_center[0]}_{box_center[1]}"
                    
                    if class_label == 'standing':
                        if box_id not in standing_duration:
                            standing_duration[box_id] = current_time
                        activity_duration = current_time - standing_duration[box_id]
                    else:
                        activity_duration = 0
                        standing_duration.pop(box_id, None)
                    
                    is_abnormal = self.is_activity_abnormal(class_label, crowd_density, sudden_movement)
                    
                    current_boxes.append((x1, y1, x2, y2, class_label))
                    
                    if is_abnormal:
                        frame_abnormal = True
                        activities_detected.append(class_label)
            
            if frame_abnormal:
                abnormal_frames += 1
            
            prev_boxes = current_boxes.copy()
        
        cap.release()
        processing_time = time.time() - start_time
        
        status = "ABNORMAL" if abnormal_frames > 0 else "NORMAL"
        activity_type = max(set(activities_detected), key=activities_detected.count) if activities_detected else "normal_activity"
        threshold = self.activity_thresholds.get(activity_type, 0.3)
        
        heatmap_id = str(uuid.uuid4())
        heatmap_path = self.save_heatmap_as_jpg(heatmap, heatmap_id)
        
        return {
            camera_id: {
                "camera_id": camera_id,
                "status": status,
                "crowd_density": round(max_crowd_density, 3),
                "activity_type": activity_type,
                "threshold": round(threshold, 3),
                "heatmap_id": heatmap_id
            }
        }, processing_time, total_frames

analyzer = VideoSurveillanceAnalyzer()

@app.get("/")
async def root():
    return {"message": "Anomatrix", "version": "1.0.0"}

@app.post("/analyze-video/", response_model=AnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    camera_id: Optional[str] = "default"
):
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        results, processing_time, total_frames = analyzer.analyze_video(temp_file_path, camera_id)
        
        return AnalysisResponse(
            results=results,
            processing_time=round(processing_time, 2),
            total_frames=total_frames
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@app.post("/analyze-multiple-videos/")
async def analyze_multiple_videos(files: List[UploadFile] = File(...)):
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
    
    all_results = {}
    total_processing_time = 0
    total_frames_all = 0
    
    for i, file in enumerate(files):
        if not file.content_type.startswith('video/'):
            continue
        
        camera_id = f"camera_{i+1}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            results, processing_time, total_frames = analyzer.analyze_video(temp_file_path, camera_id)
            all_results.update(results)
            total_processing_time += processing_time
            total_frames_all += total_frames
            
        except Exception as e:
            heatmap_id = str(uuid.uuid4())
            all_results[camera_id] = {
                "camera_id": camera_id,
                "status": "ERROR",
                "crowd_density": 0.0,
                "activity_type": "error",
                "threshold": 0.0,
                "heatmap_id": heatmap_id,
                "error": str(e)
            }
        
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    return AnalysisResponse(
        results=all_results,
        processing_time=round(total_processing_time, 2),
        total_frames=total_frames_all
    )

@app.delete("/heatmap/{heatmap_id}")
async def delete_heatmap(heatmap_id: str):
    if heatmap_id not in analyzer.heatmap_storage:
        raise HTTPException(status_code=404, detail="Heatmap not found")
    
    analyzer.cleanup_heatmap(heatmap_id)
    return {"message": f"Heatmap {heatmap_id} deleted successfully"}

@app.get("/heatmap/{heatmap_id}")
async def get_heatmap(heatmap_id: str):
    if heatmap_id not in analyzer.heatmap_storage:
        raise HTTPException(status_code=404, detail="Heatmap not found")
    
    file_path = analyzer.heatmap_storage[heatmap_id]
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Heatmap file not found")
    
    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        filename=f"heatmap_{heatmap_id}.jpg"
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
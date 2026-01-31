from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import aiohttp
import time
import cv2
import numpy as np
import base64
import logging
from contextlib import asynccontextmanager
from ultralytics import YOLO
import os
import gc
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('anomatrix.log')
    ]
)
logger = logging.getLogger(__name__)

class CameraInfo(BaseModel):
    cam_id: int
    rtsp: str

class DetectionResult(BaseModel):
    camera_id: int
    status: str
    crowd_density: float
    activity_type: str
    threshold: float
    heatmap: str

detection_results: Dict[int, DetectionResult] = {}
active_camera_processors: Dict[int, 'CameraProcessor'] = {}
start_time = time.time()
processing_executor: Optional[ThreadPoolExecutor] = None
is_system_running = False
cameras_list: List[CameraInfo] = []

CONFIG = {
    'backend_url': "https://driftersieee.duckdns.org/api/Home/rtsp",
    'camera_refresh_interval': 30,
    'model_path': "models/best_yolov8_improved.pt",
    'max_reconnect_attempts': 3,
    'frame_buffer_size': 1,
    'processing_interval': 3,
    'detection_confidence_threshold': 0.5,
    'heatmap_decay_factor': 0.995,
    'opencv_timeout': 10000000,
    'processing_duration_per_camera': 30,
    'max_concurrent_cameras': 5,
    'camera_restart_delay': 5
}

ACTIVITY_THRESHOLDS = {
    "Creeping": {"threshold": 0.25, "confidence": 0.6},
    "crawling": {"threshold": 0.2, "confidence": 0.5},
    "crawling_with_weapon": {"threshold": 0.15, "confidence": 0.8},
    "crouching": {"threshold": 0.1, "confidence": 0.5},
    "crouching_with_weapon": {"threshold": 0.2, "confidence": 0.8},
    "cycling": {"threshold": 0.05, "confidence": 0.4},
    "motor_bike": {"threshold": 0.3, "confidence": 0.6},
    "walking": {"threshold": 0.5, "confidence": 0.4},
    "walking_with_weapon": {"threshold": 0.2, "confidence": 0.5},
    "fighting": {"threshold": 0.1, "confidence": 0.5},
    "standing": {"threshold": 0.3, "confidence": 0.4}
}

os.environ.update({
    'OPENCV_FFMPEG_CAPTURE_OPTIONS': (
        'rtsp_transport;tcp|'
        'rtsp_flags;prefer_tcp|'
        'stimeout;10000000|'
        'buffer_size;1024000|'
        'max_delay;500000|'
        'fflags;nobuffer|'
        'flags;low_delay|'
        'probesize;32|'
        'analyzeduration;1000000'
    ),
    'OPENCV_VIDEOIO_PRIORITY_FFMPEG': '1',
    'OPENCV_LOG_LEVEL': 'ERROR'
})

model = YOLO(CONFIG['model_path'], task="detect")
class_names = model.names

class CameraProcessor:
    def __init__(self, camera_info: CameraInfo):
        self.camera_info = camera_info
        self.cap = None
        self.backSub = None
        self.heatmap = None
        self.prev_boxes = []
        self.standing_duration = {}
        self.frame_count = 0
        self.abnormal_threshold = 10
        self.proximity_threshold = 100
        self.crowd_density_threshold = 0.3
        self.is_running = False
        self.last_update = time.time()
        
    def start_processing(self):
        self.is_running = True
        self._reset_camera_state()
        
        logger.info(f"Starting parallel processing for camera {self.camera_info.cam_id}")
        
        try:
            self._process_camera()
        except Exception as e:
            logger.error(f"Error in camera processor {self.camera_info.cam_id}: {e}")
        finally:
            self.stop_processing()
            
    def stop_processing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"Stopped processing camera {self.camera_info.cam_id}")
        
        if self.camera_info.cam_id in active_camera_processors:
            del active_camera_processors[self.camera_info.cam_id]
    
    def _process_camera(self):
        self.cap = self._create_capture(self.camera_info.rtsp)
        if not self.cap:
            logger.error(f"Failed to create capture for camera {self.camera_info.cam_id}")
            return
        
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.is_running and is_system_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Too many consecutive failures for camera {self.camera_info.cam_id}")
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0
                self.frame_count += 1
                self.last_update = time.time()
                
                if self.frame_count % CONFIG['processing_interval'] != 0:
                    continue
                
                if not self._validate_frame(frame):
                    continue
                
                try:
                    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    logger.error(f"Error resizing frame for camera {self.camera_info.cam_id}: {e}")
                    continue
                
                self._process_frame(frame)
                
            except Exception as e:
                logger.error(f"Error processing frame for camera {self.camera_info.cam_id}: {e}")
                time.sleep(0.1)
    
    def _create_capture(self, rtsp_url: str) -> Optional[cv2.VideoCapture]:
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            
            if not cap.isOpened():
                return None
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            
            start_time = time.time()
            max_test_time = 5.0
            
            while time.time() - start_time < max_test_time:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    logger.info(f"Successfully connected to RTSP stream: {rtsp_url}")
                    return cap
                time.sleep(0.1)
            
            cap.release()
            logger.warning(f"Test frame capture failed for {rtsp_url}")
            return None
            
        except Exception as e:
            logger.error(f"Error creating capture for {rtsp_url}: {e}")
            return None
    
    def _reset_camera_state(self):
        self.heatmap = None
        self.prev_boxes = []
        self.standing_duration = {}
        self.frame_count = 0
    
    def _validate_frame(self, frame) -> bool:
        if frame is None or frame.size == 0:
            return False
        
        try:
            height, width = frame.shape[:2]
            
            if height < 240 or width < 320:
                return False
            
            mean_val = np.mean(frame)
            if mean_val < 10 or mean_val > 245:
                return False
            
            std_val = np.std(frame)
            if std_val < 5:
                return False
            
            return True
        except Exception:
            return False
    
    def _process_frame(self, frame):
        try:
            is_crowded, fg_mask, crowd_density = self._analyze_crowd_behavior(frame)
            self.heatmap = self._update_heatmap_with_decay(self.heatmap, fg_mask)
            
            try:
                results = model(frame, verbose=False, conf=CONFIG['detection_confidence_threshold'])
            except Exception as e:
                logger.error(f"YOLO detection failed for camera {self.camera_info.cam_id}: {e}")
                return
            
            current_boxes = []
            detected_activities = []
            abnormal_activities = []
            current_time = time.time()
            
            for r in results:
                if r.boxes is None:
                    continue
                    
                for box in r.boxes:
                    try:
                        cls_id = int(box.cls)
                        if cls_id >= len(class_names):
                            continue
                            
                        class_label = class_names[cls_id]
                        confidence = float(box.conf)
                        
                        if confidence < CONFIG['detection_confidence_threshold']:
                            continue
                    
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        box_id = f"{box_center[0]}_{box_center[1]}"
                        
                        activity_duration = 0
                        if class_label == 'standing':
                            if box_id not in self.standing_duration:
                                self.standing_duration[box_id] = current_time
                            activity_duration = current_time - self.standing_duration[box_id]
                        else:
                            self.standing_duration.pop(box_id, None)
                        
                        current_boxes.append((x1, y1, x2, y2, class_label, confidence))
                        detected_activities.append(class_label)
                        
                        sudden_movement = self._detect_sudden_movement(self.prev_boxes, current_boxes)
                        is_abnormal = self._is_activity_abnormal(
                            class_label, crowd_density, sudden_movement, activity_duration, confidence
                        )
                        
                        if is_abnormal:
                            abnormal_activities.append(class_label)
                            
                    except Exception as e:
                        logger.error(f"Error processing detection box: {e}")
                        continue
            
            self.prev_boxes = current_boxes.copy()
            
            try:
                heatmap_b64 = self._generate_heatmap_base64()
                status = "abnormal" if abnormal_activities else "normal"
                primary_activity = abnormal_activities[0] if abnormal_activities else (
                    detected_activities[0] if detected_activities else "none"
                )
                threshold = ACTIVITY_THRESHOLDS.get(primary_activity, {}).get("threshold", 0.3)
                
                detection_results[self.camera_info.cam_id] = DetectionResult(
                    camera_id=self.camera_info.cam_id,
                    status=status,
                    crowd_density=round(crowd_density, 3),
                    activity_type=primary_activity,
                    threshold=round(threshold, 3),
                    heatmap=heatmap_b64
                )
                
            except Exception as e:
                logger.error(f"Error generating detection results for camera {self.camera_info.cam_id}: {e}")
                
        except Exception as e:
            logger.error(f"Error in _process_frame for camera {self.camera_info.cam_id}: {e}")
    
    def _analyze_crowd_behavior(self, frame):
        fg_mask = self.backSub.apply(frame)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        motion_pixels = cv2.countNonZero(fg_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        crowd_density = motion_pixels / total_pixels
        
        is_crowd = crowd_density > self.crowd_density_threshold
        
        return is_crowd, fg_mask, crowd_density

    def _update_heatmap_with_decay(self, hm, fg_mask):
        if hm is None:
            hm = np.zeros(fg_mask.shape, dtype=np.float32)

        hm *= CONFIG['heatmap_decay_factor']
        hm += fg_mask.astype(np.float32) / 255.0
        hm = np.clip(hm, 0, 10)
        
        return hm
    
    def _generate_heatmap_base64(self) -> str:
        if self.heatmap is None:
            return ""

        try:
            heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode('.jpg', heatmap_colored, encode_param)
            heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
            return heatmap_b64
        except Exception as e:
            logger.error(f"Error generating heatmap base64: {e}")
            return ""

    def _detect_sudden_movement(self, prev_boxes, current_boxes) -> bool:
        if not prev_boxes or not current_boxes:
            return False

        for curr_box in current_boxes:
            curr_center = ((curr_box[0] + curr_box[2]) // 2, (curr_box[1] + curr_box[3]) // 2)
            
            for prev_box in prev_boxes:
                prev_center = ((prev_box[0] + prev_box[2]) // 2, (prev_box[1] + prev_box[3]) // 2)
                
                distance = np.sqrt((prev_center[0] - curr_center[0]) ** 2 + (prev_center[1] - curr_center[1]) ** 2)
                
                if distance > self.proximity_threshold:
                    return True
        return False
    
    def _is_activity_abnormal(self, activity: str, crowd_density: float, sudden_movement: bool, 
                             activity_duration: float = 0, confidence: float = 0.5) -> bool:
        activity_config = ACTIVITY_THRESHOLDS.get(activity, {"threshold": 0.3, "confidence": 0.2})

        if confidence < activity_config["confidence"]:
            return False

        always_abnormal = {"fighting", "crawling_with_weapon", "crouching_with_weapon", "walking_with_weapon"}
        if activity in always_abnormal:
            return True

        if activity == "standing" and activity_duration > self.abnormal_threshold:
            return True

        if activity in {"standing", "walking"} and sudden_movement:
            return True

        threshold = activity_config["threshold"]
        return crowd_density > threshold

class ParallelCameraManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=CONFIG['max_concurrent_cameras'])
        self.camera_futures: Dict[int, concurrent.futures.Future] = {}
        
    def start_camera_processing(self, cameras: List[CameraInfo]):
        global active_camera_processors
        
        current_camera_ids = {cam.cam_id for cam in cameras}
        cameras_to_stop = set(active_camera_processors.keys()) - current_camera_ids
        
        for cam_id in cameras_to_stop:
            self.stop_camera_processing(cam_id)
        
        for camera in cameras:
            if camera.cam_id not in active_camera_processors:
                self.start_single_camera(camera)
            else:
                processor = active_camera_processors[camera.cam_id]
                if not processor.is_running or (time.time() - processor.last_update) > 60:
                    logger.info(f"Restarting camera {camera.cam_id} - appears to be stuck")
                    self.stop_camera_processing(camera.cam_id)
                    time.sleep(CONFIG['camera_restart_delay'])
                    self.start_single_camera(camera)
    
    def start_single_camera(self, camera: CameraInfo):
        try:
            processor = CameraProcessor(camera)
            active_camera_processors[camera.cam_id] = processor
            
            future = self.executor.submit(processor.start_processing)
            self.camera_futures[camera.cam_id] = future
            
            logger.info(f"Started parallel processing for camera {camera.cam_id}")
            
        except Exception as e:
            logger.error(f"Error starting camera {camera.cam_id}: {e}")
    
    def stop_camera_processing(self, camera_id: int):
        if camera_id in active_camera_processors:
            processor = active_camera_processors[camera_id]
            processor.stop_processing()
            
        if camera_id in self.camera_futures:
            future = self.camera_futures[camera_id]
            if not future.done():
                future.cancel()
            del self.camera_futures[camera_id]
    
    def stop_all_cameras(self):
        for camera_id in list(active_camera_processors.keys()):
            self.stop_camera_processing(camera_id)
        
        self.executor.shutdown(wait=True)

camera_manager = ParallelCameraManager()

async def fetch_cameras_from_backend() -> List[CameraInfo]:
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(CONFIG['backend_url']) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Fetched {len(data)} cameras from backend")
                    cameras = []
                    for cam_data in data:
                        try:
                            camera = CameraInfo(cam_id=cam_data['id'], rtsp=cam_data['url'])
                            cameras.append(camera)
                        except Exception as e:
                            logger.warning(f"Invalid camera data: {cam_data}, error: {e}")
                    return cameras
                else:
                    logger.error(f"Failed to fetch cameras: HTTP {response.status}")
                    return []
    except asyncio.TimeoutError:
        logger.error("Timeout while fetching cameras from backend")
        return []
    except Exception as e:
        logger.error(f"Error fetching cameras from backend: {e}")
        return []

async def camera_refresh_task():
    global cameras_list
    
    while is_system_running:
        try:
            cameras = await fetch_cameras_from_backend()
            if cameras:
                cameras_list = cameras
                camera_manager.start_camera_processing(cameras)
            await asyncio.sleep(CONFIG['camera_refresh_interval'])
        except asyncio.CancelledError:
            logger.info("Camera refresh task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in camera refresh task: {e}")
            await asyncio.sleep(CONFIG['camera_refresh_interval'])

@asynccontextmanager
async def lifespan(app: FastAPI):
    global is_system_running, cameras_list
    
    logger.info("Starting Anomatrix...")
    is_system_running = True
    
    cameras = await fetch_cameras_from_backend()
    if cameras:
        cameras_list = cameras
        camera_manager.start_camera_processing(cameras)
    
    refresh_task = asyncio.create_task(camera_refresh_task())
    
    logger.info("API started successfully")
    
    yield
    
    logger.info("Shutting down...")
    is_system_running = False
    refresh_task.cancel()
    camera_manager.stop_all_cameras()
    
    gc.collect()
    logger.info("Shutdown complete")

app = FastAPI(
    title="Anomatrix", 
    version="3.0.0",
    description="Parallel Multi-Camera Anomaly Detection System",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "Anomatrix", 
        "version": "3.0.0",
        "description": "Parallel Multi-Camera Anomaly Detection System"
    }

@app.get("/status")
async def get_status():
    active_cameras = []
    for cam_id, processor in active_camera_processors.items():
        active_cameras.append({
            "camera_id": cam_id,
            "is_running": processor.is_running,
            "last_update": processor.last_update,
            "frames_processed": processor.frame_count
        })
    
    return {
        "total_cameras": len(cameras_list),
        "active_processors": len(active_camera_processors),
        "max_concurrent": CONFIG['max_concurrent_cameras'],
        "is_system_running": is_system_running,
        "total_detections": len(detection_results),
        "uptime_seconds": round(time.time() - start_time, 2),
        "active_cameras": active_cameras
    }

@app.get("/results", response_model=List[DetectionResult])
async def get_all_detections():
    return list(detection_results.values())

@app.get("/results/{camera_id}", response_model=DetectionResult)
async def get_camera_detection(camera_id: int):
    if camera_id not in detection_results:
        raise HTTPException(
            status_code=404, 
            detail=f"Camera {camera_id} not found or no detection data available"
        )
    return detection_results[camera_id]

@app.post("/refresh-cameras")
async def refresh_cameras_manually():
    try:
        cameras = await fetch_cameras_from_backend()
        if cameras:
            global cameras_list
            cameras_list = cameras
            camera_manager.start_camera_processing(cameras)
            return {
                "status": "success",
                "message": "Cameras refreshed successfully", 
                "camera_count": len(cameras),
                "active_processors": len(active_camera_processors)
            }
        else:
            return {
                "status": "warning",
                "message": "No cameras found or failed to fetch from backend", 
                "camera_count": 0
            }
    except Exception as e:
        logger.error(f"Error in manual camera refresh: {e}")
        raise HTTPException(status_code=500, detail="Failed to refresh cameras")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )
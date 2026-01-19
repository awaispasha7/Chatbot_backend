"""
Video Processor Service - YOLO inference and video encoding
"""
import os
import cv2
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import torch
from ultralytics import YOLO

from ..config import get_settings
from ..models.schemas import JobStatus
from .job_manager import job_manager
from ..utils.ffprobe import convert_to_mp4



class VideoProcessor:
    """
    Video processing service using YOLO for object detection
    
    Processes video frame-by-frame, draws bounding boxes,
    and encodes output as browser-compatible MP4
    """
    
    _model: Optional[YOLO] = None
    _model_names: dict = {}
    
    
    @classmethod
    def load_model(cls) -> bool:
        """
        Load YOLO model from configured path
        
        Returns:
            True if model loaded successfully
        """
        if cls._model is not None:
            return True
            
        settings = get_settings()
        model_path = Path(settings.model_path)
        
        # Try multiple locations for the model file
        if not model_path.exists():
            # Try relative to backend directory (Chatbot_backend-main/)
            backend_dir = Path(__file__).parent.parent.parent
            model_path = backend_dir / settings.model_path
            
        if not model_path.exists():
            # Try in current directory (if model was copied to app/)
            model_path = Path(__file__).parent / "best.pt"
            
        if not model_path.exists():
            # Try in backend root (if model is in Chatbot_backend-main/)
            backend_dir = Path(__file__).parent.parent.parent
            model_path = backend_dir / "best.pt"
            
        if not model_path.exists():
            print(f"Model not found. Tried:")
            print(f"  - {settings.model_path}")
            print(f"  - {Path(__file__).parent.parent.parent / settings.model_path}")
            print(f"  - {Path(__file__).parent / 'best.pt'}")
            print(f"  - {Path(__file__).parent.parent.parent / 'best.pt'}")
            print(f"Please set MODEL_PATH environment variable or place best.pt in one of these locations.")
            return False
        
        try:
            cls._model = YOLO(str(model_path))
            cls._model_names = cls._model.names
            print(f"Model loaded successfully. Classes: {cls._model_names}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    @classmethod
    def get_class_names(cls) -> dict:
        """Get model class names"""
        return cls._model_names
    
    @classmethod
    async def process_video(
        cls,
        job_id: str,
        input_path: str,
        output_path: str
    ) -> bool:
        """
        Process video with YOLO inference
        
        Args:
            job_id: Job identifier for progress updates
            input_path: Path to input video
            output_path: Path for output MP4
            
        Returns:
            True if processing succeeded
        """
        # Force add file handler 
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)  # Ensure INFO logs pass
        
        if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            fh = logging.FileHandler('backend_debug.log')
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            root_logger.addHandler(fh)
                
        msg = f"Processing job {job_id}: Start processing. Input: {input_path}"
        logging.info(msg)
        print(msg, flush=True)  # Backup
        
        # Check if file exists
        if not os.path.exists(input_path):
            logging.error(f"Input file not found: {input_path}")
            return False
            
        logging.info(f"Model loaded status: {cls._model is not None}")
        
        if cls._model is None:
            if not cls.load_model():
                await job_manager.update_status(
                    job_id, 
                    JobStatus.FAILED, 
                    error="Model not loaded"
                )
                return False
        
        try:
            # Update status to processing with initial progress (0% - upload complete, starting analysis)
            await job_manager.update_status(job_id, JobStatus.PROCESSING, progress=0)
            
            # Open input video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                await job_manager.update_status(
                    job_id,
                    JobStatus.FAILED,
                    error="Failed to open video file"
                )
                return False
            
            # Get video properties (5% - analyzing video properties)
            await job_manager.update_status(job_id, JobStatus.PROCESSING, progress=5)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Update progress after getting video info (10% - ready to process)
            await job_manager.update_status(job_id, JobStatus.PROCESSING, progress=10)
            
            # Create temporary output file (we'll convert to MP4 after)
            temp_dir = tempfile.gettempdir()
            temp_output = os.path.join(temp_dir, f"{job_id}_temp.avi")
            
            # Video writer with XVID codec (cross-platform)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            if not out.isOpened():
                cap.release()
                await job_manager.update_status(
                    job_id,
                    JobStatus.FAILED,
                    error="Failed to create output video"
                )
                return False
            
            # Process frames
            frame_count = 0
            last_progress = 0
            
            # Update progress more frequently (every 1% or every 10 frames, whichever comes first)
            progress_update_interval = max(1, total_frames // 100) if total_frames > 0 else 10
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 30 == 0:
                    logging.info(f"Job {job_id}: Processing frame {frame_count}")
                
                # Enhance frame for better detection (optional - helps with low-light or low-contrast videos)
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better visibility
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)
                enhanced_frame = cv2.merge([l_enhanced, a, b])
                enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)
                
                # Run YOLO detection with very low confidence threshold to detect more players
                # Use predict() first to get ALL detections (not just tracked ones)
                # Lowered confidence to 0.03 to catch more players, especially those far away or partially occluded
                # Using iou=0.7 and agnostic_nms=False to reduce filtering of valid detections
                # max_det=300 to ensure we can detect many players
                predict_results = cls._model.predict(
                    enhanced_frame,  # Use enhanced frame for better detection
                    verbose=False, 
                    conf=0.03,  # Very low threshold to catch all players
                    iou=0.7,   # IoU threshold for NMS (higher = less aggressive filtering)
                    agnostic_nms=False,  # Allow overlapping detections of different classes
                    max_det=300  # Allow up to 300 detections per frame
                )
                
                # Also get tracked results to maintain IDs across frames
                track_results = cls._model.track(
                    enhanced_frame,  # Use enhanced frame
                    persist=True, 
                    verbose=False, 
                    conf=0.03,  # Match predict threshold
                    iou=0.7,
                    agnostic_nms=False,
                    max_det=300
                )
                
                # Create tracking ID mapping from tracked results
                tracking_id_map = {}
                if track_results and len(track_results) > 0 and track_results[0].boxes is not None:
                    for box in track_results[0].boxes:
                        if box.id is not None:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            cls_id = int(box.cls[0].cpu().numpy())
                            track_id = int(box.id[0].cpu().numpy())
                            # Use box center and class as key
                            tracking_id_map[(center_x, center_y, cls_id)] = track_id
                
                # Use predict results to ensure we show ALL detections
                # Pass tracking ID map to drawing function
                if predict_results and len(predict_results) > 0:
                    annotated_frame = cls._draw_detections_and_tracks(frame, predict_results[0], tracking_id_map)
                elif track_results and len(track_results) > 0:
                    # Fallback to tracked results
                    annotated_frame = cls._draw_detections_and_tracks(frame, track_results[0])
                else:
                    annotated_frame = frame
                
                # Write frame
                out.write(annotated_frame)
                
                frame_count += 1
                
                # Update progress more frequently (10% to 90% for frame processing)
                if total_frames > 0:
                    # Calculate progress: 10% (initial) to 90% (end of processing)
                    progress = 10 + int((frame_count / total_frames) * 80)  # 10-90% range
                    # Update if progress changed by at least 1% or every N frames
                    if progress > last_progress or frame_count % progress_update_interval == 0:
                        await job_manager.update_status(
                            job_id, 
                            JobStatus.PROCESSING, 
                            progress=progress
                        )
                        last_progress = progress
                else:
                    # If we can't determine total frames, update every 30 frames
                    if frame_count % 30 == 0:
                        await job_manager.update_status(
                            job_id,
                            JobStatus.PROCESSING,
                            progress=min(90, 10 + int((frame_count / 1000) * 80))  # Estimate
                        )
            
            # Release resources
            cap.release()
            out.release()
            
            # Convert to MP4 (remaining 10%)
            await job_manager.update_status(job_id, JobStatus.PROCESSING, progress=92)
            
            success, error = convert_to_mp4(temp_output, output_path)
            
            # Cleanup temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
            
            if not success:
                await job_manager.update_status(
                    job_id,
                    JobStatus.FAILED,
                    error=f"MP4 conversion failed: {error}"
                )
                return False
            
            # Success!
            await job_manager.update_status(
                job_id,
                JobStatus.DONE,
                progress=100,
                output_file=output_path
            )
            return True
            
        except Exception as e:
            await job_manager.update_status(
                job_id,
                JobStatus.FAILED,
                error=str(e)
            )
            return False
    

    @classmethod
    def _draw_detections_and_tracks(cls, frame, result, tracking_id_map=None) -> any:
        """
        Draw all detections and tracked objects on frame
        Shows both tracked objects (with IDs) and untracked detections
        
        Args:
            frame: Original frame
            result: YOLO result object with tracking info
            tracking_id_map: Optional dict mapping (center_x, center_y, cls_id) to track_id
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        boxes = result.boxes
        if boxes is None:
            return annotated
        
        # Get class names from model
        class_names = cls._model_names
        
        # Initialize tracking_id_map if not provided
        if tracking_id_map is None:
            tracking_id_map = {}
            
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            # Get class name if available
            class_name = class_names.get(cls_id, f"class_{cls_id}")
            
            # Check if this object is tracked (either from box.id or from tracking_id_map)
            is_tracked = box.id is not None
            track_id = None
            
            if is_tracked:
                track_id = int(box.id[0].cpu().numpy())
            else:
                # Try to find tracking ID from mapping
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # Find closest match in tracking_id_map
                min_dist = float('inf')
                closest_id = None
                for (tx, ty, tcls), tid in tracking_id_map.items():
                    if tcls == cls_id:  # Same class
                        dist = ((center_x - tx)**2 + (center_y - ty)**2)**0.5
                        if dist < 50 and dist < min_dist:  # Within 50 pixels
                            min_dist = dist
                            closest_id = tid
                if closest_id is not None:
                    track_id = closest_id
                    is_tracked = True
            
            # Color based on class ID and tracking status
            # Use brighter colors for tracked objects
            if is_tracked:
                color_seed = track_id if track_id is not None else cls_id
                color = (
                    (color_seed * 123) % 200 + 55,  # Brighter for tracked
                    (color_seed * 234) % 200 + 55,
                    (color_seed * 345) % 200 + 55
                )
            else:
                # Dimmer color for untracked detections
                color_seed = cls_id
                color = (
                    (color_seed * 123) % 150 + 50,
                    (color_seed * 234) % 150 + 50,
                    (color_seed * 345) % 150 + 50
                )
            
            # Draw bounding box (thicker for tracked objects)
            thickness = 3 if is_tracked else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Build label
            if is_tracked:
                label = f"{class_name} {track_id} {conf:.2f}"
            else:
                label = f"{class_name} {conf:.2f}"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w + 4, y1), color, -1)
            
            # Draw text
            cv2.putText(
                annotated, 
                label, 
                (x1 + 2, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                1
            )
            
        return annotated


# Convenience function for background task
async def process_video_task(job_id: str, input_path: str, output_path: str):
    """
    Background task wrapper for video processing
    Runs CPU-intensive processing in thread pool to avoid blocking event loop
    """
    import asyncio
    loop = asyncio.get_running_loop()
    
    # Run the CPU-intensive video processing in a thread pool executor
    # This prevents blocking the FastAPI event loop, allowing status endpoint to respond
    await loop.run_in_executor(
        None,  # Use default ThreadPoolExecutor
        _process_video_sync,  # Synchronous wrapper
        job_id,
        input_path,
        output_path,
        loop  # Pass loop for async status updates
    )


def _process_video_sync(job_id: str, input_path: str, output_path: str, loop):
    """
    Synchronous wrapper for video processing that runs in thread pool
    Uses asyncio.run_coroutine_threadsafe for async status updates
    """
    import asyncio
    
    # Force add file handler 
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        fh = logging.FileHandler('backend_debug.log')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)
    
    msg = f"Processing job {job_id}: Start processing. Input: {input_path}"
    logging.info(msg)
    print(msg, flush=True)
    
    # Check if file exists
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        future = asyncio.run_coroutine_threadsafe(
            job_manager.update_status(job_id, JobStatus.FAILED, error="Input file not found"),
            loop
        )
        future.result()
        return
    
    logging.info(f"Model loaded status: {VideoProcessor._model is not None}")
    
    if VideoProcessor._model is None:
        if not VideoProcessor.load_model():
            future = asyncio.run_coroutine_threadsafe(
                job_manager.update_status(job_id, JobStatus.FAILED, error="Model not loaded"),
                loop
            )
            future.result()
            return
    
    try:
        # Update status to processing with initial progress
        future = asyncio.run_coroutine_threadsafe(
            job_manager.update_status(job_id, JobStatus.PROCESSING, progress=0),
            loop
        )
        future.result()
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            future = asyncio.run_coroutine_threadsafe(
                job_manager.update_status(job_id, JobStatus.FAILED, error="Failed to open video file"),
                loop
            )
            future.result()
            return
        
        # Get video properties
        future = asyncio.run_coroutine_threadsafe(
            job_manager.update_status(job_id, JobStatus.PROCESSING, progress=5),
            loop
        )
        future.result()
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Update progress after getting video info
        future = asyncio.run_coroutine_threadsafe(
            job_manager.update_status(job_id, JobStatus.PROCESSING, progress=10),
            loop
        )
        future.result()
        
        # Create temporary output file
        temp_dir = tempfile.gettempdir()
        temp_output = os.path.join(temp_dir, f"{job_id}_temp.avi")
        
        # Video writer with XVID codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        if not out.isOpened():
            cap.release()
            future = asyncio.run_coroutine_threadsafe(
                job_manager.update_status(job_id, JobStatus.FAILED, error="Failed to create output video"),
                loop
            )
            future.result()
            return
        
        # Process frames
        frame_count = 0
        last_progress = 0
        progress_update_interval = max(1, total_frames // 100) if total_frames > 0 else 10
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 30 == 0:
                logging.info(f"Job {job_id}: Processing frame {frame_count}")
            
            # Enhance frame for better detection (optional - helps with low-light or low-contrast videos)
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better visibility
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            enhanced_frame = cv2.merge([l_enhanced, a, b])
            enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)
            
            # Run YOLO detection with very low confidence threshold to detect more players
            # Use predict() first to get ALL detections (not just tracked ones)
            # Lowered confidence to 0.03 to catch more players, especially those far away or partially occluded
            # Using iou=0.7 and agnostic_nms=False to reduce filtering of valid detections
            # max_det=300 to ensure we can detect many players
            predict_results = VideoProcessor._model.predict(
                enhanced_frame,  # Use enhanced frame for better detection
                verbose=False, 
                conf=0.03,  # Very low threshold to catch all players
                iou=0.7,   # IoU threshold for NMS (higher = less aggressive filtering)
                agnostic_nms=False,  # Allow overlapping detections of different classes
                max_det=300  # Allow up to 300 detections per frame
            )
            
            # Also get tracked results to maintain IDs across frames
            track_results = VideoProcessor._model.track(
                enhanced_frame,  # Use enhanced frame
                persist=True, 
                verbose=False, 
                conf=0.03,  # Match predict threshold
                iou=0.7,
                agnostic_nms=False,
                max_det=300
            )
            
            # Create tracking ID mapping from tracked results
            tracking_id_map = {}
            if track_results and len(track_results) > 0 and track_results[0].boxes is not None:
                for box in track_results[0].boxes:
                    if box.id is not None:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        cls_id = int(box.cls[0].cpu().numpy())
                        track_id = int(box.id[0].cpu().numpy())
                        # Use box center and class as key
                        tracking_id_map[(center_x, center_y, cls_id)] = track_id
            
            # Use predict results to ensure we show ALL detections
            # Pass tracking ID map to drawing function
            if predict_results and len(predict_results) > 0:
                annotated_frame = VideoProcessor._draw_detections_and_tracks(frame, predict_results[0], tracking_id_map)
            elif track_results and len(track_results) > 0:
                # Fallback to tracked results
                annotated_frame = VideoProcessor._draw_detections_and_tracks(frame, track_results[0])
            else:
                annotated_frame = frame
            
            # Write frame
            out.write(annotated_frame)
            
            frame_count += 1
            
            # Update progress (10% to 90% for frame processing)
            if total_frames > 0:
                progress = 10 + int((frame_count / total_frames) * 80)
                if progress > last_progress or frame_count % progress_update_interval == 0:
                    future = asyncio.run_coroutine_threadsafe(
                        job_manager.update_status(job_id, JobStatus.PROCESSING, progress=progress),
                        loop
                    )
                    future.result()  # Wait for update to complete
                    last_progress = progress
            else:
                if frame_count % 30 == 0:
                    future = asyncio.run_coroutine_threadsafe(
                        job_manager.update_status(
                            job_id,
                            JobStatus.PROCESSING,
                            progress=min(90, 10 + int((frame_count / 1000) * 80))
                        ),
                        loop
                    )
                    future.result()
        
        # Release resources
        cap.release()
        out.release()
        
        # Convert to MP4
        future = asyncio.run_coroutine_threadsafe(
            job_manager.update_status(job_id, JobStatus.PROCESSING, progress=92),
            loop
        )
        future.result()
        
        success, error = convert_to_mp4(temp_output, output_path)
        
        # Cleanup temp file
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        if not success:
            future = asyncio.run_coroutine_threadsafe(
                job_manager.update_status(job_id, JobStatus.FAILED, error=f"MP4 conversion failed: {error}"),
                loop
            )
            future.result()
            return
        
        # Success!
        future = asyncio.run_coroutine_threadsafe(
            job_manager.update_status(job_id, JobStatus.DONE, progress=100, output_file=output_path),
            loop
        )
        future.result()
        
    except Exception as e:
        logging.error(f"Error processing video for job {job_id}: {e}", exc_info=True)
        future = asyncio.run_coroutine_threadsafe(
            job_manager.update_status(job_id, JobStatus.FAILED, error=str(e)),
            loop
        )
        future.result()

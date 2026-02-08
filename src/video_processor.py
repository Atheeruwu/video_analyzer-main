import cv2
import os
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
from .frame_analyzer import FrameAnalyzer
from .gemini_client import GeminiClient
from .config import API_KEY


def _tqdm(iterable=None, total=None):
    """Wrapper to disable tqdm in web/server contexts."""
    disable = os.environ.get("DISABLE_TQDM", "") == "1"
    return tqdm(iterable=iterable, total=total, disable=disable)

class VideoProcessor:
    def __init__(self, video_path, frame_sample_rate=1, scene_threshold=30):
        """
        Initialize the video processor
        
        Args:
            video_path: Path to the video file
            frame_sample_rate: Process every nth frame (default: 1)
            scene_threshold: Threshold for scene change detection (0-255)
        """
        self.video_path = video_path
        self.frame_sample_rate = frame_sample_rate
        self.scene_threshold = scene_threshold
        self.frames = []
        self.frame_data = []
        self.scenes = []
        self.analyzer = FrameAnalyzer()
        self.gemini_client = GeminiClient(api_key=API_KEY)
        self.summary = None
        self.max_frames = None
        
    def process_video(self):
        """Process the entire video and return a summary"""
        print(f"Processing video: {self.video_path}")
        # Fast demo mode: skip full analysis and only read metadata.
        if os.environ.get("FAST_DEMO", "") == "1":
            self.frames = []
            self.frame_data = []
            self.scenes = []
        else:
            self.extract_frames()
            self.analyze_frames()
        summary = self.generate_summary()
        return summary
        
    def extract_frames(self):
        """Extract frames from the video with scene change detection"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Total frames: {total_frames}, FPS: {fps}")
        print(f"Sampling every {self.frame_sample_rate} frame(s)")
        
        prev_frame = None
        current_scene = []
        scene_count = 0
        
        # Process frames with progress bar
        stop_processing = False
        with _tqdm(total=total_frames) as pbar:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Update progress bar
                pbar.update(1)
                
                # Only process every nth frame
                if frame_idx % self.frame_sample_rate == 0:
                    # Scene change detection
                    if prev_frame is not None:
                        # Convert frames to grayscale for comparison
                        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        
                        # Calculate absolute difference between frames
                        frame_diff = cv2.absdiff(curr_gray, prev_gray)
                        mean_diff = np.mean(frame_diff)
                        
                        # If difference exceeds threshold, mark as a new scene
                        if mean_diff > self.scene_threshold:
                            if current_scene:
                                self.scenes.append(current_scene)
                                scene_count += 1
                                print(f"Scene {scene_count} detected with {len(current_scene)} frames")
                            current_scene = []
                    
                    # Store the frame
                    self.frames.append(frame)
                    current_scene.append(frame)
                    prev_frame = frame.copy()

                    if self.max_frames is not None and len(self.frames) >= self.max_frames:
                        stop_processing = True
                        break
                
                frame_idx += 1
                
                # Exit loop if max frames reached
                if stop_processing:
                    break

        # Add the last scene if it exists
        if current_scene:
            self.scenes.append(current_scene)
            scene_count += 1
            
        print(f"Extracted {len(self.frames)} frames across {scene_count} scenes")
        cap.release()
        
    def analyze_frames(self, verbose=False):
        """Analyze the extracted frames"""
        print("Analyzing frames...")
        self.frame_data = []
        
        for i, frame in enumerate(_tqdm(self.frames)):
            # Analyze each frame using the FrameAnalyzer
            frame_analysis = self.analyzer.analyze_frame(frame)
            
            # Add frame index for reference
            frame_analysis['frame_index'] = i
            frame_analysis['timestamp'] = i * self.frame_sample_rate / self.get_video_fps()
            
            if verbose:
                print(f"Frame {i}: {frame_analysis.get('text', '')[:30]}...")
            
            self.frame_data.append(frame_analysis)
            
        print(f"Analyzed {len(self.frame_data)} frames")
    
    def get_video_fps(self):
        """Get the FPS of the video"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def _get_video_duration(self):
        """Get video duration in seconds without full frame analysis."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps and fps > 0:
            return float(total_frames) / float(fps)
        return 0.0
    
    def get_scene_data(self):
        """Group frame data by scenes"""
        scene_data = []
        
        frame_to_scene = {}
        for scene_idx, scene in enumerate(self.scenes):
            for frame in scene:
                for i, f in enumerate(self.frames):
                    if np.array_equal(frame, f):
                        frame_to_scene[i] = scene_idx
                        break
        
        scene_frames = defaultdict(list)
        for frame_info in self.frame_data:
            frame_idx = frame_info['frame_index']
            if frame_idx in frame_to_scene:
                scene_idx = frame_to_scene[frame_idx]
                scene_frames[scene_idx].append(frame_info)
        
        for scene_idx, frames in scene_frames.items():
            scene_text = ""
            for frame in frames:
                if 'text' in frame and frame['text']:
                    scene_text += f"{frame['text']} "
            
            scene_data.append({
                'scene_index': scene_idx,
                'start_time': frames[0]['timestamp'] if frames else 0,
                'end_time': frames[-1]['timestamp'] if frames else 0,
                'duration': frames[-1]['timestamp'] - frames[0]['timestamp'] if frames else 0,
                'text_content': scene_text.strip(),
                'frame_count': len(frames),
                'frames': frames
            })
        
        return scene_data

    def generate_summary(self):
        """Generate a summary using the GeminiClient"""
        print("Generating summary (Gemini disabled)...")
        
        scene_data = self.get_scene_data() if self.frame_data else []
        
        video_info = {
            'filename': os.path.basename(self.video_path),
            'frames_processed': len(self.frames),
            'scenes_detected': len(self.scenes),
            'total_duration': self.frame_data[-1]['timestamp'] if self.frame_data else self._get_video_duration(),
            'scenes': scene_data,
            'frame_data': self.frame_data
        }

        # Gemini call disabled to avoid API/quota usage.
        # self.summary = self.gemini_client.generate_summary(video_info)

        # Build a richer local summary without Gemini.
        summary_lines = [
            "Local summary (Gemini disabled)",
            f"Video: {video_info['filename']}",
            f"Duration: {video_info['total_duration']:.2f}s",
            f"Frames processed: {video_info['frames_processed']}",
            f"Scenes detected: {video_info['scenes_detected']}",
            "",
            "Scene details:",
        ]

        # Add per-scene info with OCR snippets, keywords, and object counts.
        if scene_data:
            for scene in scene_data:
                scene_header = (
                    f"Scene {scene['scene_index'] + 1}: "
                    f"{scene['start_time']:.2f}s - {scene['end_time']:.2f}s "
                    f"({scene['duration']:.2f}s), frames: {scene['frame_count']}"
                )
                summary_lines.append(scene_header)

                text = scene.get("text_content", "").strip()
                if text:
                    snippet = text[:400] + ("..." if len(text) > 400 else "")
                    summary_lines.append(f"OCR text: {snippet}")
                else:
                    summary_lines.append("OCR text: (none)")

                from collections import Counter
                if text:
                    tokens = text.split()
                    counts = Counter(tokens)
                    keywords = [w for w, _ in counts.most_common(10)]
                    summary_lines.append(
                        "Top keywords: " + (", ".join(keywords) if keywords else "(none)")
                    )
                else:
                    summary_lines.append("Top keywords: (none)")

                summary_lines.append("Top OCR snippets:")
                summary_lines.append("- (demo)")
        else:
            summary_lines.append("Scene details: (skipped in fast demo mode)")
            summary_lines.append("OCR text: (skipped in fast demo mode)")
            summary_lines.append("Top keywords: (skipped in fast demo mode)")
            summary_lines.append("Top OCR snippets:")
            summary_lines.append("- (skipped in fast demo mode)")

        # Add a quick object count summary across all frames.
        if self.frame_data:
            total_objects = 0
            for frame_info in self.frame_data:
                objects = frame_info.get("objects") or []
                total_objects += len(objects)
            summary_lines.append("")
            summary_lines.append(f"Total detected objects (rough): {total_objects}")

        self.summary = "\n".join(summary_lines) + "\n"
        print("Summary generation complete")
        return self.summary

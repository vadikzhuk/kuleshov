import json
import cv2
from pathlib import Path
from pydantic import BaseModel, Field
from scenedetect import detect, ContentDetector
import ollama


# Structured output schema for AI analysis of video shots
class ShotAnalysis(BaseModel):
    shot_type: str = Field(
        description="Scene zoom level: Extreme Close-up, Close-up, Medium Shot, Wide Shot, or B-roll")
    description: str = Field(description="Short summary of scene actions in English (1-2 sentences)")
    camera_movement: str = Field(description="Camera movement: Static, Pan, Tilt, Handheld/Shake, Zoom")
    quality_status: str = Field(
        description="Quality assessment: 'GOOD' if the frame is sharp and successful, 'WARNING' if there is a light defect, 'TRASH' if out of focus, strong shake or defect")
    issues: list[str] = Field(
        description="List of found issues (e.g.: out of focus, tilted horizon, overexposure, shake). Empty list if everything is fine.")


def extract_middle_frame(video_path: str, start_frame: int, end_frame: int, max_width: int = 768) -> bytes:
    """Extracts the middle frame of a scene, compresses it, and returns it as JPEG bytes."""
    cap = cv2.VideoCapture(video_path)
    mid_frame = (start_frame + end_frame) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"Failed to read frame {mid_frame}")

    # Resize for faster VLM processing without losing meaning
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


def analyze_video(video_path: str, model_name: str = "qwen3.5:9b"):
    print(f"🎬 Step 1: Looking for scene cuts in file {video_path}...")
    # threshold=27.0 — standard sensitivity of the scene cut detector
    scene_list = detect(video_path, ContentDetector(threshold=27.0))

    # If the video was shot with one take and no scene cuts, treat it as a single scene
    if not scene_list:
        print("No scene cuts found, analyzing video as a single scene.")
        return

    print(f"✅ Found scenes: {len(scene_list)}. Starting AI analysis through {model_name}...\n")
    results = []

    prompt = (
        "You are a professional video editor and colorist. "
        "Analyze this stop-frame from the original video material. "
        "Determine the shot size, describe the action, assess the technical quality (sharpness, lighting, composition) "
        "and note if this is a technical defect."
    )

    for i, (start_time, end_time) in enumerate(scene_list, start=1):
        start_tc = start_time.get_timecode()
        end_tc = end_time.get_timecode()
        print(f"[{i}/{len(scene_list)}] Scene {start_tc} -> {end_tc}...", end=" ", flush=True)

        try:
            img_bytes = extract_middle_qwframe(
                video_path,
                start_time.get_frames(),
                end_time.get_frames()
            )

            # Request to the local Ollama with a mandatory JSON schema
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt, "images": [img_bytes]}],
                format=ShotAnalysis.model_json_schema(),
                think=False,  # Disable thinking for instant JSON response
                options={"temperature": 0.1}
            )

            analysis = ShotAnalysis.model_validate_json(response.message.content)
            print(f"[{analysis.quality_status}] {analysis.shot_type} — {analysis.description}")

            results.append({
                "scene_number": i,
                "start_timecode": start_tc,
                "end_timecode": end_tc,
                "duration_sec": round((end_time - start_time).get_seconds(), 2),
                "analysis": analysis.model_dump()
            })

        except Exception as e:
            print(f"Error: {e}")

    output_file = Path(video_path).with_suffix(".ai_log.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Done! Detailed log saved to: {output_file}")


if __name__ == "__main__":
    # Specify the path to any test video (preferably with multiple cuts)
    TEST_VIDEO = "morzhilebed.mp4"
    analyze_video(TEST_VIDEO)

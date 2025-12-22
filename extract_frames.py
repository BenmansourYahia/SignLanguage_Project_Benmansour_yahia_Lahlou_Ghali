"""
Video to Training Images Extractor

How to use:
1. Record videos for each sign (5-10 seconds each)
2. Name them like: A.mp4, B.mp4, C.mp4, etc.
3. Put all videos in a folder called 'videos'
4. Run: python extract_frames.py
5. Frames will be saved to trainingdata/ automatically!
"""

import cv2
import os
from pathlib import Path

# Configuration
VIDEO_FOLDER = "Videos"  # Put your videos here
OUTPUT_FOLDER = "trainingdata"  # Frames go here
FRAME_INTERVAL = 5  # Extract 1 frame every 5 frames (higher = fewer images)
IMAGE_SIZE = 224  # Resize to this size

print("="*60)
print("Video Frame Extractor for ASL Training")
print("="*60)

# Check if video folder exists
if not os.path.exists(VIDEO_FOLDER):
    print(f"\n❌ ERROR: '{VIDEO_FOLDER}' folder not found!")
    print("\nPlease create it and add your videos:")
    print(f"""
{VIDEO_FOLDER}/
├── A.mp4 (or .avi, .mov, .mkv)
├── B.mp4
├── C.mp4
└── ... (one video per sign)
    """)
    exit(1)

# Create output folder if needed
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Supported video formats
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']

# Find all letter folders
letter_folders = [f for f in Path(VIDEO_FOLDER).iterdir() if f.is_dir()]

if len(letter_folders) == 0:
    print(f"\n❌ ERROR: No letter folders found in '{VIDEO_FOLDER}'!")
    print(f"\nExpected structure:")
    print(f"{VIDEO_FOLDER}/A/ (with video files inside)")
    print(f"{VIDEO_FOLDER}/B/ (with video files inside)")
    exit(1)

print(f"\n✓ Found {len(letter_folders)} letter folder(s)")
print("\nProcessing videos...\n")

total_frames = 0
total_videos = 0

for letter_folder in sorted(letter_folders):
    sign_name = letter_folder.name  # e.g., "A", "B", "C"
    
    # Find all videos in this letter folder
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(letter_folder.glob(f"*{ext}"))
    
    if len(video_files) == 0:
        print(f"[{sign_name}] ⚠ No videos found, skipping...")
        continue
    
    print(f"[{sign_name}] Found {len(video_files)} video(s)")
    
    # Create output folder for this sign
    output_dir = os.path.join(OUTPUT_FOLDER, sign_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Count existing images
    existing_count = len(list(Path(output_dir).glob("*.jpg")))
    
    frames_for_letter = 0
    
    for video_path in video_files:
        print(f"  Processing {video_path.name}...")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"    ⚠ Failed to open, skipping...")
            continue
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at intervals
            if frame_count % FRAME_INTERVAL == 0:
                # Resize frame
                resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                
                # Generate filename
                img_filename = f"img_{existing_count + frames_for_letter + saved_count + 1:04d}.jpg"
                img_path = os.path.join(output_dir, img_filename)
                
                # Save frame
                cv2.imwrite(img_path, resized)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        frames_for_letter += saved_count
        total_videos += 1
        
        print(f"    ✓ Extracted {saved_count} frames")
    
    total_frames += frames_for_letter
    new_total = existing_count + frames_for_letter
    
    print(f"  [{sign_name}] Total images now: {new_total}\n")

print("\n" + "="*60)
print("EXTRACTION COMPLETE!")
print("="*60)
print(f"\n✓ Videos processed: {total_videos}")
print(f"✓ Total frames extracted: {total_frames}")
print(f"✓ Frames saved to: {OUTPUT_FOLDER}/")
print(f"\nNext steps:")
print(f"1. Check the images look good")
print(f"2. Run: python train_improved_model.py")
print(f"3. Wait for training to complete (~10-15 min)")
print(f"4. Run: flutter run")
print("\nWith 15-second videos, you'll get ~90 frames each!")
print("Your model will be MUCH more accurate! 🎉")
print("="*60)

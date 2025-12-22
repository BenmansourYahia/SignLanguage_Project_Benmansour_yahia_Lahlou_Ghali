# Video Frame Extraction Guide

## Quick Start

### 1. Record Videos

Use your phone or webcam to record short videos:

**Format:**
- **Length**: 5-10 seconds per sign
- **Content**: Make the sign and move your hand slightly
- **Quality**: Any resolution (script will resize)
- **Format**: .mp4, .avi, .mov, .mkv

**Tips for good videos:**
- Wave your hand slightly while signing
- Try different angles during the video
- Keep the sign clear and visible
- Record in good lighting

### 2. Organize Videos

Create a `videos` folder and name files by sign:

```
videos/
├── A.mp4
├── B.mp4
├── C.mp4
├── D.mp4
└── ... (one video per letter)
```

**Naming is important!** The filename (without extension) becomes the folder name.

### 3. Extract Frames

```bash
python extract_frames.py
```

The script will:
- Find all videos in `videos/` folder
- Extract 1 frame every 5 frames (~6 frames per second)
- Resize each frame to 224x224
- Save to `trainingdata/A/`, `trainingdata/B/`, etc.
- Add to any existing images (won't delete old ones!)

### 4. How Many Frames?

**From a 10-second video:**
- At 30fps: ~60 frames extracted
- At 60fps: ~120 frames extracted

**Perfect for reaching your 60-image goal!**

## Example Workflow

### For ALL 26 letters:

1. **Record**: 
   - 26 videos (one per letter)
   - 5-10 seconds each
   - Total time: ~15 minutes

2. **Extract**:
   ```bash
   python extract_frames.py
   ```
   - Processes all 26 videos
   - Extracts ~60 frames each
   - Total: ~1,560 images

3. **Train**:
   ```bash
   python train_improved_model.py
   ```
   - Trains on all images
   - Takes ~10-15 minutes
   - Gives excellent accuracy!

## Advanced Tips

### Extract More/Fewer Frames

Edit `extract_frames.py` and change:
```python
FRAME_INTERVAL = 5  # Lower = more frames (3 = more, 10 = fewer)
```

### Different Image Size

Change:
```python
IMAGE_SIZE = 224  # Or 64, 128, 256, etc.
```

### Video Quality Tips

**Best results:**
- Record in 1080p or higher
- Good lighting (avoid shadows)
- Plain background
- Hand clearly visible
- Move hand naturally (don't stay perfectly still)

## Troubleshooting

**"No videos found"**
- Check folder name is `videos/` (lowercase)
- Check video format is supported
- Try renaming file extension to lowercase (.mp4 not .MP4)

**"Failed to open video"**
- Video file might be corrupted
- Try converting to .mp4 format
- Check file isn't empty

**Too many/few frames**
- Adjust `FRAME_INTERVAL` in the script
- Lower number = more frames
- Higher number = fewer frames

## Ready!

Just record your videos, put them in `videos/`, and run:
```bash
python extract_frames.py
```

Super fast way to create a huge training dataset! 🚀

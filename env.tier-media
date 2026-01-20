# PMOVES.AI Tier Environment: Media
# For Media processing services (Whisper, YOLO, FFmpeg, etc.)
# Source after env.shared: source env.shared && source env.tier-media.sh

# ============================================================================
# Media Tier Configuration
# ============================================================================

export TIER=media

# Media processing limits
export MAX_CONCURRENT_JOBS=${MAX_CONCURRENT_JOBS:-4}  # Limited for GPU
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-2000}
export JOB_TIMEOUT_MS=${JOB_TIMEOUT_MS:-1800000}  # 30 minutes

# GPU configuration
export GPU_ENABLED=${GPU_ENABLED:-true}
export GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-0.9}
export GPU_ALLOW_GROWTH=${GPU_ALLOW_GROWTH:-true}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Video processing
export VIDEO_MAX_FRAMES=${VIDEO_MAX_FRAMES:-10000}
export VIDEO_SAMPLE_RATE=${VIDEO_SAMPLE_RATE:-1}  # Sample every Nth frame
export VIDEO_MAX_RESOLUTION=${VIDEO_MAX_RESOLUTION:-1080p}

# Audio processing
export AUDIO_SAMPLE_RATE=${AUDIO_SAMPLE_RATE:-16000}
export AUDIO_CHANNELS=${AUDIO_CHANNELS:-1}
export AUDIO_MAX_DURATION_SECONDS=${AUDIO_MAX_DURATION_SECONDS:-3600}

# Image processing
export IMAGE_MAX_SIZE=${IMAGE_MAX_SIZE:-4096}  # Max dimension
export IMAGE_FORMAT=${IMAGE_FORMAT:-jpg}
export IMAGE_QUALITY=${IMAGE_QUALITY:-90}

# Whisper transcription
export WHISPER_MODEL=${WHISPER_MODEL:-small}
export WHISPER_LANGUAGE=${WHISPER_LANGUAGE:-auto}
export WHISPER_TASK=${WHISPER_TASK:-transcribe}

# YOLO object detection
export YOLO_MODEL=${YOLO_MODEL:-yolov8m}
export YOLO_CONFIDENCE_THRESHOLD=${YOLO_CONFIDENCE_THRESHOLD:-0.25}
export YOLO_IOU_THRESHOLD=${YOLO_IOU_THRESHOLD:-0.45}

# FFmpeg configuration
export FFMPEG_NUM_THREADS=${FFMPEG_NUM_THREADS:-4}
export FFMPEG_PRESET=${FFMPEG_PRESET:-medium}
export FFMPEG_CRF=${FFMPEG_CRF:-23}

# Output storage
export OUTPUT_BUCKET=${OUTPUT_BUCKET:-outputs}
export OUTPUT_FORMAT=${OUTPUT_FORMAT:-json}
export INCLUDE_METADATA=${INCLUDE_METADATA:-true}

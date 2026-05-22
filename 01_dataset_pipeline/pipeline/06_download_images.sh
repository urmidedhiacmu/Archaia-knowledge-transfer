#!/bin/bash
#SBATCH --job-name=archaia_images
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Job 3: Download Images
# Downloads filtered images at 1024px using img2dataset

echo "========================================"
echo "Job 3: Download Images"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# Move logs to logs directory when job finishes
trap 'mv ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.{out,err} ${HOME}/archaia_project/logs/ 2>/dev/null || true' EXIT

# Set paths
OUTPUT_DIR="/data/user_data/${USER}/archaia"
TSV_FILE="${OUTPUT_DIR}/img2dataset_download_clean.tsv"
IMG_OUTPUT="${OUTPUT_DIR}/artifact_images_1024_full"

# ==========================
# INSTALL IMG2DATASET
# ==========================

echo "Checking img2dataset..."

if command -v img2dataset &> /dev/null; then
    echo "✓ img2dataset found"
else
    echo "Installing img2dataset..."
    pip install --user img2dataset
    export PATH="${HOME}/.local/bin:${PATH}"
    
    if command -v img2dataset &> /dev/null; then
        echo "✓ img2dataset installed"
    else
        echo "✗ Error: img2dataset installation failed"
        exit 1
    fi
fi

echo ""

# ==========================
# VERIFY INPUT
# ==========================

echo "Verifying input file..."

if [ ! -f "${TSV_FILE}" ]; then
    echo "✗ Error: TSV file not found: ${TSV_FILE}"
    echo "  Job 2 may not have completed successfully"
    exit 1
fi

IMG_COUNT=$(wc -l < "${TSV_FILE}")
echo "✓ Found TSV with ${IMG_COUNT} images"
echo ""

# ==========================
# DOWNLOAD CONFIGURATION
# ==========================

IMAGE_SIZE=1024
PROCESSES=16
THREADS=8
RESIZE_MODE="keep_ratio"

echo "Download configuration:"
echo "  Image size: ${IMAGE_SIZE}px (max dimension)"
echo "  Processes: ${PROCESSES}"
echo "  Threads: ${THREADS}"
echo "  Resize mode: ${RESIZE_MODE}"
echo "  Output: ${IMG_OUTPUT}"

echo ""

echo "========================================"
echo "Starting download..."
echo "========================================"
echo ""

img2dataset \
    --url_list "${TSV_FILE}" \
    --input_format "tsv" \
    --url_col "url" \
    --caption_col "caption" \
    --output_folder "${IMG_OUTPUT}" \
    --processes_count ${PROCESSES} \
    --thread_count ${THREADS} \
    --image_size ${IMAGE_SIZE} \
    --resize_mode "${RESIZE_MODE}" \
    --resize_only_if_bigger True \
    --output_format "files" \
    --enable_wandb False \
    --skip_reencode False \
    --incremental True \
    --retries 5 \
    --timeout 30


DOWNLOAD_EXIT=$?

# ==========================
# VERIFY DOWNLOAD
# ==========================

echo ""
echo "========================================"
echo "Verifying download..."
echo "========================================"
echo ""

if [ ${DOWNLOAD_EXIT} -eq 0 ]; then
    echo "✓ img2dataset completed successfully"
else
    echo "⚠ img2dataset exited with code ${DOWNLOAD_EXIT}"
fi

# Count downloaded images
if [ -d "${IMG_OUTPUT}" ]; then
    DOWNLOADED=$(find "${IMG_OUTPUT}" -name "*.jpg" -o -name "*.png" | wc -l)
    echo "  Downloaded images: ${DOWNLOADED}"
    echo "  Expected images: ${IMG_COUNT}"
    
    if [ ${DOWNLOADED} -gt 0 ]; then
        SUCCESS_RATE=$(echo "scale=1; ${DOWNLOADED} * 100 / ${IMG_COUNT}" | bc)
        echo "  Success rate: ${SUCCESS_RATE}%"
    fi
    
    # Check stats file
    if [ -f "${IMG_OUTPUT}/stats.json" ]; then
        echo ""
        echo "Download stats:"
        cat "${IMG_OUTPUT}/stats.json"
    fi
else
    echo "✗ Output directory not found"
    exit 1
fi

# ==========================
# STORAGE INFO
# ==========================

echo ""
echo "Storage usage:"
du -sh "${IMG_OUTPUT}"

# ==========================
# SUMMARY
# ==========================

echo ""
echo "========================================"
if [ ${DOWNLOAD_EXIT} -eq 0 ] && [ ${DOWNLOADED} -gt 0 ]; then
    echo "Job 3 Complete - SUCCESS"
else
    echo "Job 3 Complete - CHECK RESULTS"
fi
echo "========================================"
echo "End: $(date)"
echo ""
echo "Images saved to:"
echo "  ${IMG_OUTPUT}/"
echo ""
echo "All jobs complete!"
echo "========================================"

exit ${DOWNLOAD_EXIT}

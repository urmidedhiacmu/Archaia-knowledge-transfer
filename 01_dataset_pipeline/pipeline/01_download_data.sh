#!/bin/bash
#SBATCH --job-name=archaia_download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# # Job 1: Download OpenContext Parquet Files
# # Downloads manifest, resources, and assertions to ~/archaia_project/data/

# echo "========================================"
# echo "Job 1: Download OpenContext Data"
# echo "========================================"
# echo "Job ID: ${SLURM_JOB_ID}"
# echo "Node: $(hostname)"
# echo "Start: $(date)"
# echo ""

# # Move logs to logs directory when job finishes
# trap 'mv ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.{out,err} ${HOME}/archaia_project/logs/ 2>/dev/null || true' EXIT

# # Go to data directory
# DATA_DIR="${HOME}/archaia_project/data"
# mkdir -p "${DATA_DIR}"
# cd "${DATA_DIR}"

# echo "Downloading to: ${DATA_DIR}"
# echo ""

# # ==========================
# # DOWNLOAD FILES
# # ==========================

# echo "Downloading manifest..."
# wget -q --show-progress \
#     https://storage.googleapis.com/opencontext-parquet/oc_all_manifest.parquet

# echo "✓ Manifest downloaded"
# echo ""

# echo "Downloading resources..."
# wget -q --show-progress \
#     https://storage.googleapis.com/opencontext-parquet/oc_all_resources.parquet

# echo "✓ Resources downloaded"
# echo ""

# echo "Downloading assertions..."
# wget -q --show-progress \
#     https://storage.googleapis.com/opencontext-parquet/oc_all_assertions.parquet

# echo "✓ Assertions downloaded"
# echo ""

# ==========================
# VERIFY
# ==========================

echo "========================================"
echo "Verifying downloads..."
echo "========================================"
echo ""

ls -lh *.parquet

# Check file sizes
for file in oc_all_manifest.parquet oc_all_resources.parquet oc_all_assertions.parquet; do
    if [ ! -f "$file" ]; then
        echo "✗ Error: $file not found!"
        exit 1
    fi
    
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file")
    if [ "$size" -eq 0 ]; then
        echo "✗ Error: $file is empty!"
        exit 1
    fi
done

echo ""
echo "✓ All files downloaded successfully!"
echo ""

# ==========================
# SUMMARY
# ==========================

echo "========================================"
echo "Job 1 Complete"
echo "========================================"
echo "End: $(date)"
echo ""
echo "Files saved to: ${DATA_DIR}/"
echo ""
echo "Next: Job 2 will process this data"
echo "========================================"

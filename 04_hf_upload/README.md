# HuggingFace Dataset Upload

Scripts for uploading the dataset to HuggingFace.

| Script | What it does |
|---|---|
| `upload_to_hf.py` + `upload_job.sh` | Uploads full dataset to `archaia/dataset_v1` (private). Has a retry loop — safe to resubmit if job times out. |
| `create_sample_upload_hf_eric.py` + `eric_upload_sample_hf.sh` | Creates a 100-artifact sample with images and uploads to `archaia/dataset_sample_100_v4` (private). |

**HF org:** `archaia` (private). You need to be added to the org to push.
Contact Eric Kansa or Prof. Ippolito for access.

## Running the sample upload (for sharing with collaborators)

```bash
source ~/archaia_env/bin/activate
huggingface-cli login   # paste your HF token

# Edit create_sample_upload_hf_eric.py to update HF_REPO and PARQUET paths if needed
sbatch eric_upload_sample_hf.sh
tail -f scripts/sample_upload_<jobid>.out
```

When done it prints the HF URL to share.

## Notes

- Use `--partition=cpu` for HF upload jobs — they don't need GPU and `normal` QoS requires one
- `api.upload_large_folder()` handles resuming interrupted uploads
- The retry loop in `upload_job.sh` restarts the script every 60s if it exits early

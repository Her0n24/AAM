scratch_path="/work/scratch-nopw2/hhhn2"
monthly_mean_path_base="${scratch_path}/ERA5/monthly_mean/variables"

# Check for corrupt files in the monthly mean variables directory
for f in ${monthly_mean_path_base}/ERA5_*.nc; do
  if ! ncdump -h "$f" >/dev/null 2>&1; then
    echo "CORRUPT: $f"
  fi
done
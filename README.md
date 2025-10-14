# average_ti3s_rgbgeom
average_ti3s_rgbgeom.py

Purpose:
  - Create a single averaged .ti3 from multiple ArgyllCMS .ti3 files (repeated scans).
  - RGB_R/G/B -> geometric median of the RGB triplets (true 3D median).
  - STDEV_R/G/B -> pooled standard deviation combining within-scan and between-scan variance:
        total_var = mean(STDEV_i^2) + Var(R_i across scans)
        total_std = sqrt(total_var)
    If STDEV_i fields are missing in inputs, falls back to stddev of the per-scan means.
  - XYZ_/LAB_/other fields -> copied unchanged from the first .ti3.
  - All numeric output formatted with fixed six decimals (e.g. 0.123000).

Usage:
  python3 average_ti3s_rgbgeom.py input1.ti3 input2.ti3 ... -o output.ti3

Example:
  python3 average_ti3s_rgbgeom_fixeddecimals.py \
    ScannedIT8WFTarget01.ti3 ScannedIT8WFTarget02.ti3 \
    ScannedIT8WFTarget03.ti3 ScannedIT8WFTarget04.ti3 \
    ScannedIT8WFTarget05.ti3 -o AveragedWF_geom.ti3 -v

Then profile with:
  colprof -v -D "Scanner Geometric Median" -qm -as AveragedWF_geom

----------------------------------------------------------------------

Flags:
  -o, --out <filename>   Output .ti3 filename (required)
  -v, --verbose          Verbose progress output (optional)

----------------------------------------------------------------------

Notes:
  - Requires Python 3.7+ and NumPy installed.
  - Assumes all input files share the same patch order and field names.
  - The pooled formula assumes roughly equal numbers of sampled pixels per patch
    across scans. If counts differ and you have sample counts, weight the within-scan variances.
    
Fixes included:
 - Robust per-file parsing using each file's BEGIN_DATA_FORMAT
 - Normalization of sample IDs so "A01", "A1", " a01 " match
 - Proper scoping (no NameError for loop variable)
 - Warnings when patches are missing in any file

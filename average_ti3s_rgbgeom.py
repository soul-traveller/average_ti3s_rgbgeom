#!/usr/bin/env python3
"""
average_ti3s_rgbgeom.py
=====================================

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

Author:
  Knut Larsson with support from ChatGPT (OpenAI), 2025
"""

import sys
import re
import argparse
from pathlib import Path
import numpy as np

# ---------- helpers ----------
def read_lines(path):
    """Read file into list of lines (keep newline chars)."""
    return path.read_text(encoding='utf-8', errors='ignore').splitlines(keepends=True)

def find_marker(lines, marker):
    """Return the index of a marker line, or -1 if not found."""
    for i, L in enumerate(lines):
        if L.strip() == marker:
            return i
    return -1

def parse_fields(line):
    """Split a BEGIN_DATA_FORMAT line into field tokens."""
    return re.split(r'\s+', line.strip())

def parse_data_lines(lines, data_start, data_end):
    """Return tokenized data lines between BEGIN_DATA and END_DATA."""
    data = []
    for L in lines[data_start + 1 : (data_end if data_end != -1 else None)]:
        s = L.strip()
        if not s:
            continue
        toks = re.split(r'\s+', s)
        data.append(toks)
    return data

def build_out_line(tokens):
    """Format output tokens to a single .ti3 data line with trailing space+newline."""
    return ' '.join(tokens) + ' \n'

# ---------- geometric median (Weiszfeld) ----------
def geometric_median(points, eps=1e-12, max_iter=100000):
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 1:
        return pts[0]
    median = pts.mean(axis=0)
    for _ in range(max_iter):
        diff = pts - median
        dist = np.linalg.norm(diff, axis=1)
        # avoid division by zero; eps small
        w = 1.0 / np.maximum(dist, eps)
        new_med = (w[:, None] * pts).sum(axis=0) / w.sum()
        if np.linalg.norm(new_med - median) < 1e-15:
            return new_med
        median = new_med
    return median
# ---------- sample id normalization ----------
def normalize_id(s):
    """
    Normalize patch/sample ID for robust matching:
    - strip whitespace
    - uppercase
    - if pattern PREFIX + NUMBER, remove leading zeros in number (A01 -> A1)
    This makes 'A01', 'A1', ' a01 ' map to the same normalized key.
    """
    if s is None:
        return s
    s2 = s.strip().upper()
    m = re.match(r'^([A-Z]+)(0*)(\d+)$', s2)
    if m:
        prefix = m.group(1)
        num = m.group(3)
        return f"{prefix}{int(num)}"
    return s2

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Average .ti3 files: RGB geometric median + pooled STDEV (fixed decimals)")
    parser.add_argument("files", nargs='+', help=".ti3 input files (2 or more)")
    parser.add_argument("-o", "--out", required=True, help="Output .ti3 filename")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    args = parser.parse_args()

    input_paths = [Path(f) for f in args.files]
    if len(input_paths) < 2:
        sys.exit("Need at least two .ti3 files.")

    # --- Read first file to get header/format/order (this defines output ordering) ---
    base_lines = read_lines(input_paths[0])
    fmt_s = find_marker(base_lines, "BEGIN_DATA_FORMAT")
    fmt_e = find_marker(base_lines, "END_DATA_FORMAT")
    if fmt_s == -1 or fmt_e == -1:
        sys.exit("First file missing BEGIN_DATA_FORMAT/END_DATA_FORMAT")

    fields = parse_fields(base_lines[fmt_s + 1])
    field_index = {name: idx for idx, name in enumerate(fields)}

    dstart = find_marker(base_lines, "BEGIN_DATA")
    dend = find_marker(base_lines, "END_DATA")
    if dstart == -1:
        sys.exit("First file missing BEGIN_DATA")

    data_lines = parse_data_lines(base_lines, dstart, dend)

    # Keep base sample order and tokens
    base_ids = [tokens[0] for tokens in data_lines]               # original IDs in first file
    normalized_to_base = {normalize_id(sid): sid for sid in base_ids}
    base_map = {tokens[0]: tokens for tokens in data_lines}       # original tokens by original ID

    # Prepare containers
    RGB_data = {sid: [] for sid in base_ids}                      # sid = original base id
    STDEV_within = {sid: {"R": [], "G": [], "B": []} for sid in base_ids}
    have_stdev_fields = False

    # --- Collect data from all input files (each file parsed independently) ---
    for p in input_paths:
        if args.verbose:
            print("Reading:", p.name)
        lines = read_lines(p)

        # Use this file's data format line (robust per-file)
        local_fmt_s = find_marker(lines, "BEGIN_DATA_FORMAT")
        local_fmt_e = find_marker(lines, "END_DATA_FORMAT")
        if local_fmt_s == -1 or local_fmt_e == -1:
            sys.exit(f"File {p} missing BEGIN_DATA_FORMAT/END_DATA_FORMAT")
        local_fields = parse_fields(lines[local_fmt_s + 1])
        local_index = {name: idx for idx, name in enumerate(local_fields)}

        # data block for this file
        ds = find_marker(lines, "BEGIN_DATA")
        de = find_marker(lines, "END_DATA")
        if ds == -1:
            sys.exit(f"File {p} missing BEGIN_DATA")
        local_data = parse_data_lines(lines, ds, de)

        # Build normalized local_map: normalized_id -> tokens
        local_map = {normalize_id(t[0]): t for t in local_data}

        # detect STDEV fields presence in this file
        for key in ("STDEV_R", "STDEV_G", "STDEV_B"):
            if key in local_index:
                have_stdev_fields = True

        # For each sample in the base ordering, find the corresponding local tokens
        for base_id in base_ids:
            base_norm = normalize_id(base_id)
            if base_norm not in local_map:
                # warn but continue; this means this patch was not found in this file
                if args.verbose:
                    print(f"Warning: patch {base_id} (normalized {base_norm}) not found in {p.name}")
                continue
            toks = local_map[base_norm]

            # read RGB using local field indices (robust to order differences)
            try:
                r = float(toks[local_index["RGB_R"]])
                g = float(toks[local_index["RGB_G"]])
                b = float(toks[local_index["RGB_B"]])
            except Exception as e:
                sys.exit(f"Failed to parse RGB values for patch {base_id} in {p.name}: {e}")

            RGB_data[base_id].append((r, g, b))

            # read within-scan STDEV fields if present
            if "STDEV_R" in local_index:
                try:
                    STDEV_within[base_id]["R"].append(float(toks[local_index["STDEV_R"]]))
                    STDEV_within[base_id]["G"].append(float(toks[local_index["STDEV_G"]]))
                    STDEV_within[base_id]["B"].append(float(toks[local_index["STDEV_B"]]))
                except Exception:
                    # ignore parse errors for stdev values
                    pass

    # --- Build averaged data lines in base order ---
    new_data_lines = []
    for i, base_id in enumerate(base_ids, 1):
        entries = RGB_data[base_id]
        if len(entries) == 0:
            # No data found for this patch in any file (unexpected)
            print(f"Warning: no RGB entries collected for patch {base_id}; copying base tokens unchanged.")
            new_data_lines.append(build_out_line(base_map[base_id][:]))
            continue
        
        rgbs = np.array(entries, dtype=float)  # shape (n_scans_found, 3)
        # geometric median
        median_rgb = geometric_median(rgbs)
        # Round geometric median components to fixed 6 decimals for deterministic output
        out_val = [round(float(v), 6) for v in median_rgb]
        # between-scan variance of per-scan means
        var_between = np.var(rgbs, axis=0, ddof=0)

        # pooled total variance = mean(within-scan variance_i) + var_between
        if have_stdev_fields and len(STDEV_within[base_id]["R"]) > 0:
            within_var_R = np.mean(np.array(STDEV_within[base_id]["R"], dtype=float)**2) if len(STDEV_within[base_id]["R"])>0 else 0.0
            within_var_G = np.mean(np.array(STDEV_within[base_id]["G"], dtype=float)**2) if len(STDEV_within[base_id]["G"])>0 else 0.0
            within_var_B = np.mean(np.array(STDEV_within[base_id]["B"], dtype=float)**2) if len(STDEV_within[base_id]["B"])>0 else 0.0
            within = np.array([within_var_R, within_var_G, within_var_B])
            total_var = within + var_between
        else:
            # fallback to between-scan variance only
            total_var = var_between

        total_std = np.sqrt(total_var)

        # start from base tokens and overwrite RGB and STDEV fields
        # --- Build output tokens by field name ---
        toks_out = []

        for fname in fields:
            if fname == "SAMPLE_ID":
                toks_out.append(base_id)
            elif fname == "RGB_R":
                toks_out.append(f"{out_val[0]:.6f}")
            elif fname == "RGB_G":
                toks_out.append(f"{out_val[1]:.6f}")
            elif fname == "RGB_B":
                toks_out.append(f"{out_val[2]:.6f}")
            elif fname == "STDEV_R":
                toks_out.append(f"{total_std[0]:.6f}")
            elif fname == "STDEV_G":
                toks_out.append(f"{total_std[1]:.6f}")
            elif fname == "STDEV_B":
                toks_out.append(f"{total_std[2]:.6f}")
            else:
                # copy unchanged from base line if available
                idx = field_index.get(fname)
                if idx is not None and idx < len(base_map[base_id]):
                    toks_out.append(base_map[base_id][idx])
                else:
                    toks_out.append("0.000000")  # fallback

        new_data_lines.append(build_out_line(toks_out))

        if args.verbose and (i % 50 == 0 or i == len(base_ids)):
            print(f"Processed {i}/{len(base_ids)} patches")

    # --- Write output file preserving header/footer from base file ---
    pre = base_lines[: dstart + 1]
    post = base_lines[dend:] if dend != -1 else []
    out_lines = pre + new_data_lines + post
    Path(args.out).write_text(''.join(out_lines), encoding='utf-8')
    print(f"Wrote averaged .ti3 (geometric median RGB, pooled STDEV) -> {args.out}")

if __name__ == "__main__":
    main()

import bz2
import json
import os
import struct
import sys
import time
import gzip
from datetime import datetime, timedelta, timezone

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import geojsoncontour
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.ndimage import gaussian_filter, map_coordinates
from urllib.request import Request, urlopen
from urllib.error import HTTPError

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# --- ตั้งค่าโฟลเดอร์และโหมดการทำงาน (Local vs Production) ---
NOWCAST_OUT_DIR = os.environ.get("NOWCAST_OUT_DIR", "../Doofah_fe/public/nowcast")
ECMWF_PUBLIC_DIR = os.environ.get("ECMWF_PUBLIC_DIR", "../Doofah_fe/public")
STATE_MARKER_FILE = f"{NOWCAST_OUT_DIR}/_last_obs_time.txt"
STATUS_FILE = f"{NOWCAST_OUT_DIR}/status.json"
SHADOW_ALERTS_FILE = f"{NOWCAST_OUT_DIR}/shadow_alerts.json"

IS_PRODUCTION_CI = os.environ.get("GITHUB_ACTIONS", "").lower() == "true" and os.environ.get("LOCAL_ONLY", "").lower() != "true"
os.makedirs(NOWCAST_OUT_DIR, exist_ok=True)

EXPECTED_SHAPE = (320, 180)
NEW_LON = np.arange(97.0, 106.0, 0.05)   # 180 จุด
NEW_LAT = np.arange(21.0, 5.0, -0.05)    # 320 จุด
LON_GRID, LAT_GRID = np.meshgrid(NEW_LON, NEW_LAT)

BUCKET_NAME = "noaa-himawari9"
PRODUCT_PREFIX = "AHI-L1b-FLDK"
THAILAND_SEGMENTS = ["S0410", "S0510"]

NOWCAST_STEP_MINUTES = 10
NOWCAST_HORIZON_MINUTES = 120
FRESH_MAX_AGE_MINUTES = 25.0
DELAYED_MAX_AGE_MINUTES = 45.0
KV_MAX_SAFE_BYTES = 24 * 1024 * 1024


# --- 1. คำนวณพิกัดภาพดาวเทียม Himawari-9 จากพารามิเตอร์ใน HSD Header ---
def build_himawari_lookup_indices(
        lat_deg, lon_deg, height_m=0.0, lon_0=140.7,
        cfac=20466275.0, lfac=20466275.0, coff=2750.5,
        loff=2750.5, first_line_0based=1650.0,
    ):
        r_eq = 6378137.0
        r_pol = 6356752.3
        satellite_distance = 42164000.0
        e2 = 1.0 - (r_pol ** 2 / r_eq ** 2)

        lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
        lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
        lam_0 = np.deg2rad(float(lon_0))
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        n = r_eq / np.sqrt(1.0 - e2 * sin_lat ** 2)

        x_ecef = (n + height_m) * cos_lat * np.cos(lon - lam_0)
        y_ecef = (n + height_m) * cos_lat * np.sin(lon - lam_0)
        z_ecef = (n * (1.0 - e2) + height_m) * sin_lat

        # Coordinates in the satellite-fixed frame; vector points from ground
        # target toward the satellite.
        sat_x = satellite_distance - x_ecef
        sat_y = -y_ecef
        sat_z = -z_ecef

        with np.errstate(invalid="ignore", divide="ignore"):
            # Geostationary scan angles (radians).
            x_rad = np.arctan2(-sat_y, sat_x)
            y_rad = np.arctan2(
                -sat_z,
                np.sqrt(sat_x ** 2 + sat_y ** 2),
            )

        col_global = (float(coff) - 1.0) + np.rad2deg(x_rad) * float(cfac) / (2 ** 16)
        line_global = (float(loff) - 1.0) - np.rad2deg(y_rad) * float(lfac) / (2 ** 16)

        # A target is visible only if the outward ellipsoid normal has a positive
        # dot product with the ground-to-satellite vector.
        normal_dot_view = (
            (satellite_distance - x_ecef) * (x_ecef / (r_eq ** 2))
            + (-y_ecef) * (y_ecef / (r_eq ** 2))
            + (-z_ecef) * (z_ecef / (r_pol ** 2))
        )
        visible = (
            np.isfinite(line_global)
            & np.isfinite(col_global)
            & (normal_dot_view > 0.0)
        )
        local_line = line_global - float(first_line_0based)
        local_line = np.where(visible, local_line, np.nan)
        col_idx = np.where(visible, col_global, np.nan)
        return local_line.astype(np.float32), col_idx.astype(np.float32)


RAY_HEIGHTS_KM = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]


# --- 2. ดึงลำดับภาพย้อนหลังทุก 10 นาที และตรวจสอบ Header ของไฟล์ HSD (P0.5 & P1.2) ---
s3_client = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))


def find_himawari_sequence_10m(max_lookback_minutes=120, max_frames=7):
    now_utc = datetime.now(timezone.utc)
    base_time = now_utc.replace(minute=(now_utc.minute // 10) * 10, second=0, microsecond=0)

    found = None
    for delta_min in range(0, max_lookback_minutes + 10, 10):
        candidate = base_time - timedelta(minutes=delta_min)
        prefix = f"{PRODUCT_PREFIX}/{candidate.strftime('%Y/%m/%d/%H%M')}/"
        resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        contents = resp.get("Contents", [])
        keys_b13 = sorted(obj["Key"] for obj in contents if "_B13_FLDK_R20_" in obj["Key"])
        keys_b15 = sorted(obj["Key"] for obj in contents if "_B15_FLDK_R20_" in obj["Key"])
        has_b13_s0410 = any("S0410" in k for k in keys_b13)
        has_b13_s0510 = any("S0510" in k for k in keys_b13)
        if has_b13_s0410 and has_b13_s0510:
            has_b15 = any("S0410" in k for k in keys_b15) and any("S0510" in k for k in keys_b15)
            found = (candidate, keys_b13, keys_b15 if has_b15 else None)
            break

    if found is None:
        return [], None

    t_curr, keys_curr_b13, keys_curr_b15 = found
    frames = [(t_curr, keys_curr_b13)]
    for step_back in range(10, max_lookback_minutes + 10, 10):
        if len(frames) >= max_frames:
            break
        t_prev = t_curr - timedelta(minutes=step_back)
        prefix_prev = f"{PRODUCT_PREFIX}/{t_prev.strftime('%Y/%m/%d/%H%M')}/"
        resp_prev = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix_prev)
        keys_prev = sorted(obj["Key"] for obj in resp_prev.get("Contents", []) if "_B13_FLDK_R20_" in obj["Key"])
        if any("S0410" in k for k in keys_prev) and any("S0510" in k for k in keys_prev):
            frames.append((t_prev, keys_prev))

    frames.sort(key=lambda item: item[0])
    return frames, keys_curr_b15


def parse_hsd_segment_to_tb_celsius(raw_bytes, expected_band=None):
    """Parse a decompressed HSD file, respecting its byte-order flag and header lengths."""
    if len(raw_bytes) < 282:
        raise ValueError("HSD header สั้นเกินไป")
    if raw_bytes[0] != 1:
        raise ValueError(f"ไม่พบ HSD Basic Information Block #1 (พบ {raw_bytes[0]})")

    byte_order_flag = raw_bytes[5]
    if byte_order_flag == 0:
        endian = "<"
    elif byte_order_flag == 1:
        endian = ">"
    else:
        raise ValueError(f"Byte order flag ของ HSD ไม่ถูกต้อง: {byte_order_flag}")

    def u16(buf, start):
        return struct.unpack(endian + "H", buf[start:start + 2])[0]

    def u32(buf, start):
        return struct.unpack(endian + "I", buf[start:start + 4])[0]

    def i32(buf, start):
        return struct.unpack(endian + "i", buf[start:start + 4])[0]

    def f32(buf, start):
        return struct.unpack(endian + "f", buf[start:start + 4])[0]

    def f64(buf, start):
        return struct.unpack(endian + "d", buf[start:start + 8])[0]

    first_block_len = u16(raw_bytes, 1)
    if first_block_len < 80 or first_block_len > len(raw_bytes):
        raise ValueError(f"ความยาว HSD Block #1 ผิดปกติ: {first_block_len}")

    number_of_header_blocks = u16(raw_bytes, 3)
    if not 1 <= number_of_header_blocks <= 20:
        raise ValueError(f"จำนวน HSD header blocks ผิดปกติ: {number_of_header_blocks}")

    total_header_length = u32(raw_bytes, 70)
    total_data_length = u32(raw_bytes, 74)
    quality_flag_1 = raw_bytes[78] if len(raw_bytes) > 78 else 0

    # JMA HSD Quality Flag 1: MSB=1 means this flag is invalid.
    if quality_flag_1 & 0x80:
        raise ValueError("HSD Quality Flag 1 ระบุว่า flag ไม่ valid")

    n_cols = None
    n_lines = None
    compression_flag = 0
    band_no = None
    lam_um = None
    error_val = 65535
    outside_val = 65534
    gain = const_rad = None
    c0 = c1 = c2 = None
    proj_info = {
        "sub_lon": None,
        "cfac": None,
        "lfac": None,
        "coff": None,
        "loff": None,
        "first_line_0based": None,
        "seg_no": None,
        "total_segments": None,
        "quality_flag_1": int(quality_flag_1),
        "byte_order": "little" if endian == "<" else "big",
        "navigation_correction_block_present": False,
    }
    seen_blocks = set()
    offset = 0
    valid_bits = None

    for _ in range(number_of_header_blocks):
        if offset + 3 > len(raw_bytes):
            raise ValueError("HSD เสียหาย: header block ไม่ครบ")
        block_no = raw_bytes[offset]
        if block_no == 10:
            # HSD Block #10 stores block length as a 4-byte I4.
            block_len = u32(raw_bytes, offset + 1)
        else:
            block_len = u16(raw_bytes, offset + 1)
        if block_len < 3:
            raise ValueError(f"HSD Block #{block_no} มีความยาวไม่ถูกต้อง: {block_len}")
        if offset + block_len > len(raw_bytes):
            raise ValueError(f"HSD Block #{block_no} เกินขอบเขตไฟล์")
        seen_blocks.add(block_no)

        if block_no == 1:
            if block_len != first_block_len:
                raise ValueError("ความยาว Block #1 ไม่ตรงกัน")
        elif block_no == 2:
            if block_len < 10:
                raise ValueError("HSD Block #2 สั้นเกินไป")
            n_cols = u16(raw_bytes, offset + 5)
            n_lines = u16(raw_bytes, offset + 7)
            compression_flag = raw_bytes[offset + 9]
            if not (100 <= n_cols <= 22000 and 100 <= n_lines <= 22000):
                raise ValueError(f"ขนาด HSD ผิดปกติ: {n_cols} x {n_lines}")
            if compression_flag not in (0, 1, 2):
                raise ValueError(f"Compression flag ไม่รองรับ: {compression_flag}")
        elif block_no == 3:
            if block_len < 27:
                raise ValueError("HSD Block #3 สั้นเกินไป")
            sub_lon = f64(raw_bytes, offset + 3)
            cfac = u32(raw_bytes, offset + 11)
            lfac = u32(raw_bytes, offset + 15)
            coff = f32(raw_bytes, offset + 19)
            loff = f32(raw_bytes, offset + 23)
            if not (100.0 <= sub_lon <= 180.0 and 1e5 <= cfac <= 1e8 and 1e5 <= lfac <= 1e8):
                raise ValueError("พารามิเตอร์ Projection ใน HSD ไม่สมเหตุสมผล")
            if not (np.isfinite(coff) and np.isfinite(loff)):
                raise ValueError("COFF/LOFF ใน HSD ไม่เป็น finite number")
            proj_info.update({"sub_lon": sub_lon, "cfac": float(cfac), "lfac": float(lfac), "coff": coff, "loff": loff})
        elif block_no == 5:
            if block_len < 59:
                raise ValueError("HSD Block #5 สั้นเกินไป")
            band_no = u16(raw_bytes, offset + 3)
            lam_um = f64(raw_bytes, offset + 5)
            # Valid bits are at +13..15; count sentinels at +15 and +17.
            error_val = u16(raw_bytes, offset + 15)
            outside_val = u16(raw_bytes, offset + 17)
            gain = f64(raw_bytes, offset + 19)
            const_rad = f64(raw_bytes, offset + 27)
            c0 = f64(raw_bytes, offset + 35)
            c1 = f64(raw_bytes, offset + 43)
            c2 = f64(raw_bytes, offset + 51)

            valid_bits = u16(raw_bytes, offset + 13)
            if not 1 <= valid_bits <= 16:
                raise ValueError(f"จำนวน valid bits ผิดปกติ: {valid_bits}")

            if valid_bits is None:
                raise ValueError("HSD ไม่มีค่า valid number of bits per pixel")

        elif block_no == 7:
            if block_len < 7:
                raise ValueError("HSD Block #7 สั้นเกินไป")
            proj_info["total_segments"] = int(raw_bytes[offset + 3])
            proj_info["seg_no"] = int(raw_bytes[offset + 4])
            first_line_1based = u16(raw_bytes, offset + 5)
            if first_line_1based < 1:
                raise ValueError("First line number ใน HSD ไม่ถูกต้อง")
            proj_info["first_line_0based"] = float(first_line_1based - 1)
        elif block_no == 8:
            # Record presence for diagnostics. Block #8 correction data are not
            # applied by this function; doing so requires a separately validated
            # implementation of the JMA navigation-correction specification.
            proj_info["navigation_correction_block_present"] = block_len > 3

        offset += block_len

    if not {1, 2, 3, 5, 7}.issubset(seen_blocks):
        raise ValueError(f"HSD ขาด header blocks ที่จำเป็น: {sorted({1, 2, 3, 5, 7} - seen_blocks)}")
    if expected_band is not None and band_no != expected_band:
        raise ValueError(f"หมายเลข Band ใน HSD ({band_no}) ไม่ตรงกับที่ร้องขอ ({expected_band})")
    if any(v is None for v in (gain, const_rad, c0, c1, c2, lam_um)):
        raise ValueError("HSD ไม่มีพารามิเตอร์ Calibration ครบ")
    if lam_um <= 0 or not np.isfinite(lam_um):
        raise ValueError(f"Wavelength ผิดปกติ: {lam_um}")
    if not np.all(np.isfinite([gain, const_rad, c0, c1, c2])):
        raise ValueError("พารามิเตอร์ Calibration มีค่า NaN/Inf")

    # The official header gives the total header length. It should agree with
    # the sum of individual header block lengths.
    header_len = total_header_length or offset
    if header_len != offset:
        raise ValueError(f"Total header length ({header_len}) ไม่ตรงกับผลรวม block lengths ({offset})")
    if header_len > len(raw_bytes):
        raise ValueError("Header length เกินขนาดไฟล์ HSD")

    # Total data length in the Basic Information Block describes the stored
    # Data Block #12 bytes (i.e. before the optional inner gzip/bzip2 decode).
    stored_payload = raw_bytes[header_len:]
    if total_data_length and len(stored_payload) != total_data_length:
        raise ValueError(
            f"Total data length ไม่ตรงกับ bytes หลัง header "
            f"({total_data_length} != {len(stored_payload)})"
        )

    payload = stored_payload
    try:
        if compression_flag == 1:
            payload = gzip.decompress(payload)
        elif compression_flag == 2:
            payload = bz2.decompress(payload)
    except (OSError, EOFError) as exc:
        raise ValueError("คลาย compression ของ HSD Data Block #12 ไม่สำเร็จ") from exc

    n_pixels = int(n_lines) * int(n_cols)
    bytes_needed = n_pixels * 2
    if len(payload) != bytes_needed:
        raise ValueError(
            f"ขนาดข้อมูลพิกเซล HSD ไม่ตรงกับ shape "
            f"({len(payload)} != {bytes_needed} bytes)"
        )

    counts = np.frombuffer(payload, dtype=endian + "u2").reshape((n_lines, n_cols))
    if valid_bits is None:
        raise ValueError("HSD ไม่มีค่า valid number of bits per pixel")

    if expected_band in (13, 15) and valid_bits != 12:
        raise ValueError(
            f"Band {expected_band} ควรมี 12 valid bits "
            f"แต่พบ {valid_bits}"
        )

    max_valid_count = (1 << valid_bits) - 1

    valid_counts = (
        (counts != error_val)
        & (counts != outside_val)
        & (counts <= max_valid_count)
    )
    radiance = gain * counts.astype(np.float64) + const_rad
    valid_radiance = valid_counts & np.isfinite(radiance) & (radiance > 0.0)
    radiance = np.where(valid_radiance, radiance, np.nan)

    h = 6.62606957e-34
    c = 2.99792458e8
    k_b = 1.3806488e-23
    lam_m = lam_um * 1e-6
    planck_c1 = 2.0 * h * c**2 / (lam_m**5) * 1e-6
    planck_c2 = h * c / (k_b * lam_m)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t_eff = planck_c2 / np.log((planck_c1 / radiance) + 1.0)
        tb_k = c0 + c1 * t_eff + c2 * (t_eff**2)
    tb_c = tb_k - 273.15
    tb_c = np.where(np.isfinite(tb_c) & (tb_c > -150.0) & (tb_c < 100.0), tb_c, np.nan)
    return tb_c.astype(np.float32), proj_info


def download_full_strip(s3_keys, expected_band=13):
    parsed_segments = []
    for seg_id in THAILAND_SEGMENTS:
        matches = [k for k in s3_keys if seg_id in k]
        if len(matches) != 1:
            raise ValueError(f"คาดว่าไฟล์ {seg_id} ต้องมี 1 ไฟล์ แต่พบ {len(matches)} ไฟล์")
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=matches[0])
        compressed = obj["Body"].read()
        try:
            raw_bytes = bz2.decompress(compressed)
        except OSError as exc:
            raise ValueError(f"ไฟล์ NOAA ไม่สามารถคลาย BZip2 ได้: {matches[0]}") from exc
        tb_seg, proj_info = parse_hsd_segment_to_tb_celsius(raw_bytes, expected_band=expected_band)
        if proj_info["seg_no"] is None or proj_info["first_line_0based"] is None:
            raise ValueError(f"HSD segment metadata ไม่ครบ: {matches[0]}")
        parsed_segments.append((proj_info["first_line_0based"], tb_seg, proj_info, matches[0]))

    parsed_segments.sort(key=lambda item: item[0])
    reference_proj = parsed_segments[0][2]
    for _, seg_grid, proj_info, key in parsed_segments:
        if seg_grid.shape[1] != parsed_segments[0][1].shape[1]:
            raise ValueError("จำนวนคอลัมน์ของ HSD segments ไม่ตรงกัน")
        for name in ("sub_lon", "cfac", "lfac", "coff", "loff"):
            a = float(proj_info[name])
            b = float(reference_proj[name])
            if not np.isclose(a, b, rtol=1e-7, atol=1e-5):
                raise ValueError(f"Projection {name} ไม่ตรงกันระหว่าง segments: {key}")

    # Reject missing/overlapping rows; a silent vstack would shift geography.
    for i in range(1, len(parsed_segments)):
        prev_start, prev_grid, _, _ = parsed_segments[i - 1]
        cur_start, cur_grid, _, cur_key = parsed_segments[i]
        expected_start = prev_start + prev_grid.shape[0]
        if abs(cur_start - expected_start) > 0.5:
            raise ValueError(
                f"HSD segments ไม่ต่อเนื่อง: คาด first line {expected_start}, "
                f"แต่ {cur_key} เริ่ม {cur_start}"
            )

    strip = np.vstack([item[1] for item in parsed_segments])
    proj = dict(reference_proj)
    proj["first_line_0based"] = float(parsed_segments[0][0])
    proj["segments_used"] = [item[3] for item in parsed_segments]
    return strip, proj


def sample_strip_safely(full_strip, line_idx, col_idx):
    """Bilinear sampling that preserves off-image and missing values as NaN."""
    strip = np.asarray(full_strip, dtype=np.float32)
    line = np.asarray(line_idx, dtype=np.float32)
    col = np.asarray(col_idx, dtype=np.float32)
    if line.shape != col.shape:
        raise ValueError("line_idx และ col_idx ต้องมี shape ตรงกัน")

    h, w = strip.shape
    in_bounds = (
        np.isfinite(line) & np.isfinite(col)
        & (line >= 0.0) & (line <= h - 1)
        & (col >= 0.0) & (col <= w - 1)
    )
    # map_coordinates should never receive NaN/Inf coordinates. Move invalid
    # positions to a harmless coordinate and mask them out after sampling.
    safe_line = np.where(in_bounds, line, -1.0)
    safe_col = np.where(in_bounds, col, -1.0)
    valid_src = np.isfinite(strip)
    safe_values = np.where(valid_src, strip, 0.0)
    coords = np.asarray([safe_line, safe_col], dtype=np.float32)
    values = map_coordinates(safe_values, coords, order=1, mode="constant", cval=0.0)
    weights = map_coordinates(valid_src.astype(np.float32), coords, order=1, mode="constant", cval=0.0)
    sampled = values / np.maximum(weights, 1e-6)
    return np.where(in_bounds & (weights > 0.5) & np.isfinite(sampled), sampled, np.nan).astype(np.float32)


def fetch_thailand_grids_with_parallax(
        keys_b13, keys_b15=None, sfc_temp_grid=None, ecmwf_2t_available=False
    ):
        strip_b13, proj = download_full_strip(keys_b13, expected_band=13)

        strip_b15 = None
        if keys_b15:
            try:
                candidate_b15, proj_b15 = download_full_strip(keys_b15, expected_band=15)
                projection_fields = (
                    "sub_lon", "cfac", "lfac", "coff", "loff", "first_line_0based"
                )
                projection_matches = all(
                    np.isclose(
                        float(proj[name]), float(proj_b15[name]),
                        rtol=1e-7, atol=1e-5,
                    )
                    for name in projection_fields
                )
                if candidate_b15.shape != strip_b13.shape or not projection_matches:
                    raise ValueError("B15 shape/Projection ไม่ตรงกับ B13")
                strip_b15 = candidate_b15
            except Exception as exc:
                print(f"⚠️ ปิดใช้ B15 ในรอบนี้; จะใช้ B13 อย่างเดียว: {exc}")
                strip_b15 = None

        if (
            ecmwf_2t_available
            and sfc_temp_grid is not None
            and np.shape(sfc_temp_grid) == EXPECTED_SHAPE
        ):
            base_sfc_t = np.where(
                np.isfinite(sfc_temp_grid), sfc_temp_grid, 28.5
            ).astype(np.float32)
        else:
            base_sfc_t = np.full(EXPECTED_SHAPE, 28.5, dtype=np.float32)

        lookup_by_h = {
            h_km: build_himawari_lookup_indices(
                LAT_GRID, LON_GRID,
                height_m=h_km * 1000.0,
                lon_0=proj["sub_lon"],
                cfac=proj["cfac"],
                lfac=proj["lfac"],
                coff=proj["coff"],
                loff=proj["loff"],
                first_line_0based=proj["first_line_0based"],
            )
            for h_km in RAY_HEIGHTS_KM
        }

        line_0, col_0 = lookup_by_h[0.0]
        tb0 = sample_strip_safely(strip_b13, line_0, col_0)
        valid0 = np.isfinite(tb0)
        phys_h0 = np.clip((base_sfc_t - tb0) / 6.1, 0.0, 16.5)

        resolved = valid0 & np.isfinite(phys_h0) & (phys_h0 <= 3.2)
        tb_corr = np.where(resolved, tb0, np.nan).astype(np.float32)
        win_line = line_0.copy()
        win_col = col_0.copy()

        for h_km in RAY_HEIGHTS_KM[1:]:
            line_h, col_h = lookup_by_h[h_km]
            tb_h = sample_strip_safely(strip_b13, line_h, col_h)
            phys_h = np.clip((base_sfc_t - tb_h) / 6.1, 0.0, 16.5)
            if h_km >= 15.5:
                consistent = phys_h >= 13.0
            else:
                consistent = np.abs(phys_h - h_km) <= 2.6
            valid_h = np.isfinite(tb_h) & np.isfinite(phys_h)
            take = valid_h & consistent & (~resolved | (tb_h < tb_corr))
            tb_corr = np.where(take, tb_h, tb_corr)
            win_line = np.where(take, line_h, win_line)
            win_col = np.where(take, col_h, win_col)
            resolved |= take

        fallback_unshifted = valid0 & ~resolved
        tb_corr = np.where(fallback_unshifted, tb0, tb_corr)
        win_line = np.where(fallback_unshifted, line_0, win_line)
        win_col = np.where(fallback_unshifted, col_0, win_col)

        tb_b15_corr = (
            sample_strip_safely(strip_b15, win_line, win_col)
            if strip_b15 is not None else None
        )
        valid_ratio = float(np.mean(np.isfinite(tb_corr)))
        return tb_corr.astype(np.float32), tb_b15_corr, valid_ratio, proj


# --- 3. ตรวจสอบเวลาและขนาดกริดของ ECMWF ก่อนนำมาใช้ (P0.4) ---

def load_ecmwf_environment(target_utc_dt):
    env_factor = np.ones(EXPECTED_SHAPE, dtype=np.float32)
    sfc_temp = None
    env_info = {
        "ecmwf_env_available": False,
        "ecmwf_2t_available": False,
        "ecmwf_2t_units": None,
        "ecmwf_2t_coverage": 0.0,
        "ecmwf_mucape_coverage": 0.0,
        "ecmwf_tcwv_coverage": 0.0,
        "ecmwf_run_id": None,
        "ecmwf_matched_step": None,
        "ecmwf_time_diff_hours": None,
    }

    # ECMWF data is hosted on Cloudflare Pages
    ecmwf_base_url = os.environ.get(
        "ECMWF_BASE_URL",
        "https://doofah-forecast-data.pages.dev"
    ).rstrip("/")

    print("========== PATH DEBUG ==========")
    print(f"Working directory: {os.getcwd()}")
    print(f"NOWCAST_OUT_DIR: {os.path.abspath(NOWCAST_OUT_DIR)}")
    print(f"NOWCAST output directory exists: {os.path.isdir(NOWCAST_OUT_DIR)}")
    print(f"ECMWF source URL: {ecmwf_base_url}")
    print(f"ECMWF metadata URL: {ecmwf_base_url}/metadata.json")
    print("ECMWF mode: remote HTTP")
    print("================================")

    def fetch_json(url, missing_ok=False):
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 DooFah-Nowcast/1.0",
                "Accept": "application/json,*/*",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            body = exc.read(500).decode("utf-8", errors="replace")
            print(f"❌ ECMWF HTTP {exc.code}: {url}")
            print(f"   Response: {body[:300]}")

            if missing_ok and exc.code == 404:
                return None
            raise

    def read_grid(filename):
        url = f"{ecmwf_base_url}/{filename}"
        raw = fetch_json(url, missing_ok=True)

        if raw is None:
            print(f"⚠️ ไม่พบ ECMWF grid: {url}")
            return None, None

        units = raw.get("units") if isinstance(raw, dict) else None

        if isinstance(raw, dict):
            arr_list = raw.get("values")
            if arr_list is None:
                arr_list = raw.get("data")
        else:
            arr_list = raw

        if arr_list is None:
            return None, units

        arr = np.asarray(arr_list, dtype=np.float32)

        if arr.shape != EXPECTED_SHAPE:
            print(
                f"⚠️ รูปร่างกริดไม่ตรง {filename}: "
                f"ได้ {arr.shape}, ต้องการ {EXPECTED_SHAPE}"
            )
            return None, units

        arr[~np.isfinite(arr)] = np.nan
        return arr, units

    try:
        # Fetch metadata from Cloudflare Pages
        meta_url = f"{ecmwf_base_url}/metadata.json"
        meta = fetch_json(meta_url)

        forecasts = meta.get("forecasts", {})
        best_step = None
        min_diff_hours = float("inf")

        for step_str, valid_str in forecasts.items():
            try:
                if isinstance(valid_str, dict):
                    valid_str = (
                        valid_str.get("valid_time")
                        or valid_str.get("valid")
                    )

                valid_dt = datetime.strptime(
                    str(valid_str).strip(),
                    "%Y-%m-%d %H"
                ).replace(tzinfo=timezone.utc)

            except (TypeError, ValueError):
                continue

            diff_h = abs(
                (target_utc_dt - valid_dt).total_seconds()
            ) / 3600.0

            if diff_h < min_diff_hours:
                min_diff_hours = diff_h
                best_step = str(step_str)

        # Only accept a forecast time within 3.5 hours
        if best_step is None or min_diff_hours > 3.5:
            print(
                "⚠️ ไม่พบ ECMWF step ที่ใกล้เวลาเป้าหมายพอ "
                f"(ส่วนต่าง {min_diff_hours:.2f} ชั่วโมง)"
            )
            return env_factor, sfc_temp, env_info

        env_info["ecmwf_run_id"] = meta.get("data_as_of")

        try:
            env_info["ecmwf_matched_step"] = int(best_step)
        except ValueError:
            env_info["ecmwf_matched_step"] = best_step

        env_info["ecmwf_time_diff_hours"] = round(
            min_diff_hours, 2
        )

        print(
            f"🌦️ ECMWF matched step={best_step}, "
            f"time difference={min_diff_hours:.2f}h"
        )

        # 2-metre temperature
        t2m, t2m_units = read_grid(
            f"2t_{best_step}_grid.json"
        )

        if t2m is not None:
            coverage = float(np.mean(np.isfinite(t2m)))
            env_info["ecmwf_2t_coverage"] = round(coverage, 4)
            finite_vals = t2m[np.isfinite(t2m)]

            if finite_vals.size:
                med = float(np.nanmedian(finite_vals))
                unit_str = (
                    str(t2m_units or "")
                    .strip()
                    .lower()
                    .replace("°", "")
                )

                if unit_str in {"k", "kelvin", "degk"}:
                    t2m_c = t2m - 273.15
                    normalized_unit = "K_to_C"

                elif unit_str in {
                    "c", "degc", "celsius",
                    "degree_celsius", "degrees_celsius"
                }:
                    t2m_c = t2m.copy()
                    normalized_unit = "C"

                elif 150.0 <= med <= 350.0:
                    t2m_c = t2m - 273.15
                    normalized_unit = "K_to_C_inferred_from_range"

                elif -90.0 <= med <= 70.0:
                    t2m_c = t2m.copy()
                    normalized_unit = "C_inferred_from_range"

                else:
                    t2m_c = None
                    normalized_unit = "unrecognized"

                if t2m_c is not None and coverage >= 0.85:
                    physically_plausible = (
                        np.isfinite(t2m_c)
                        & (t2m_c >= -90.0)
                        & (t2m_c <= 65.0)
                    )

                    if float(np.mean(physically_plausible)) >= 0.80:
                        sfc_temp = np.where(
                            physically_plausible,
                            np.clip(t2m_c, -50.0, 60.0),
                            np.nan
                        ).astype(np.float32)

                        env_info["ecmwf_2t_available"] = True

                env_info["ecmwf_2t_units"] = normalized_unit

        # MUCAPE and total column water vapour
        mucape, _ = read_grid(
            f"mucape_{best_step}_grid.json"
        )
        tcwv, _ = read_grid(
            f"tcwv_{best_step}_grid.json"
        )

        mucape_cov = (
            float(np.mean(np.isfinite(mucape)))
            if mucape is not None else 0.0
        )
        tcwv_cov = (
            float(np.mean(np.isfinite(tcwv)))
            if tcwv is not None else 0.0
        )

        env_info["ecmwf_mucape_coverage"] = round(
            mucape_cov, 4
        )
        env_info["ecmwf_tcwv_coverage"] = round(
            tcwv_cov, 4
        )

        if (
            mucape is not None
            and tcwv is not None
            and mucape_cov >= 0.85
            and tcwv_cov >= 0.85
        ):
            cape_safe = np.where(
                np.isfinite(mucape),
                np.clip(mucape, 0.0, 10000.0),
                500.0
            )
            tcwv_safe = np.where(
                np.isfinite(tcwv),
                np.clip(tcwv, 0.0, 100.0),
                50.0
            )

            cape_norm = np.clip(
                cape_safe / 2000.0, 0.5, 1.3
            )
            tcwv_norm = np.clip(
                tcwv_safe / 55.0, 0.7, 1.2
            )

            env_factor = np.clip(
                0.5 * cape_norm + 0.5 * tcwv_norm,
                0.6, 1.35
            ).astype(np.float32)

            env_info["ecmwf_env_available"] = True

    except Exception as exc:
        print(f"⚠️ ไม่สามารถจับคู่/ตรวจ ECMWF ได้: {exc}")

    return env_factor, sfc_temp, env_info


# --- 4. อัลกอริทึม Optical Flow พร้อมการตรวจคุณภาพความสอดคล้องของเวกเตอร์ (P0.2) ---
def compute_pair_optical_flow(tb_prev, tb_curr, dt_minutes):
    dt = float(dt_minutes)

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Optical Flow ต้องใช้ dt > 0 นาที: {dt}")

    prev = np.asarray(tb_prev, dtype=np.float32)
    curr = np.asarray(tb_curr, dtype=np.float32)

    if prev.shape != curr.shape or prev.ndim != 2:
        raise ValueError("tb_prev และ tb_curr ต้องเป็นกริด 2 มิติขนาดเดียวกัน")

    input_valid = np.isfinite(prev) & np.isfinite(curr)

    # Fill missing values for the optical-flow calculation only.
    # Original missing pixels are masked again before returning.
    img1 = np.clip(
        (15.0 - np.nan_to_num(prev, nan=25.0)) / 95.0,
        0.0, 1.0,
    )
    img2 = np.clip(
        (15.0 - np.nan_to_num(curr, nan=25.0)) / 95.0,
        0.0, 1.0,
    )

    scale_30m = 30.0 / dt

    if HAS_CV2:
        u8_prev = np.uint8(img1 * 255.0)
        u8_curr = np.uint8(img2 * 255.0)

        flow = cv2.calcOpticalFlowFarneback(
            u8_prev,
            u8_curr,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=4,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        dx_30m = flow[..., 0] * scale_30m
        dy_30m = flow[..., 1] * scale_30m

    else:
        s1 = gaussian_filter(img1, sigma=1.8)
        s2 = gaussian_filter(img2, sigma=1.8)

        Iy, Ix = np.gradient(0.5 * (s1 + s2))
        It = s2 - s1

        Ixx = gaussian_filter(Ix * Ix, sigma=5.0) + 1e-4
        Iyy = gaussian_filter(Iy * Iy, sigma=5.0) + 1e-4
        Ixy = gaussian_filter(Ix * Iy, sigma=5.0)
        Ixt = gaussian_filter(Ix * It, sigma=5.0)
        Iyt = gaussian_filter(Iy * It, sigma=5.0)

        det = Ixx * Iyy - Ixy ** 2
        det_safe = np.maximum(det, 1e-6)

        dx_30m = -(Iyy * Ixt - Ixy * Iyt) / det_safe * scale_30m
        dy_30m = -(Ixx * Iyt - Ixy * Ixt) / det_safe * scale_30m

    flow_finite = np.isfinite(dx_30m) & np.isfinite(dy_30m)

    dx_30m = np.clip(
        np.nan_to_num(dx_30m, nan=0.0, posinf=15.0, neginf=-15.0),
        -15.0, 15.0,
    )
    dy_30m = np.clip(
        np.nan_to_num(dy_30m, nan=0.0, posinf=15.0, neginf=-15.0),
        -15.0, 15.0,
    )

    # Restore no-data pixels and reject non-finite flow estimates.
    dx_30m = np.where(input_valid & flow_finite, dx_30m, np.nan)
    dy_30m = np.where(input_valid & flow_finite, dy_30m, np.nan)

    return dy_30m.astype(np.float32), dx_30m.astype(np.float32)

def _masked_gaussian(data, sigma):
    """Gaussian smoothing that preserves missing pixels."""
    data = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(data)

    if not np.any(valid):
        return np.full(data.shape, np.nan, dtype=np.float32)

    numerator = gaussian_filter(
        np.where(valid, data, 0.0),
        sigma=sigma,
    )
    denominator = gaussian_filter(
        valid.astype(np.float32),
        sigma=sigma,
    )

    return np.where(
        denominator > 1e-3,
        numerator / np.maximum(denominator, 1e-6),
        np.nan,
    ).astype(np.float32)


def compute_multistep_optical_flow_with_quality(decoded_frames):
    if not decoded_frames:
        zeros = np.zeros(EXPECTED_SHAPE, dtype=np.float32)
        meta = {
            "frames_count": 0, "span_minutes": 0.0, "latest_gap_minutes": None,
            "max_gap_minutes": None, "directional_consistency": None,
            "active_pixel_fraction": 0.0, "active_pixel_count": 0,
            "flow_quality": "insufficient",
        }
        return zeros, zeros, meta
    decoded_frames = sorted(decoded_frames, key=lambda item: item[0])
    n = len(decoded_frames)
    base_shape = np.asarray(decoded_frames[-1][1]).shape
    meta = {
        "frames_count": n, "span_minutes": 0.0, "latest_gap_minutes": None,
        "max_gap_minutes": None, "directional_consistency": None,
        "active_pixel_fraction": 0.0, "active_pixel_count": 0,
        "flow_quality": "insufficient",
    }
    if n < 2:
        zeros = np.zeros(base_shape, dtype=np.float32)
        return zeros, zeros, meta

    dy_list, dx_list, valid_list, gaps = [], [], [], []
    for i in range(n - 1):
        t1, prev = decoded_frames[i]
        t2, curr = decoded_frames[i + 1]
        dt_min = (t2 - t1).total_seconds() / 60.0
        if dt_min <= 0:
            continue
        gaps.append(float(dt_min))
        dy, dx = compute_pair_optical_flow(prev, curr, dt_min)
        joint_valid = np.isfinite(prev) & np.isfinite(curr) & np.isfinite(dy) & np.isfinite(dx)
        dy_list.append(np.where(joint_valid, dy, np.nan).astype(np.float32))
        dx_list.append(np.where(joint_valid, dx, np.nan).astype(np.float32))
        valid_list.append(joint_valid)

    if not dy_list:
        zeros = np.zeros(base_shape, dtype=np.float32)
        meta["flow_quality"] = "low"
        return zeros, zeros, meta

    weights = np.asarray([2.0**i for i in range(len(dy_list))], dtype=np.float32)
    weights /= weights.sum()
    valid_stack = np.stack(valid_list, axis=0)
    dy_stack = np.stack(dy_list, axis=0)
    dx_stack = np.stack(dx_list, axis=0)
    weight_stack = weights[:, None, None] * valid_stack.astype(np.float32)
    denom = np.sum(weight_stack, axis=0)
    dy_combined = np.sum(np.nan_to_num(dy_stack) * weight_stack, axis=0) / np.maximum(denom, 1e-6)
    dx_combined = np.sum(np.nan_to_num(dx_stack) * weight_stack, axis=0) / np.maximum(denom, 1e-6)
    dy_combined = np.where(denom > 0, dy_combined, np.nan)
    dx_combined = np.where(denom > 0, dx_combined, np.nan)
    dy_combined = _masked_gaussian(dy_combined, sigma=1.2)
    dx_combined = _masked_gaussian(dx_combined, sigma=1.2)

    total_span = (decoded_frames[-1][0] - decoded_frames[0][0]).total_seconds() / 60.0
    latest_gap = gaps[-1] if gaps else None
    max_gap = max(gaps) if gaps else None
    finite_flow = np.isfinite(dy_combined) & np.isfinite(dx_combined)
    mag = np.hypot(dy_combined, dx_combined)
    active = finite_flow & (mag >= 0.5)  # pixels per 30 minutes; conservative activity floor
    active_count = int(np.count_nonzero(active))
    active_fraction = float(active_count / max(1, finite_flow.size))

    consistency = None
    if len(dy_list) >= 2:
        mag_a = np.hypot(dy_list[-2], dx_list[-2])
        mag_b = np.hypot(dy_list[-1], dx_list[-1])
        common = np.isfinite(mag_a) & np.isfinite(mag_b) & (mag_a >= 0.5) & (mag_b >= 0.5)
        min_consistency_pixels = max(50, int(0.001 * common.size))
        if int(np.count_nonzero(common)) >= min_consistency_pixels:
            cos_sim = (
                dy_list[-2][common] * dy_list[-1][common]
                + dx_list[-2][common] * dx_list[-1][common]
            ) / (mag_a[common] * mag_b[common] + 1e-6)
            consistency = float(np.clip(np.nanmedian(cos_sim), -1.0, 1.0))

    # Activity must be observed; no active pixels is not a perfect consistency score.
    if (
        n >= 3 and total_span >= 20.0 and latest_gap is not None and latest_gap <= 20.0
        and max_gap is not None and max_gap <= 20.0
        and consistency is not None and consistency >= 0.45
        and active_count >= max(100, int(0.002 * finite_flow.size))
    ):
        flow_quality = "high"
    elif (
        latest_gap is not None and latest_gap <= 30.0 and max_gap is not None and max_gap <= 40.0
        and active_count >= max(50, int(0.001 * finite_flow.size))
        and (consistency is None or consistency >= 0.20)
    ):
        flow_quality = "moderate"
    else:
        flow_quality = "low"

    meta.update({
        "span_minutes": round(float(total_span), 1),
        "latest_gap_minutes": round(float(latest_gap), 1) if latest_gap is not None else None,
        "max_gap_minutes": round(float(max_gap), 1) if max_gap is not None else None,
        "directional_consistency": round(consistency, 3) if consistency is not None else None,
        "active_pixel_fraction": round(active_fraction, 5),
        "active_pixel_count": active_count,
        "flow_quality": flow_quality,
    })
    return dy_combined, dx_combined, meta


def advect_field_minutes(field, dy_30m, dx_30m, lead_minutes):
        field = np.asarray(field, dtype=np.float32)
        dy = np.asarray(dy_30m, dtype=np.float32)
        dx = np.asarray(dx_30m, dtype=np.float32)
        if field.shape != dy.shape or field.shape != dx.shape:
            raise ValueError("field, dy_30m และ dx_30m ต้องมี shape ตรงกัน")
        if lead_minutes == 0:
            return field.copy()

        flow_valid = np.isfinite(dy) & np.isfinite(dx)
        dy_safe = np.where(flow_valid, dy, 0.0)
        dx_safe = np.where(flow_valid, dx, 0.0)
        h, w = field.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        scale = float(lead_minutes) / 30.0
        coords = np.asarray(
            [yy - dy_safe * scale, xx - dx_safe * scale], dtype=np.float32
        )

        source_valid = np.isfinite(field)
        values = map_coordinates(
            np.where(source_valid, field, 0.0), coords,
            order=1, mode="constant", cval=0.0,
        )
        weights = map_coordinates(
            source_valid.astype(np.float32), coords,
            order=1, mode="constant", cval=0.0,
        )
        advected = np.where(
            flow_valid & (weights > 0.5),
            values / np.maximum(weights, 1e-6),
            np.nan,
        )
        sigma = 0.25 * (float(lead_minutes) / 30.0)
        if sigma >= 0.25:
            advected = _masked_gaussian(advected, sigma=sigma)
            advected = np.where(flow_valid, advected, np.nan)
        return advected.astype(np.float32)

def make_cloud_motion_diagnostics(tb_prev, tb_curr, dy_30m, dx_30m, dt_prev_minutes):
        """Return speed (m/s), meteorological-from, bearing-to and cooling proxy."""
        dy = np.asarray(dy_30m, dtype=np.float32)
        dx = np.asarray(dx_30m, dtype=np.float32)
        valid_flow = np.isfinite(dy) & np.isfinite(dx)
        meters_per_pixel_lat = 5556.0
        meters_per_pixel_lon = 5556.0 * np.cos(np.deg2rad(LAT_GRID))
        u_ms = dx * meters_per_pixel_lon / 1800.0
        v_ms = -dy * meters_per_pixel_lat / 1800.0
        speed = np.where(valid_flow, np.hypot(u_ms, v_ms), np.nan)
        motion_mag_px = np.hypot(dy, dx)
        has_direction = valid_flow & (motion_mag_px >= 0.1)
        bearing_to = (np.degrees(np.arctan2(u_ms, v_ms)) + 360.0) % 360.0
        bearing_from = (bearing_to + 180.0) % 360.0

        bearing_to = np.where(has_direction, bearing_to, np.nan)
        bearing_from = np.where(has_direction, bearing_from, np.nan)

        h, w = tb_curr.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        dt = max(1.0, float(dt_prev_minutes))
        scale = dt / 30.0
        dy_safe = np.where(valid_flow, dy, 0.0)
        dx_safe = np.where(valid_flow, dx, 0.0)
        coords = np.asarray(
            [yy - dy_safe * scale, xx - dx_safe * scale], dtype=np.float32
        )

        prev_valid = np.isfinite(tb_prev)
        prev_values = map_coordinates(
            np.where(prev_valid, tb_prev, 0.0), coords,
            order=1, mode="constant", cval=0.0,
        )
        prev_weights = map_coordinates(
            prev_valid.astype(np.float32), coords,
            order=1, mode="constant", cval=0.0,
        )
        prev_advected = np.where(
            prev_weights > 0.5,
            prev_values / np.maximum(prev_weights, 1e-6),
            np.nan,
        )
        can_compare = (
            np.isfinite(tb_curr) & np.isfinite(prev_advected) & valid_flow
        )
        cooling_value = np.clip((prev_advected - tb_curr) * (30.0 / dt), 0.0, 40.0)
        cooling = np.where(
            can_compare,
            np.where(tb_curr < 8.0, cooling_value, 0.0),
            np.nan,
        )
        return (
            speed.astype(np.float32),
            bearing_from.astype(np.float32),
            bearing_to.astype(np.float32),
            cooling.astype(np.float32),
        )


def export_nowcast_layer(data_2d, param_name, levels, colors, precision=1, extend="max"):
    arr = np.asarray(data_2d, dtype=np.float32)
    if arr.shape != EXPECTED_SHAPE:
        raise ValueError(f"[{param_name}] grid shape {arr.shape} ไม่ตรงกับ EXPECTED_SHAPE={EXPECTED_SHAPE}")
    valid = np.isfinite(arr)
    rounded = np.where(valid, np.round(arr, precision), np.nan)
    json_grid = [[None if not np.isfinite(v) else float(v) for v in row] for row in rounded]
    grid_key = f"{param_name}_grid.json"
    grid_bundle[grid_key] = json_grid
    with open(f"{NOWCAST_OUT_DIR}/{grid_key}", "w", encoding="utf-8") as f:
        json.dump(json_grid, f, separators=(",", ":"), allow_nan=False)

    expected_colors = len(levels) - 1
    if extend in ("min", "max"):
        expected_colors += 1
    elif extend == "both":
        expected_colors += 2
    if len(colors) != expected_colors:
        raise ValueError(f"[{param_name}] จำนวนสี ({len(colors)}) ไม่ตรงกับที่ต้องการ ({expected_colors})")

    geo_key = f"{param_name}.geojson"
    if not np.any(valid):
        geojson_str = json.dumps({"type": "FeatureCollection", "features": []}, separators=(",", ":"))
    else:
        fig = plt.figure()
        try:
            ax = fig.add_subplot(111)
            masked = np.ma.masked_invalid(arr)
            contour = ax.contourf(NEW_LON, NEW_LAT, masked, levels=levels, colors=colors, extend=extend)
            geojson_str = geojsoncontour.contourf_to_geojson(
                contourf=contour, ndigits=3, stroke_width=0, fill_opacity=0.6
            )
        finally:
            plt.close(fig)
    geojson_bundle[geo_key] = json.loads(geojson_str)
    with open(f"{NOWCAST_OUT_DIR}/{geo_key}", "w", encoding="utf-8") as f:
        f.write(geojson_str)

def encode_compact_frame(field_2d):
    arr = np.asarray(field_2d, dtype=np.float32)
    valid = np.isfinite(arr)
    clipped = np.clip(np.rint(np.where(valid, arr, 0.0)), 0, 100).astype(np.uint8)
    encoded = np.where(valid, clipped, 255).astype(np.uint8)
    return encoded.tolist()


# --- 5. สร้างเหตุการณ์แจ้งเตือนโหมดทดลอง (Shadow Mode Alerts สำหรับเตรียมขึ้น Phase 1) ---
def generate_shadow_mode_alerts(
        compact_frames, cloud_speed, cloud_to_deg, obs_id,
        alerts_allowed, block_reason=None,
    ):
        candidates = []
        check_leads = [m for m in (10, 20, 30, 40, 50, 60) if str(m) in compact_frames]
        min_valid_pixels = 15       # อย่างน้อย 15/25 pixels ต่อ regional cell
        min_high_pixels = 5         # ต้องมีอย่างน้อย 5 pixels ที่ risk >= 65
        risk_threshold = 65

        if alerts_allowed and "0" in compact_frames and check_leads:
            frame_0 = np.asarray(compact_frames["0"], dtype=np.int16)
            if frame_0.shape != EXPECTED_SHAPE:
                raise ValueError("Shadow alert: frame 0 มีขนาด grid ไม่ตรงกัน")
            future_arrays = {
                m: np.asarray(compact_frames[str(m)], dtype=np.int16)
                for m in check_leads
            }
            if any(a.shape != EXPECTED_SHAPE for a in future_arrays.values()):
                raise ValueError("Shadow alert: future frame มีขนาด grid ไม่ตรงกัน")
            if np.shape(cloud_speed) != EXPECTED_SHAPE or np.shape(cloud_to_deg) != EXPECTED_SHAPE:
                raise ValueError("Shadow alert: cloud motion grid มีขนาดไม่ตรงกัน")

            for r_blk in range(0, EXPECTED_SHAPE[0], 5):
                for c_blk in range(0, EXPECTED_SHAPE[1], 5):
                    current_sub = frame_0[r_blk:r_blk + 5, c_blk:c_blk + 5]
                    current_valid = current_sub[(current_sub >= 0) & (current_sub <= 100)]
                    if current_valid.size < min_valid_pixels:
                        continue
                    current_p80 = float(np.percentile(current_valid, 80))
                    earliest_eta = None
                    peak_p80 = 0.0

                    for lead_m in check_leads:
                        sub = future_arrays[lead_m][r_blk:r_blk + 5, c_blk:c_blk + 5]
                        valid_sub = sub[(sub >= 0) & (sub <= 100)]
                        if valid_sub.size < min_valid_pixels:
                            continue
                        p80 = float(np.percentile(valid_sub, 80))
                        high_count = int(np.count_nonzero(valid_sub >= risk_threshold))
                        peak_p80 = max(peak_p80, p80)
                        rising = (current_p80 < 55.0) or (p80 >= current_p80 + 15.0)
                        if (
                            p80 >= risk_threshold
                            and high_count >= min_high_pixels
                            and rising
                            and earliest_eta is None
                        ):
                            earliest_eta = lead_m

                    if earliest_eta is None:
                        continue

                    center_r = min(EXPECTED_SHAPE[0] - 1, r_blk + 2)
                    center_c = min(EXPECTED_SHAPE[1] - 1, c_blk + 2)
                    speed_value = cloud_speed[center_r, center_c]
                    bearing_value = cloud_to_deg[center_r, center_c]
                    if not np.isfinite(speed_value) or not np.isfinite(bearing_value):
                        continue
                    speed_ms = float(speed_value)
                    if speed_ms < 2.0:
                        continue

                    lat_c = round(21.0 - center_r * 0.05, 2)
                    lon_c = round(97.0 + center_c * 0.05, 2)
                    cell_id = f"TH_{lat_c:.2f}_{lon_c:.2f}"
                    eta_min = max(10, earliest_eta - 10)
                    eta_max = min(60, earliest_eta + 10)
                    safe_obs_id = obs_id.replace(" ", "_").replace(":", "")
                    event_id = f"nowcast_{safe_obs_id}_{cell_id}"
                    candidates.append({
                        "event_id": event_id,
                        "cell_id": cell_id,
                        "center_lat": lat_c,
                        "center_lon": lon_c,
                        "eta_window_minutes": [eta_min, eta_max],
                        "current_risk_score_p80": round(current_p80, 1),
                        "peak_risk_score_p80": round(peak_p80, 1),
                        "cloud_speed_kmh": round(speed_ms * 3.6, 1),
                        "cloud_motion_bearing_to_deg": int(round(float(bearing_value))) % 360,
                        "message_th": (
                            "ตรวจพบกลุ่มเมฆที่มีสัญญาณเอื้อต่อการเกิดฝนและอาจเคลื่อนเข้าใกล้พื้นที่ของคุณ "
                            f"ในช่วงประมาณ {eta_min}–{eta_max} นาทีข้างหน้า โปรดติดตามสถานการณ์ต่อเนื่อง"
                        ),
                    })

        candidates.sort(
            key=lambda item: (-item["peak_risk_score_p80"], item["eta_window_minutes"][0])
        )
        shadow_payload = {
            "mode": "shadow_evaluation_only",
            "push_dispatched": False,
            "observed_at_utc": obs_id,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "alerts_allowed": bool(alerts_allowed),
            "block_reason": block_reason,
            "candidate_count": len(candidates),
            "candidates": candidates[:150],
        }
        with open(SHADOW_ALERTS_FILE, "w", encoding="utf-8") as f:
            json.dump(shadow_payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        return len(candidates)


# --- 6. ฟังก์ชันสร้าง GeoJSON + Grid และอัปโหลดเข้า Cloudflare KV อย่างปลอดภัย (P1.3) ---
geojson_bundle = {}
grid_bundle = {}


def upload_to_cloudflare_kv(metadata_dict, frames_payload):
    cf_account = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    cf_token = os.getenv("CLOUDFLARE_WORKER_API_TOKEN")
    cf_ns = os.getenv("NOWCAST_KV_NAMESPACE_ID")

    # 1. ตรวจสอบ Environment Variables
    missing = [
        name for name, value in [
            ("CLOUDFLARE_ACCOUNT_ID", cf_account),
            ("CLOUDFLARE_WORKER_API_TOKEN", cf_token),
            ("NOWCAST_KV_NAMESPACE_ID", cf_ns),
        ]
        if not value
    ]

    if missing:
        raise RuntimeError(
            "ขาด GitHub Actions Secrets: " + ", ".join(missing)
        )

    # 2. กำหนด Cloudflare API
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{cf_account}/storage/kv/namespaces/{cf_ns}/bulk"
    )

    headers = {
        "Authorization": f"Bearer {cf_token}",
        "Content-Type": "application/json",
    }

    # 3. เตรียม Metadata และ GeoJSON
    geojson_bundle["nowcast_metadata.json"] = metadata_dict

    kv_items = {}

    for filename, data in geojson_bundle.items():
        kv_items[f"nowcast/{filename}"] = json.dumps(
            data, separators=(",", ":"), ensure_ascii=False
        )

    # อัปโหลด Grid files แยก Key ด้วย
    for filename, data in grid_bundle.items():
        # ไม่อัปโหลด bundle เฟรมแบบเก่าทั้งก้อน
        if filename == "nowcast_frames.json":
            continue

        kv_items[f"nowcast/{filename}"] = json.dumps(
            data, separators=(",", ":"), ensure_ascii=False
        )

    # 4. แยกแต่ละเฟรมเป็น KV Key
    compact_frames = frames_payload.get("frames", {})
    print("🔎 Frame keys:", list(compact_frames.keys()))

    if not compact_frames:
        raise RuntimeError(
            "ไม่พบ frames ใน frames_payload จึงยกเลิกการอัปโหลด"
        )

    for lead_min, frame_data in compact_frames.items():
        filename = f"nowcast_frame_{lead_min}.json"
        kv_items[f"nowcast/{filename}"] = json.dumps(
            frame_data, separators=(",", ":"), ensure_ascii=False
        )

    # 5. เก็บ Metadata สำหรับ /nowcast/latest
    metadata_str = json.dumps(
        metadata_dict, separators=(",", ":"), ensure_ascii=False
    )
    kv_items["nowcast/latest.json"] = metadata_str

    # 6. ตรวจขนาดข้อมูลแต่ละ KV Value
    for key, value in kv_items.items():
        size = len(value.encode("utf-8"))

        if size > KV_MAX_SAFE_BYTES:
            raise RuntimeError(
                f"KV value ใหญ่เกินกำหนด: {key} "
                f"({size / (1024 * 1024):.2f} MiB)"
            )

    print(f"📦 จำนวน KV Keys: {len(kv_items)}")
    print(
        "📦 ขนาดข้อมูลรวม: "
        f"{sum(len(v.encode('utf-8')) for v in kv_items.values()) / (1024 * 1024):.2f} MiB"
    )

    payload = [
        {"key": key, "value": value}
        for key, value in kv_items.items()
    ]

    request_body = json.dumps(
        payload, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")

    if len(request_body) >= 100 * 1024 * 1024:
        raise RuntimeError(
            "Bulk upload request มีขนาดถึงหรือเกิน 100 MB"
        )

    # 7. อัปโหลดเข้า Cloudflare KV
    resp = requests.put(
        url,
        headers=headers,
        data=request_body,
        timeout=60,
    )

    try:
        result = resp.json()
    except ValueError:
        raise RuntimeError(
            f"Cloudflare ตอบกลับข้อมูลที่ไม่ใช่ JSON: "
            f"{resp.status_code} {resp.text[:1000]}"
        )

    if resp.ok and result.get("success"):
        errors = result.get("errors", [])

        if errors:
            raise RuntimeError(
                f"Cloudflare Bulk API รายงานข้อผิดพลาด: {errors}"
            )

        print("✅ อัปโหลด Nowcast แยก Key เข้า KV สำเร็จ!")
        return True

    raise RuntimeError(
        f"อัปโหลดเข้า Cloudflare KV ล้มเหลว: "
        f"{resp.status_code} {resp.text[:2000]}"
    )


# --- 7. เริ่มรัน Pipeline หลักของ DooFah Nowcast ---
def main():
    t_start = time.time()

    geojson_bundle.clear()
    grid_bundle.clear()

    print("🛰️ กำลังค้นหาลำดับภาพดาวเทียม Himawari-9 (Band 13 + Band 15) จาก NOAA S3...")
    raw_sequence, keys_curr_b15 = find_himawari_sequence_10m(
        max_lookback_minutes=120, max_frames=7
    )

    if not raw_sequence:
        print("❌ ไม่พบข้อมูลดาวเทียม Himawari-9 ในช่วง 120 นาทีที่ผ่านมา")
        sys.exit(1)

    t_curr, _ = raw_sequence[-1]
    obs_id = t_curr.strftime("%Y-%m-%d %H:%M UTC")
    age_minutes = round((datetime.now(timezone.utc) - t_curr).total_seconds() / 60.0, 1)

    if os.path.exists(STATE_MARKER_FILE):
        with open(STATE_MARKER_FILE, "r") as f:
            if f.read().strip() == obs_id:
                print(f"⏭️ ข้อมูลรอบ {obs_id} ถูกประมวลผลแล้ว ข้ามการทำงานซ้ำ")
                sys.exit(0)

    # 1) ตรวจสอบและจับคู่เวลาข้อมูล ECMWF ก่อนใช้แก้ Parallax และถ่วงน้ำหนัก (P0.4)
    env_weight, sfc_temp_grid, ecmwf_info = load_ecmwf_environment(t_curr)

    decoded_frames = []
    tb_b15_curr = None
    latest_valid_ratio = 0.0

    for idx, (t_frame, keys_frame) in enumerate(raw_sequence):
        is_latest = (idx == len(raw_sequence) - 1)
        diff_min = int(round((t_curr - t_frame).total_seconds() / 60.0))
        label = "ล่าสุด 0m (B13+B15)" if (is_latest and keys_curr_b15) else f"ย้อนหลัง -{diff_min} นาที (B13)"
        print(f"📥 กำลังดาวน์โหลดและประมวลผลภาพเวลา {t_frame.strftime('%H:%M UTC')} [{label}]...")

        try:
            b13_grid, b15_grid, valid_ratio, _ = fetch_thailand_grids_with_parallax(
                keys_frame,
                keys_b15=(keys_curr_b15 if is_latest else None),
                sfc_temp_grid=sfc_temp_grid,
                ecmwf_2t_available=ecmwf_info["ecmwf_2t_available"],
            )
        except Exception as exc:
            print(f"⚠️ ประมวลผลเฟรม {t_frame.strftime('%H:%M UTC')} ไม่สำเร็จ: {exc}")
            if is_latest:
                print("❌ เฟรมล่าสุด B13 ใช้งานไม่ได้ จึงหยุดรอบนี้")
                raise
            continue
    
        if valid_ratio >= 0.90:
            decoded_frames.append((t_frame, b13_grid))
            if is_latest:
                tb_b15_curr = b15_grid
                latest_valid_ratio = valid_ratio
        else:
            print(f"⚠️ ข้ามเฟรม {t_frame.strftime('%H:%M UTC')} เนื่องจากพิกเซลสมบูรณ์ต่ำกว่าเกณฑ์ ({valid_ratio*100:.1f}%)")

    if not decoded_frames:
        print("❌ ไม่มีเฟรมดาวเทียมที่ผ่านเกณฑ์คุณภาพ")
        sys.exit(1)

    if decoded_frames[-1][0] != t_curr:
        print("❌ เฟรมล่าสุดเสีย/ผ่านเกณฑ์ไม่สำเร็จ จึงไม่ใช้เฟรมเก่าปลอมเป็นข้อมูลปัจจุบัน")
        sys.exit(1)


    tb_curr = decoded_frames[-1][1]
    joint_b13_b15 = np.zeros(tb_curr.shape, dtype=bool)
    if tb_b15_curr is not None and tb_b15_curr.shape == tb_curr.shape:
        joint_b13_b15 = np.isfinite(tb_curr) & np.isfinite(tb_b15_curr)
    split_window_available = float(np.mean(joint_b13_b15)) >= 0.90
    if split_window_available:
        btd_13_15 = np.where(
            joint_b13_b15, np.clip(tb_curr - tb_b15_curr, -2.0, 10.0), np.nan
        )
        weight_btd = np.clip(1.25 - 0.30 * np.maximum(0.0, np.nan_to_num(btd_13_15, nan=1.0) - 1.0), 0.20, 1.20)
        # Pixels where B15 alone is missing get a neutral factor, never an inflated risk.
        optical_depth_weight = np.where(joint_b13_b15, weight_btd, 1.0).astype(np.float32)
    else:
        btd_13_15 = None
        optical_depth_weight = np.ones_like(tb_curr, dtype=np.float32)

        
    # --- หมวด 07A: ภาพอินฟราเรดความสว่างยอดเมฆ (sat_b13) ---
    export_nowcast_layer(
        tb_curr, "sat_b13",
        levels=[-80, -70, -60, -50, -40, -30, -20, -10],
        colors=["#ff00ff", "#8e44ad", "#c0392b", "#e74c3c", "#e67e22", "#f1c40f", "#3498db", "#74add1"],
        precision=1, extend="min"
    )

    # --- หมวด 07B: ดัชนีความปกคลุมเมฆจากอินฟราเรด (cloud_cover / ir_cloudiness_proxy, P1.1) ---
    ref_sfc_t = (
        np.where(np.isfinite(sfc_temp_grid), sfc_temp_grid, 28.5)
        if ecmwf_info["ecmwf_2t_available"] and sfc_temp_grid is not None
        else np.full_like(tb_curr, 28.5, dtype=np.float32)
    )
    temp_depression = np.clip(ref_sfc_t - tb_curr, 0.0, 90.0)
    ir_cloud_pct = np.clip((temp_depression - 6.5) / 38.0, 0.0, 1.0) * 100.0
    if split_window_available:
        thin_cirrus_boost = np.where(
            np.isfinite(btd_13_15) & (temp_depression > 4.0),
            np.clip((np.nan_to_num(btd_13_15, nan=1.2) - 1.2) / 3.5, 0.0, 1.0) * 40.0,
            0.0,
        )
        cloud_cover = np.clip(np.maximum(ir_cloud_pct, thin_cirrus_boost), 0.0, 100.0)
    else:
        cloud_cover = np.clip(ir_cloud_pct, 0.0, 100.0)

    cloud_cover = _masked_gaussian(cloud_cover, sigma=0.6)
    cloud_cover = np.where(np.isfinite(tb_curr), cloud_cover, np.nan)

    export_nowcast_layer(
        cloud_cover, "cloud_cover",
        levels=[15, 35, 55, 75, 90],
        colors=["#94a3b8", "#cbd5e1", "#e2e8f0", "#f1f5f9", "#ffffff"],
        precision=0, extend="max"
    )

    # --- หมวด 08: Cloud Tracking & Quality Gate (P0.2, P0.3, P0.6) ---
    dy_30m, dx_30m, flow_q = compute_multistep_optical_flow_with_quality(decoded_frames)
    flow_available = flow_q["flow_quality"] in ["high", "moderate"]

    # กำหนดสถานะความสดใหม่ 3 ระดับ (fresh / delayed / insufficient_data)
    if not flow_available or age_minutes > DELAYED_MAX_AGE_MINUTES:
        freshness_state = "insufficient_data" if not flow_available else "delayed"
    elif age_minutes > FRESH_MAX_AGE_MINUTES:
        freshness_state = "delayed"
    else:
        freshness_state = "fresh"

    # อนุญาตให้สร้างการแจ้งเตือนได้เฉพาะเมื่อข้อมูลสด (fresh) และคุณภาพ Flow ระดับ high เท่านั้น
    alerts_allowed = (freshness_state == "fresh") and (flow_q["flow_quality"] == "high")
    alert_block_reason = None
    if not alerts_allowed:
        if freshness_state != "fresh":
            alert_block_reason = f"data_status_{freshness_state} (age={age_minutes}m)"
        else:
            alert_block_reason = f"flow_quality_{flow_q['flow_quality']}"

    if len(decoded_frames) >= 2:
        t_prev, tb_prev = decoded_frames[-2]
        dt_prev = max(1.0, (t_curr - t_prev).total_seconds() / 60.0)
        cloud_speed, cloud_from_deg, cloud_to_deg, cooling_30m = make_cloud_motion_diagnostics(
            tb_prev, tb_curr, dy_30m, dx_30m, dt_prev
        )
    else:
        cloud_speed = np.full_like(tb_curr, np.nan, dtype=np.float32)
        cloud_from_deg = np.full_like(tb_curr, np.nan, dtype=np.float32)
        cloud_to_deg = np.full_like(tb_curr, np.nan, dtype=np.float32)
        cooling_30m = np.full_like(tb_curr, np.nan, dtype=np.float32)

    export_nowcast_layer(
        cloud_speed, "cloud_motion",
        levels=[2, 5, 10, 15, 20, 30],
        colors=["#e0f3db", "#a8ddb5", "#4eb3d3", "#2b8cbe", "#08589e", "#08306b"],
        precision=1, extend="max"
    )
    dir_from_clean = np.where(
        np.isfinite(cloud_from_deg), np.round(cloud_from_deg, 0), np.nan
    ).tolist()
    dir_from_clean = [
        [None if not np.isfinite(v) else float(v) for v in row]
        for row in dir_from_clean
    ]
    dir_to_clean = np.where(
        np.isfinite(cloud_to_deg), np.round(cloud_to_deg, 0), np.nan
    ).tolist()
    dir_to_clean = [
        [None if not np.isfinite(v) else float(v) for v in row]
        for row in dir_to_clean
    ]
    grid_bundle["cloud_motion_dir_grid.json"] = dir_from_clean
    grid_bundle["cloud_motion_to_dir_grid.json"] = dir_to_clean
    with open(f"{NOWCAST_OUT_DIR}/cloud_motion_dir_grid.json", "w") as f:
        json.dump(dir_from_clean, f, separators=(",", ":"))
    with open(f"{NOWCAST_OUT_DIR}/cloud_motion_to_dir_grid.json", "w") as f:
        json.dump(dir_to_clean, f, separators=(",", ":"))

    export_nowcast_layer(
        cooling_30m, "cloud_top_cooling",
        levels=[3, 6, 10, 15, 20],
        colors=["#fee090", "#fdae61", "#f46d43", "#d73027", "#800026"],
        precision=1, extend="max"
    )

    coldness_score = np.clip((-10.0 - tb_curr) / 58.0, 0.0, 1.0) * 50.0
    cooling_score = np.clip(np.nan_to_num(cooling_30m, nan=0.0) / 14.0, 0.0, 1.0) * 50.0
    cloud_growth = coldness_score + cooling_score
    export_nowcast_layer(
        cloud_growth, "cloud_growth",
        levels=[20, 40, 60, 80],
        colors=["#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"],
        precision=0, extend="max"
    )

    # --- หมวด 09: Convective Risk Score (ปลอดบั๊ก B15 Fallback Inflation) ---
    tb_smooth_meso = _masked_gaussian(tb_curr, sigma=2.5)
    local_prominence = np.clip(tb_smooth_meso - tb_curr, 0.0, 25.0)

    valid_tb = np.isfinite(tb_curr)
    tb_for_gradient = np.nan_to_num(tb_curr, nan=0.0)
    gy, gx = np.gradient(tb_for_gradient)

    # Zero gradients where their finite-difference neighbors are missing.
    valid_y = valid_tb.copy()
    valid_x = valid_tb.copy()
    valid_y[1:-1, :] &= valid_tb[:-2, :] & valid_tb[2:, :]
    valid_x[:, 1:-1] &= valid_tb[:, :-2] & valid_tb[:, 2:]
    valid_y[0, :] &= valid_tb[1, :]
    valid_y[-1, :] &= valid_tb[-2, :]
    valid_x[:, 0] &= valid_tb[:, 1]
    valid_x[:, -1] &= valid_tb[:, -2]
    gy = np.where(valid_y, gy, 0.0)
    gx = np.where(valid_x, gx, 0.0)

    flow_mag = np.hypot(dy_30m, dx_30m)
    valid_flow_vec = np.isfinite(flow_mag) & (flow_mag > 1e-4)
    unit_dy = np.divide(dy_30m, flow_mag, out=np.zeros_like(dy_30m), where=valid_flow_vec)
    unit_dx = np.divide(dx_30m, flow_mag, out=np.zeros_like(dx_30m), where=valid_flow_vec)
    upshear_grad = np.clip((gy * unit_dy + gx * unit_dx) / 4.5, -0.35, 0.45)
    upshear_grad = np.where(valid_flow_vec, upshear_grad, 0.0)
    core_alignment = np.clip(0.85 + upshear_grad + (local_prominence / 16.0), 0.25, 1.35)

    deep_cold_term = np.clip((-12.0 - tb_curr) / 55.0, 0.0, 1.0) * optical_depth_weight * core_alignment
    cooling_term = np.clip(np.nan_to_num(cooling_30m, nan=0.0) / 10.5, 0.0, 1.0)

    # คำนวณคะแนนฝนเมฆอุ่นเฉพาะเมื่อมีข้อมูล B15 ยืนยันความหนาแน่นของเมฆจริงเท่านั้น (P0.1)
    if split_window_available:
        tb_smooth_micro = _masked_gaussian(tb_curr, sigma=1.2)
        micro_tower_drop = np.clip(tb_smooth_micro - tb_curr, 0.0, 12.0)
        is_warm_congestus = (tb_curr < 6.0) & (tb_curr >= -18.0) & (btd_13_15 < 1.8)
        cooling_safe = np.nan_to_num(
                        cooling_30m,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
        warm_shower_score = np.where(
            is_warm_congestus,
            np.clip((micro_tower_drop - 1.5) / 4.5, 0.0, 1.0) * 35.0 + np.clip(cooling_safe / 6.5, 0.0, 1.0) * 22.0,
            0.0,
        )
    else:
        warm_shower_score = np.zeros_like(tb_curr)

    base_conv_score = np.maximum(
        (0.48 * deep_cold_term + 0.52 * cooling_term) * 100.0,
        warm_shower_score,
    )
    convective_risk = np.clip(base_conv_score * env_weight, 0.0, 100.0)

    risk_levels = [20, 40, 60, 80]
    risk_colors = ["#74add1", "#2ecc71", "#f1c40f", "#e74c3c"]
    export_nowcast_layer(convective_risk, "convective_risk", risk_levels, risk_colors, precision=0, extend="max")

    # High + fresh: keep 120 min for shadow evaluation.
    # Moderate + fresh: limit to 60 min. Delayed/low quality: observed frame only.
    if freshness_state == "fresh" and flow_q["flow_quality"] == "high":
        effective_horizon_minutes = 120
    elif freshness_state == "fresh" and flow_q["flow_quality"] == "moderate":
        effective_horizon_minutes = 60
    else:
        effective_horizon_minutes = 0

    allow_future_advection = effective_horizon_minutes > 0
    lead_minutes_list = (
        list(range(0, effective_horizon_minutes + 1, NOWCAST_STEP_MINUTES))
        if allow_future_advection else [0]
    )

    compact_frames = {}
    for lead_min in lead_minutes_list:
        adv_field = advect_field_minutes(convective_risk, dy_30m, dx_30m, lead_min)
        if lead_min > 0 and ecmwf_info["ecmwf_env_available"]:
            blend_env = min(0.20, (lead_min / 120.0) * 0.20)
            adv_field = np.clip(adv_field * ((1.0 - blend_env) + blend_env * env_weight), 0.0, 100.0)
        compact_frames[str(lead_min)] = encode_compact_frame(adv_field)

    parallax_method = (
        "heuristic_ray_march_with_ecmwf_2t"
        if ecmwf_info["ecmwf_2t_available"]
        else "heuristic_ray_march_standard_lapse"
    )

    nowcast_frames_payload = {
        "observed_at_utc": obs_id,
        "step_minutes": NOWCAST_STEP_MINUTES,
        "horizon_minutes": effective_horizon_minutes,
        "shape": list(EXPECTED_SHAPE),
        "lead_minutes": lead_minutes_list,
        "value_type": "convective_risk_score",
        "probability_calibrated": False,
        "parallax_method": parallax_method,
        "split_window_available": split_window_available,
        "flow_available": flow_available,
        "flow_quality": flow_q["flow_quality"],
        "freshness_state": freshness_state,
        "alerts_allowed": alerts_allowed,
        "frames": compact_frames,
    }

    with open(f"{NOWCAST_OUT_DIR}/nowcast_frames.json", "w") as f:
        json.dump(nowcast_frames_payload, f, separators=(",", ":"))

    # ประเมินและบันทึก Shadow Mode Alerts (สำหรับทดสอบความแม่นยำย้อนหลังก่อนเปิด Phase 1)
    shadow_count = generate_shadow_mode_alerts(
        compact_frames, cloud_speed, cloud_to_deg, obs_id, alerts_allowed, alert_block_reason
    )

    nowcast_metadata = {
        "platform": "DooFah Nowcast",
        "source": "NOAA/JMA Himawari-9 B13+B15 + 10m Multi-Step Optical Flow",
        "observed_at_utc": obs_id,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data_age_minutes": age_minutes,
        "freshness_state": freshness_state,
        "is_delayed": freshness_state == "delayed",
        "alerts_allowed": alerts_allowed,
        "alert_block_reason": alert_block_reason,
        "shadow_candidate_alerts": shadow_count,
        "flow_available": flow_available,
        "flow_quality": flow_q["flow_quality"],
        "flow_frames_used": flow_q["frames_count"],
        "flow_span_minutes": flow_q["span_minutes"],
        "flow_latest_gap_minutes": flow_q["latest_gap_minutes"],
        "flow_directional_consistency": flow_q["directional_consistency"],
        "flow_active_pixel_fraction": flow_q.get("active_pixel_fraction"),
        "flow_active_pixel_count": flow_q.get("active_pixel_count"),
        "flow_max_gap_minutes": flow_q.get("max_gap_minutes"),
        "effective_horizon_policy": "fresh_high_120m_fresh_moderate_60m_otherwise_0m",
        "split_window_available": split_window_available,
        "parallax_method": parallax_method,
        "direction_convention": {
            "cloud_motion_dir_grid": "meteorological_from_deg",
            "cloud_motion_to_dir_grid": "bearing_to_deg",
        },
        "ecmwf_environment": ecmwf_info,
        "step_minutes": NOWCAST_STEP_MINUTES,
        "horizon_minutes": effective_horizon_minutes,
        "lead_minutes": lead_minutes_list,
        "value_type": "convective_risk_score",
        "probability_calibrated": False,
        "layers": [
            "sat_b13",
            "cloud_cover",
            "cloud_motion",
            "cloud_top_cooling",
            "cloud_growth",
            "convective_risk",
        ],
    }

    with open(f"{NOWCAST_OUT_DIR}/nowcast_metadata.json", "w") as f:
        json.dump(nowcast_metadata, f, ensure_ascii=False, indent=2)

    if os.path.exists(SHADOW_ALERTS_FILE):
        with open(SHADOW_ALERTS_FILE, "r", encoding="utf-8") as sf:
            grid_bundle["shadow_alerts_latest.json"] = json.load(sf)

    uploaded_to_kv = upload_to_cloudflare_kv(nowcast_metadata, nowcast_frames_payload)

    if IS_PRODUCTION_CI and not uploaded_to_kv:
        raise RuntimeError("Production CI ไม่ได้เผยแพร่ข้อมูลขึ้น Cloudflare KV")

    # สร้างไฟล์ status.json ตามเกณฑ์ส่งมอบงาน Phase 0
    status_payload = {
        "service": "doofah-nowcast",
        "status": freshness_state,
        "observed_at_utc": obs_id,
        "published_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "published_to_cloudflare_kv": uploaded_to_kv,
        "data_age_minutes": age_minutes,
        "valid_pixel_ratio": round(latest_valid_ratio, 4),
        "flow_quality": flow_q["flow_quality"],
        "split_window_available": split_window_available,
        "ecmwf_env_available": ecmwf_info["ecmwf_env_available"],
        "alerts_allowed": alerts_allowed,
        "shadow_candidate_alerts": shadow_count,
        "last_error": None,
    }
    with open(STATUS_FILE, "w") as f:
        json.dump(status_payload, f, ensure_ascii=False, indent=2)

    with open(STATE_MARKER_FILE, "w") as f:
        f.write(obs_id)

    print(
        f"🎉 ประมวลผล DooFah Nowcast สำเร็จใน {time.time() - t_start:.2f} วินาที "
        f"[สถานะ: {freshness_state}, Flow: {flow_q['flow_quality']}, Shadow Alerts: {shadow_count} จุด]"
    )


if __name__ == "__main__":
    main()
import bz2
import json
import os
import struct
import sys
import time
from datetime import datetime, timedelta, timezone

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import geojsoncontour
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.ndimage import gaussian_filter, map_coordinates

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# --- ตั้งค่าโฟลเดอร์และกริดมาตรฐานของ "ดูฟ้า" (0.05°) ---
# NOWCAST_OUT_DIR = "public_nowcast"
NOWCAST_OUT_DIR = '../Doofah_fe/public/nowcast'
ECMWF_PUBLIC_DIR = "public"
STATE_MARKER_FILE = f"{NOWCAST_OUT_DIR}/_last_obs_time.txt"
os.makedirs(NOWCAST_OUT_DIR, exist_ok=True)

NEW_LON = np.arange(97.0, 106.0, 0.05)   # 180 จุด
NEW_LAT = np.arange(21.0, 5.0, -0.05)    # 320 จุด
LON_GRID, LAT_GRID = np.meshgrid(NEW_LON, NEW_LAT)

BUCKET_NAME = "noaa-himawari9"
PRODUCT_PREFIX = "AHI-L1b-FLDK"
THAILAND_SEGMENTS = ["S0410", "S0510"]   # แถบที่ 4 และ 5 ครอบคลุมละติจูด 5°N - 21°N

# --- 1. คำนวณพิกัดภาพดาวเทียม Himawari-9 มายังกริดประเทศไทย ---
def build_himawari_lookup_indices(lat_deg, lon_deg, lon_0=140.7):
    """
    แปลงพิกัด (lat, lon) ของไทยเป็นพิกัดแถวและคอลัมน์ (line, col)
    บนภาพ Full Disk 2km (5500 x 5500) ของ Himawari-9
    """
    r_eq = 6378137.0
    r_pol = 6356752.3
    H = 42164000.0
    e2 = 1.0 - (r_pol**2 / r_eq**2)

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lam_0 = np.deg2rad(lon_0)

    phi_c = np.arctan((r_pol**2 / r_eq**2) * np.tan(lat))
    r_c = r_pol / np.sqrt(1.0 - e2 * (np.cos(phi_c) ** 2))

    s_x = H - r_c * np.cos(phi_c) * np.cos(lon - lam_0)
    s_y = -r_c * np.cos(phi_c) * np.sin(lon - lam_0)
    s_z = r_c * np.sin(phi_c)
    s_norm = np.sqrt(s_x**2 + s_y**2 + s_z**2)

    x_rad = np.arcsin(-s_y / s_norm)
    y_rad = np.arctan(s_z / s_x)

    cfac = 20466275.0
    lfac = 20466275.0
    coff = 2750.5
    loff = 2750.5

    col_idx = coff + np.rad2deg(x_rad) * (2**-16) * cfac
    line_idx = loff - np.rad2deg(y_rad) * (2**-16) * lfac

    # เนื่องจากเราโหลดเฉพาะ Segment 4 และ 5 (เริ่มที่บรรทัด 3 * 550 = 1650)
    local_line_idx = line_idx - 1650.0
    return local_line_idx, col_idx

LOCAL_LINE_IDX, COL_IDX = build_himawari_lookup_indices(LAT_GRID, LON_GRID)

# --- 2. ดึงและแกะไฟล์ Himawari Standard Data (HSD) จาก NOAA S3 ---
s3_client = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))

def find_latest_himawari_times(max_lookback_minutes=120):
    """ค้นหาเวลาภาพดาวเทียมล่าสุดที่มีครบทั้ง S0410 และ S0510 และหาเวลา 30 นาทีก่อนหน้า"""
    now_utc = datetime.now(timezone.utc)
    # ปัดเศษลงให้ตรงหลัก 10 นาที
    base_time = now_utc.replace(minute=(now_utc.minute // 10) * 10, second=0, microsecond=0)

    found_latest = None
    for delta_min in range(10, max_lookback_minutes + 10, 10):
        candidate = base_time - timedelta(minutes=delta_min)
        prefix = f"{PRODUCT_PREFIX}/{candidate.strftime('%Y/%m/%d/%H%M')}/"
        resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        contents = resp.get("Contents", [])
        keys = [obj["Key"] for obj in contents if "_B13_FLDK_R20_" in obj["Key"]]
        if any("S0410" in k for k in keys) and any("S0510" in k for k in keys):
            found_latest = (candidate, keys)
            break

    if not found_latest:
        return None, None

    t_curr, keys_curr = found_latest
    # ค้นหาภาพก่อนหน้า (เป้าหมายคือ 30 นาทีก่อนหน้า หรือ 20/40 นาทีหากบางสล็อตขาด)
    for step_back in [30, 20, 40]:
        t_prev = t_curr - timedelta(minutes=step_back)
        prefix_prev = f"{PRODUCT_PREFIX}/{t_prev.strftime('%Y/%m/%d/%H%M')}/"
        resp_prev = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix_prev)
        keys_prev = [obj["Key"] for obj in resp_prev.get("Contents", []) if "_B13_FLDK_R20_" in obj["Key"]]
        if any("S0410" in k for k in keys_prev) and any("S0510" in k for k in keys_prev):
            return (t_curr, keys_curr), (t_prev, keys_prev)

    return (t_curr, keys_curr), None

def parse_hsd_segment_to_tb_celsius(raw_bytes):
    """แกะไบนารี JMA HSD และแปลงค่า Count เป็น Brightness Temperature (°C)"""
    offset = 0
    gain, const_rad = None, None
    c0, c1, c2 = 0.0, 1.0, 0.0
    lam_um = 10.4073

    # อ่าน Header 11 บล็อก
    for _ in range(11):
        block_no = raw_bytes[offset]
        block_len = struct.unpack("<H", raw_bytes[offset + 1 : offset + 3])[0]
        if block_no == 5:
            lam_um = struct.unpack("<d", raw_bytes[offset + 5 : offset + 13])[0]
            gain = struct.unpack("<d", raw_bytes[offset + 19 : offset + 27])[0]
            const_rad = struct.unpack("<d", raw_bytes[offset + 27 : offset + 35])[0]
            c0 = struct.unpack("<d", raw_bytes[offset + 35 : offset + 43])[0]
            c1 = struct.unpack("<d", raw_bytes[offset + 43 : offset + 51])[0]
            c2 = struct.unpack("<d", raw_bytes[offset + 51 : offset + 59])[0]
        offset += block_len

    # ข้อมูลภาพขนาด 550 บรรทัด x 5500 คอลัมน์ (uint16) อยู่ต่อจาก Header
    n_pixels = 550 * 5500
    counts = np.frombuffer(raw_bytes[offset : offset + n_pixels * 2], dtype="<u2").reshape((550, 5500))

    # กรองค่า Error / นอกพื้นที่สแกน (>= 65534)
    valid_mask = counts < 65534
    radiance = np.where(valid_mask, gain * counts + const_rad, np.nan)
    radiance = np.clip(radiance, 1e-5, None)

    # สมการ Planck สำหรับแปลง Radiance -> Brightness Temperature (K -> °C)
    h = 6.62606957e-34
    c = 2.99792458e8
    k_b = 1.3806488e-23
    lam_m = lam_um * 1e-6

    c1_p = 2.0 * h * (c**2) / (lam_m**5) * 1e-6
    c2_p = (h * c) / (k_b * lam_m)

    t_eff = c2_p / np.log((c1_p / radiance) + 1.0)
    tb_k = c0 + c1 * t_eff + c2 * (t_eff**2)
    return tb_k - 273.15

def fetch_thailand_b13_grid(s3_keys):
    """ดาวน์โหลด S0410 และ S0510 มาต่อกัน แล้วดึงเฉพาะพิกัดประเทศไทย (320 x 180)"""
    segments_data = []
    for seg_id in THAILAND_SEGMENTS:
        seg_key = next(k for k in s3_keys if seg_id in k)
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=seg_key)
        compressed = obj["Body"].read()
        raw_bytes = bz2.decompress(compressed)
        segments_data.append(parse_hsd_segment_to_tb_celsius(raw_bytes))

    # ต่อแถบที่ 4 และ 5 เข้าด้วยกัน ได้ขนาด (1100, 5500)
    full_strip = np.vstack(segments_data)

    # Interpolate มายังกริด 0.05° ของประเทศไทย
    coords = np.array([LOCAL_LINE_IDX, COL_IDX])
    th_tb = map_coordinates(full_strip, coords, order=1, mode="nearest")
    return th_tb

# --- 3. อัลกอริทึม Optical Flow, Cloud Tracking และ Rain Nowcast ---
def compute_optical_flow(tb_prev, tb_curr, dt_minutes):
    """
    คำนวณเวกเตอร์การเคลื่อนที่ของกลุ่มเมฆ (dy, dx ในหน่วยพิกเซลต่อ 30 นาที)
    โดยโฟกัสที่กลุ่มเมฆที่ยอดเมฆเย็นกว่า +5°C
    """
    scale_30m = 30.0 / max(dt_minutes, 1.0)
    img1 = np.clip((5.0 - np.nan_to_num(tb_prev, nan=15.0)) / 85.0, 0.0, 1.0)
    img2 = np.clip((5.0 - np.nan_to_num(tb_curr, nan=15.0)) / 85.0, 0.0, 1.0)

    if HAS_CV2:
        u8_prev = np.uint8(img1 * 255.0)
        u8_curr = np.uint8(img2 * 255.0)
        flow = cv2.calcOpticalFlowFarneback(
            u8_prev, u8_curr, None,
            pyr_scale=0.5, levels=3, winsize=25,
            iterations=3, poly_n=7, poly_sigma=1.5, flags=0
        )
        dx_30m = flow[..., 0] * scale_30m
        dy_30m = flow[..., 1] * scale_30m
    else:
        s1 = gaussian_filter(img1, sigma=2.0)
        s2 = gaussian_filter(img2, sigma=2.0)
        Iy, Ix = np.gradient(0.5 * (s1 + s2))
        It = s2 - s1
        Ixx = gaussian_filter(Ix * Ix, sigma=6.0) + 1e-4
        Iyy = gaussian_filter(Iy * Iy, sigma=6.0) + 1e-4
        Ixy = gaussian_filter(Ix * Iy, sigma=6.0)
        Ixt = gaussian_filter(Ix * It, sigma=6.0)
        Iyt = gaussian_filter(Iy * It, sigma=6.0)
        det = Ixx * Iyy - Ixy**2
        dx_30m = -(Iyy * Ixt - Ixy * Iyt) / det * scale_30m
        dy_30m = -(Ixx * Iyt - Ixy * Ixt) / det * scale_30m

    # จำกัดความเร็วสูงสุดไม่ให้เกิดเวกเตอร์กระโดดผิดธรรมชาติ (~45 m/s)
    dx_30m = np.clip(dx_30m, -15.0, 15.0)
    dy_30m = np.clip(dy_30m, -15.0, 15.0)
    return dy_30m, dx_30m

def advect_field(field, dy_step, dx_step, n_steps_30m):
    """พัดพาสนามข้อมูลไปข้างหน้าตามเวกเตอร์การไหล (Semi-Lagrangian Extrapolation)"""
    H, W = field.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    current = field.copy()
    for _ in range(n_steps_30m):
        coords = np.array([yy - dy_step, xx - dx_step])
        current = map_coordinates(current, coords, order=1, mode="nearest")
    return current

def load_ecmwf_environment_weight():
    """
    ดึงบริบทความไม่เสถียรของบรรยากาศ (MUCAPE และ TCWV) จากไฟล์ ECMWF ล่าสุดถ้ามี
    เพื่อใช้ถ่วงน้ำหนักโอกาสพัฒนาตัวของพายุในระยะ 1-6 ชั่วโมง
    """
    env_factor = np.ones((len(NEW_LAT), len(NEW_LON)))
    try:
        mucape_path = os.path.join(ECMWF_PUBLIC_DIR, "mucape_3_grid.json")
        tcwv_path = os.path.join(ECMWF_PUBLIC_DIR, "tcwv_3_grid.json")
        if os.path.exists(mucape_path) and os.path.exists(tcwv_path):
            with open(mucape_path, "r") as f:
                mucape = np.array(json.load(f), dtype=float)
            with open(tcwv_path, "r") as f:
                tcwv = np.array(json.load(f), dtype=float)
            cape_norm = np.clip( np.nan_to_num(mucape, nan=500.0) / 2000.0, 0.5, 1.3 )
            tcwv_norm = np.clip( np.nan_to_num(tcwv, nan=50.0) / 55.0, 0.7, 1.2 )
            env_factor = np.clip(0.5 * cape_norm + 0.5 * tcwv_norm, 0.6, 1.35)
    except Exception:
        pass
    return env_factor

# --- 4. ฟังก์ชันสร้าง GeoJSON + Grid และจัดเก็บใน Bundle ---
geojson_bundle = {}
grid_bundle = {}

def export_nowcast_layer(data_2d, param_name, levels, colors, precision=1, extend="max"):
    rounded = np.round(data_2d, precision)
    clean_list = np.where(np.isnan(rounded), None, rounded).tolist()
    grid_key = f"{param_name}_grid.json"
    grid_bundle[grid_key] = clean_list

    with open(f"{NOWCAST_OUT_DIR}/{grid_key}", "w") as f:
        json.dump(clean_list, f, separators=(",", ":"))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    expected_colors = len(levels) - 1
    if extend in ["min", "max"]:
        expected_colors += 1
    elif extend == "both":
        expected_colors += 2
    if len(colors) != expected_colors:
        plt.close(fig)
        raise ValueError(f"[{param_name}] จำนวนสี ({len(colors)}) ไม่ตรงกับที่ต้องการ ({expected_colors})")

    contour = ax.contourf(NEW_LON, NEW_LAT, data_2d, levels=levels, colors=colors, extend=extend)
    geojson_str = geojsoncontour.contourf_to_geojson(
        contourf=contour, ndigits=3, stroke_width=0, fill_opacity=0.6
    )
    plt.close(fig)

    geo_key = f"{param_name}.geojson"
    geojson_bundle[geo_key] = json.loads(geojson_str)
    with open(f"{NOWCAST_OUT_DIR}/{geo_key}", "w") as f:
        f.write(geojson_str)

def upload_to_cloudflare_kv(metadata_dict):
    """อัปโหลดเข้า Cloudflare KV โดยใช้เพียง 2 Bundle Keys เพื่อประหยัดโควตา Writes รายวัน"""
    cf_account = os.getenv("CF_ACCOUNT_ID")
    cf_token = os.getenv("CF_API_TOKEN")
    cf_ns = os.getenv("CF_KV_NAMESPACE_ID")

    if not (cf_account and cf_token and cf_ns):
        print("ℹ️ ไม่ได้ตั้งค่า CF_ACCOUNT_ID / CF_API_TOKEN / CF_KV_NAMESPACE_ID -> บันทึกเฉพาะในเครื่อง")
        return

    geojson_bundle["nowcast_metadata.json"] = metadata_dict
    url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/storage/kv/namespaces/{cf_ns}/bulk"
    headers = {
        "Authorization": f"Bearer {cf_token}",
        "Content-Type": "application/json",
    }
    payload = [
        {"key": "nowcast_geojson_bundle", "value": json.dumps(geojson_bundle, separators=(",", ":"))},
        {"key": "nowcast_grid_bundle", "value": json.dumps(grid_bundle, separators=(",", ":"))},
    ]
    resp = requests.put(url, headers=headers, json=payload, timeout=60)
    if resp.status_code == 200 and resp.json().get("success"):
        print("✅ อัปโหลด Nowcast Bundles เข้า Cloudflare Workers KV สำเร็จ! (ใช้โควตาเพียง 2 Writes)")
    else:
        print(f"❌ อัปโหลดเข้า Cloudflare KV ล้มเหลว: {resp.status_code} {resp.text}")
        sys.exit(1)

# --- 5. เริ่มรัน Pipeline หลักของ DooFah Nowcast ---
def main():
    t_start = time.time()
    print("🛰️ กำลังค้นหาภาพดาวเทียม Himawari-9 Band 13 ล่าสุดจาก NOAA S3...")
    latest_pair, prev_pair = find_latest_himawari_times()

    if not latest_pair:
        print("❌ ไม่พบข้อมูลดาวเทียม Himawari-9 ในช่วง 120 นาทีที่ผ่านมา")
        sys.exit(1)

    t_curr, keys_curr = latest_pair
    obs_id = t_curr.strftime("%Y-%m-%d %H:%M UTC")
    age_minutes = round((datetime.now(timezone.utc) - t_curr).total_seconds() / 60.0, 1)

    # ตรวจสอบว่าภาพเวลานี้เคยประมวลผลไปแล้วหรือยัง
    if os.path.exists(STATE_MARKER_FILE):
        with open(STATE_MARKER_FILE, "r") as f:
            if f.read().strip() == obs_id:
                print(f"⏭️ ข้อมูลรอบ {obs_id} ถูกประมวลผลแล้ว ข้ามการอัปโหลดซ้ำ")
                sys.exit(0)

    print(f"📥 กำลังดาวน์โหลดและถอดรหัสภาพเวลา {obs_id} (อายุข้อมูล: {age_minutes} นาที)...")
    tb_curr = fetch_thailand_b13_grid(keys_curr)

    if prev_pair:
        t_prev, keys_prev = prev_pair
        dt_min = (t_curr - t_prev).total_seconds() / 60.0
        print(f"📥 กำลังดาวน์โหลดภาพเปรียบเทียบย้อนหลัง {int(dt_min)} นาที ({t_prev.strftime('%H:%M UTC')})...")
        tb_prev = fetch_thailand_b13_grid(keys_prev)
    else:
        dt_min = 30.0
        tb_prev = tb_curr.copy()

    # --- หมวด 07: Observations (sat_b13) ---
    # แสดงอุณหภูมิความสว่างยอดเมฆ (°C) เฉพาะบริเวณที่มีเมฆเย็นกว่า -10°C
    export_nowcast_layer(
        tb_curr, "sat_b13",
        levels=[-80, -70, -60, -50, -40, -30, -20, -10],
        colors=["#ff00ff", "#8e44ad", "#c0392b", "#e74c3c", "#e67e22", "#f1c40f", "#3498db", "#74add1", "#ffffff00"],
        precision=1, extend="both"
    )

    # --- หมวด 08: Cloud Tracking (cloud_motion, cloud_top_cooling, cloud_growth) ---
    dy_30m, dx_30m = compute_optical_flow(tb_prev, tb_curr, dt_min)
    # แปลงพิกเซล/30นาที เป็น m/s (1 พิกเซล = 0.05° ≈ 5550 เมตร, 30 นาที = 1800 วินาที)
    u_cloud = (dx_30m * 5550.0) / 1800.0
    v_cloud = (-dy_30m * 5550.0) / 1800.0  # แกนละติจูดเรียงจากเหนือลงใต้ จึงกลับเครื่องหมาย
    cloud_speed = np.sqrt(u_cloud**2 + v_cloud**2)
    cloud_dir = (270.0 - np.rad2deg(np.arctan2(v_cloud, u_cloud))) % 360.0

    export_nowcast_layer(
        cloud_speed, "cloud_motion",
        levels=[2, 5, 10, 15, 20, 30],
        colors=["#e0f3db", "#a8ddb5", "#4eb3d3", "#2b8cbe", "#08589e", "#08306b"],
        precision=1, extend="max"
    )
    dir_clean = np.where(np.isnan(cloud_dir), None, np.round(cloud_dir, 0)).tolist()
    grid_bundle["cloud_motion_dir_grid.json"] = dir_clean
    with open(f"{NOWCAST_OUT_DIR}/cloud_motion_dir_grid.json", "w") as f:
        json.dump(dir_clean, f, separators=(",", ":"))

    # ชดเชยตำแหน่งเมฆก่อนคำนวณอัตราการเย็นตัวของยอดเมฆ (Lagrangian Cooling Rate °C / 30 นาที)
    H, W = tb_curr.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    tb_prev_advected = map_coordinates(tb_prev, np.array([yy - dy_30m, xx - dx_30m]), order=1, mode="nearest")

    # ค่าบวก = ยอดเมฆเย็นลง (เมฆกำลังก่อตัวสูงขึ้นในแนวดิ่ง) / กรองเฉพาะเมฆที่เย็นกว่า -5°C
    cooling_30m = np.where(tb_curr < -5.0, np.clip(tb_prev_advected - tb_curr, 0.0, 40.0), 0.0)
    export_nowcast_layer(
        cooling_30m, "cloud_top_cooling",
        levels=[3, 6, 10, 15, 20],
        colors=["#fee090", "#fdae61", "#f46d43", "#d73027", "#800026"],
        precision=1, extend="max"
    )

    # ดัชนีการเติบโตของเมฆ (Cloud Growth Score 0-100)
    coldness_score = np.clip((-15.0 - tb_curr) / 55.0, 0.0, 1.0) * 50.0
    cooling_score = np.clip(cooling_30m / 15.0, 0.0, 1.0) * 50.0
    cloud_growth = coldness_score + cooling_score
    export_nowcast_layer(
        cloud_growth, "cloud_growth",
        levels=[20, 40, 60, 80],
        colors=["#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"],
        precision=0, extend="max"
    )

    # --- หมวด 09: Rain Nowcast (convective_risk & rain_probability 0-1h, 1-3h, 3-6h) ---
    env_weight = load_ecmwf_environment_weight()
    # ความเสี่ยงการเกิดฝนฟ้าคะนอง ณ ปัจจุบัน (0-100%)
    base_conv_prob = (
        0.55 * np.clip((-20.0 - tb_curr) / 50.0, 0.0, 1.0)
        + 0.45 * np.clip(cooling_30m / 12.0, 0.0, 1.0)
    ) * 100.0
    convective_risk = np.clip(base_conv_prob * env_weight, 0.0, 100.0)

    prob_levels = [20, 40, 60, 80]
    prob_colors = ["#74add1", "#2ecc71", "#f1c40f", "#e74c3c"]

    export_nowcast_layer(convective_risk, "convective_risk", prob_levels, prob_colors, precision=0, extend="max")

    # พยากรณ์ความน่าจะเป็นฝนระยะ 0-1 ชม. (เฉลี่ย step +30m และ +60m)
    p_30m = advect_field(convective_risk, dy_30m, dx_30m, 1)
    p_60m = advect_field(convective_risk, dy_30m, dx_30m, 2)
    prob_0_1h = np.clip(gaussian_filter(np.maximum(p_30m, p_60m), sigma=1.0) * 0.95, 0.0, 100.0)
    export_nowcast_layer(prob_0_1h, "rain_prob_0_1h", prob_levels, prob_colors, precision=0, extend="max")

    # พยากรณ์ความน่าจะเป็นฝนระยะ 1-3 ชม. (step +2h ถึง +3h พร้อมเกลี่ยความไม่แน่นอนเชิงพื้นที่)
    p_2h = advect_field(convective_risk, dy_30m, dx_30m, 4)
    p_3h = advect_field(convective_risk, dy_30m, dx_30m, 6)
    prob_1_3h = np.clip(gaussian_filter(0.5 * (p_2h + p_3h) * env_weight, sigma=2.5) * 0.85, 0.0, 100.0)
    export_nowcast_layer(prob_1_3h, "rain_prob_1_3h", prob_levels, prob_colors, precision=0, extend="max")

    # พยากรณ์ความน่าจะเป็นฝนระยะ 3-6 ชม. (เน้นพื้นที่เสี่ยงและสภาพแวดล้อม ECMWF)
    p_5h = advect_field(convective_risk, dy_30m, dx_30m, 10)
    prob_3_6h = np.clip(gaussian_filter(p_5h * env_weight, sigma=4.5) * 0.70, 0.0, 100.0)
    export_nowcast_layer(prob_3_6h, "rain_prob_3_6h", prob_levels, prob_colors, precision=0, extend="max")

    # สร้าง Metadata ของ Nowcast พร้อมสถานะความสดใหม่ (Freshness Gate)
    nowcast_metadata = {
        "platform": "DooFah Nowcast",
        "source": "NOAA/JMA Himawari-9 Band 13 (10.4 um) + Optical Flow",
        "observed_at_utc": obs_id,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data_age_minutes": age_minutes,
        "is_delayed": age_minutes > 35.0,
        "layers": [
            "sat_b13",
            "cloud_motion",
            "cloud_top_cooling",
            "cloud_growth",
            "convective_risk",
            "rain_prob_0_1h",
            "rain_prob_1_3h",
            "rain_prob_3_6h",
        ],
    }

    with open(f"{NOWCAST_OUT_DIR}/nowcast_metadata.json", "w") as f:
        json.dump(nowcast_metadata, f)

    upload_to_cloudflare_kv(nowcast_metadata)

    with open(STATE_MARKER_FILE, "w") as f:
        f.write(obs_id)

    print(f"🎉 ประมวลผล DooFah Nowcast เสร็จสมบูรณ์ใน {time.time() - t_start:.2f} วินาที!")

if __name__ == "__main__":
    main()
import pickle
from ecmwf.opendata import Client
import time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geojsoncontour
import json
import os
import shutil
import sys
import cfgrib
from datetime import datetime, timezone
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#### Local Development Environment
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "../Doofah_fe/public")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#### Production Environment
OUT_DIR = 'temp_dist'
# FINAL_DIR = 'public'
FINAL_DIR = '../Doofah_fe/public'
os.makedirs(FINAL_DIR, exist_ok=True)  # ใช้ตัวแปรเดียวกันกับ OUTPUT_DIR เพื่อให้สอดคล้องกับการตั้งค่าใน Dockerfile

# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(FINAL_DIR, exist_ok=True)

STATE_FILE = f"{OUT_DIR}/_pipeline_state.pkl"
RUN_MARKER_FILE = f"{OUT_DIR}/_current_run.txt"

def save_pipeline_state(state_dict):
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state_dict, f)

def load_pipeline_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "rb") as f:
            return pickle.load(f)
    return None

# Reuse Client (azure -> google -> aws)
SOURCES_ORDER = [s.strip() for s in os.getenv("ECMWF_SOURCES", "azure,google,aws").split(",") if s.strip()]
clients = {src: Client(source=src) for src in SOURCES_ORDER}
active_source_log = set()

def retrieve_with_fallback(allow_exit=True, **kwargs):
    for src in SOURCES_ORDER:
        try:
            clients[src].retrieve(**kwargs)
            active_source_log.add(src)
            return True
        except Exception as e:
            print(f"      [!] โหลดจาก {src} ไม่สำเร็จ กำลังสลับแหล่งข้อมูล... ({type(e).__name__}: {e})")
            if os.path.exists(kwargs.get("target", "")):
                try:
                    os.remove(kwargs["target"])
                except OSError:
                    pass
            
    if allow_exit:
        print(f"❌ ข้อผิดพลาดร้ายแรง: ไม่สามารถโหลดข้อมูล Step {kwargs.get('step')} จากทุกเซิร์ฟเวอร์ได้!")
        sys.exit(1)
    return False

# Quality Gate
def validate_grid_data(da, param_name, step):
    arr = da.values
    finite_ratio = np.isfinite(arr).sum() / arr.size
    if finite_ratio < 0.8:
        raise ValueError(f"[Quality Gate Failed] ตัวแปร {param_name} ที่ Step {step} มีข้อมูลสมบูรณ์เพียง {finite_ratio*100:.1f}%")

def export_geojson(data_smooth, param_name, step, levels, colors, precision=1, extend='max'):
    validate_grid_data(data_smooth, param_name, step)
    
    rounded_values = np.round(data_smooth.values, precision)
    values_clean = np.where(np.isnan(rounded_values), None, rounded_values)
    
    with open(f'{OUT_DIR}/{param_name}_{step}_grid.json', 'w') as f:
        json.dump(values_clean.tolist(), f, separators=(',', ':'))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    expected_colors = len(levels) - 1
    if extend in ['min', 'max']:
        expected_colors += 1
    elif extend == 'both':
        expected_colors += 2
        
    if len(colors) != expected_colors:
        plt.close(fig)
        raise ValueError(f"[{param_name}] จำนวนสี ({len(colors)}) ไม่ตรงกับที่ matplotlib ต้องการสำหรับ extend='{extend}' ({expected_colors})")
            
    contour = ax.contourf(
        data_smooth.longitude,
        data_smooth.latitude,
        data_smooth.values,
        levels=levels,
        colors=colors,
        extend=extend
    )
    geojson_str = geojsoncontour.contourf_to_geojson(
        contourf=contour,
        ndigits=3,
        stroke_width=0,
        fill_opacity=0.6
    )
    
    with open(f'{OUT_DIR}/{param_name}_{step}.geojson', 'w') as f:
        f.write(geojson_str)
    plt.close(fig)

def export_wind_extras(u, v, param_name, step):
    wdir = (270 - np.rad2deg(np.arctan2(v.values, u.values))) % 360
    wdir_rounded = np.round(wdir, 0)
    with open(f'{OUT_DIR}/{param_name}_dir_{step}_grid.json', 'w') as f:
        json.dump(np.where(np.isnan(wdir_rounded), None, wdir_rounded).tolist(), f, separators=(',', ':'))

# --- สมการทางอุตุนิยมวิทยา ---
def calc_heat_index_xr(t2m_c, d2m_c):
    e = 6.11 * 10.0 ** (7.5 * d2m_c / (237.3 + d2m_c))
    es = 6.11 * 10.0 ** (7.5 * t2m_c / (237.3 + t2m_c))
    rh = (e / es) * 100.0
    rh = xr.where(rh > 100, 100, xr.where(rh < 0, 0, rh))

    t_f = (t2m_c * 9/5) + 32
    # สมการ Steadman และค่าเฉลี่ยกับอุณหภูมิตามเกณฑ์มาตรฐาน NWS WPC
    hi_steadman = 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094)
    hi_simple = 0.5 * (t_f + hi_steadman)

    hi_full = (-42.379 + 2.04901523*t_f + 10.14333127*rh 
               - 0.22475541*t_f*rh - 0.00683783*t_f**2 
               - 0.05481717*rh**2 + 0.00122874*t_f**2*rh 
               + 0.00085282*t_f*rh**2 - 0.00000199*t_f**2*rh**2)
               
    dry_term = ((17 - np.abs(t_f - 95)) / 17).clip(min=0)
    adj_dry = xr.where((rh < 13) & (t_f >= 80) & (t_f <= 112), -((13 - rh) / 4) * np.sqrt(dry_term), 0)
    adj_humid = xr.where((rh > 85) & (t_f >= 80) & (t_f <= 87), ((rh - 85) / 10) * ((87 - t_f) / 5), 0)
                         
    hi_f = xr.where(hi_simple >= 80, hi_full + adj_dry + adj_humid, hi_simple)
    hi_c = (hi_f - 32) * 5/9
    return xr.where(t2m_c < 27, t2m_c, hi_c)

def calc_vpd_xr(t2m_c, d2m_c):
    es = 6.11 * 10.0 ** (7.5 * t2m_c / (237.3 + t2m_c))
    ea = 6.11 * 10.0 ** (7.5 * d2m_c / (237.3 + d2m_c))
    vpd_kpa = (es - ea) / 10.0
    return xr.where(vpd_kpa < 0, 0, vpd_kpa)

def calc_mfc_native(q_kg_kg, u, v):
    """
    คำนวณ Moisture Flux Convergence (MFC ระดับ 850 hPa) บนกริดต้นฉบับ (0.25°)
    ก่อนนำไปทำ Interpolation เพื่อไม่ให้ขยายความคลาดเคลื่อนเชิงพื้นที่
    """
    q_g_kg = q_kg_kg * 1000.0
    flux_u = q_g_kg * u
    flux_v = q_g_kg * v
    
    lats = q_kg_kg.latitude.values
    lons = q_kg_kg.longitude.values
    
    # คำนวณระยะทางจริง (เมตร) ตามพิกัดภูมิศาสตร์ของกริดต้นฉบับ
    dy_m = np.gradient(lats)[:, np.newaxis] * 111132.954
    dx_m = np.gradient(lons)[np.newaxis, :] * 111319.491 * np.cos(np.deg2rad(lats))[:, np.newaxis]
    
    div_u = np.gradient(flux_u.values, axis=1) / dx_m
    div_v = np.gradient(flux_v.values, axis=0) / dy_m
    
    mfc = -(div_u + div_v) * 86400.0
    return xr.DataArray(mfc, coords=q_kg_kg.coords, dims=q_kg_kg.dims)

def calc_et0_3h(t_c, td_c, sp_pa, ws_10m, rn_mj_m2_3h):
    u2 = ws_10m * 0.748
    p_kpa = sp_pa / 1000.0
    gamma = 0.000665 * p_kpa
    es = 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))
    ea = 0.6108 * np.exp((17.27 * td_c) / (td_c + 237.3))
    delta = (4098.0 * es) / ((t_c + 237.3)**2)
    rn_h = rn_mj_m2_3h / 3.0
    g_h = xr.where(rn_h > 0, 0.1 * rn_h, 0.5 * rn_h)
    et0_h = (0.408 * delta * (rn_h - g_h) + gamma * (37.0 / (t_c + 273.16)) * u2 * (es - ea)) / (delta + gamma * (1.0 + 0.34 * u2))
    et0_3h = et0_h * 3.0
    return xr.where(et0_3h < 0, 0, et0_3h)

# --- เตรียมข้อมูลพิกัด, DEM และตรวจสอบรอบพยากรณ์ ---
new_lon = np.arange(97.0, 106.0, 0.05)
new_lat = np.arange(21.0, 5.0, -0.05)
G = 9.80665
data_status = {
    "dem_applied": False,
    "gis_features_applied": False,
    "chirps_applied": False,
    "formula_version": "doofah-v2.0"
}

print("กำลังเตรียมข้อมูลภูมิประเทศและภูมิอากาศ...")
Z_dem = None
urban_fraction = None

# รองรับทั้งไฟล์ gis_features_th.npz (สำหรับเฟส GIS) และ dem_th_005.nc (ปัจจุบัน)
if os.path.exists('assets/gis_features_th.npz'):
    try:
        gis_npz = np.load('assets/gis_features_th.npz')
        Z_dem = xr.DataArray(gis_npz['elevation'], coords=[new_lat, new_lon], dims=['latitude', 'longitude'])
        if 'urban_fraction' in gis_npz:
            urban_fraction = xr.DataArray(gis_npz['urban_fraction'], coords=[new_lat, new_lon], dims=['latitude', 'longitude'])
            data_status["gis_features_applied"] = True
        data_status["dem_applied"] = True
        print("  > โหลดข้อมูลจาก assets/gis_features_th.npz สำเร็จ")
    except Exception as e:
        print(f"  [!] โหลด gis_features_th.npz ไม่สำเร็จ: {e}")

if Z_dem is None:
    try:
        ds_dem = xr.open_dataset('assets/dem_th_005.nc')
        if 'lat' in ds_dem.dims and 'lon' in ds_dem.dims:
            ds_dem = ds_dem.rename({'lat': 'latitude', 'lon': 'longitude'})
        var_name = 'elevation' if 'elevation' in ds_dem else list(ds_dem.data_vars)[0]
        Z_dem = ds_dem[var_name].interp(latitude=new_lat, longitude=new_lon, method="cubic")
        ds_dem.close()
        data_status["dem_applied"] = True
        print("  > โหลดข้อมูล DEM จาก assets/dem_th_005.nc สำเร็จ")
    except Exception as e:
        print(f"  [!] ไม่พบ DEM หรือเกิดข้อผิดพลาด ({e}) -> ปิดการปรับอุณหภูมิด้วยความสูง (ใช้ค่า ECMWF โดยตรง)")
        Z_dem = None
        data_status["dem_applied"] = False

# ดึง Step 0 เพื่อตรวจสอบรอบเวลาล่าสุดและคำนวณระดับความสูงอ้างอิงของ ECMWF
retrieve_with_fallback(step=0, type="fc", param="z", target="forecast_z_0.grib")
ds_z = xr.open_dataset("forecast_z_0.grib", engine="cfgrib", backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}, 'indexpath': ''})

base_time_dt = ds_z.time.values.astype('datetime64[s]').item()
pinned_date = base_time_dt.strftime("%Y%m%d")
pinned_time = base_time_dt.hour

# แก้บั๊ก P0: คำนวณ delta_Z เฉพาะเมื่อมีข้อมูล DEM จริงเท่านั้น หากไม่มีให้เป็น 0 ทุกจุด
if data_status["dem_applied"] and Z_dem is not None:
    Z_ecmwf_interp = (ds_z['z'] / G).sel(latitude=slice(21.0, 5.0), longitude=slice(97.0, 106.0)).interp(latitude=new_lat, longitude=new_lon, method="cubic")
    delta_Z = Z_dem - Z_ecmwf_interp
else:
    delta_Z = xr.DataArray(np.zeros((len(new_lat), len(new_lon))), coords=[new_lat, new_lon], dims=['latitude', 'longitude'])

ds_z.close()
os.remove("forecast_z_0.grib")

# --- กำหนดช่วงพยากรณ์ตามรอบเวลา (00Z/12Z = 144 ชม., 06Z/18Z = 90 ชม.) ---
max_step = 144 if pinned_time in (0, 12) else 90
forecast_steps = list(range(3, max_step + 1, 3))

# --- Pre-flight Quality Gate: ตรวจสอบว่าไฟล์สเต็ปสุดท้ายของรอบนี้พร้อมหรือยัง ---
print(f"🔍 [Quality Gate] ตรวจสอบความพร้อมของรอบ {pinned_date} {pinned_time:02d}z ที่ Step สุดท้าย ({max_step})...")
check_file = f"check_{max_step}.grib"
is_ready = retrieve_with_fallback(
    allow_exit=False,
    date=pinned_date,
    time=pinned_time,
    step=max_step,
    type="fc",
    param="msl",
    target=check_file
)
if os.path.exists(check_file):
    os.remove(check_file)

if not is_ready:
    print(f"⏳ ข้อมูลรอบ {pinned_date} {pinned_time:02d}z ยังอัปเดตไม่ครบถึง Step {max_step} ระบบขอข้ามรอบนี้เพื่อรักษาข้อมูลชุดเดิม")
    sys.exit(0)
print(f"✅ ข้อมูลรอบ {pinned_date} {pinned_time:02d}z พร้อมครบถ้วน เริ่มการประมวลผล...")

# --- โหลดตัวคูณ CHIRPS ---
try:
    chirps_multiplier = np.load('assets/chirps_multiplier_thailand.npy')
    data_status["chirps_applied"] = True
except Exception as e:
    print(f"  [!] ไม่พบ CHIRPS หรือเกิดข้อผิดพลาด ({e}) -> ใช้ตัวคูณ 1.0")
    chirps_multiplier = np.ones((12, len(new_lat), len(new_lon)))
    data_status["chirps_applied"] = False

# --- ตรวจสอบรอบพยากรณ์และกู้คืน State (Guard Same Run & Resume) ---
current_run_id = f"{pinned_date}_{pinned_time:02d}z"

os.makedirs(os.path.dirname(RUN_MARKER_FILE) or ".", exist_ok=True)

if os.path.exists(RUN_MARKER_FILE):
    with open(RUN_MARKER_FILE, "r") as f:
        saved_run_id = f.read().strip()
    if saved_run_id != current_run_id:
        print(f"🔄 ตรวจพบรอบพยากรณ์ใหม่ ({current_run_id} != {saved_run_id}) ล้างข้อมูลแคชเดิม...")
        shutil.rmtree(OUT_DIR)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(RUN_MARKER_FILE, "w") as f:
            f.write(current_run_id)
else:
    with open(RUN_MARKER_FILE, "w") as f:
        f.write(current_run_id)

saved_state = load_pipeline_state()
if saved_state:
    print(f"📦 กู้คืน State เดิมสำเร็จ: สเต็ปล่าสุดที่ทำเสร็จคือ Step {saved_state['last_step']}")
    previous_raw_tp = saved_state['previous_raw_tp']
    previous_tp_adj = saved_state.get('previous_tp_adj', None)
    previous_ttr = saved_state['previous_ttr']
    previous_ssrd = saved_state['previous_ssrd']
    previous_ssr = saved_state['previous_ssr']
    previous_str = saved_state['previous_str']
    tp_adj_history = saved_state.get('tp_adj_history', {})
    msl_history = saved_state['msl_history']
else:
    previous_raw_tp = None
    previous_tp_adj = None
    previous_ttr = None
    previous_ssrd = None
    previous_ssr = None
    previous_str = None
    tp_adj_history = {}
    msl_history = {}

# --- เตรียมโครงสร้าง Metadata ---
run_time_str = base_time_dt.strftime("%Y-%m-%d %H:00 UTC")
metadata_path = f'{OUT_DIR}/metadata.json'

if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
else:
    metadata = {
        "platform": "DooFah",
        "model": "ECMWF IFS Open Data",
        "cycle": current_run_id,
        "data_as_of": run_time_str,
        "max_step": max_step,
        "tp24_valid_from_step": 24,
        "experimental_layers": ["mfc", "storm", "ttr", "et0"],
        "forecasts": {},
        "status": data_status,
        "skipped": []
    }

sfc_params_batches = [
    ['tp', '2t', 'msl', '10u', '10v', '2d'],
    ['tcc', 'mucape', 'tcwv', '10fg', '100u', '100v'], 
    ['ttr', 'ssrd', 'ssr', 'str', 'sp']
]
pl_params = ['u', 'v', 'vo', 'q']
REQUIRED_SFC_VARS = ['tp', 't2m', 'd2m', 'msl', 'u10', 'v10', 'tcc', 'mucape', 'tcwv']

timeseries_store = saved_state.get('timeseries_store', {
    'steps': [],
    'tp3h': [],
    '2t': [],
    'msl': [],
    'ws': [],
    '10fg': []
}) if saved_state else {
    'steps': [],
    'tp3h': [],
    '2t': [],
    'msl': [],
    'ws': [],
    '10fg': []
}

# --- ประมวลผลลูปหลัก ---
for step in forecast_steps:
    step_start_time = time.time()
    current_time_str = datetime.now().strftime('%H:%M:%S')
    print(f"\n[{current_time_str}] ⏳ กำลังประมวลผล Step: {step}/{max_step} (Run: {pinned_date} {pinned_time:02d}z)...")
    
    # เช็กการ Resume: ต้องบันทึกเสร็จใน State และมีไฟล์ storm_{step}.geojson และ tp3h_{step}.geojson
    if (
        saved_state
        and step <= saved_state.get('last_step', 0)
        and os.path.exists(f'{OUT_DIR}/storm_{step}.geojson')
        and os.path.exists(f'{OUT_DIR}/tp3h_{step}.geojson')
    ):
        print(f"  > ข้ามการประมวลผล (พบผลลัพธ์ของ Step {step} ใน State เดิมแล้ว)")
        continue
    
    # --- โหลดข้อมูล ---
    t0 = time.time()
    datasets_sfc = []
    sfc_filenames = []
    
    for i, batch in enumerate(sfc_params_batches):
        sfc_filename = f"sfc_{step}_part{i}.grib"
        sfc_filenames.append(sfc_filename)
        retrieve_with_fallback(date=pinned_date, time=pinned_time, step=step, type="fc", param=batch, target=sfc_filename)
        ds_batch = cfgrib.open_datasets(sfc_filename, backend_kwargs={'indexpath': ''})
        datasets_sfc.extend(ds_batch)
    
    sfc_vars = {}
    valid_time_np = None
    
    for ds in datasets_sfc:
        if valid_time_np is None and 'valid_time' in ds:
            valid_time_np = ds.valid_time.values
        ds_smooth = ds.sel(latitude=slice(21.0, 5.0), longitude=slice(97.0, 106.0)).interp(longitude=new_lon, latitude=new_lat, method="cubic")
        for var_name in ds_smooth.data_vars:
            sfc_vars[var_name] = ds_smooth[var_name]
            
    # Quality Gate: ตรวจสอบว่าตัวแปรผิวดินหลักมาครบหรือไม่
    missing_req = [v for v in REQUIRED_SFC_VARS if v not in sfc_vars]
    if missing_req:
        raise RuntimeError(f"[Quality Gate Failed] ขาดตัวแปรสำคัญใน Step {step}: {missing_req}")
    
    pl_filename = f"pl850_{step}.grib"
    retrieve_with_fallback(date=pinned_date, time=pinned_time, step=step, type="fc", param=pl_params, levelist=850, target=pl_filename)
    ds_pl = xr.open_dataset(pl_filename, engine="cfgrib", backend_kwargs={'indexpath': ''})
    
    # เก็บกริดต้นฉบับ (0.25°) บริเวณประเทศไทยพร้อมขอบ 1 องศา สำหรับคำนวณอนุพันธ์เชิงพื้นที่ (MFC)
    ds_pl_native_sub = ds_pl.sel(latitude=slice(22.0, 4.0), longitude=slice(96.0, 107.0))
    # กริดที่ Interpolate เป็น 0.05° สำหรับแสดงผลทั่วไป
    ds_pl_smooth = ds_pl.sel(latitude=slice(21.0, 5.0), longitude=slice(97.0, 106.0)).interp(longitude=new_lon, latitude=new_lat, method="cubic")
    
    metadata["forecasts"][str(step)] = np.datetime_as_string(valid_time_np, unit='h').replace('T', ' ')
    dt_valid = valid_time_np.astype('datetime64[s]').item()
    month_idx = dt_valid.month - 1
    thai_hour = (dt_valid.hour + 7) % 24
    print(f"  > โหลดไฟล์ GRIB และจัดเตรียมตัวแปรสำเร็จ (ใช้เวลา: {time.time() - t0:.2f} วิ)")
    
    # --- 1. หมวดฝน (tp, tp3h, tp24, rr) พร้อมคูณ CHIRPS รายคาบ 3 ชั่วโมง ---
    t0 = time.time()
    raw_tp = sfc_vars['tp'].clip(min=0) * 1000.0
    
    # 1.1 แยกหาฝนสะสมราย 3 ชั่วโมงดิบก่อน
    if previous_raw_tp is None:
        raw_tp3h = raw_tp
    else:
        raw_tp3h = xr.where((raw_tp - previous_raw_tp) < 0, 0, (raw_tp - previous_raw_tp))
    previous_raw_tp = raw_tp.copy()
    
    # 1.2 คูณตัวปรับแก้ CHIRPS ประจำเดือนของช่วงเวลา 3 ชั่วโมงนั้นๆ
    tp3h_adj = raw_tp3h * chirps_multiplier[month_idx]
    
    # 1.3 บวกทบเพื่อสร้างฝนสะสมรวมตั้งแต่ต้นรอบที่ผ่านการปรับแก้รายช่วงแล้ว
    if previous_tp_adj is None:
        tp_adj = tp3h_adj
    else:
        tp_adj = previous_tp_adj + tp3h_adj
    previous_tp_adj = tp_adj.copy()
    tp_adj_history[step] = tp_adj.copy()
    
    # 1.4 คำนวณอัตราฝนเฉลี่ยรายชั่วโมง (มม./ชม.)
    rr_adj = tp3h_adj / 3.0
    
    # 1.5 คำนวณฝนสะสมย้อนหลัง 24 ชั่วโมง (8 สเต็ป)
    if step >= 24 and (step - 24) in tp_adj_history:
        tp24_adj = xr.where((tp_adj - tp_adj_history[step - 24]) < 0, 0, (tp_adj - tp_adj_history[step - 24]))
    else:
        # ช่วง < 24 ชม. แรก ส่งค่าสะสมตั้งแต่เริ่มรอบไปแสดงผลชั่วคราว (กำกับใน metadata["tp24_valid_from_step"] = 24)
        tp24_adj = tp_adj
        
    export_geojson(tp_adj, 'tp', step, [1, 10, 35, 90, 150, 250, 400], ['#00FFFF', '#0000FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#FF00FF'], precision=1)
    export_geojson(tp3h_adj, 'tp3h', step, [0.1, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0], ['#00FFFF', '#0000FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#FF00FF'], precision=1)
    export_geojson(tp24_adj, 'tp24', step, [0.1, 10, 35, 90, 150, 250, 400], ['#00FFFF', '#0000FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#FF00FF'], precision=1)
    export_geojson(rr_adj, 'rr', step, [0.1, 1.0, 2.5, 5.0, 7.6, 15.0, 50.0], ['#00FFFF', '#0000FF', '#00FF00', '#FFFF00', '#FFA500', '#FF0000', '#FF00FF'], precision=1)
    print(f"  > 1. ฝน (tp, tp3h, tp24, rr) เสร็จสิ้น (ใช้เวลา: {time.time() - t0:.2f} วิ)")

    # --- 2. อุณหภูมิ, RH, Heat Index, VPD ---
    t0 = time.time()
    lr = 0.0065 if 7 <= thai_hour < 18 else 0.0050
    t_adj = (sfc_vars['t2m'] - 273.15) - (lr * delta_Z)
    td_raw = (sfc_vars['d2m'] - 273.15) - (0.0018 * delta_Z)
    td_adj = xr.where(td_raw > t_adj, t_adj, td_raw)

    export_geojson(t_adj, '2t', step, [8, 16, 23, 28, 32, 35, 40], ['#000080', '#0000ff', '#00ffff', '#ccff66', '#ffff00', '#ff9900', '#ff0000', '#800000'], precision=1, extend='both')
    export_geojson(td_adj, '2d', step, [10, 16, 18, 21, 24, 26], ['#ffffff', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695', '#000033'], precision=1, extend='both')

    e = 6.11 * 10.0 ** (7.5 * td_adj / (237.3 + td_adj))
    es = 6.11 * 10.0 ** (7.5 * t_adj / (237.3 + t_adj))
    rh = (e / es) * 100.0
    rh = xr.where(rh > 100, 100, xr.where(rh < 0, 0, rh))
    export_geojson(rh, 'rh', step, [30, 50, 60, 70, 80, 90], ['#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1'], precision=0, extend='both')
    
    heat_index = calc_heat_index_xr(t_adj, td_adj)
    export_geojson(heat_index, 'heat', step, [27, 33, 42, 52], ['#27ae60', '#f1c40f', '#f39c12', '#e74c3c', '#8e44ad'], precision=1, extend='both')

    vpd = calc_vpd_xr(t_adj, td_adj)
    export_geojson(vpd, 'vpd', step, [0.4, 0.8, 1.2, 1.6, 2.0, 3.0, 4.0], ['#3498db', '#74add1', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#800000', '#4a0000'], precision=2, extend='both')
    print(f"  > 2. อุณหภูมิ, RH, Heat Index, VPD เสร็จสิ้น (ใช้เวลา: {time.time() - t0:.2f} วิ)")

    # --- 3. ลมผิวดิน, ลม 100 เมตร และ ลมกระโชกแรง ---
    t0 = time.time()
    ws_10m = np.sqrt((sfc_vars['u10']**2 + sfc_vars['v10']**2).clip(min=0))
    export_geojson(ws_10m, 'ws', step, [1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2], ['#b3d1ff', '#66a3ff', '#1a75ff', '#0052cc', '#003d99', '#002966', '#001433'], precision=1)
    export_wind_extras(sfc_vars['u10'], sfc_vars['v10'], 'ws', step)

    if 'u100' in sfc_vars and 'v100' in sfc_vars:
        ws_100m = np.sqrt((sfc_vars['u100']**2 + sfc_vars['v100']**2).clip(min=0))
        export_geojson(ws_100m, 'ws100', step, [3, 5, 7, 9, 12, 25], ['#b3d1ff', '#66a3ff', '#1a75ff', '#0052cc', '#003d99', '#001433'], precision=1)
        export_wind_extras(sfc_vars['u100'], sfc_vars['v100'], 'ws100', step)

    if '10fg' in sfc_vars:
        export_geojson(sfc_vars['10fg'], '10fg', step, [10.8, 13.9, 17.2, 20.8, 25.7], ['#27ae60', '#f1c40f', '#f39c12', '#e74c3c', '#8e44ad'], precision=1)
    else:
        if "10fg" not in metadata.get("skipped", []):
            metadata.setdefault("skipped", []).append("10fg")
    print(f"  > 3. หมวดลม (ws, ws100, 10fg) เสร็จสิ้น (ใช้เวลา: {time.time() - t0:.2f} วิ)")

    # --- 4. อุณหภูมิเทียบเท่าการแผ่รังสีคลื่นยาว (ttr proxy) ---
    t0 = time.time()
    if 'ttr' in sfc_vars:
        raw_ttr = sfc_vars['ttr']
        ttr_diff = raw_ttr if previous_ttr is None else (raw_ttr - previous_ttr)
        previous_ttr = raw_ttr.copy()
        olr = -(ttr_diff) / 10800.0  
        olr = xr.where(olr < 1e-6, 1e-6, olr)
        tb_celsius = ((olr / 5.670374e-8) ** 0.25) - 273.15
        export_geojson(tb_celsius, 'ttr', step, [-46, -30, -18, -8], ['#8e44ad', '#9b59b6', '#3498db', '#f39c12', '#27ae60'], precision=1, extend='both')
    print(f"  > 4. รังสีคลื่นยาวเทียบเท่า (ttr) เสร็จสิ้น (ใช้เวลา: {time.time() - t0:.2f} วิ)")

    # --- 5. โซลาร์ / รังสีแสงอาทิตย์คลื่นสั้นที่พื้นผิว (ssrd) ---
    t0 = time.time()
    if 'ssrd' in sfc_vars:
        raw_ssrd = sfc_vars['ssrd'].clip(min=0)
        if previous_ssrd is None:
            ssrd_diff = raw_ssrd
        else:
            ssrd_diff = xr.where((raw_ssrd - previous_ssrd) < 0, 0, (raw_ssrd - previous_ssrd))
        previous_ssrd = raw_ssrd.copy()
        ssrd_wm2 = ssrd_diff / 10800.0
        export_geojson(ssrd_wm2, 'ssrd', step, [50, 200, 400, 600, 800], ['#34495e', '#d35400', '#e67e22', '#f1c40f', '#f39c12', '#ffff66'], precision=0, extend='both')
    print(f"  > 5. พลังงานแสงอาทิตย์ (ssrd) เสร็จสิ้น (ใช้เวลา: {time.time() - t0:.2f} วิ)")

    # --- 6. เลเยอร์ความกดอากาศ, สภาวะพายุ และเกษตรกรรม ---
    t0 = time.time()
    msl_levels = np.arange(990, 1042, 2).tolist()
    export_geojson(sfc_vars['msl'] / 100.0, 'msl', step, msl_levels, ['#2c3e50'] * (len(msl_levels) + 1), precision=1, extend='both')

    current_msl = sfc_vars['msl'] / 100.0  
    msl_history[step] = current_msl.copy()
    if step - 24 in msl_history:
        export_geojson(current_msl - msl_history[step - 24], 'msl_diff', step, [-6, -4, -2, -1, 1, 2, 4, 6], ['#8e44ad', '#9b59b6', '#e74c3c', '#e67e22', '#ffffff', '#74add1', '#4575b4', '#313695', '#000033'], precision=1, extend='both')

    export_geojson(sfc_vars['mucape'].clip(min=0), 'mucape', step, [100, 500, 1000, 1500, 2500, 4000], ['#e6ffcc', '#b3ff66', '#ffff00', '#ffb366', '#ff6600', '#ff0000'], precision=0)
    export_geojson(sfc_vars['tcc'].clip(min=0, max=1), 'tcc', step, [0.125, 0.375, 0.625, 0.875], ['#e6e6e6', '#cccccc', '#999999', '#666666'], precision=2)
    export_geojson(sfc_vars['tcwv'].clip(min=0), 'tcwv', step, [30, 40, 50, 55, 60, 65, 70], ['#d9e6f2', '#b3cce6', '#8cb3d9', '#6699cc', '#4080bf', '#264d73', '#1b3752'], precision=1)

    ws_850 = np.sqrt((ds_pl_smooth['u']**2 + ds_pl_smooth['v']**2).clip(min=0))
    export_geojson(ws_850, 'ws850', step, [2, 5, 10, 15, 20, 25], ['#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4', '#4eb3d3', '#2b8cbe'], precision=1)
    export_wind_extras(ds_pl_smooth['u'], ds_pl_smooth['v'], 'ws850', step)
    
    vo850 = ds_pl_smooth['vo'] * 100000.0
    export_geojson(vo850, 'vo850', step, [-10, -5, -2, 2, 5, 10, 20], ['#000080', '#0000ff', '#8080ff', '#ffffff', '#ff8080', '#ff0000', '#800000', '#4d0000'], precision=1, extend='both')

    # คำนวณ MFC บนกริดต้นฉบับ (0.25°) แล้วจึงทำ Cubic Interpolation ลงกริดแสดงผล (0.05°)
    if 'q' in ds_pl_native_sub:
        mfc_native = calc_mfc_native(ds_pl_native_sub['q'], ds_pl_native_sub['u'], ds_pl_native_sub['v'])
        mfc_smooth = mfc_native.sel(latitude=slice(21.0, 5.0), longitude=slice(97.0, 106.0)).interp(longitude=new_lon, latitude=new_lat, method="cubic")
        export_geojson(mfc_smooth, 'mfc', step, [-40, -20, -10, 10, 20, 40], ['#a50026', '#f46d43', '#fee090', '#ffffff', '#e0f3f8', '#74add1', '#313695'], precision=1, extend='both')

    # วิเคราะห์ Storm Index (ระดับ 0.5, 1.5, 2.5 ครอบคลุมคะแนน 0, 1, 2, 3)
    mucape = sfc_vars['mucape'].clip(min=0)
    tcwv = sfc_vars['tcwv'].clip(min=0)
    storm_score = xr.where(mucape >= 1000, 1, 0) + xr.where(tcwv >= 55, 1, 0) + xr.where(vo850 >= 5, 1, 0)
    export_geojson(storm_score, 'storm', step, [0.5, 1.5, 2.5], ['#f39c12', '#e74c3c', '#8e44ad'], precision=0, extend='max')

    if 'ssr' in sfc_vars and 'str' in sfc_vars and 'sp' in sfc_vars:
        raw_ssr = sfc_vars['ssr']
        raw_str = sfc_vars['str']
        ssr_diff = xr.where((raw_ssr if previous_ssr is None else (raw_ssr - previous_ssr)) < 0, 0, (raw_ssr if previous_ssr is None else (raw_ssr - previous_ssr)))
        str_diff = raw_str if previous_str is None else (raw_str - previous_str)
        previous_ssr, previous_str = raw_ssr.copy(), raw_str.copy()
        
        et0 = calc_et0_3h(t_adj, td_adj, sfc_vars['sp'], ws_10m, (ssr_diff + str_diff) / 1000000.0)
        export_geojson(et0, 'et0', step, [0.2, 0.5, 1.0, 1.5, 2.0, 3.0], ['#fef0d9', '#fdd49e', '#fdbb84', '#fc8d59', '#e34a33', '#b30000'], precision=2)
    print(f"  > 6. เลเยอร์อื่นๆ (MSL, CAPE, TCC, MFC, ET0, Storm) เสร็จสิ้น (ใช้เวลา: {time.time() - t0:.2f} วิ)")

    # ปิดไฟล์และลบไฟล์ชั่วคราว
    for ds in datasets_sfc:
        ds.close()
    ds_pl.close()
    for f in sfc_filenames:
        os.remove(f)
    os.remove(pl_filename)
    
    # อัปเดต Metadata ระหว่างรัน
    metadata["sources_used"] = sorted(list(active_source_log))
    metadata["generated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

    if step not in timeseries_store['steps']:
        timeseries_store['steps'].append(step)
        timeseries_store['tp3h'].append(np.round(tp3h_adj.values, 1))
        timeseries_store['2t'].append(np.round(t_adj.values, 1))
        timeseries_store['msl'].append(np.round(current_msl.values, 1))
        timeseries_store['ws'].append(np.round((ws_10m.values * 3.6), 1)) # เก็บเป็น กม./ชม.
        fg_val = sfc_vars['10fg'].values * 3.6 if '10fg' in sfc_vars else ws_10m.values * 3.6
        timeseries_store['10fg'].append(np.round(fg_val, 1))
        
    # บันทึก Checkpoint State ประจำสเต็ป
    save_pipeline_state({
        'last_step': step,
        'previous_raw_tp': previous_raw_tp,
        'previous_tp_adj': previous_tp_adj,
        'previous_ttr': previous_ttr,
        'previous_ssrd': previous_ssrd,
        'previous_ssr': previous_ssr,
        'previous_str': previous_str,
        'tp_adj_history': tp_adj_history,
        'msl_history': msl_history,
        'timeseries_store': timeseries_store
    })
        
    print(f"✅ รวมเวลาประมวลผล Step {step}: {time.time() - step_start_time:.2f} วินาที\n" + "-"*50)

# --- ย้ายไฟล์เข้าโฟลเดอร์สำหรับ Deploy แบบปลอดภัย (Atomic-like Swap) ---
new_files = [
    f for f in os.listdir(OUT_DIR)
    if not f.startswith('_') and f != 'metadata.json'
]

# --- สร้างไฟล์ Time-Series แบบแบ่งบล็อก 20x20 (สำหรับหน้าดูรายละเอียดพื้นที่ 6 วัน) ---
print("📊 กำลังสร้างไฟล์ Time-series สำหรับกราฟและสรุป 6 วัน...")
TILE_SIZE = 20
n_lat, n_lon = len(new_lat), len(new_lon)
ts_arrays = {k: np.stack(timeseries_store[k], axis=-1) for k in ['tp3h', '2t', 'msl', 'ws', '10fg']}

for r_idx in range(0, n_lat, TILE_SIZE):
    for c_idx in range(0, n_lon, TILE_SIZE):
        tile_r = r_idx // TILE_SIZE
        tile_c = c_idx // TILE_SIZE
        tile_payload = {
            "steps": timeseries_store['steps'],
            "tp3h": ts_arrays['tp3h'][r_idx:r_idx+TILE_SIZE, c_idx:c_idx+TILE_SIZE, :].tolist(),
            "2t": ts_arrays['2t'][r_idx:r_idx+TILE_SIZE, c_idx:c_idx+TILE_SIZE, :].tolist(),
            "msl": ts_arrays['msl'][r_idx:r_idx+TILE_SIZE, c_idx:c_idx+TILE_SIZE, :].tolist(),
            "ws": ts_arrays['ws'][r_idx:r_idx+TILE_SIZE, c_idx:c_idx+TILE_SIZE, :].tolist(),
            "10fg": ts_arrays['10fg'][r_idx:r_idx+TILE_SIZE, c_idx:c_idx+TILE_SIZE, :].tolist(),
        }
        with open(f"{OUT_DIR}/ts_{tile_r}_{tile_c}_grid.json", "w") as f:
            json.dump(tile_payload, f, separators=(',', ':'))

# 1. ย้ายไฟล์ GeoJSON และ Grid ทั้งหมดไปทับใน public/ ก่อน
for filename in new_files:
    src_path = os.path.join(OUT_DIR, filename)
    dst_path = os.path.join(FINAL_DIR, filename)
    
    # ตรวจสอบว่ามีไฟล์อยู่จริงไหมก่อนย้าย เพื่อป้องกัน FileNotFoundError ในรอบ 90 ชม. (เช่น 06z หรือ 18z)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
    else:
        print(f"⚠️ ข้าม: ไม่พบไฟล์ {filename} (อาจเกินช่วงสเต็ปสูงสุดของรอบนี้)")

# 2. ลบไฟล์ข้อมูลพยากรณ์ของรอบเก่าที่ไม่ได้อยู่ในรอบนี้ (เช่น กรณีสลับจาก 144 ชม. มาเป็น 90 ชม.)
valid_Dynamic_files = set(new_files) | {'metadata.json', '_headers'}
for existing_file in os.listdir(FINAL_DIR):
    if (existing_file.endswith('.geojson') or existing_file.endswith('_grid.json')) and existing_file not in valid_Dynamic_files:
        try:
            os.remove(os.path.join(FINAL_DIR, existing_file))
        except OSError:
            pass

# 3. ย้าย metadata.json เป็นลำดับสุดท้าย เพื่อไม่ให้หน้าเว็บอ่านเจอสเต็ปที่ยังย้ายไฟล์ไม่เสร็จ
if os.path.exists(metadata_path):
    shutil.move(metadata_path, os.path.join(FINAL_DIR, 'metadata.json'))

# 4. สร้างไฟล์ _headers สำหรับเปิด CORS บน Cloudflare Pages
with open(os.path.join(FINAL_DIR, '_headers'), 'w') as f:
    f.write("/*\n  Access-Control-Allow-Origin: *\n")

shutil.rmtree(OUT_DIR)
print(f"✅ ประมวลผลและย้ายไฟล์ไปยัง {FINAL_DIR}/ เสร็จสมบูรณ์!")
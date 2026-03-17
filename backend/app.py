import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import folium
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from shapely.geometry import LineString, Point
from shapely.ops import transform as shp_transform
from pyproj import Transformer

import openrouteservice
from geopy.geocoders import Nominatim

# ======================
# FastAPI + CORS
# ======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://192.168.50.158:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ======================
# 提供前端頁面
# ======================
BASE_DIR = Path(__file__).resolve().parent          # backend/
PROJECT_DIR = BASE_DIR.parent                       # 專案根目錄
WEB_DIR = PROJECT_DIR / "web"

app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

@app.get("/")
def serve_home():
    return FileResponse(WEB_DIR / "index.html")


class Req(BaseModel):
    start: str
    end: str
    dist_m: Optional[float] = None
    target_routes: Optional[int] = None
    cap: Optional[float] = None
    share_factor: Optional[float] = None
    weight_factor: Optional[float] = None


@app.get("/health")
def health():
    return {"ok": True}

# ======================
# 設定（你可調）
# ======================
API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjMxYTMyOGFlNzQzYTQ1NjBhZWQ2NTY0ZDVlZTE1NDc0IiwiaCI6Im11cm11cjY0In0="
client = openrouteservice.Client(key=API_KEY)

geolocator = Nominatim(user_agent="route_safety_app", timeout=10)

DEFAULT_DIST_THRESHOLD_M = 10.0
DEFAULT_CAP = 100.0
DEFAULT_TARGET_ROUTES = 2
DEFAULT_SHARE_FACTOR = 0.2
DEFAULT_WEIGHT_FACTOR = 2.5


# ======================
# 事故資料：啟動時讀一次（超重要，才不會每次都讀很慢）
# ======================
BASE_DIR = Path(__file__).resolve().parent          # backend/
PROJECT_DIR = BASE_DIR.parent                       # 專案根目錄

DATA_DIR = None
for p in [PROJECT_DIR / "DATA", PROJECT_DIR / "data", BASE_DIR / "DATA", BASE_DIR / "data"]:
    if p.exists():
        DATA_DIR = p
        break

if DATA_DIR is None:
    # 沒找到也沒關係，API 會回傳錯誤訊息
    ACCIDENTS = None
else:
    def read_one_file(path: str) -> pd.DataFrame:
        if path.lower().endswith(".csv"):
            for enc in ["utf-8", "utf-8-sig", "cp950", "big5", "latin-1"]:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    df["__source_file__"] = os.path.basename(path)
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception:
                    return pd.DataFrame()
            return pd.DataFrame()
        else:
            try:
                df = pd.read_excel(path)
                df["__source_file__"] = os.path.basename(path)
                return df
            except Exception:
                return pd.DataFrame()

    files = []
    files += list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.CSV"))
    files += list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.XLSX"))

    dfs = [read_one_file(str(p)) for p in files]
    dfs = [d for d in dfs if not d.empty]

    ACCIDENTS = pd.concat(dfs, ignore_index=True) if dfs else None

# ======================
# 工具：欄位對應 + 年份解析 + 嚴重度
# ======================
def pick_column(cols, candidates):
    norm = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        k = cand.strip().lower()
        if k in norm:
            return norm[k]
    return None

time_col  = "發生時間"
loc_col   = "發生地點"
death_injury_col = "死亡受傷人數"

death_pat  = re.compile(r"死亡\s*([-+]?\d+(?:\.\d+)?)")
injury_pat = re.compile(r"受傷\s*([-+]?\d+(?:\.\d+)?)")

def parse_death_injury(cell):
    if pd.isna(cell):
        return 0.0, 0.0
    s = str(cell)
    m1 = death_pat.search(s)
    m2 = injury_pat.search(s)
    death  = float(m1.group(1)) if m1 else 0.0
    injury = float(m2.group(1)) if m2 else 0.0
    return death, injury

def infer_severity(row) -> str:
    death, injury = parse_death_injury(row.get(death_injury_col, None))
    if death > 0:
        return "A1"
    if injury > 0:
        return "A2"
    return "A3"

def extract_year_roc(val):
    """支援：107年..、107/..、2018-..、1070101.. 等格式，回傳民國年"""
    if pd.isna(val):
        return None
    if hasattr(val, "year"):
        y = int(val.year)
        return y - 1911 if y > 1911 else y

    s = str(val).strip()
    m = re.search(r"(\d{2,3})\s*年", s)
    if m:
        return int(m.group(1))

    m = re.search(r"^\s*(\d{2,3})\s*[\/\-.]", s)
    if m:
        return int(m.group(1))

    m = re.search(r"^\s*(20\d{2})\s*[\/\-.]", s)
    if m:
        return int(m.group(1)) - 1911

    m = re.search(r"^\s*(\d{7,})", s)
    if m:
        return int(m.group(1)[:3])

    return None

def place_to_lonlat(name: str):
    loc = geolocator.geocode(name, country_codes="tw")
    if loc is None:
        loc = geolocator.geocode(f"{name}, Taiwan")
    if loc is None:
        return None
    return (loc.longitude, loc.latitude)  # (lon, lat)

# ======================
# 主流程：分析一條路線
# ======================
tf_4326_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
tf_3857_to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

def to_m(lon, lat):
    return tf_4326_to_3857.transform(lon, lat)

def to_deg(x, y):
    return tf_3857_to_4326.transform(x, y)

def eval_one_route(feature, route_idx: int, accidents_df: pd.DataFrame, lat_col: str, lon_col: str,
                   num_years: int, dist_threshold_m: float, cap: float):

    coords = feature["geometry"]["coordinates"]
    route_line = LineString(coords)

    route_line_m  = shp_transform(lambda x, y, z=None: tf_4326_to_3857.transform(x, y), route_line)
    buffer_geom_m = route_line_m.buffer(dist_threshold_m)
    minx, miny, maxx, maxy = buffer_geom_m.bounds

    lats = accidents_df[lat_col].to_numpy()
    lons = accidents_df[lon_col].to_numpy()

    danger_records = []
    count_a1 = count_a2 = count_a3 = 0

    for idx, (lat, lon) in enumerate(zip(lats, lons)):
        x, y = to_m(lon, lat)
        if not (minx <= x <= maxx and miny <= y <= maxy):
            continue

        p_m = Point(x, y)
        if buffer_geom_m.contains(p_m):
            lon2, lat2 = to_deg(p_m.x, p_m.y)
            row = accidents_df.iloc[idx]
            sev = infer_severity(row)

            if sev == "A1":
                count_a1 += 1
            elif sev == "A2":
                count_a2 += 1
            else:
                count_a3 += 1

            parts = [f"路線#{route_idx}｜嚴重度：{sev}"]
            if time_col in row:
                parts.append(f"時間：{row[time_col]}")
            if loc_col in row:
                parts.append(f"地點：{row[loc_col]}")
            parts.append(f"{death_injury_col}：{row.get(death_injury_col, '')}")
            popup = "<br>".join(parts)
            danger_records.append((lat2, lon2, sev, popup))

    route_length_km = max(0.0001, route_line_m.length / 1000)

    weight_a1, weight_a2, weight_a3 = 5, 3, 1
    weighted_accidents = weight_a1 * count_a1 + weight_a2 * count_a2 + weight_a3 * count_a3

    annual_weighted_accidents = weighted_accidents / max(1, num_years)
    accidents_per_km_year = annual_weighted_accidents / route_length_km

    risk = min(1.0, accidents_per_km_year / max(1e-9, cap))
    safety_score = 100 * (1.0 - risk)

    summary = feature.get("properties", {}).get("summary", {})
    duration_s = float(summary.get("duration", 0.0))
    distance_m = float(summary.get("distance", 0.0))

    return {
        "route_idx": route_idx,
        "feature": feature,
        "duration_min": duration_s / 60.0 if duration_s else None,
        "distance_km": distance_m / 1000.0 if distance_m else None,
        "count_a1": count_a1,
        "count_a2": count_a2,
        "count_a3": count_a3,
        "accidents_per_km_year": accidents_per_km_year,
        "safety_score": safety_score,
        "danger_records": danger_records,
    }

# ======================
# API：即時輸入 start/end → 回傳 map_html
# ======================
@app.post("/analyze")
def analyze(req: Req):
    if ACCIDENTS is None or len(ACCIDENTS) == 0:
        return {"error": f"找不到事故資料。請確認有 data/ 或 DATA/ 資料夾，路徑：{DATA_DIR}"}

    start = place_to_lonlat(req.start)
    end = place_to_lonlat(req.end)
    if not start:
        return {"error": f"找不到出發地：{req.start}"}
    if not end:
        return {"error": f"找不到目的地：{req.end}"}

    dist_m = float(req.dist_m) if req.dist_m is not None else DEFAULT_DIST_THRESHOLD_M
    cap = float(req.cap) if req.cap is not None else DEFAULT_CAP
    target_routes = int(req.target_routes) if req.target_routes is not None else DEFAULT_TARGET_ROUTES
    target_routes = max(1, min(3, target_routes))
    share_factor = float(req.share_factor) if req.share_factor is not None else DEFAULT_SHARE_FACTOR
    weight_factor = float(req.weight_factor) if req.weight_factor is not None else DEFAULT_WEIGHT_FACTOR

    # 事故資料清理 & 欄位對應
    accidents = ACCIDENTS.copy()

    lat_col = pick_column(accidents.columns, ["緯度", "lat", "latitude", "y", "Lat", "LAT", "Y"])
    lon_col = pick_column(accidents.columns, ["經度", "lon", "longitude", "x", "Lon", "LON", "X"])
    if not lat_col or not lon_col:
        return {"error": f"事故資料找不到經緯度欄位。欄位有：{list(accidents.columns)[:30]}"}

    accidents[lat_col] = pd.to_numeric(accidents[lat_col], errors="coerce")
    accidents[lon_col] = pd.to_numeric(accidents[lon_col], errors="coerce")
    accidents = accidents.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

    # 年數（用於年平均）
    num_years = 1
    if time_col in accidents.columns:
        accidents["發生年份"] = accidents[time_col].apply(extract_year_roc)
        valid_years = accidents["發生年份"].dropna().unique()
        if len(valid_years) > 0:
            num_years = len(valid_years)

    t0 = time.time()

    # 取得路線（先嘗試多路線，太遠就自動退回單一路線）
    from openrouteservice.exceptions import ApiError

    fallback_to_single = False

    try:
        if target_routes > 1:
            route_geojson = client.directions(
                coordinates=[start, end],
                profile="driving-car",
                format="geojson",
                alternative_routes={
                    "target_count": target_routes,
                    "share_factor": share_factor,
                    "weight_factor": weight_factor
                }
            )
        else:
            route_geojson = client.directions(
                coordinates=[start, end],
                profile="driving-car",
                format="geojson"
            )

    except ApiError as e:
        msg = str(e)

        # 如果是 ORS 多路線距離限制，就自動退回單一路線
        if "alternative Routes algorithm" in msg or "must not be greater than 100000.0 meters" in msg:
            fallback_to_single = True
            route_geojson = client.directions(
                coordinates=[start, end],
                profile="driving-car",
                format="geojson"
            )
        else:
            return {"error": f"ORS 路線計算失敗：{msg}"}

    routes_features = route_geojson.get("features", [])
    if not routes_features:
        return {"error": "ORS 沒有回傳路線，請換個起終點試試。"}

    # 每條路線評估
    results = []
    for i, ft in enumerate(routes_features, start=1):
        results.append(eval_one_route(ft, i, accidents, lat_col, lon_col, num_years, dist_m, cap))

    fastest = min(
        [r for r in results if r["duration_min"] is not None],
        key=lambda x: x["duration_min"],
        default=None
    )
    safest = max(results, key=lambda x: x["safety_score"], default=None)

    # folium 地圖
    center_lat = (start[1] + end[1]) / 2
    center_lon = (start[0] + end[0]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    folium.Marker([start[1], start[0]], popup=f"起點：{req.start}").add_to(m)
    folium.Marker([end[1], end[0]], popup=f"終點：{req.end}").add_to(m)

    # routes（所有）
    layer_all = folium.FeatureGroup(name="routes（所有）", show=False)
    folium.GeoJson(route_geojson).add_to(layer_all)
    layer_all.add_to(m)

    color_map = {"A1": "red", "A2": "orange", "A3": "green"}

    def add_points_to_layer(records, layer):
        for lat, lon, sev, popup in records:
            c = color_map.get(sev, "gray")
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                color=c,
                fill=True,
                fill_color=c,
                fill_opacity=0.7,
                popup=folium.Popup(popup, max_width=320),
            ).add_to(layer)

    fast_idx = fastest["route_idx"] if fastest else None
    safe_idx = safest["route_idx"] if safest else None

    for r in results:
        ridx = r["route_idx"]
        if ridx == fast_idx and ridx == safe_idx:
            tag = f"🏁🛡 最快&最安全（#{ridx}）"
        elif ridx == fast_idx:
            tag = f"🏁 最快路線（#{ridx}）"
        elif ridx == safe_idx:
            tag = f"🛡 最安全路線（#{ridx}）"
        else:
            tag = f"路線 #{ridx}"

        show_route = (ridx == fast_idx) or (ridx == safe_idx)
        layer_route = folium.FeatureGroup(name=tag, show=show_route)
        folium.GeoJson(r["feature"]).add_to(layer_route)
        layer_route.add_to(m)

        layer_pts = folium.FeatureGroup(name=f"事故點（{tag}）", show=False)
        add_points_to_layer(r["danger_records"], layer_pts)
        layer_pts.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # 左下角資訊內容
    fast_html = ""
    safe_html = ""
    notice_html = ""

    if fallback_to_single:
        notice_html = """
        <div style="margin-bottom:8px; color:#b45309;">
          ⚠ 距離較遠，已自動改為單一路線模式
        </div>
        """

    if fastest:
        fast_html = f"""
        <b>🏁 最快路線（#{fastest['route_idx']}）</b><br>
        時間：約 {fastest['duration_min']:.1f} 分<br>
        距離：約 {fastest['distance_km']:.2f} km<br>
        A1={fastest['count_a1']} / A2={fastest['count_a2']} / A3={fastest['count_a3']}<br>
        每公里年平均事故：{fastest['accidents_per_km_year']:.2f}<br>
        安全分數：{fastest['safety_score']:.1f}<br>
        """

    if safest:
        safe_html = f"""
        <b>🛡 最安全路線（#{safest['route_idx']}）</b><br>
        時間：約 {safest['duration_min']:.1f} 分<br>
        距離：約 {safest['distance_km']:.2f} km<br>
        A1={safest['count_a1']} / A2={safest['count_a2']} / A3={safest['count_a3']}<br>
        每公里年平均事故：{safest['accidents_per_km_year']:.2f}<br>
        安全分數：{safest['safety_score']:.1f}<br>
        """

    # 可開關資訊框
    info_html = f"""
    <div id="info-toggle-btn"
         onclick="toggleInfoPanel()"
         style="
            position: fixed;
            bottom: 20px;
            left: 20px;
            z-index: 10001;
            background: #111;
            color: white;
            padding: 10px 14px;
            border-radius: 10px;
            font-size: 14px;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
         ">
      顯示路線資訊
    </div>

    <div id="info-panel"
         style="
            position: fixed;
            bottom: 70px;
            left: 20px;
            z-index: 10000;
            background: white;
            padding: 10px 12px;
            border: 1px solid #ccc;
            border-radius: 10px;
            font-size: 13px;
            max-width: 320px;
            max-height: 55vh;
            overflow-y: auto;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            display: none;
         ">
      {notice_html}
      <b>同一起終點：多路線比較</b><br>
      距離門檻：{dist_m:.0f} m<br>
      資料涵蓋：{num_years} 年事故<br>
      cap（飽和上限）：{cap:.0f}<br>
      <hr style="margin:6px 0;">
      {fast_html}
      <hr style="margin:6px 0;">
      {safe_html}
      <hr style="margin:6px 0;">
      伺服器計算耗時：約 {time.time() - t0:.2f} 秒
    </div>

    <script>
    function toggleInfoPanel() {{
        const panel = document.getElementById("info-panel");
        const btn = document.getElementById("info-toggle-btn");

        if (panel.style.display === "none" || panel.style.display === "") {{
            panel.style.display = "block";
            btn.innerText = "隱藏路線資訊";
        }} else {{
            panel.style.display = "none";
            btn.innerText = "顯示路線資訊";
        }}
    }}
    </script>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    html = m.get_root().render()

    return {
        "fastest": {
            "route_idx": fastest["route_idx"] if fastest else 1,
            "duration_min": round(fastest["duration_min"], 1) if fastest and fastest["duration_min"] else None
        },
        "safest": {
            "route_idx": safest["route_idx"] if safest else 1,
            "duration_min": round(safest["duration_min"], 1) if safest and safest["duration_min"] else None
        },
        "map_html": html,
        "notice": "距離較遠，已自動改為單一路線模式" if fallback_to_single else ""
    }


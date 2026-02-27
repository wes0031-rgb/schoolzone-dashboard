"""
스쿨존 안전 분석 대시보드 — 내 아이가 살기 좋은 동네
성남시 어린이 보호구역 안전등급 시각화
팀원 데이터 통합: 시언(V6 142개소) + 광민(50/30/20 60개소) + 경민(가중치)

실행: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# ──────────────────────────────────────────────
# 1. Page Config & Custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="스쿨존 안전 분석 — 성남시",
    page_icon="\U0001f3eb",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #132743 100%);
}
section[data-testid="stSidebar"] * { color: #D6EAF8 !important; }
section[data-testid="stSidebar"] .stMultiSelect > div > div,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important;
}
h1 { color: #1B4F72 !important; font-weight: 700 !important; letter-spacing: -0.5px; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1B4F72, #2E86C1);
    padding: 16px 20px; border-radius: 12px;
    box-shadow: 0 4px 15px rgba(27,79,114,0.25);
}
div[data-testid="stMetric"] label {
    color: rgba(255,255,255,0.75) !important; font-size: 13px !important; font-weight: 400 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #FFFFFF !important; font-size: 28px !important; font-weight: 700 !important;
}
button[data-baseweb="tab"] {
    font-size: 15px !important; font-weight: 500 !important;
    color: #5DADE2 !important; padding: 10px 24px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1B4F72 !important; border-bottom: 3px solid #1B4F72 !important;
}
h2, h3 { color: #1B4F72 !important; font-weight: 600 !important; }
div[data-testid="stDataFrame"] {
    border: 1px solid #D6EAF8; border-radius: 8px; overflow: hidden;
}
section[data-testid="stSidebar"] .stCheckbox label span { font-size: 14px !important; }
.footer-text {
    text-align: center; color: #85929E; font-size: 12px; padding: 10px 0 20px 0;
}
.warning-banner {
    background: linear-gradient(135deg, #FDEDEC, #F9EBEA);
    border-left: 4px solid #E74C3C; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 16px;
}
.warning-banner b { color: #C0392B; }
.warning-banner span { color: #555; font-size: 13px; }
.suggestion-card {
    background: #F0F6FC; border-radius: 8px; padding: 14px 16px;
    border-left: 4px solid #2E86C1; margin-bottom: 8px;
}
.suggestion-card .school-name { color: #1B4F72; font-weight: 700; font-size: 15px; }
.suggestion-card .suggestion { color: #2E86C1; font-size: 13px; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 2. Constants
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

GRADE_COLORS = {"A": "#154360", "B": "#2471A3", "C": "#85C1E9", "D": "#E74C3C"}
GRADE_LABELS = {"A": "A (우수)", "B": "B (양호)", "C": "C (보통)", "D": "D (주의)"}

MAP_CENTER = [37.42, 127.13]

# 시설물 컬럼 (6개 — 시언/광민 공통)
COMMON_FACILITY_COLS = ["도로적색표면", "신호등", "횡단보도", "도로안전표지", "생활안전CCTV", "무인교통단속카메라"]
# 광민 전용 추가 시설 컬럼
EXTRA_FACILITY_COLS = ["보호구역표지판", "옐로카펫", "무단횡단방지펜스"]

PLOTLY_LAYOUT = dict(
    font=dict(family="Noto Sans KR, sans-serif"),
    plot_bgcolor="#FAFCFF",
    paper_bgcolor="#FFFFFF",
    title_font=dict(size=18, color="#1B4F72"),
)

# ──────────────────────────────────────────────
# 3. Data Loading (cached)
# ──────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "스쿨존_팀통합_최종.csv", encoding="utf-8-sig")


@st.cache_data
def load_guardhouses():
    return pd.read_csv(DATA_DIR / "아동안전지킴이집_성남시.csv", encoding="utf-8-sig")


@st.cache_data
def load_accidents():
    return pd.read_csv(DATA_DIR / "사고다발지_성남시.csv", encoding="utf-8-sig")


@st.cache_data
def load_cctv():
    return pd.read_csv(DATA_DIR / "생활안전CCTV_정제.csv", encoding="utf-8-sig")


@st.cache_data
def load_cameras():
    return pd.read_csv(DATA_DIR / "무인교통단속카메라_정제.csv", encoding="utf-8-sig")


@st.cache_data
def load_signs():
    return pd.read_csv(DATA_DIR / "도로안전표지_정제.csv", encoding="utf-8-sig")


@st.cache_data
def load_population():
    return pd.read_csv(DATA_DIR / "연령별인구_성남시_행정동.csv", encoding="utf-8-sig")


@st.cache_data
def load_geojson():
    with open(DATA_DIR / "성남시_행정동_경계.geojson", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_national_stats():
    return pd.read_csv(DATA_DIR / "전국_어린이보호구역_5년통계.csv", encoding="utf-8-sig")


@st.cache_data
def load_traffic():
    return pd.read_csv(DATA_DIR / "교통량_성남인근_등하교시간대.csv", encoding="utf-8-sig")


# ──────────────────────────────────────────────
# 4. Helper Functions
# ──────────────────────────────────────────────

def calculate_custom_score(df, weights):
    """경민 가중치 기반 안전점수 계산 (0~100 MinMax)"""
    scores = pd.Series(0.0, index=df.index)
    for feat, w in weights.items():
        if feat not in df.columns or w == 0:
            continue
        col = df[feat].fillna(0)
        mn, mx = col.min(), col.max()
        if mx > mn:
            norm = (col - mn) / (mx - mn)
        else:
            norm = pd.Series(0.5, index=df.index)
        # 감산 피처: 높을수록 위험 → 반전
        if feat in ("발생건수", "생활안전CCTV", "무인교통단속카메라"):
            norm = 1 - norm
        scores += norm * abs(w)
    total_w = sum(abs(v) for v in weights.values() if v != 0)
    if total_w > 0:
        scores = scores / total_w * 100
    return scores


def assign_grade_by_quartile(scores):
    """사분위수 기반 등급 부여"""
    q25, q50, q75 = scores.quantile([0.25, 0.5, 0.75])
    def _grade(s):
        if s >= q75:
            return "A"
        elif s >= q50:
            return "B"
        elif s >= q25:
            return "C"
        return "D"
    return scores.apply(_grade)


def make_popup_v6(row):
    """시언 V6 점수 기반 팝업"""
    grade_key = row["등급"]
    color = GRADE_COLORS.get(grade_key, "#999")
    grade_label = GRADE_LABELS.get(grade_key, grade_key)
    return f"""
    <div style="font-family:'Noto Sans KR',sans-serif;width:260px;padding:4px;">
      <div style="font-size:15px;font-weight:700;color:#1B4F72;margin-bottom:4px;">
        {row['시설물명']}
        <span style="font-size:11px;color:#85929E;font-weight:400;margin-left:4px;">{row['시설유형']}</span>
      </div>
      <div style="display:inline-block;background:{color};color:#fff;
           padding:2px 10px;border-radius:20px;font-size:12px;font-weight:500;">
        {grade_label}
      </div>
      <span style="color:#555;font-size:13px;margin-left:6px;">
        {row['활성_안전점수']:.1f}점
      </span>
      <hr style="margin:8px 0;border:none;border-top:1px solid #D6EAF8;">
      <table style="font-size:11px;color:#555;width:100%;border-collapse:collapse;">
        <tr style="background:#F0F6FC;"><td colspan="2" style="padding:3px 4px;font-weight:600;color:#1B4F72;">V6 점수 구조</td></tr>
        <tr><td style="padding:2px 4px;">가산점(시설)</td><td style="text-align:right;font-weight:600;">{row['가산점_시설_V6']:.1f}점</td></tr>
        <tr><td style="padding:2px 4px;">가산점(보너스)</td><td style="text-align:right;font-weight:600;">{int(row['가산점_보너스_V6'])}점</td></tr>
        <tr style="background:#FDEDEC;"><td style="padding:2px 4px;">감산점 합계</td><td style="text-align:right;font-weight:600;color:#E74C3C;">-{row['감산점_합계_V6']:.1f}점</td></tr>
        <tr><td style="padding:2px 4px;">기본점(50)</td><td style="text-align:right;">50.0점</td></tr>
        <tr style="background:#EBF5FB;"><td style="padding:2px 4px;font-weight:700;color:#1B4F72;">최종 V6</td><td style="text-align:right;font-weight:700;color:#1B4F72;">{row['최종안전점수_V6']:.1f}점</td></tr>
      </table>
      <hr style="margin:6px 0;border:none;border-top:1px solid #D6EAF8;">
      <table style="font-size:10px;color:#888;width:100%;border-collapse:collapse;">
        <tr><td>도로적색표면 {int(row['도로적색표면'])}</td><td>신호등 {int(row['신호등'])}</td><td>횡단보도 {int(row['횡단보도'])}</td></tr>
        <tr><td>안전표지 {int(row['도로안전표지'])}</td><td>CCTV {int(row['생활안전CCTV'])}</td><td>카메라 {int(row['무인교통단속카메라'])}</td></tr>
        <tr><td>발생건수 {int(row['발생건수'])}</td><td>어린이비율 {row['어린이비율']:.1f}%</td><td></td></tr>
      </table>
    </div>
    """


def make_popup_gm(row):
    """광민 50/30/20 점수 기반 팝업"""
    grade_key = row["등급"]
    color = GRADE_COLORS.get(grade_key, "#999")
    grade_label = GRADE_LABELS.get(grade_key, grade_key)
    return f"""
    <div style="font-family:'Noto Sans KR',sans-serif;width:260px;padding:4px;">
      <div style="font-size:15px;font-weight:700;color:#1B4F72;margin-bottom:4px;">
        {row['시설물명']}
        <span style="font-size:11px;color:#85929E;font-weight:400;margin-left:4px;">{row['시설유형']}</span>
      </div>
      <div style="display:inline-block;background:{color};color:#fff;
           padding:2px 10px;border-radius:20px;font-size:12px;font-weight:500;">
        {grade_label}
      </div>
      <span style="color:#555;font-size:13px;margin-left:6px;">
        {row['활성_안전점수']:.1f}점
      </span>
      <hr style="margin:8px 0;border:none;border-top:1px solid #D6EAF8;">
      <table style="font-size:11px;color:#555;width:100%;border-collapse:collapse;">
        <tr style="background:#F0F6FC;"><td colspan="2" style="padding:3px 4px;font-weight:600;color:#1B4F72;">100점 만점 구조 (50/30/20)</td></tr>
        <tr><td style="padding:2px 4px;">시설물 (50점)</td><td style="text-align:right;font-weight:600;">{row['시설물_점수(50점)']:.1f}점</td></tr>
        <tr><td style="padding:2px 4px;">사고이력 (30점)</td><td style="text-align:right;font-weight:600;">{row['사고이력_점수(30점)']:.1f}점</td></tr>
        <tr><td style="padding:2px 4px;">인구환경 (20점)</td><td style="text-align:right;font-weight:600;">{row['인구환경_점수(20점)']:.1f}점</td></tr>
        <tr style="background:#EBF5FB;"><td style="padding:2px 4px;font-weight:700;color:#1B4F72;">최종 안전점수</td><td style="text-align:right;font-weight:700;color:#1B4F72;">{row['최종_안전점수']:.1f}점</td></tr>
      </table>
      <hr style="margin:6px 0;border:none;border-top:1px solid #D6EAF8;">
      <table style="font-size:10px;color:#888;width:100%;border-collapse:collapse;">
        <tr><td>적색표면 {int(row['도로적색표면'])}</td><td>신호등 {int(row['신호등'])}</td><td>횡단보도 {int(row['횡단보도'])}</td></tr>
        <tr><td>안전표지 {int(row['도로안전표지'])}</td><td>CCTV {int(row['생활안전CCTV'])}</td><td>카메라 {int(row['무인교통단속카메라'])}</td></tr>
        <tr><td>표지판 {int(row.get('보호구역표지판', 0))}</td><td>옐로카펫 {int(row.get('옐로카펫', 0))}</td><td>펜스 {int(row.get('무단횡단방지펜스', 0))}</td></tr>
      </table>
    </div>
    """


def make_popup_custom(row):
    """가중치 직접 설정 모드 팝업"""
    grade_key = row["등급"]
    color = GRADE_COLORS.get(grade_key, "#999")
    grade_label = GRADE_LABELS.get(grade_key, grade_key)
    return f"""
    <div style="font-family:'Noto Sans KR',sans-serif;width:260px;padding:4px;">
      <div style="font-size:15px;font-weight:700;color:#1B4F72;margin-bottom:4px;">
        {row['시설물명']}
        <span style="font-size:11px;color:#85929E;font-weight:400;margin-left:4px;">{row['시설유형']}</span>
      </div>
      <div style="display:inline-block;background:{color};color:#fff;
           padding:2px 10px;border-radius:20px;font-size:12px;font-weight:500;">
        {grade_label}
      </div>
      <span style="color:#555;font-size:13px;margin-left:6px;">
        {row['활성_안전점수']:.1f}점 (사용자 가중치)
      </span>
      <hr style="margin:8px 0;border:none;border-top:1px solid #D6EAF8;">
      <table style="font-size:10px;color:#888;width:100%;border-collapse:collapse;">
        <tr><td>적색표면 {int(row['도로적색표면'])}</td><td>신호등 {int(row['신호등'])}</td><td>횡단보도 {int(row['횡단보도'])}</td></tr>
        <tr><td>안전표지 {int(row['도로안전표지'])}</td><td>CCTV {int(row['생활안전CCTV'])}</td><td>카메라 {int(row['무인교통단속카메라'])}</td></tr>
        <tr><td>발생건수 {int(row['발생건수'])}</td><td>어린이비율 {row['어린이비율']:.1f}%</td><td></td></tr>
      </table>
    </div>
    """


def create_legend_html():
    items = "".join(
        f'<li style="margin:3px 0;"><span style="background:{GRADE_COLORS[g]};width:14px;height:14px;'
        f'display:inline-block;border-radius:50%;margin-right:8px;vertical-align:middle;'
        f'box-shadow:0 1px 3px rgba(0,0,0,.2);"></span>'
        f'<span style="vertical-align:middle;">{GRADE_LABELS[g]}</span></li>'
        for g in ["A", "B", "C", "D"]
    )
    return f"""
    <div style="position:fixed;bottom:30px;right:30px;z-index:1000;
         background:white;padding:14px 18px;border-radius:10px;
         box-shadow:0 4px 12px rgba(0,0,0,.15);font-size:13px;
         font-family:'Noto Sans KR',sans-serif;border:1px solid #D6EAF8;">
      <div style="font-weight:700;color:#1B4F72;margin-bottom:6px;">안전등급</div>
      <ul style="list-style:none;padding:0;margin:0;">{items}</ul>
    </div>
    """


def create_map(filtered_df, overlay_flags, pop_df, geo, popup_fn):
    m = folium.Map(location=MAP_CENTER, zoom_start=12, tiles="cartodbpositron")

    if geo and geo.get("features"):
        choropleth_data = pop_df[["구명", "동명", "어린이_비율"]].copy()
        choropleth_data["adm_nm"] = "경기도 성남시" + choropleth_data["구명"] + " " + choropleth_data["동명"]
        folium.Choropleth(
            geo_data=geo,
            data=choropleth_data,
            columns=["adm_nm", "어린이_비율"],
            key_on="feature.properties.adm_nm",
            fill_color="PuBu",
            fill_opacity=0.25,
            line_opacity=0.4,
            legend_name="어린이 비율 (%)",
            name="행정동 경계",
        ).add_to(m)

    for _, row in filtered_df.iterrows():
        grade_key = row["등급"]
        color = GRADE_COLORS.get(grade_key, "#999")
        grade_label = GRADE_LABELS.get(grade_key, grade_key)
        # Marker size: bigger for 초등학교
        radius = 9 if row["시설유형"] == "초등학교" else 6
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=radius,
            color="#FFFFFF",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_fn(row), max_width=290),
            tooltip=f"{row['시설물명']} ({grade_label}) {row['활성_안전점수']:.1f}점",
        ).add_to(m)

    if overlay_flags.get("지킴이집"):
        gh = load_guardhouses()
        for _, r in gh.iterrows():
            if pd.notna(r["위도"]) and pd.notna(r["경도"]):
                folium.Marker(
                    [r["위도"], r["경도"]],
                    icon=folium.Icon(color="green", icon="home", prefix="fa"),
                    tooltip=r["안전시설명"],
                ).add_to(m)

    if overlay_flags.get("사고다발지"):
        acc = load_accidents()
        for _, r in acc.iterrows():
            if pd.notna(r["위도"]) and pd.notna(r["경도"]):
                folium.CircleMarker(
                    [r["위도"], r["경도"]],
                    radius=6, color="#E74C3C", fill=True,
                    fill_color="#E74C3C", fill_opacity=0.6,
                    tooltip=f"사고다발지: {r['사고지역위치명']}",
                ).add_to(m)

    if overlay_flags.get("CCTV"):
        cctv = load_cctv()
        for _, r in cctv.iterrows():
            if pd.notna(r["위도"]) and pd.notna(r["경도"]):
                folium.CircleMarker(
                    [r["위도"], r["경도"]],
                    radius=3, color="#8E44AD", fill=True,
                    fill_color="#8E44AD", fill_opacity=0.4, tooltip="CCTV",
                ).add_to(m)

    if overlay_flags.get("카메라"):
        cam = load_cameras()
        for _, r in cam.iterrows():
            if pd.notna(r["위도"]) and pd.notna(r["경도"]):
                folium.CircleMarker(
                    [r["위도"], r["경도"]],
                    radius=3, color="#2980B9", fill=True,
                    fill_color="#2980B9", fill_opacity=0.4, tooltip="단속카메라",
                ).add_to(m)

    if overlay_flags.get("표지판"):
        signs = load_signs()
        for _, r in signs.iterrows():
            if pd.notna(r["위도"]) and pd.notna(r["경도"]):
                folium.CircleMarker(
                    [r["위도"], r["경도"]],
                    radius=2, color="#F39C12", fill=True,
                    fill_color="#F39C12", fill_opacity=0.3, tooltip="안전표지",
                ).add_to(m)

    m.get_root().html.add_child(folium.Element(create_legend_html()))
    return m


# ──────────────────────────────────────────────
# 5. Sidebar
# ──────────────────────────────────────────────
df_raw = load_data()
df = df_raw.copy()

st.sidebar.markdown(
    "<h2 style='text-align:center;margin-bottom:0;'>스쿨존 안전 분석</h2>"
    "<p style='text-align:center;opacity:0.6;font-size:13px;'>성남시 어린이 보호구역</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

# ── 점수 산출 방식 ──
st.sidebar.markdown(
    "<p style='font-weight:600;font-size:14px;margin-bottom:8px;'>점수 산출 방식</p>",
    unsafe_allow_html=True,
)
scoring_mode = st.sidebar.radio(
    "점수 모드",
    ["시언 V6 안전점수 (142개소)", "광민 100점 안전점수 (60개소)", "가중치 직접 설정"],
    label_visibility="collapsed",
)

feature_weights = None
if scoring_mode == "시언 V6 안전점수 (142개소)":
    df["활성_안전점수"] = df["최종안전점수_V6"]
    df["등급"] = df["등급_V6"]
    df["안전등급"] = df["등급_V6"].map(GRADE_LABELS)
    score_label = "V6 안전점수"
    popup_fn = make_popup_v6
elif scoring_mode == "광민 100점 안전점수 (60개소)":
    df = df[df["최종_안전점수"].notna()].copy()
    df["활성_안전점수"] = df["최종_안전점수"]
    df["등급"] = df["등급_광민"]
    df["안전등급"] = df["등급_광민"].map(GRADE_LABELS)
    score_label = "100점 안전점수"
    popup_fn = make_popup_gm
else:
    # 가중치 직접 설정 모드
    st.sidebar.markdown(
        "<p style='font-size:12px;opacity:0.7;margin-bottom:4px;'>카테고리별 가중치 (0~10)</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<p style='font-size:11px;opacity:0.5;margin:0;'>── 위험 (감산) ──</p>", unsafe_allow_html=True)
    w_acc = st.sidebar.slider("발생건수 (감산)", 0, 10, 3, key="w_acc")
    w_cctv = st.sidebar.slider("CCTV (감산)", 0, 10, 1, key="w_cctv")
    w_cam = st.sidebar.slider("무인카메라 (감산)", 0, 10, 1, key="w_cam")
    st.sidebar.markdown("<p style='font-size:11px;opacity:0.5;margin:0;'>── 안전 (가산) ──</p>", unsafe_allow_html=True)
    w_red = st.sidebar.slider("도로적색표면", 0, 10, 3, key="w_red")
    w_signal = st.sidebar.slider("신호등", 0, 10, 2, key="w_sig")
    w_cross = st.sidebar.slider("횡단보도", 0, 10, 1, key="w_cross")
    w_rsign = st.sidebar.slider("안전표지", 0, 10, 1, key="w_rsign")
    w_child = st.sidebar.slider("어린이비율 (가산)", 0, 10, 1, key="w_child")
    feature_weights = {
        "발생건수": w_acc,
        "생활안전CCTV": w_cctv,
        "무인교통단속카메라": w_cam,
        "도로적색표면": w_red,
        "신호등": w_signal,
        "횡단보도": w_cross,
        "도로안전표지": w_rsign,
        "어린이비율": w_child,
    }
    df["활성_안전점수"] = calculate_custom_score(df, feature_weights)
    df["등급"] = assign_grade_by_quartile(df["활성_안전점수"])
    df["안전등급"] = df["등급"].map(GRADE_LABELS)
    score_label = "사용자 가중치 점수"
    popup_fn = make_popup_custom

st.sidebar.markdown("---")

# ── 시설 유형 필터 ──
available_types = sorted(df["시설유형"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect("시설 유형", options=available_types, default=available_types)

# ── 구 필터 ──
available_gu = sorted(df["구"].dropna().unique().tolist())
selected_gu = st.sidebar.multiselect("구 선택", options=available_gu, default=available_gu)

# ── 등급 필터 ──
available_grades = [GRADE_LABELS[g] for g in ["A", "B", "C", "D"] if GRADE_LABELS[g] in df["안전등급"].values]
selected_grades = st.sidebar.multiselect("안전등급", options=available_grades, default=available_grades)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-weight:600;font-size:14px;margin-bottom:8px;'>시설물 레이어</p>",
    unsafe_allow_html=True,
)
ov_guardhouse = st.sidebar.checkbox("아동안전지킴이집", value=True)
ov_accident = st.sidebar.checkbox("사고다발지", value=True)
ov_cctv = st.sidebar.checkbox("생활안전 CCTV", value=False)
ov_camera = st.sidebar.checkbox("무인교통단속카메라", value=False)
ov_sign = st.sidebar.checkbox("도로안전표지", value=False)
overlay_flags = {
    "지킴이집": ov_guardhouse, "사고다발지": ov_accident,
    "CCTV": ov_cctv, "카메라": ov_camera, "표지판": ov_sign,
}

st.sidebar.markdown("---")
school_list = ["(전체)"] + sorted(df["시설물명"].tolist())
selected_school = st.sidebar.selectbox("개별 시설 선택", school_list)

# CSV 다운로드
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-weight:600;font-size:14px;margin-bottom:8px;'>데이터 내보내기</p>",
    unsafe_allow_html=True,
)
csv_cols = ["시설물명", "시설유형", "구", "안전등급", "활성_안전점수"]
# Add score-specific columns
if scoring_mode == "광민 100점 안전점수 (60개소)":
    csv_cols += ["시설물_점수(50점)", "사고이력_점수(30점)", "인구환경_점수(20점)"]
elif scoring_mode == "시언 V6 안전점수 (142개소)":
    csv_cols += ["가산점_시설_V6", "가산점_보너스_V6", "감산점_합계_V6"]
csv_cols += COMMON_FACILITY_COLS + ["발생건수", "어린이비율"]
csv_cols = [c for c in csv_cols if c in df.columns]
csv_export = df[csv_cols].copy().rename(columns={"활성_안전점수": "안전점수"})
st.sidebar.download_button(
    label="CSV 다운로드",
    data=csv_export.to_csv(index=False, encoding="utf-8-sig"),
    file_name="스쿨존_안전분석.csv",
    mime="text/csv",
)

# ──────────────────────────────────────────────
# 6. Main Content
# ──────────────────────────────────────────────
filtered_df = df[
    df["시설유형"].isin(selected_types)
    & df["구"].isin(selected_gu)
    & df["안전등급"].isin(selected_grades)
]

# Header
mode_tag = {
    "시언 V6 안전점수 (142개소)": "시언 V6",
    "광민 100점 안전점수 (60개소)": "광민 50/30/20",
    "가중치 직접 설정": "사용자 가중치",
}.get(scoring_mode, "")

st.markdown(f"""
<div style="margin-bottom:8px;">
    <span style="font-size:36px;font-weight:700;color:#1B4F72;">
        내 아이가 살기 좋은 동네
    </span>
    <span style="font-size:14px;color:#85929E;margin-left:12px;">
        성남시 어린이 보호구역 안전 분석 대시보드 &nbsp;|&nbsp; {mode_tag}
    </span>
</div>
""", unsafe_allow_html=True)

# D등급 경고 배너
d_grade = filtered_df[filtered_df["등급"] == "D"]
if len(d_grade) > 0:
    names = " / ".join(d_grade["시설물명"].tolist()[:10])
    extra = f" 외 {len(d_grade)-10}개" if len(d_grade) > 10 else ""
    st.markdown(
        f'<div class="warning-banner">'
        f'<b>주의 필요 {len(d_grade)}개소</b> &nbsp; '
        f'<span>{names}{extra}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("시설 수", f"{len(filtered_df)}개소")
avg_score = filtered_df["활성_안전점수"].mean() if len(filtered_df) else 0
k2.metric("평균 안전점수", f"{avg_score:.1f}")
safe_ratio = (
    (filtered_df["등급"].isin(["A", "B"])).sum() / len(filtered_df) * 100
    if len(filtered_df) else 0
)
k3.metric("안전(A+B) 비율", f"{safe_ratio:.0f}%")
total_accidents = int(filtered_df["발생건수"].sum())
k4.metric("사고건수 합계", f"{total_accidents}건")

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# Tabs
tab_map, tab_analysis, tab_district = st.tabs(["지도", "상세분석", "동네정보"])

# ============================
# Tab 1: 지도
# ============================
with tab_map:
    pop_df = load_population()
    geo = load_geojson()
    m = create_map(filtered_df, overlay_flags, pop_df, geo, popup_fn)
    st_folium(m, height=550, use_container_width=True, returned_objects=[])

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    col_top, col_bot = st.columns(2)

    top5 = (
        filtered_df.nlargest(5, "활성_안전점수")[
            ["시설물명", "시설유형", "구", "안전등급", "활성_안전점수"]
        ]
        .rename(columns={"활성_안전점수": "안전점수"})
        .reset_index(drop=True)
    )
    top5.index = top5.index + 1
    with col_top:
        st.markdown("##### 안전점수 상위 5")
        st.dataframe(top5, use_container_width=True)

    bot5 = (
        filtered_df.nsmallest(5, "활성_안전점수")[
            ["시설물명", "시설유형", "구", "안전등급", "활성_안전점수"]
        ]
        .rename(columns={"활성_안전점수": "안전점수"})
        .reset_index(drop=True)
    )
    bot5.index = bot5.index + 1
    with col_bot:
        st.markdown("##### 안전점수 하위 5")
        st.dataframe(bot5, use_container_width=True)

# ============================
# Tab 2: 상세분석
# ============================
with tab_analysis:
    # ── 점수 구조 시각화 ──
    if scoring_mode == "시언 V6 안전점수 (142개소)":
        st.markdown("##### V6 점수 구조: 기본(50) + 가산점(시설+보너스) - 감산점")
        v6_struct = pd.DataFrame({
            "항목": ["가산점_시설", "가산점_보너스", "감산점_합계"],
            "평균": [
                df["가산점_시설_V6"].mean(),
                df["가산점_보너스_V6"].mean(),
                -df["감산점_합계_V6"].mean(),
            ],
            "색상": ["#2E86C1", "#5DADE2", "#E74C3C"],
        })
        fig_struct = px.bar(
            v6_struct, x="항목", y="평균",
            title="V6 점수 구성 요소 평균",
            color="항목",
            color_discrete_map={
                "가산점_시설": "#2E86C1",
                "가산점_보너스": "#5DADE2",
                "감산점_합계": "#E74C3C",
            },
            text="평균",
        )
        fig_struct.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_struct.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
        st.plotly_chart(fig_struct, use_container_width=True)

    elif scoring_mode == "광민 100점 안전점수 (60개소)":
        st.markdown("##### 100점 만점 구조: 시설물(50) + 사고이력(30) + 인구환경(20)")
        gm_struct = pd.DataFrame({
            "항목": ["시설물(50점)", "사고이력(30점)", "인구환경(20점)"],
            "평균": [
                df["시설물_점수(50점)"].mean(),
                df["사고이력_점수(30점)"].mean(),
                df["인구환경_점수(20점)"].mean(),
            ],
        })
        fig_struct = px.bar(
            gm_struct, x="항목", y="평균",
            title="50/30/20 점수 구성 요소 평균",
            color="항목",
            color_discrete_map={
                "시설물(50점)": "#1B4F72",
                "사고이력(30점)": "#E74C3C",
                "인구환경(20점)": "#F39C12",
            },
            text="평균",
        )
        fig_struct.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_struct.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
        st.plotly_chart(fig_struct, use_container_width=True)

    col_hist, col_radar = st.columns(2)

    with col_hist:
        fig_hist = px.histogram(
            df, x="활성_안전점수", nbins=20,
            title=f"안전점수 분포 ({score_label})",
            labels={"활성_안전점수": score_label},
            color_discrete_sequence=["#2E86C1"],
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="시설 수", bargap=0.08)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_radar:
        if selected_school != "(전체)":
            school_row = df[df["시설물명"] == selected_school].iloc[0]

            radar_feats = COMMON_FACILITY_COLS
            radar_labels = radar_feats.copy()

            vals = []
            for f in radar_feats:
                mx = df[f].max()
                vals.append(school_row[f] / mx * 100 if mx > 0 else 0)
            vals.append(vals[0])
            radar_labels_closed = radar_labels + [radar_labels[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=radar_labels_closed,
                fill="toself", name=selected_school,
                fillcolor="rgba(46,134,193,0.2)",
                line=dict(color="#1B4F72", width=2),
            ))
            fig_radar.update_layout(
                **PLOTLY_LAYOUT,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#D6EAF8"),
                    angularaxis=dict(gridcolor="#D6EAF8"),
                    bgcolor="#FAFCFF",
                ),
                title=f"{selected_school} 시설물 현황",
                height=380, showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # 상세 카드
            grade_color = GRADE_COLORS.get(school_row["등급"], "#999")
            grade_label = GRADE_LABELS.get(school_row["등급"], school_row["등급"])
            st.markdown(
                f"<div style='background:#F0F6FC;padding:12px 16px;border-radius:8px;"
                f"border-left:4px solid {grade_color};'>"
                f"<b style='color:#1B4F72;'>{selected_school}</b> &nbsp; "
                f"<span style='background:{grade_color};color:#fff;padding:2px 10px;"
                f"border-radius:20px;font-size:12px;'>{grade_label}</span> &nbsp; "
                f"<span style='color:#555;'>{score_label}: <b>{school_row['활성_안전점수']:.1f}</b></span> &nbsp; "
                f"<span style='color:#555;'>발생건수: <b>{int(school_row['발생건수'])}</b>건</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#F0F6FC;padding:20px;border-radius:8px;"
                "text-align:center;color:#5DADE2;margin-top:40px;'>"
                "사이드바에서 개별 시설을 선택하면<br>레이더 차트가 표시됩니다."
                "</div>",
                unsafe_allow_html=True,
            )

    # ── 등급별 현황 ──
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 등급별 현황")
    grade_stats = df.groupby("등급").agg(
        시설수=("시설물명", "count"),
        평균_발생건수=("발생건수", "mean"),
        평균_안전점수=("활성_안전점수", "mean"),
    ).reindex(["A", "B", "C", "D"]).reset_index()
    grade_stats["평균_발생건수"] = grade_stats["평균_발생건수"].round(1)
    grade_stats["평균_안전점수"] = grade_stats["평균_안전점수"].round(1)
    grade_stats.columns = ["등급", "시설 수", "평균 발생건수", "평균 안전점수"]

    col_table, col_bar = st.columns(2)
    with col_table:
        st.dataframe(grade_stats, use_container_width=True, hide_index=True)

    with col_bar:
        fig_acc = px.bar(
            grade_stats, x="등급", y="평균 발생건수",
            title="등급별 평균 사고 발생건수",
            color="등급",
            color_discrete_map={g: GRADE_COLORS[g] for g in ["A", "B", "C", "D"]},
            text="평균 발생건수",
        )
        fig_acc.update_traces(texttemplate="%{text}", textposition="outside")
        fig_acc.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)

    # ── 시설유형별 현황 ──
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 시설유형별 안전점수")
    type_stats = df.groupby("시설유형").agg(
        시설수=("시설물명", "count"),
        평균점수=("활성_안전점수", "mean"),
        평균_발생건수=("발생건수", "mean"),
    ).reset_index()
    type_stats["평균점수"] = type_stats["평균점수"].round(1)
    type_stats["평균_발생건수"] = type_stats["평균_발생건수"].round(1)

    fig_type = px.bar(
        type_stats.sort_values("평균점수"), x="평균점수", y="시설유형",
        orientation="h",
        title="시설유형별 평균 안전점수",
        labels={"평균점수": "평균 안전점수", "시설유형": ""},
        color="평균점수",
        color_continuous_scale=[[0, "#E74C3C"], [0.5, "#85C1E9"], [1, "#154360"]],
        text="시설수",
    )
    fig_type.update_traces(texttemplate="%{text}개소", textposition="outside")
    fig_type.update_layout(**PLOTLY_LAYOUT, height=280, coloraxis_showscale=False)
    st.plotly_chart(fig_type, use_container_width=True)

    # ── D등급 개선 제안 ──
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 주의 필요 시설 (D등급)")
    low_facilities = df[df["등급"] == "D"].sort_values("활성_안전점수")
    if len(low_facilities) > 0:
        for _, row in low_facilities.iterrows():
            grade_color = GRADE_COLORS["D"]
            # 가장 부족한 시설 찾기
            worst = None
            worst_pct = 1.0
            for f in COMMON_FACILITY_COLS:
                mx = df[f].max()
                if mx > 0:
                    pct = row[f] / mx
                    if pct < worst_pct:
                        worst_pct = pct
                        worst = f
            suggestion = f"{worst} 보강 필요 (현재 {int(row[worst])}개)" if worst else "추가 분석 필요"
            st.markdown(
                f'<div class="suggestion-card">'
                f'<span class="school-name">{row["시설물명"]}</span> '
                f'<span style="font-size:11px;color:#85929E;">({row["시설유형"]})</span> &nbsp; '
                f'<span style="background:{grade_color};color:#fff;padding:2px 10px;'
                f'border-radius:20px;font-size:11px;">D ({row["활성_안전점수"]:.1f}점)</span>'
                f'<div class="suggestion">개선 제안: {suggestion}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("모든 시설이 양호한 상태입니다.")


# ============================
# Tab 3: 동네정보
# ============================
with tab_district:
    pop_df = load_population()
    nat_df = load_national_stats()

    # 어린이 비율
    pop_sorted = pop_df.sort_values("어린이_비율", ascending=True)
    fig_pop = px.bar(
        pop_sorted, x="어린이_비율", y="동명", orientation="h",
        title="성남시 행정동별 어린이(0~14세) 비율",
        labels={"어린이_비율": "어린이 비율 (%)", "동명": ""},
        color="어린이_비율",
        color_continuous_scale=[[0, "#D6EAF8"], [0.5, "#5DADE2"], [1, "#1B4F72"]],
    )
    fig_pop.update_layout(**PLOTLY_LAYOUT, height=900, coloraxis_showscale=False)
    st.plotly_chart(fig_pop, use_container_width=True)

    # 전국 추이
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=nat_df["발생년"], y=nat_df["사고건수"],
        mode="lines+markers", name="사고건수",
        line=dict(color="#1B4F72", width=3),
        marker=dict(size=9, color="#1B4F72"),
    ))
    fig_trend.add_trace(go.Scatter(
        x=nat_df["발생년"], y=nat_df["사망자수"],
        mode="lines+markers", name="사망자수",
        line=dict(color="#E74C3C", width=2, dash="dash"),
        marker=dict(size=7, color="#E74C3C"),
        yaxis="y2",
    ))
    fig_trend.update_layout(
        **PLOTLY_LAYOUT,
        title="전국 어린이보호구역 사고 추이 (2020~2024)",
        xaxis_title="연도", yaxis_title="사고건수",
        yaxis2=dict(
            title=dict(text="사망자수", font=dict(color="#E74C3C")),
            overlaying="y", side="right",
            tickfont=dict(color="#E74C3C"),
        ),
        height=400,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#D6EAF8", borderwidth=1),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # 교통량
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    traffic_df = load_traffic()
    if len(traffic_df) > 0:
        traffic_agg = traffic_df.groupby("호선명").agg(
            등교=("등교시간_합계", "mean"),
            하교=("하교시간_합계", "mean"),
        ).reset_index().sort_values("등교", ascending=True)
        traffic_melted = traffic_agg.melt(
            id_vars="호선명", value_vars=["등교", "하교"],
            var_name="시간대", value_name="평균교통량",
        )
        fig_traffic = px.bar(
            traffic_melted, x="평균교통량", y="호선명", color="시간대",
            orientation="h", barmode="group",
            title="성남 인근 주요 국도 등하교 시간대 평균 교통량",
            labels={"평균교통량": "평균 교통량 (대)", "호선명": "", "시간대": ""},
            color_discrete_map={"등교": "#1B4F72", "하교": "#5DADE2"},
        )
        fig_traffic.update_layout(**PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig_traffic, use_container_width=True)

    # 구별 안전점수
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    gu_scores = df.groupby("구").agg(
        평균안전점수=("활성_안전점수", "mean"),
        시설수=("시설물명", "count"),
    ).reset_index().sort_values("평균안전점수", ascending=True)

    fig_gu = px.bar(
        gu_scores, x="평균안전점수", y="구", orientation="h",
        title="구별 평균 안전점수",
        labels={"평균안전점수": "평균 안전점수", "구": ""},
        color="평균안전점수",
        color_continuous_scale=[[0, "#E74C3C"], [0.5, "#85C1E9"], [1, "#154360"]],
        text="시설수",
    )
    fig_gu.update_traces(texttemplate="%{text}개소", textposition="outside")
    fig_gu.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=False)
    st.plotly_chart(fig_gu, use_container_width=True)


# ──────────────────────────────────────────────
# 7. Footer
# ──────────────────────────────────────────────
st.markdown(
    '<div class="footer-text">'
    "데이터 출처: 공공데이터포털, 도로교통공단, 경기데이터드림, 성남시 &nbsp;|&nbsp; "
    "팀원 데이터 통합: 시언(V6) + 광민(50/30/20) + 경민(가중치)"
    "</div>",
    unsafe_allow_html=True,
)

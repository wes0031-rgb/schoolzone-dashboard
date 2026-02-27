"""
스쿨존 안전 분석 대시보드 — 내 아이가 살기 좋은 동네
성남시 초등학교 73개 어린이 보호구역 안전등급 시각화 (v11)

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
    page_icon="🏫",
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
# 2. 상수 & 경로
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

GRADE_COLORS = {
    "A": "#154360",
    "B": "#2471A3",
    "C": "#85C1E9",
    "D": "#E74C3C",
}
GRADE_LABELS = {
    "A": "A (우수)",
    "B": "B (양호)",
    "C": "C (보통)",
    "D": "D (주의)",
}

MAP_CENTER = [37.42, 127.13]

# v11 피처 11개 (감산 41% / 가산 59%)
V11_FEATURES = {
    # 위험 (감산 41%)
    "사고건수_300m":       {"label": "발생건수(300m)",      "weight": -0.30, "category": "위험(감산)"},
    "CCTV_300m":           {"label": "생활안전CCTV(300m)",   "weight": -0.06, "category": "위험(감산)"},
    "무인카메라_300m":     {"label": "무인카메라(300m)",     "weight": -0.05, "category": "위험(감산)"},
    # 안전 (가산 59%)
    "도로적색표면_300m":   {"label": "도로적색표면(300m)",   "weight": 0.13,  "category": "안전(가산)"},
    "신호등_300m":         {"label": "신호등(300m)",         "weight": 0.11,  "category": "안전(가산)"},
    "횡단보도_300m":       {"label": "횡단보도(300m)",       "weight": 0.07,  "category": "안전(가산)"},
    "도로안전표지_300m":   {"label": "안전표지(300m)",       "weight": 0.07,  "category": "안전(가산)"},
    "보호구역표지판_300m": {"label": "표지판(300m)",         "weight": 0.07,  "category": "안전(가산)"},
    "무단횡단방지펜스_300m": {"label": "펜스(300m)",         "weight": 0.07,  "category": "안전(가산)"},
    "옐로카펫_300m":       {"label": "옐로카펫(300m)",       "weight": 0.05,  "category": "안전(가산)"},
    "어린이비율":          {"label": "어린이비율(%)",        "weight": 0.02,  "category": "안전(가산)"},
}

# 개선 제안 매핑 (가산 피처만 — 추가 설치 가능 시설)
IMPROVEMENT_SUGGESTIONS = {
    "도로적색표면_300m":     "도로적색표면 추가 설치",
    "신호등_300m":           "신호등 추가 설치",
    "횡단보도_300m":         "횡단보도 추가 설치",
    "도로안전표지_300m":     "도로안전표지 추가 설치",
    "보호구역표지판_300m":   "보호구역표지판 추가 설치",
    "무단횡단방지펜스_300m": "무단횡단방지펜스 추가 설치",
    "옐로카펫_300m":         "옐로카펫 추가 설치",
}

PLOTLY_LAYOUT = dict(
    font=dict(family="Noto Sans KR, sans-serif"),
    plot_bgcolor="#FAFCFF",
    paper_bgcolor="#FFFFFF",
    title_font=dict(size=18, color="#1B4F72"),
)

# ──────────────────────────────────────────────
# 3. 데이터 로딩 (캐시)
# ──────────────────────────────────────────────

@st.cache_data
def load_v11():
    df = pd.read_csv(DATA_DIR / "스쿨존_안전점수_v11.csv", encoding="utf-8-sig")
    df["안전등급"] = df["등급"].map(GRADE_LABELS)
    return df


@st.cache_data
def load_cameras():
    return pd.read_csv(DATA_DIR / "무인교통단속카메라_정제.csv", encoding="utf-8-sig")


@st.cache_data
def load_cctv():
    return pd.read_csv(DATA_DIR / "생활안전CCTV_정제.csv", encoding="utf-8-sig")


@st.cache_data
def load_signs():
    return pd.read_csv(DATA_DIR / "도로안전표지_정제.csv", encoding="utf-8-sig")


@st.cache_data
def load_guardhouses():
    return pd.read_csv(DATA_DIR / "아동안전지킴이집_성남시.csv", encoding="utf-8-sig")


@st.cache_data
def load_accidents():
    return pd.read_csv(DATA_DIR / "사고다발지_성남시.csv", encoding="utf-8-sig")


@st.cache_data
def load_population():
    return pd.read_csv(DATA_DIR / "연령별인구_성남시_행정동.csv", encoding="utf-8-sig")


@st.cache_data
def load_geojson():
    with open(DATA_DIR / "성남시_행정동_경계.geojson", encoding="utf-8") as f:
        geo = json.load(f)
    return geo


@st.cache_data
def load_national_stats():
    return pd.read_csv(DATA_DIR / "전국_어린이보호구역_5년통계.csv", encoding="utf-8-sig")


@st.cache_data
def load_traffic():
    return pd.read_csv(DATA_DIR / "교통량_성남인근_등하교시간대.csv", encoding="utf-8-sig")


# ──────────────────────────────────────────────
# 4. 헬퍼 함수
# ──────────────────────────────────────────────

def calculate_custom_score(df, weights):
    """사용자 가중치 기반 안전점수 계산 (v11 피처 10개, 0~100 MinMax)"""
    scores = pd.Series(0.0, index=df.index)
    for feat, w in weights.items():
        col = df[feat]
        mn, mx = col.min(), col.max()
        if mx > mn:
            norm = (col - mn) / (mx - mn)
        else:
            norm = pd.Series(0.5, index=df.index)
        # 가중치 부호: +면 높을수록 좋고, -면 높을수록 나쁨
        info = V11_FEATURES[feat]
        if info["weight"] < 0:
            norm = 1 - norm  # 감산 피처: 높을수록 위험 → 반전
        scores += norm * abs(w)

    total_w = sum(abs(w) for w in weights.values())
    if total_w > 0:
        scores = scores / total_w * 100
    else:
        scores = pd.Series(50.0, index=df.index)
    return scores


def assign_custom_grade(score):
    """사용자 가중치 점수의 등급 (사분위수 대신 고정 임계값)"""
    if score >= 75:
        return "A (우수)"
    elif score >= 65:
        return "B (양호)"
    elif score >= 55:
        return "C (보통)"
    else:
        return "D (주의)"


def make_popup_html(row):
    grade = row["안전등급"]
    grade_key = row["등급"]
    color = GRADE_COLORS[grade_key]
    return f"""
    <div style="font-family:'Noto Sans KR',sans-serif;width:260px;padding:4px;">
      <div style="font-size:15px;font-weight:700;color:#1B4F72;margin-bottom:4px;">
        {row['시설명']}
      </div>
      <div style="display:inline-block;background:{color};color:#fff;
           padding:2px 10px;border-radius:20px;font-size:12px;font-weight:500;">
        {grade}
      </div>
      <span style="color:#555;font-size:13px;margin-left:6px;">
        {row['활성_안전점수']:.1f}점
      </span>
      <hr style="margin:8px 0;border:none;border-top:1px solid #D6EAF8;">
      <table style="font-size:11px;color:#555;width:100%;border-collapse:collapse;">
        <tr style="background:#FDEDEC;"><td colspan="2" style="padding:3px 4px;font-weight:600;color:#C0392B;">위험 — 감산 (41%)</td></tr>
        <tr><td style="padding:2px 4px;">발생건수(300m) 30%</td><td style="text-align:right;">{int(row['사고건수_300m'])}건</td></tr>
        <tr><td style="padding:2px 4px;">생활안전CCTV 6%</td><td style="text-align:right;">{int(row['CCTV_300m'])}대</td></tr>
        <tr><td style="padding:2px 4px;">무인카메라 5%</td><td style="text-align:right;">{int(row['무인카메라_300m'])}대</td></tr>
        <tr style="background:#F0F6FC;"><td colspan="2" style="padding:3px 4px;font-weight:600;color:#1B4F72;">안전 — 가산 (59%)</td></tr>
        <tr><td style="padding:2px 4px;">도로적색표면 13%</td><td style="text-align:right;">{int(row['도로적색표면_300m'])}개</td></tr>
        <tr><td style="padding:2px 4px;">신호등 11%</td><td style="text-align:right;">{int(row['신호등_300m'])}개</td></tr>
        <tr><td style="padding:2px 4px;">횡단보도 7%</td><td style="text-align:right;">{int(row['횡단보도_300m'])}개</td></tr>
        <tr><td style="padding:2px 4px;">안전표지 7%</td><td style="text-align:right;">{int(row['도로안전표지_300m'])}개</td></tr>
        <tr><td style="padding:2px 4px;">표지판 7%</td><td style="text-align:right;">{int(row['보호구역표지판_300m'])}개</td></tr>
        <tr><td style="padding:2px 4px;">펜스 7%</td><td style="text-align:right;">{int(row['무단횡단방지펜스_300m'])}개</td></tr>
        <tr><td style="padding:2px 4px;">옐로카펫 5%</td><td style="text-align:right;">{int(row['옐로카펫_300m'])}개</td></tr>
        <tr><td style="padding:2px 4px;">어린이비율 2%</td><td style="text-align:right;">{row['어린이비율']:.1f}%</td></tr>
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


def get_improvement_suggestion(row, df):
    """예방시설 중 가장 부족한 항목 기반 개선 제안"""
    prevention_feats = [f for f, info in V11_FEATURES.items() if info["weight"] > 0]
    worst_feat = None
    worst_percentile = 1.0
    for feat in prevention_feats:
        if feat in row.index and feat in df.columns:
            val = row[feat]
            mx = df[feat].max()
            pct = val / mx if mx > 0 else 1.0
            if pct < worst_percentile:
                worst_percentile = pct
                worst_feat = feat
    if worst_feat and worst_feat in IMPROVEMENT_SUGGESTIONS:
        current = int(row[worst_feat])
        median = int(df[worst_feat].median())
        return f"{IMPROVEMENT_SUGGESTIONS[worst_feat]} (현재 {current}개, 중앙값 {median}개)"
    return "추가 분석 필요"


def create_map(filtered_df, overlay_flags, pop_df, geo):
    m = folium.Map(location=MAP_CENTER, zoom_start=12, tiles="cartodbpositron")

    if geo and geo["features"]:
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
        color = GRADE_COLORS[grade_key]
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=9,
            color="#FFFFFF",
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(make_popup_html(row), max_width=290),
            tooltip=f"{row['시설명']} ({row['안전등급']})",
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
# 5. 사이드바
# ──────────────────────────────────────────────
df = load_v11()

st.sidebar.markdown(
    "<h2 style='text-align:center;margin-bottom:0;'>스쿨존 안전 분석</h2>"
    "<p style='text-align:center;opacity:0.6;font-size:13px;'>성남시 초등학교 73개</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

available_gu = sorted(df["구"].dropna().unique().tolist())
selected_gu = st.sidebar.multiselect("구 선택", options=available_gu, default=available_gu)

available_grades = [GRADE_LABELS[g] for g in ["A", "B", "C", "D"]]
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
school_list = ["(전체)"] + sorted(df["시설명"].tolist())
selected_school = st.sidebar.selectbox("개별 학교 선택", school_list)

# 점수 산출 방식
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-weight:600;font-size:14px;margin-bottom:8px;'>점수 산출 방식</p>",
    unsafe_allow_html=True,
)
scoring_mode = st.sidebar.radio(
    "점수 산출",
    ["v11 안전점수 (감산41%/가산59%)", "가중치 직접 설정"],
    label_visibility="collapsed",
)

feature_weights = None
if scoring_mode == "가중치 직접 설정":
    st.sidebar.markdown(
        "<p style='font-size:12px;opacity:0.7;margin-bottom:4px;'>카테고리별 가중치 (0~10)</p>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<p style='font-size:11px;opacity:0.5;margin:0;'>── 위험 (감산) ──</p>", unsafe_allow_html=True)
    w_accident = st.sidebar.slider("발생건수 (감산)", 0, 10, 3, key="w_acc")
    w_cctv = st.sidebar.slider("CCTV (감산)", 0, 10, 1, key="w_cctv")
    w_cam = st.sidebar.slider("무인카메라 (감산)", 0, 10, 1, key="w_cam")
    st.sidebar.markdown("<p style='font-size:11px;opacity:0.5;margin:0;'>── 안전 (가산) ──</p>", unsafe_allow_html=True)
    w_red = st.sidebar.slider("도로적색표면", 0, 10, 3, key="w_red")
    w_signal = st.sidebar.slider("신호등", 0, 10, 2, key="w_sig")
    w_cross = st.sidebar.slider("횡단보도", 0, 10, 1, key="w_cross")
    w_roadsign = st.sidebar.slider("안전표지", 0, 10, 1, key="w_rsign")
    w_zonesign = st.sidebar.slider("표지판", 0, 10, 1, key="w_zsign")
    w_fence = st.sidebar.slider("펜스", 0, 10, 1, key="w_fence")
    w_yellow = st.sidebar.slider("옐로카펫", 0, 10, 1, key="w_yel")
    w_child = st.sidebar.slider("어린이비율 (가산)", 0, 10, 1, key="w_child")
    feature_weights = {
        "사고건수_300m": w_accident,
        "CCTV_300m": w_cctv,
        "무인카메라_300m": w_cam,
        "도로적색표면_300m": w_red,
        "신호등_300m": w_signal,
        "횡단보도_300m": w_cross,
        "도로안전표지_300m": w_roadsign,
        "보호구역표지판_300m": w_zonesign,
        "무단횡단방지펜스_300m": w_fence,
        "옐로카펫_300m": w_yellow,
        "어린이비율": w_child,
    }

# ── 점수 계산 (모드에 따라) ──
if scoring_mode == "가중치 직접 설정" and feature_weights is not None:
    df["활성_안전점수"] = calculate_custom_score(df, feature_weights)
    df["안전등급"] = df["활성_안전점수"].apply(assign_custom_grade)
    score_label = "사용자 가중치 안전점수"
else:
    df["활성_안전점수"] = df["안전점수"]
    score_label = "안전점수 v11"

# CSV 다운로드
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-weight:600;font-size:14px;margin-bottom:8px;'>데이터 내보내기</p>",
    unsafe_allow_html=True,
)
csv_cols = ["시설명", "구", "도로명주소", "안전등급", "활성_안전점수",
            "사고건수_300m", "CCTV_300m", "무인카메라_300m",
            "도로적색표면_300m", "신호등_300m", "횡단보도_300m",
            "도로안전표지_300m", "보호구역표지판_300m", "무단횡단방지펜스_300m",
            "옐로카펫_300m", "어린이비율"]
csv_export = df[csv_cols].copy()
csv_export = csv_export.rename(columns={"활성_안전점수": "안전점수"})
st.sidebar.download_button(
    label="CSV 다운로드",
    data=csv_export.to_csv(index=False, encoding="utf-8-sig"),
    file_name="스쿨존_안전분석_v11.csv",
    mime="text/csv",
)


# ──────────────────────────────────────────────
# 6. 메인 콘텐츠
# ──────────────────────────────────────────────
filtered_df = df[
    df["구"].isin(selected_gu)
    & df["안전등급"].isin(selected_grades)
]

st.markdown("""
<div style="margin-bottom:8px;">
    <span style="font-size:36px;font-weight:700;color:#1B4F72;">
        내 아이가 살기 좋은 동네
    </span>
    <span style="font-size:14px;color:#85929E;margin-left:12px;">
        성남시 초등학교 73개 어린이 보호구역 안전 분석 대시보드 (v11)
    </span>
</div>
""", unsafe_allow_html=True)

# — D등급 경고 배너 —
d_grade_schools = df[df["등급"] == "D"]
if len(d_grade_schools) > 0:
    school_names = " / ".join(d_grade_schools["시설명"].tolist()[:10])
    extra = f" 외 {len(d_grade_schools)-10}개" if len(d_grade_schools) > 10 else ""
    st.markdown(
        f'<div class="warning-banner">'
        f'<b>주의 필요 {len(d_grade_schools)}개소</b> &nbsp; '
        f'<span>{school_names}{extra}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

# — KPI —
k1, k2, k3, k4 = st.columns(4)
k1.metric("초등학교 수", f"{len(filtered_df)}개교")
avg_score = filtered_df["활성_안전점수"].mean()
k2.metric("평균 안전점수", f"{avg_score:.1f}" if len(filtered_df) else "-")
safe_ratio = (
    (filtered_df["등급"].isin(["A", "B"])).sum()
    / len(filtered_df) * 100
    if len(filtered_df) else 0
)
k3.metric("안전(A+B) 비율", f"{safe_ratio:.0f}%")
total_accidents = int(filtered_df["사고건수_300m"].sum())
k4.metric("사고건수 합계(300m)", f"{total_accidents}건")

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# — 탭 —
tab_map, tab_analysis, tab_district = st.tabs(["지도", "상세분석", "동네정보"])

# ============================
# 탭1: 지도
# ============================
with tab_map:
    pop_df = load_population()
    geo = load_geojson()
    m = create_map(filtered_df, overlay_flags, pop_df, geo)
    st_folium(m, height=550, use_container_width=True, returned_objects=[])

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    col_top, col_bot = st.columns(2)

    top5 = (
        filtered_df.nlargest(5, "활성_안전점수")[
            ["시설명", "구", "안전등급", "활성_안전점수"]
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
            ["시설명", "구", "안전등급", "활성_안전점수"]
        ]
        .rename(columns={"활성_안전점수": "안전점수"})
        .reset_index(drop=True)
    )
    bot5.index = bot5.index + 1
    with col_bot:
        st.markdown("##### 안전점수 하위 5")
        st.dataframe(bot5, use_container_width=True)

# ============================
# 탭2: 상세분석
# ============================
with tab_analysis:
    # — 가중치 구조 시각화 —
    weight_data = pd.DataFrame([
        {"피처": info["label"], "가중치": abs(info["weight"]) * 100,
         "카테고리": info["category"], "방향": "감산" if info["weight"] < 0 else "가산"}
        for feat, info in V11_FEATURES.items()
    ])
    weight_sorted = weight_data.sort_values("가중치", ascending=True)
    colors = {"가산": "#2E86C1", "감산": "#E74C3C"}
    fig_weight = px.bar(
        weight_sorted,
        x="가중치", y="피처", orientation="h",
        color="방향",
        title="v11 안전점수 가중치 구조 (감산 41% / 가산 59%)",
        labels={"가중치": "가중치 (%)", "피처": ""},
        color_discrete_map=colors,
        text="가중치",
    )
    fig_weight.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig_weight.update_layout(**PLOTLY_LAYOUT, height=420, bargap=0.15)
    st.plotly_chart(fig_weight, use_container_width=True)

    col_hist, col_radar = st.columns(2)

    with col_hist:
        fig_hist = px.histogram(
            df, x="활성_안전점수", nbins=20,
            title="안전점수 분포",
            labels={"활성_안전점수": score_label},
            color_discrete_sequence=["#2E86C1"],
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="학교 수", bargap=0.08)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_radar:
        if selected_school != "(전체)":
            school_row = df[df["시설명"] == selected_school].iloc[0]

            # 레이더 차트 — 주요 6개 피처 선별
            radar_feats = [
                "사고건수_300m", "도로적색표면_300m", "신호등_300m",
                "횡단보도_300m", "CCTV_300m", "옐로카펫_300m",
            ]
            radar_labels = [V11_FEATURES[f]["label"] for f in radar_feats]

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

            # 학교 상세 카드
            grade = school_row["안전등급"]
            grade_color = GRADE_COLORS[school_row["등급"]]
            suggestion = get_improvement_suggestion(school_row, df)
            st.markdown(
                f"<div style='background:#F0F6FC;padding:12px 16px;border-radius:8px;"
                f"border-left:4px solid {grade_color};'>"
                f"<b style='color:#1B4F72;'>{selected_school}</b> &nbsp; "
                f"<span style='background:{grade_color};color:#fff;padding:2px 10px;"
                f"border-radius:20px;font-size:12px;'>{grade}</span> &nbsp; "
                f"<span style='color:#555;'>{score_label}: <b>{school_row['활성_안전점수']:.1f}</b></span> &nbsp; "
                f"<span style='color:#555;'>사고건수: <b>{int(school_row['사고건수_300m'])}</b>건</span>"
                f"<div style='margin-top:8px;color:#2E86C1;font-size:13px;'>"
                f"개선 제안: {suggestion}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#F0F6FC;padding:20px;border-radius:8px;"
                "text-align:center;color:#5DADE2;margin-top:40px;'>"
                "사이드바에서 개별 학교를 선택하면<br>레이더 차트가 표시됩니다."
                "</div>",
                unsafe_allow_html=True,
            )

    # — 등급별 사고율 비교 —
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 등급별 사고 현황 (v11 검증)")
    grade_stats = df.groupby("등급").agg(
        학교수=("시설명", "count"),
        사고경험_비율=("사고유무_300m", "mean"),
        평균_사고건수=("사고건수_300m", "mean"),
        평균_안전점수=("활성_안전점수", "mean"),
    ).reindex(["A", "B", "C", "D"]).reset_index()
    grade_stats["사고경험_비율"] = (grade_stats["사고경험_비율"] * 100).round(1)
    grade_stats["평균_사고건수"] = grade_stats["평균_사고건수"].round(1)
    grade_stats["평균_안전점수"] = grade_stats["평균_안전점수"].round(1)
    grade_stats.columns = ["등급", "학교 수", "사고율(%)", "평균 사고건수", "평균 안전점수"]

    col_table, col_bar = st.columns(2)
    with col_table:
        st.dataframe(grade_stats, use_container_width=True, hide_index=True)

    with col_bar:
        fig_acc = px.bar(
            grade_stats, x="등급", y="사고율(%)",
            title="등급별 사고율 (A < B < C < D 단조감소 확인)",
            color="등급",
            color_discrete_map={g: GRADE_COLORS[g] for g in ["A", "B", "C", "D"]},
            text="사고율(%)",
        )
        fig_acc.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_acc.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)

    # — 개선이 필요한 스쿨존 —
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 개선이 필요한 스쿨존")
    low_schools = df[df["등급"].isin(["D", "C"])].sort_values("활성_안전점수")
    if len(low_schools) > 0:
        for _, row in low_schools.iterrows():
            grade = row["안전등급"]
            grade_color = GRADE_COLORS[row["등급"]]
            suggestion = get_improvement_suggestion(row, df)
            st.markdown(
                f'<div class="suggestion-card">'
                f'<span class="school-name">{row["시설명"]}</span> &nbsp; '
                f'<span style="background:{grade_color};color:#fff;padding:2px 10px;'
                f'border-radius:20px;font-size:11px;">{grade} ({row["활성_안전점수"]:.1f}점)</span>'
                f'<div class="suggestion">개선 제안: {suggestion}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("모든 스쿨존이 양호한 상태입니다.")

# ============================
# 탭3: 동네정보
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

    # 전국 5년 추이
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

    # 등하교 시간대 교통량
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    traffic_df = load_traffic()
    if len(traffic_df) > 0:
        traffic_agg = traffic_df.groupby("호선명").agg(
            등교=("등교시간_합계", "mean"),
            하교=("하교시간_합계", "mean"),
        ).reset_index()
        traffic_agg = traffic_agg.sort_values("등교", ascending=True)

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

    # 구별 안전점수 집계
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    gu_scores = df.groupby("구").agg(
        평균안전점수=("활성_안전점수", "mean"),
        학교수=("시설명", "count"),
    ).reset_index().sort_values("평균안전점수", ascending=True)

    fig_gu = px.bar(
        gu_scores, x="평균안전점수", y="구", orientation="h",
        title="구별 평균 안전점수",
        labels={"평균안전점수": "평균 안전점수", "구": ""},
        color="평균안전점수",
        color_continuous_scale=[[0, "#E74C3C"], [0.5, "#85C1E9"], [1, "#154360"]],
        text="학교수",
    )
    fig_gu.update_traces(texttemplate="%{text}개교", textposition="outside")
    fig_gu.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=False)
    st.plotly_chart(fig_gu, use_container_width=True)


# ──────────────────────────────────────────────
# 7. 푸터
# ──────────────────────────────────────────────
st.markdown(
    '<div class="footer-text">'
    "데이터 출처: 공공데이터포털, 도로교통공단, 경기데이터드림, 성남시 &nbsp;|&nbsp; "
    "안전점수: v11 (11개 피처, 감산41%/가산59%, MinMax 정규화, 5년 사고) &nbsp;|&nbsp; "
    "상관계수: -0.547"
    "</div>",
    unsafe_allow_html=True,
)

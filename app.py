"""
스쿨존 안전 분석 대시보드 — 내 아이가 살기 좋은 동네
성남시 어린이 보호구역 142개소 안전등급 시각화

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
    color: #FFFFFF !important; font-size: 13px !important; font-weight: 500 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #FFFFFF !important; font-size: 28px !important; font-weight: 700 !important;
}
button[data-baseweb="tab"] {
    font-size: 15px !important; font-weight: 500 !important;
    color: #1A5276 !important; padding: 10px 24px !important;
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
    text-align: center; color: #566573; font-size: 12px; padding: 10px 0 20px 0;
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

FACILITY_COLS = [
    "도로적색표면", "신호등", "횡단보도", "도로안전표지",
    "생활안전CCTV", "무인교통단속카메라",
    "보호구역표지판", "옐로카펫", "무단횡단방지펜스",
]

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


@st.cache_data
def load_cv_features():
    return pd.read_csv(DATA_DIR / "커스텀비전_시설물별.csv", encoding="utf-8-sig")


@st.cache_data
def load_gyosan_schools():
    return pd.read_csv(DATA_DIR / "교산_교육시설.csv", encoding="utf-8-sig")


@st.cache_data
def load_gyosan_public():
    return pd.read_csv(DATA_DIR / "교산_공공청사.csv", encoding="utf-8-sig")


@st.cache_data
def load_accident_images():
    return pd.read_csv(DATA_DIR / "accidentlevel.csv", encoding="utf-8-sig")


@st.cache_resource
def train_safety_model():
    from sklearn.linear_model import LinearRegression
    _df = load_data()
    feat = ["도로적색표면", "신호등", "횡단보도", "도로안전표지",
            "생활안전CCTV", "무인교통단속카메라",
            "보호구역표지판", "옐로카펫", "무단횡단방지펜스",
            "발생건수", "어린이비율"]
    valid = _df.dropna(subset=feat + ["최종안전점수_V6"])
    X = valid[feat]
    y = valid["최종안전점수_V6"]
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    return model, feat, r2


@st.cache_resource
def train_structure_model():
    """1단계: 도로 구조 → 사고 부근 여부 (로지스틱 회귀)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    img_df = load_accident_images()
    img_df["accident_label"] = img_df["image"].str.contains("부근").astype(int)

    feat_cols = ["p_wide", "p_barrier_yes", "road_width_relative",
                 "sidewalk_ratio", "parked_density"]
    X = img_df[feat_cols].values
    y = img_df["accident_label"].values

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    cv_auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    pipe.fit(X, y)

    # 시설물별 평균 structure_risk
    img_df["structure_risk"] = pipe.predict_proba(X)[:, 1]
    facility_risk = img_df.groupby("시설물명")["structure_risk"].mean().reset_index()

    return pipe, float(cv_auc.mean()), facility_risk


@st.cache_resource
def train_integrated_model():
    """3단계: 구조위험 + 시설 + 어린이비율 → 사고 등급 (3-class 통합 모델)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    _df = load_data()
    _, _, fac_risk = train_structure_model()

    merged = _df.merge(fac_risk, on="시설물명", how="left")
    merged["structure_risk"] = merged["structure_risk"].fillna(
        merged["structure_risk"].median()
    )
    merged["accident_label"] = pd.cut(
        merged["발생건수"], bins=[-0.1, 0, 6, np.inf], labels=[0, 1, 2]
    ).astype(int)

    # 3개 추가 시설은 데이터 미수집 시설을 0으로 처리
    for _fc in ["보호구역표지판", "옐로카펫", "무단횡단방지펜스"]:
        merged[_fc] = merged[_fc].fillna(0)

    feat_cols = ["structure_risk"] + FACILITY_COLS + ["어린이비율"]
    valid = merged.dropna(subset=feat_cols)
    X = valid[feat_cols].values
    y = valid["accident_label"].values

    model = SkPipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.01, class_weight="balanced",
            solver="lbfgs", max_iter=1000, random_state=42,
        )),
    ])
    cv_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc_ovr_weighted")
    model.fit(X, y)

    # 위험(2) 클래스 계수 — class 2 = 7건+ 위험
    coef_df = pd.DataFrame({
        "변수": feat_cols,
        "계수": model.named_steps["lr"].coef_[2],
    }).sort_values("계수")

    return model, feat_cols, float(cv_auc.mean()), coef_df


# ──────────────────────────────────────────────
# 4. Helper Functions
# ──────────────────────────────────────────────

def make_popup(row):
    """마커 팝업 생성"""
    grade_key = row["등급"]
    color = GRADE_COLORS.get(grade_key, "#999")
    grade_label = GRADE_LABELS.get(grade_key, grade_key)
    return f"""
    <div style="font-family:'Noto Sans KR',sans-serif;width:260px;padding:4px;">
      <div style="font-size:15px;font-weight:700;color:#1B4F72;margin-bottom:4px;">
        {row['시설물명']}
        <span style="font-size:11px;color:#34495E;font-weight:400;margin-left:4px;">{row['시설유형']}</span>
      </div>
      <div style="display:inline-block;background:{color};color:#fff;
           padding:2px 10px;border-radius:20px;font-size:12px;font-weight:500;">
        {grade_label}
      </div>
      <span style="color:#2C3E50;font-size:13px;margin-left:6px;">
        {row['활성_안전점수']:.1f}점
      </span>
      <hr style="margin:8px 0;border:none;border-top:1px solid #D6EAF8;">
      <table style="font-size:11px;color:#2C3E50;width:100%;border-collapse:collapse;">
        <tr style="background:#F0F6FC;"><td colspan="2" style="padding:3px 4px;font-weight:600;color:#1B4F72;">점수 구조</td></tr>
        <tr><td style="padding:2px 4px;">가산점(시설)</td><td style="text-align:right;font-weight:600;">{row['가산점_시설_V6']:.1f}점</td></tr>
        <tr><td style="padding:2px 4px;">가산점(보너스)</td><td style="text-align:right;font-weight:600;">{int(row['가산점_보너스_V6'])}점</td></tr>
        <tr style="background:#FDEDEC;"><td style="padding:2px 4px;">감산점 합계</td><td style="text-align:right;font-weight:600;color:#E74C3C;">-{row['감산점_합계_V6']:.1f}점</td></tr>
        <tr><td style="padding:2px 4px;">기본점(50)</td><td style="text-align:right;">50.0점</td></tr>
        <tr style="background:#EBF5FB;"><td style="padding:2px 4px;font-weight:700;color:#1B4F72;">최종 안전점수</td><td style="text-align:right;font-weight:700;color:#1B4F72;">{row['활성_안전점수']:.1f}점</td></tr>
      </table>
      <hr style="margin:6px 0;border:none;border-top:1px solid #D6EAF8;">
      <table style="font-size:10px;color:#444;width:100%;border-collapse:collapse;">
        <tr><td>적색표면 {int(row['도로적색표면'])}</td><td>신호등 {int(row['신호등'])}</td><td>횡단보도 {int(row['횡단보도'])}</td></tr>
        <tr><td>안전표지 {int(row['도로안전표지'])}</td><td>CCTV {int(row['생활안전CCTV'])}</td><td>카메라 {int(row['무인교통단속카메라'])}</td></tr>
        <tr><td>표지판 {int(row.get('보호구역표지판', 0))}</td><td>옐로카펫 {int(row.get('옐로카펫', 0))}</td><td>펜스 {int(row.get('무단횡단방지펜스', 0))}</td></tr>
        <tr><td>발생건수 {int(row['발생건수'])}건 ({"안전" if row["발생건수"] == 0 else "주의" if row["발생건수"] <= 6 else "위험"})</td><td>어린이비율 {row['어린이비율']:.1f}%</td><td>구조위험 {row.get('structure_risk', 0):.0%}</td></tr>
      </table>
      {"" if pd.isna(row.get('CV_도로폭확률')) else f'''
      <hr style="margin:6px 0;border:none;border-top:1px solid #E8DAEF;">
      <table style="font-size:10px;color:#444;width:100%;border-collapse:collapse;">
        <tr style="background:#F5EEF8;"><td colspan="2" style="padding:3px 4px;font-weight:600;color:#6C3483;">도로환경 (CV)</td></tr>
        <tr><td style="padding:2px 4px;">넓은도로</td><td style="text-align:right;">{row["CV_도로폭확률"]:.0%}</td></tr>
        <tr><td style="padding:2px 4px;">분리장치</td><td style="text-align:right;">{row["CV_분리장치확률"]:.0%}</td></tr>
        <tr><td style="padding:2px 4px;">주정차</td><td style="text-align:right;">{row["CV_주정차밀도"]:.1f}대</td></tr>
      </table>
      '''}
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


def create_map(filtered_df, overlay_flags, pop_df, geo):
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
            popup=folium.Popup(make_popup(row), max_width=290),
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

# CV 데이터 병합
cv_df = load_cv_features()
df = df.merge(cv_df, on="시설물명", how="left")

# 3개 추가 시설 NaN → 0 처리
for _fc in ["보호구역표지판", "옐로카펫", "무단횡단방지펜스"]:
    if _fc in df.columns:
        df[_fc] = df[_fc].fillna(0)

# structure_risk 병합
_, _, _fac_risk = train_structure_model()
df = df.merge(_fac_risk, on="시설물명", how="left")
df["structure_risk"] = df["structure_risk"].fillna(df["structure_risk"].median())

st.sidebar.markdown(
    "<h2 style='text-align:center;margin-bottom:0;'>스쿨존 안전 분석</h2>"
    "<p style='text-align:center;opacity:0.6;font-size:13px;'>성남시 어린이 보호구역</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

# ── 안전점수 설정 ──
df["활성_안전점수"] = df["최종안전점수_V6"]
df["등급"] = df["등급_V6"]
df["안전등급"] = df["등급_V6"].map(GRADE_LABELS)

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
csv_cols = ["시설물명", "시설유형", "구", "안전등급", "활성_안전점수",
            "가산점_시설_V6", "가산점_보너스_V6", "감산점_합계_V6"]
csv_cols += FACILITY_COLS + ["발생건수", "어린이비율"]
csv_cols = [c for c in csv_cols if c in df.columns]
csv_export = df[csv_cols].copy().rename(columns={
    "활성_안전점수": "안전점수",
    "가산점_시설_V6": "가산점_시설",
    "가산점_보너스_V6": "가산점_보너스",
    "감산점_합계_V6": "감산점_합계",
})
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
st.markdown("""
<div style="margin-bottom:8px;">
    <span style="font-size:36px;font-weight:700;color:#1B4F72;">
        내 아이가 살기 좋은 동네
    </span>
    <span style="font-size:14px;color:#34495E;margin-left:12px;">
        성남시 어린이 보호구역 142개소 안전 분석 대시보드
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
tab_map, tab_analysis, tab_cv, tab_district, tab_sim = st.tabs(["지도", "상세분석", "도로환경 (CV)", "동네정보", "교산 시뮬레이션"])

# ============================
# Tab 1: 지도
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
    st.markdown("##### 점수 구조: 기본(50) + 가산점(시설+보너스) - 감산점")
    score_struct = pd.DataFrame({
        "항목": ["가산점(시설)", "가산점(보너스)", "감산점 합계"],
        "평균": [
            df["가산점_시설_V6"].mean(),
            df["가산점_보너스_V6"].mean(),
            -df["감산점_합계_V6"].mean(),
        ],
    })
    fig_struct = px.bar(
        score_struct, x="항목", y="평균",
        title="안전점수 구성 요소 평균",
        color="항목",
        color_discrete_map={
            "가산점(시설)": "#2E86C1",
            "가산점(보너스)": "#5DADE2",
            "감산점 합계": "#E74C3C",
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
            title="안전점수 분포",
            labels={"활성_안전점수": "안전점수"},
            color_discrete_sequence=["#2E86C1"],
        )
        fig_hist.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="시설 수", bargap=0.08)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_radar:
        if selected_school != "(전체)":
            school_row = df[df["시설물명"] == selected_school].iloc[0]

            radar_feats = FACILITY_COLS
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
                f"<span style='color:#2C3E50;'>안전점수: <b>{school_row['활성_안전점수']:.1f}</b></span> &nbsp; "
                f"<span style='color:#2C3E50;'>발생건수: <b>{int(school_row['발생건수'])}</b>건</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#F0F6FC;padding:20px;border-radius:8px;"
                "text-align:center;color:#1A5276;margin-top:40px;'>"
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
            for f in FACILITY_COLS:
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
                f'<span style="font-size:11px;color:#34495E;">({row["시설유형"]})</span> &nbsp; '
                f'<span style="background:{grade_color};color:#fff;padding:2px 10px;'
                f'border-radius:20px;font-size:11px;">D ({row["활성_안전점수"]:.1f}점)</span>'
                f'<div class="suggestion">개선 제안: {suggestion}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.success("모든 시설이 양호한 상태입니다.")

    # ── 정책 시뮬레이션: 시설물 추가 효과 ──
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 정책 시뮬레이션: 시설물 추가 효과")
    st.caption("선택한 시설에 시설물 1개를 추가할 때 위험(7건+) 확률 변화량을 예측합니다.")

    _integ_model_pol, integ_feats_pol, _, _ = train_integrated_model()

    if selected_school != "(전체)":
        pol_row = df[df["시설물명"] == selected_school]
        if len(pol_row) > 0:
            pol_row = pol_row.iloc[0]
            pol_base = pol_row[integ_feats_pol].values.reshape(1, -1)
            pol_base_prob = float(_integ_model_pol.predict_proba(pol_base)[0, 2])

            pol_results = []
            for i, feat in enumerate(integ_feats_pol):
                if feat in FACILITY_COLS:
                    pol_modified = pol_base.copy()
                    pol_modified[0, i] += 1
                    pol_new_prob = float(_integ_model_pol.predict_proba(pol_modified)[0, 2])
                    pol_delta = pol_new_prob - pol_base_prob
                    pol_results.append({
                        "시설물": feat,
                        "현재 수량": int(pol_row[feat]),
                        "현재 위험확률": pol_base_prob,
                        "추가 후 위험확률": pol_new_prob,
                        "변화량 (%p)": pol_delta,
                    })

            pol_df = pd.DataFrame(pol_results).sort_values("변화량 (%p)")

            # TOP 3 효과적 시설
            top3 = pol_df.head(3)
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#D4EFDF,#EBF5FB);'
                f'padding:14px 18px;border-radius:10px;border-left:4px solid #27AE60;">'
                f'<b style="color:#1B4F72;">{selected_school}</b> — '
                f'현재 위험확률: <b>{pol_base_prob:.1%}</b><br>'
                f'<span style="font-size:13px;color:#2C3E50;">'
                f'위험확률 감소 TOP 3: '
                + " / ".join(
                    f'<b>{r["시설물"]}</b> +1 → {r["변화량 (%p)"]:+.1%}p'
                    for _, r in top3.iterrows()
                )
                + '</span></div>',
                unsafe_allow_html=True,
            )

            # 전체 효과 차트
            fig_pol = go.Figure()
            fig_pol.add_trace(go.Bar(
                y=pol_df["시설물"], x=pol_df["변화량 (%p)"] * 100,
                orientation="h",
                marker_color=["#27AE60" if v < 0 else "#E74C3C" for v in pol_df["변화량 (%p)"]],
                text=[f"{v*100:+.2f}%p" for v in pol_df["변화량 (%p)"]],
                textposition="outside",
            ))
            fig_pol.add_vline(x=0, line_color="#555", line_width=1)
            fig_pol.update_layout(
                **PLOTLY_LAYOUT, height=350,
                title=f"{selected_school}: 시설물 +1개 추가 시 위험확률 변화",
                xaxis=dict(title="위험확률 변화 (%p)"),
                yaxis=dict(title=""),
            )
            st.plotly_chart(fig_pol, use_container_width=True)
        else:
            st.info("선택한 시설의 데이터를 찾을 수 없습니다.")
    else:
        st.markdown(
            "<div style='background:#D4EFDF;padding:20px;border-radius:8px;"
            "text-align:center;color:#27AE60;'>"
            "사이드바에서 개별 시설을 선택하면<br>시설물 추가 효과 시뮬레이션이 표시됩니다."
            "</div>",
            unsafe_allow_html=True,
        )


# ============================
# Tab 3: 도로환경 (CV)
# ============================
with tab_cv:
    st.markdown("### 캡스톤 연구: 3단계 사고 예측 모델")
    st.caption(
        "핵심 메시지: 스쿨존 사고는 구조 + 정책(시설) + 노출(어린이)의 결합 결과이다. "
        "도로 구조 단독으로는 설명력이 부족하며, 정책 시설 투입이 핵심이다."
    )

    # ── 모델 학습 ──
    _struct_model, struct_auc, _fac_risk_cv = train_structure_model()
    _integ_model, integ_feats, integ_auc, integ_coef = train_integrated_model()

    # ── (a) 3단계 모델 AUC 비교 차트 ──
    st.markdown("##### 3단계 모델 AUC 비교")
    auc_data = pd.DataFrame({
        "단계": [
            "1단계\n구조 (CV 이미지)",
            "2단계\n외부검증 (구조→시설데이터)",
            "3단계\n통합 (구조+시설+노출)",
        ],
        "AUC": [struct_auc, 0.39, integ_auc],
        "색상": ["#F39C12", "#E74C3C", "#27AE60"],
    })
    fig_auc = go.Figure()
    fig_auc.add_trace(go.Bar(
        x=auc_data["단계"], y=auc_data["AUC"],
        marker_color=auc_data["색상"].tolist(),
        text=[f"{v:.3f}" for v in auc_data["AUC"]],
        textposition="outside", textfont=dict(size=16, color="#1B4F72"),
    ))
    fig_auc.add_hline(y=0.5, line_dash="dash", line_color="#E74C3C",
                      annotation_text="무작위 기준선 (0.5)", annotation_position="top left")
    fig_auc.update_layout(
        **PLOTLY_LAYOUT, height=380,
        title="3단계 모델 성능 비교 (5-Fold CV Weighted OVR AUC)",
        yaxis=dict(title="ROC-AUC", range=[0, 1]),
        xaxis=dict(title=""),
    )
    st.plotly_chart(fig_auc, use_container_width=True)

    st.markdown(
        '<div style="background:linear-gradient(135deg,#EBF5FB,#D4EFDF);padding:14px 18px;'
        'border-radius:10px;border-left:4px solid #27AE60;margin-bottom:16px;">'
        '<b style="color:#1B4F72;">핵심 발견:</b> '
        f'도로 구조만으로는 AUC {struct_auc:.3f}로 제한적이며, '
        '구조 위험도를 시설 데이터에 직접 적용하면 AUC 0.39로 오히려 하락합니다. '
        f'그러나 <b>구조 + 9개 시설 + 어린이비율</b>을 통합하면 3-class Weighted AUC <b>{integ_auc:.3f}</b>으로 '
        '유의미한 예측력을 확보합니다. (라벨: 안전 0건 / 주의 1~6건 / 위험 7건+)'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── (b) 통합 모델 계수 해석 차트 ──
    st.markdown("##### 통합 모델 변수 계수 해석")
    coef_sorted = integ_coef.sort_values("계수")
    coef_colors = ["#E74C3C" if c > 0 else "#2E86C1" for c in coef_sorted["계수"]]

    fig_coef = go.Figure()
    fig_coef.add_trace(go.Bar(
        y=coef_sorted["변수"], x=coef_sorted["계수"],
        orientation="h",
        marker_color=coef_colors,
        text=[f"{c:+.3f}" for c in coef_sorted["계수"]],
        textposition="outside",
    ))
    fig_coef.add_vline(x=0, line_color="#555", line_width=1)
    fig_coef.update_layout(
        **PLOTLY_LAYOUT, height=420,
        title="통합 모델 — 위험(7건+) 예측 변수 계수 (양수=위험 증가 / 음수=보호 효과)",
        xaxis=dict(title="계수"),
        yaxis=dict(title=""),
    )
    st.plotly_chart(fig_coef, use_container_width=True)

    # 계수 인사이트 카드
    pos_vars = coef_sorted[coef_sorted["계수"] > 0].sort_values("계수", ascending=False)
    neg_vars = coef_sorted[coef_sorted["계수"] < 0].sort_values("계수")
    cv_ins1, cv_ins2 = st.columns(2)
    with cv_ins1:
        st.markdown(
            '<div style="background:#FDEDEC;padding:12px 16px;border-radius:8px;'
            'border-left:4px solid #E74C3C;">'
            '<b style="color:#C0392B;">위험 증가 (+) 변수</b><br>'
            '<span style="font-size:13px;color:#2C3E50;">'
            + "<br>".join(f"{r['변수']}: {r['계수']:+.3f}" for _, r in pos_vars.iterrows())
            + '<br><br>사고가 많은 곳에 사후 설치되는 패턴</span></div>',
            unsafe_allow_html=True,
        )
    with cv_ins2:
        st.markdown(
            '<div style="background:#EBF5FB;padding:12px 16px;border-radius:8px;'
            'border-left:4px solid #2E86C1;">'
            '<b style="color:#1B4F72;">보호 효과 (-) 변수</b><br>'
            '<span style="font-size:13px;color:#2C3E50;">'
            + "<br>".join(f"{r['변수']}: {r['계수']:+.3f}" for _, r in neg_vars.iterrows())
            + '<br><br>예방적 시설의 사고 억제 효과</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── (c) 시설별 사고 확률 예측 (3단계) ──
    st.markdown("##### 시설별 사고 확률 예측 — 3단계 (통합 모델)")

    df_for_prob = df.dropna(subset=integ_feats).copy()
    if len(df_for_prob) > 0:
        X_prob = df_for_prob[integ_feats].values
        _proba = _integ_model.predict_proba(X_prob)
        df_for_prob["안전확률"] = _proba[:, 0]
        df_for_prob["주의확률"] = _proba[:, 1]
        df_for_prob["위험확률"] = _proba[:, 2]

        prob_display = df_for_prob.nlargest(20, "위험확률")[
            ["시설물명", "시설유형", "구", "등급",
             "안전확률", "주의확률", "위험확률", "발생건수"]
        ].copy()
        prob_display.columns = [
            "시설물명", "시설유형", "구", "등급",
            "안전확률", "주의확률", "위험확률", "실제발생건수",
        ]
        prob_display["안전확률"] = prob_display["안전확률"].map("{:.1%}".format)
        prob_display["주의확률"] = prob_display["주의확률"].map("{:.1%}".format)
        prob_display["위험확률"] = prob_display["위험확률"].map("{:.1%}".format)
        st.dataframe(prob_display, use_container_width=True, hide_index=True)
        st.caption(f"위험확률 상위 20개소 (전체 {len(df_for_prob)}개소 분석)")

    st.markdown("---")

    # ── (d) 기존 유지: 등급별 CV 특성 + 레이더 ──
    st.markdown("##### 등급별 도로환경 특성 (CV)")
    cv_cols = ["CV_도로폭확률", "CV_분리장치확률", "CV_도로상대폭", "CV_보행공간비율", "CV_주정차밀도"]
    cv_labels = ["넓은 도로 확률", "분리장치 확률", "도로 상대폭", "보행공간 비율", "주정차 밀도"]
    df_cv = df.dropna(subset=["CV_도로폭확률"])

    cv_col1, cv_col2 = st.columns(2)

    with cv_col1:
        cv_grade = df_cv.groupby("등급")[cv_cols].mean()
        cv_grade.columns = cv_labels
        cv_grade = cv_grade.reindex(["A", "B", "C", "D"])
        cv_melt = cv_grade.reset_index().melt(
            id_vars="등급", var_name="특성", value_name="값",
        )
        fig_cv_grade = px.bar(
            cv_melt[cv_melt["특성"].isin(["넓은 도로 확률", "분리장치 확률"])],
            x="등급", y="값", color="특성", barmode="group",
            title="등급별 도로폭 vs 분리장치 확률",
            color_discrete_map={
                "넓은 도로 확률": "#E74C3C", "분리장치 확률": "#2E86C1",
            },
        )
        fig_cv_grade.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig_cv_grade, use_container_width=True)

    with cv_col2:
        if selected_school != "(전체)" and selected_school in df_cv["시설물명"].values:
            cv_row = df_cv[df_cv["시설물명"] == selected_school].iloc[0]
            cv_avg = df_cv[cv_cols].mean()

            cv_radar_vals = [
                cv_row["CV_도로폭확률"] * 100,
                cv_row["CV_분리장치확률"] * 100,
                cv_row["CV_도로상대폭"] * 200,
                cv_row["CV_보행공간비율"] * 300,
                min(cv_row["CV_주정차밀도"] * 20, 100),
            ]
            cv_avg_vals = [
                cv_avg["CV_도로폭확률"] * 100,
                cv_avg["CV_분리장치확률"] * 100,
                cv_avg["CV_도로상대폭"] * 200,
                cv_avg["CV_보행공간비율"] * 300,
                min(cv_avg["CV_주정차밀도"] * 20, 100),
            ]
            cv_theta = cv_labels + [cv_labels[0]]

            fig_cv_radar = go.Figure()
            fig_cv_radar.add_trace(go.Scatterpolar(
                r=cv_radar_vals + [cv_radar_vals[0]],
                theta=cv_theta,
                fill="toself", name=selected_school,
                fillcolor="rgba(108,52,131,0.2)",
                line=dict(color="#6C3483", width=2),
            ))
            fig_cv_radar.add_trace(go.Scatterpolar(
                r=cv_avg_vals + [cv_avg_vals[0]],
                theta=cv_theta,
                fill="toself", name="전체 평균",
                fillcolor="rgba(46,134,193,0.1)",
                line=dict(color="#2E86C1", width=1, dash="dash"),
            ))
            fig_cv_radar.update_layout(
                **PLOTLY_LAYOUT,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#E8DAEF"),
                    angularaxis=dict(gridcolor="#E8DAEF"),
                    bgcolor="#FAFCFF",
                ),
                title=f"{selected_school} 도로환경 프로필",
                height=380, showlegend=True,
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_cv_radar, use_container_width=True)
        else:
            st.markdown(
                "<div style='background:#F5EEF8;padding:20px;border-radius:8px;"
                "text-align:center;color:#6C3483;margin-top:40px;'>"
                "사이드바에서 개별 시설을 선택하면<br>도로환경 레이더 차트가 표시됩니다."
                "</div>",
                unsafe_allow_html=True,
            )


# ============================
# Tab 4: 동네정보
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


# ============================
# Tab 4: 교산 시뮬레이션
# ============================
with tab_sim:
    st.markdown("### 교산 신도시 스쿨존 시뮬레이션")
    st.caption(
        "2기 신도시(성남시) 142개소 분석 결과를 학습하여, "
        "3기 신도시(교산) 계획 학교의 예상 안전점수를 시뮬레이션합니다."
    )

    gs_schools = load_gyosan_schools()
    gs_public = load_gyosan_public()
    safety_model, model_features, model_r2 = train_safety_model()

    # 등급 기준선 (V6 사분위수)
    gs_q1, gs_q2, gs_q3 = df["최종안전점수_V6"].quantile([0.25, 0.5, 0.75]).values

    def classify_grade(score):
        if score >= gs_q3:
            return "A"
        if score >= gs_q2:
            return "B"
        if score >= gs_q1:
            return "C"
        return "D"

    # ── (a) 교산 지도 ──
    gs_col_map, gs_col_info = st.columns([3, 2])

    GS_TYPE_STYLE = {
        "초등학교": ("#1B4F72", 10),
        "유치원": ("#5DADE2", 6),
        "중학교": ("#2E86C1", 8),
        "고등학교": ("#85C1E9", 8),
        "특수학교": ("#F39C12", 8),
    }

    with gs_col_map:
        gs_map = folium.Map(
            location=[37.515, 127.200], zoom_start=13, tiles="cartodbpositron",
        )
        for _, gs_r in gs_schools.iterrows():
            gs_color, gs_radius = GS_TYPE_STYLE.get(gs_r["시설유형"], ("#999", 6))
            folium.CircleMarker(
                [gs_r["위도"], gs_r["경도"]],
                radius=gs_radius, color="#fff", weight=2,
                fill=True, fill_color=gs_color, fill_opacity=0.9,
                tooltip=f"{gs_r['시설물명']} ({gs_r['시설유형']})",
            ).add_to(gs_map)

        if st.checkbox("공공청사 표시", value=False, key="gs_pub_ov"):
            for _, gs_r in gs_public.iterrows():
                folium.Marker(
                    [gs_r["위도"], gs_r["경도"]],
                    icon=folium.Icon(color="gray", icon="building", prefix="fa"),
                    tooltip=gs_r["시설물명"],
                ).add_to(gs_map)

        st_folium(gs_map, height=450, use_container_width=True, returned_objects=[])

    with gs_col_info:
        st.markdown("##### 계획 교육시설")
        gs_type_counts = gs_schools["시설유형"].value_counts()
        for gs_t, gs_n in gs_type_counts.items():
            gs_tc, _ = GS_TYPE_STYLE.get(gs_t, ("#999", 6))
            st.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                f'background:{gs_tc};margin-right:6px;vertical-align:middle;"></span>'
                f'{gs_t}: **{gs_n}**개',
                unsafe_allow_html=True,
            )
        st.markdown(f"합계: **{len(gs_schools)}**개소")
        st.markdown(f"<br>안전점수 모델 R² = **{model_r2:.3f}**", unsafe_allow_html=True)
        st.markdown(f"사고확률 모델 AUC = **{integ_auc:.3f}**", unsafe_allow_html=True)
        st.caption("성남시 142개소 데이터 기반")

    st.markdown("---")

    # ── (b)(c) 시나리오 시뮬레이션 ──
    st.markdown("##### 시설물 시나리오 시뮬레이션")

    SIM_SCENARIOS = {
        "최소 설치 (D등급 평균)": [7, 4, 2, 2, 20, 2, 15, 0, 5],
        "표준 설치 (전체 평균)": [10, 5, 4, 4, 19, 2, 20, 1, 10],
        "권장 설치 (A등급 평균)": [13, 8, 6, 6, 20, 2, 30, 2, 15],
    }

    gs_col_sc, gs_col_ct = st.columns([1, 2])

    with gs_col_sc:
        gs_scenario = st.radio(
            "시나리오", list(SIM_SCENARIOS.keys()) + ["직접 설정"], key="gs_radio",
        )

    with gs_col_ct:
        if gs_scenario == "직접 설정":
            gs_sl_cols = st.columns(3)
            gs_fvals = []
            for gs_i, gs_f in enumerate(FACILITY_COLS):
                with gs_sl_cols[gs_i % 3]:
                    gs_fvals.append(
                        st.slider(gs_f, 0, int(df[gs_f].max()) + 10,
                                  int(df[gs_f].mean()), key=f"gs_{gs_f}")
                    )
        else:
            gs_fvals = list(SIM_SCENARIOS[gs_scenario])
            st.markdown(
                "**설정 시설물:** "
                + " / ".join(f"{f}: **{v}**" for f, v in zip(FACILITY_COLS, gs_fvals))
            )

        gs_child = st.slider(
            "예상 어린이 비율 (%)", 5.0, 25.0, 15.0, 0.5, key="gs_child",
        )

    # 안전점수 예측 (LinearRegression)
    gs_input = dict(zip(
        model_features,
        list(gs_fvals) + [0, gs_child],
    ))
    gs_X = pd.DataFrame([gs_input])
    gs_pred = max(0.0, min(100.0, float(safety_model.predict(gs_X)[0])))
    gs_grade = classify_grade(gs_pred)

    # 사고확률 예측 (LogisticRegression 통합 3-class 모델)
    gs_sr = df["structure_risk"].median()  # 신도시 → 중앙값 사용
    gs_integ_input = dict(zip(
        integ_feats,
        [gs_sr] + list(gs_fvals) + [gs_child],
    ))
    gs_integ_X = pd.DataFrame([gs_integ_input])
    gs_proba = _integ_model.predict_proba(gs_integ_X)[0]
    gs_safe_prob = float(gs_proba[0])
    gs_caution_prob = float(gs_proba[1])
    gs_danger_prob = float(gs_proba[2])

    st.markdown("---")
    st.markdown("##### 시뮬레이션 결과")

    gs_k1, gs_k2, gs_k3, gs_k4, gs_k5, gs_k6 = st.columns(6)
    gs_k1.metric("예상 안전점수", f"{gs_pred:.1f}")
    gs_k2.metric("예상 등급", GRADE_LABELS[gs_grade])
    gs_k3.metric("안전확률", f"{gs_safe_prob:.1%}")
    gs_k4.metric("주의확률", f"{gs_caution_prob:.1%}")
    gs_k5.metric("위험확률", f"{gs_danger_prob:.1%}")
    gs_k6.metric("어린이 비율", f"{gs_child:.1f}%")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── 시설물 민감도 분석 ──
    st.markdown("##### 시설물 +1개 효과 분석")
    st.caption("현재 시나리오에서 각 시설물을 1개 추가했을 때 안전점수·위험확률 변화량")

    gs_sens = []
    for gs_si, gs_sf in enumerate(FACILITY_COLS):
        # 안전점수 민감도 (LinearRegression)
        gs_inp_plus = gs_input.copy()
        gs_inp_plus[gs_sf] = gs_inp_plus[gs_sf] + 1
        gs_pred_plus = max(0.0, min(100.0, float(
            safety_model.predict(pd.DataFrame([gs_inp_plus]))[0]
        )))
        # 위험확률 민감도 (LogisticRegression 3-class)
        gs_int_plus = gs_integ_input.copy()
        gs_int_plus[gs_sf] = gs_int_plus[gs_sf] + 1
        gs_risk_plus = float(
            _integ_model.predict_proba(pd.DataFrame([gs_int_plus]))[0, 2]
        )
        gs_sens.append({
            "시설물": gs_sf,
            "안전점수 변화": gs_pred_plus - gs_pred,
            "위험확률 변화(%p)": (gs_risk_plus - gs_danger_prob) * 100,
        })

    gs_sens_df = pd.DataFrame(gs_sens).sort_values("위험확률 변화(%p)")

    gs_sc1, gs_sc2 = st.columns(2)
    with gs_sc1:
        fig_sens_sc = go.Figure()
        fig_sens_sc.add_trace(go.Bar(
            y=gs_sens_df["시설물"], x=gs_sens_df["안전점수 변화"],
            orientation="h",
            marker_color=[
                "#27AE60" if v > 0 else "#E74C3C"
                for v in gs_sens_df["안전점수 변화"]
            ],
            text=[f"{v:+.2f}" for v in gs_sens_df["안전점수 변화"]],
            textposition="outside",
        ))
        fig_sens_sc.add_vline(x=0, line_color="#555", line_width=1)
        fig_sens_sc.update_layout(
            **PLOTLY_LAYOUT, height=350,
            title="시설물 +1개 → 안전점수 변화",
            xaxis=dict(title="안전점수 변화"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_sens_sc, use_container_width=True)

    with gs_sc2:
        fig_sens_rk = go.Figure()
        fig_sens_rk.add_trace(go.Bar(
            y=gs_sens_df["시설물"], x=gs_sens_df["위험확률 변화(%p)"],
            orientation="h",
            marker_color=[
                "#27AE60" if v < 0 else "#E74C3C"
                for v in gs_sens_df["위험확률 변화(%p)"]
            ],
            text=[f"{v:+.2f}%p" for v in gs_sens_df["위험확률 변화(%p)"]],
            textposition="outside",
        ))
        fig_sens_rk.add_vline(x=0, line_color="#555", line_width=1)
        fig_sens_rk.update_layout(
            **PLOTLY_LAYOUT, height=350,
            title="시설물 +1개 → 위험확률 변화",
            xaxis=dict(title="위험확률 변화 (%p)"),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig_sens_rk, use_container_width=True)

    # 우선 투자 추천 TOP 3
    gs_top3_sens = gs_sens_df.head(3)
    st.markdown(
        '<div style="background:linear-gradient(135deg,#D4EFDF,#EBF5FB);'
        'padding:14px 18px;border-radius:10px;border-left:4px solid #27AE60;">'
        '<b style="color:#1B4F72;">우선 투자 추천 TOP 3</b>'
        ' — 위험확률 감소 효과 기준<br>'
        '<span style="font-size:13px;color:#2C3E50;">'
        + " / ".join(
            f'<b>{r["시설물"]}</b> +1 → 점수 {r["안전점수 변화"]:+.1f}'
            f' · 위험 {r["위험확률 변화(%p)"]:+.1f}%p'
            for _, r in gs_top3_sens.iterrows()
        )
        + '</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── (d) 성남시 대비 비교 차트 ──
    gs_col_c1, gs_col_c2 = st.columns(2)

    gs_gavg_score = (
        df.groupby("등급_V6")["최종안전점수_V6"].mean().reindex(["A", "B", "C", "D"])
    )
    gs_gavg_fac = (
        df.groupby("등급_V6")[FACILITY_COLS].mean().reindex(["A", "B", "C", "D"])
    )

    with gs_col_c1:
        gs_scomp = pd.DataFrame({
            "구분": ["교산 시뮬레이션"]
                    + [f"성남시 {g}등급" for g in ["A", "B", "C", "D"]],
            "안전점수": [gs_pred] + gs_gavg_score.tolist(),
        })
        fig_gs_sc = px.bar(
            gs_scomp, x="구분", y="안전점수",
            title="예상 안전점수 vs 성남시 등급 평균",
            color="구분",
            color_discrete_sequence=[
                "#F39C12", "#154360", "#2471A3", "#85C1E9", "#E74C3C",
            ],
            text="안전점수",
        )
        fig_gs_sc.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_gs_sc.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
        st.plotly_chart(fig_gs_sc, use_container_width=True)

    with gs_col_c2:
        gs_fcomp = []
        for gs_f, gs_v in zip(FACILITY_COLS, gs_fvals):
            gs_fcomp.append({"시설물": gs_f, "구분": "교산", "수량": gs_v})
            gs_fcomp.append({
                "시설물": gs_f, "구분": "성남 A등급",
                "수량": round(gs_gavg_fac.loc["A", gs_f], 1),
            })
            gs_fcomp.append({
                "시설물": gs_f, "구분": "성남 D등급",
                "수량": round(gs_gavg_fac.loc["D", gs_f], 1),
            })
        fig_gs_fc = px.bar(
            pd.DataFrame(gs_fcomp), x="시설물", y="수량",
            color="구분", barmode="group",
            title="시설물 수량: 교산 vs 성남시 A/D등급",
            color_discrete_map={
                "교산": "#F39C12", "성남 A등급": "#154360", "성남 D등급": "#E74C3C",
            },
        )
        fig_gs_fc.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig_gs_fc, use_container_width=True)

    # ── (e) A등급 달성 권장 시설물 테이블 ──
    st.markdown("##### A등급 달성 권장 시설물")
    gs_rec = []
    for gs_f, gs_v in zip(FACILITY_COLS, gs_fvals):
        gs_aavg = gs_gavg_fac.loc["A", gs_f]
        gs_gap = gs_aavg - gs_v
        gs_rec.append({
            "시설물": gs_f,
            "현재 설정": gs_v,
            "A등급 평균": round(gs_aavg, 1),
            "추가 필요": max(0, int(round(gs_gap))),
            "상태": "충족" if gs_v >= gs_aavg else "미달",
        })
    st.dataframe(pd.DataFrame(gs_rec), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# 7. Footer
# ──────────────────────────────────────────────
st.markdown(
    '<div class="footer-text">'
    "데이터 출처: 공공데이터포털, 도로교통공단, 경기데이터드림, 성남시"
    "</div>",
    unsafe_allow_html=True,
)

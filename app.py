"""
스쿨존 안전 분석 대시보드 — 내 아이가 살기 좋은 동네
성남시 142개소 + 광명시 51개소 어린이 보호구역 안전등급 시각화

실행: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import FastMarkerCluster, MiniMap, Fullscreen, MeasureControl
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# ──────────────────────────────────────────────
# 1. Page Config & Custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="스쿨존 안전 분석 — 성남시·광명시",
    page_icon="\U0001f3eb",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

/* ── 사이드바: 따뜻한 차콜 ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
}
section[data-testid="stSidebar"] * { color: #FEF9E7 !important; }
section[data-testid="stSidebar"] .stMultiSelect > div > div,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(243,156,18,0.35) !important;
    border-radius: 8px !important;
}

/* ── 헤딩: 스쿨존 다크그린 ── */
h1 { color: #1E8449 !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2, h3 { color: #2C3E50 !important; font-weight: 600 !important; }

/* ── KPI 카드: 앰버-옐로 그라데이션 ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #F39C12, #F1C40F);
    padding: 16px 20px; border-radius: 12px;
    box-shadow: 0 4px 15px rgba(243,156,18,0.30);
}
div[data-testid="stMetric"] label {
    color: #2C3E50 !important; font-size: 13px !important; font-weight: 500 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #2C3E50 !important; font-size: 28px !important; font-weight: 700 !important;
}

/* ── 탭: 앰버 액센트 ── */
button[data-baseweb="tab"] {
    font-size: 15px !important; font-weight: 500 !important;
    color: #566573 !important; padding: 10px 24px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #E67E22 !important; border-bottom: 3px solid #F39C12 !important;
}

/* ── 데이터프레임 ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #F5CBA7; border-radius: 8px; overflow: hidden;
}
section[data-testid="stSidebar"] .stCheckbox label span { font-size: 14px !important; }
.footer-text {
    text-align: center; color: #566573; font-size: 12px; padding: 10px 0 20px 0;
}

/* ── 경고 배너 ── */
.warning-banner {
    background: linear-gradient(135deg, #FDEDEC, #F9EBEA);
    border-left: 4px solid #E74C3C; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 16px;
}
.warning-banner b { color: #C0392B; }
.warning-banner span { color: #555; font-size: 13px; }

/* ── 제안 카드 ── */
.suggestion-card {
    background: #FEF9E7; border-radius: 8px; padding: 14px 16px;
    border-left: 4px solid #F39C12; margin-bottom: 8px;
}
.suggestion-card .school-name { color: #2C3E50; font-weight: 700; font-size: 15px; }
.suggestion-card .suggestion { color: #E67E22; font-size: 13px; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 2. Constants
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

GRADE_COLORS = {"A": "#27AE60", "B": "#F1C40F", "C": "#E67E22", "D": "#E74C3C"}
GRADE_LABELS = {"A": "A (우수)", "B": "B (양호)", "C": "C (보통)", "D": "D (주의)"}

CITY_CONFIG = {
    "성남시": {
        "center": [37.42, 127.13],
        "zoom": 12,
        "geojson": "성남시_행정동_경계.geojson",
        "population": "연령별인구_성남시_행정동.csv",
    },
    "광명시": {
        "center": [37.445, 126.870],
        "zoom": 13,
        "geojson": "광명시_행정동_경계.geojson",
        "population": "광명시_인구_행정동.csv",
    },
}

FACILITY_COLS = [
    "도로적색표면", "신호등", "횡단보도", "도로안전표지",
    "생활안전CCTV", "무인교통단속카메라",
    "보호구역표지판", "옐로카펫", "무단횡단방지펜스",
]

PLOTLY_LAYOUT = dict(
    font=dict(family="Noto Sans KR, sans-serif"),
    plot_bgcolor="#FFFDF5",
    paper_bgcolor="#FFFFFF",
    title_font=dict(size=18, color="#2C3E50"),
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
def load_red_surface():
    return pd.read_csv(DATA_DIR / "도로적색표면_전처리1.csv", encoding="utf-8-sig")

@st.cache_data
def load_traffic_lights():
    return pd.read_csv(DATA_DIR / "신호등_전처리1.csv", encoding="utf-8-sig")

@st.cache_data
def load_crosswalks():
    return pd.read_csv(DATA_DIR / "횡단보도_전처리1.csv", encoding="utf-8-sig")

@st.cache_data
def load_zone_signs():
    return pd.read_csv(DATA_DIR / "보호구역표지판_전처리1.csv", encoding="utf-8-sig")

@st.cache_data
def load_yellow_carpet():
    return pd.read_csv(DATA_DIR / "옐로카펫_전처리1.csv", encoding="utf-8-sig")

@st.cache_data
def load_fences():
    return pd.read_csv(DATA_DIR / "무단횡단방지펜스_전처리1.csv", encoding="utf-8-sig")


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
def load_gwangmyung():
    _gm = pd.read_csv(DATA_DIR / "광명_스쿨존.csv", encoding="utf-8-sig")
    for _fc in FACILITY_COLS:
        if _fc not in _gm.columns:
            _gm[_fc] = 0
    if "구" not in _gm.columns:
        _gm["구"] = "광명시"
    if "시설유형" not in _gm.columns:
        _gm["시설유형"] = "초등학교"
    return _gm


@st.cache_data
def load_gm_geojson():
    path = DATA_DIR / "광명시_행정동_경계.geojson"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_gm_population():
    path = DATA_DIR / "광명시_인구_행정동.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame()


@st.cache_data
def load_accident_images():
    return pd.read_csv(DATA_DIR / "accidentlevel_addData.csv", encoding="utf-8-sig")


@st.cache_data
def load_improved_scores():
    """개선 2차 모델 결과 (SMOTE + Calibration + 상호작용 피처)"""
    return pd.read_csv(DATA_DIR / "3_final_scoring_results_improved.csv", encoding="utf-8-sig")


@st.cache_data
def load_2nd_dataset():
    """2차 모델 학습 데이터 (117개소, structure_risk 포함)"""
    return pd.read_csv(DATA_DIR / "2_DatasetFor2ndData.csv", encoding="utf-8-sig")


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
    # accident_label이 데이터에 이미 포함된 경우 사용, 없으면 파일명 기반 추론
    if "accident_label" not in img_df.columns:
        img_df["accident_label"] = img_df["image"].str.contains("부근").astype(int)
    img_df["accident_label"] = img_df["accident_label"].astype(int)

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
    """2차 통합 모델: 구조위험 + 시설 + 어린이비율 → 사고 발생 여부 (이진 분류, 개선)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV

    ds = load_2nd_dataset()
    ds["accident_label"] = (ds["발생건수"] >= 1).astype(int)

    feat_cols = ["structure_risk"] + FACILITY_COLS + ["어린이 비율(%)"]
    X = ds[feat_cols].fillna(ds[feat_cols].median())
    y = ds["accident_label"]

    model = SkPipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=2000, random_state=42,
        )),
    ])
    cv_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    model.fit(X, y)

    # 사고 발생 클래스(1) 계수
    coef_df = pd.DataFrame({
        "변수": feat_cols,
        "계수": model.named_steps["lr"].coef_[0],
    }).sort_values("계수")

    auc_score = round(cv_auc.mean(), 2)
    return model, feat_cols, auc_score, coef_df


# ──────────────────────────────────────────────
# 4. Helper Functions
# ──────────────────────────────────────────────

def make_popup(row, city="성남시"):
    """마커 팝업 생성"""
    grade_key = row["등급"]
    color = GRADE_COLORS.get(grade_key, "#999")
    grade_label = GRADE_LABELS.get(grade_key, grade_key)

    # 점수 구조 섹션: 성남시는 가산/감산, 광명시는 모델 추정
    if city == "광명시":
        score_section = f"""
      <table style="font-size:11px;color:#2C3E50;width:100%;border-collapse:collapse;">
        <tr style="background:#FEF5E7;"><td colspan="2" style="padding:3px 4px;font-weight:600;color:#2C3E50;">모델 추정값</td></tr>
        <tr><td style="padding:2px 4px;">추정 방식</td><td style="text-align:right;font-size:10px;">성남시 모델 적용</td></tr>
        <tr style="background:#FEF9E7;"><td style="padding:2px 4px;font-weight:700;">예상 안전점수</td><td style="text-align:right;font-weight:700;">{row['활성_안전점수']:.1f}점</td></tr>
      </table>"""
    else:
        score_section = f"""
      <table style="font-size:11px;color:#2C3E50;width:100%;border-collapse:collapse;">
        <tr style="background:#FEF5E7;"><td colspan="2" style="padding:3px 4px;font-weight:600;color:#2C3E50;">점수 구조</td></tr>
        <tr><td style="padding:2px 4px;">가산점(시설)</td><td style="text-align:right;font-weight:600;">{row['가산점_시설_V6']:.1f}점</td></tr>
        <tr><td style="padding:2px 4px;">가산점(보너스)</td><td style="text-align:right;font-weight:600;">{int(row['가산점_보너스_V6'])}점</td></tr>
        <tr style="background:#FDEDEC;"><td style="padding:2px 4px;">감산점 합계</td><td style="text-align:right;font-weight:600;color:#E74C3C;">-{row['감산점_합계_V6']:.1f}점</td></tr>
        <tr><td style="padding:2px 4px;">기본점(50)</td><td style="text-align:right;">50.0점</td></tr>
        <tr style="background:#FEF9E7;"><td style="padding:2px 4px;font-weight:700;">최종 안전점수</td><td style="text-align:right;font-weight:700;">{row['활성_안전점수']:.1f}점</td></tr>
      </table>"""

    # CV 도로환경 섹션 (성남시만)
    cv_section = ""
    if city == "성남시" and not pd.isna(row.get('CV_도로폭확률')):
        cv_section = f"""
      <hr style="margin:6px 0;border:none;border-top:1px solid #FDEBD0;">
      <table style="font-size:10px;color:#444;width:100%;border-collapse:collapse;">
        <tr style="background:#FEF9E7;"><td colspan="2" style="padding:3px 4px;font-weight:600;color:#E67E22;">도로환경 (CV)</td></tr>
        <tr><td style="padding:2px 4px;">넓은도로</td><td style="text-align:right;">{row["CV_도로폭확률"]:.0%}</td></tr>
        <tr><td style="padding:2px 4px;">분리장치</td><td style="text-align:right;">{row["CV_분리장치확률"]:.0%}</td></tr>
        <tr><td style="padding:2px 4px;">주정차</td><td style="text-align:right;">{row["CV_주정차밀도"]:.1f}대</td></tr>
      </table>"""

    acc_count = int(row['발생건수']) if pd.notna(row.get('발생건수')) else 0
    child_ratio = row['어린이비율'] if pd.notna(row.get('어린이비율')) else 0

    return f"""
    <div style="font-family:'Noto Sans KR',sans-serif;width:260px;padding:4px;">
      <div style="font-size:15px;font-weight:700;color:#2C3E50;margin-bottom:4px;">
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
      <hr style="margin:8px 0;border:none;border-top:1px solid #F5CBA7;">
      {score_section}
      <hr style="margin:6px 0;border:none;border-top:1px solid #F5CBA7;">
      <table style="font-size:10px;color:#444;width:100%;border-collapse:collapse;">
        <tr><td>적색표면 {int(row.get('도로적색표면', 0))}</td><td>신호등 {int(row.get('신호등', 0))}</td><td>횡단보도 {int(row.get('횡단보도', 0))}</td></tr>
        <tr><td>안전표지 {int(row.get('도로안전표지', 0))}</td><td>CCTV {int(row.get('생활안전CCTV', 0))}</td><td>카메라 {int(row.get('무인교통단속카메라', 0))}</td></tr>
        <tr><td>표지판 {int(row.get('보호구역표지판', 0))}</td><td>옐로카펫 {int(row.get('옐로카펫', 0))}</td><td>펜스 {int(row.get('무단횡단방지펜스', 0))}</td></tr>
        <tr><td>발생건수 {acc_count}건 ({"안전" if acc_count == 0 else "주의" if acc_count <= 6 else "위험"})</td><td>어린이비율 {child_ratio:.1f}%</td><td>{f"구조위험 {row.get('structure_risk', 0):.0%}" if city == "성남시" else ""}</td></tr>
      </table>
      {cv_section}
    </div>
    """


def create_legend_html():
    grade_items = "".join(
        f'<li style="margin:3px 0;"><span style="background:{GRADE_COLORS[g]};width:12px;height:12px;'
        f'display:inline-block;border-radius:50%;margin-right:6px;vertical-align:middle;'
        f'box-shadow:0 1px 3px rgba(0,0,0,.2);"></span>'
        f'<span style="vertical-align:middle;">{GRADE_LABELS[g]}</span></li>'
        for g in ["A", "B", "C", "D"]
    )
    layer_colors = [
        ("green", "지킴이집"), ("#E74C3C", "사고다발지"),
        ("#8E44AD", "CCTV"), ("#2980B9", "단속카메라"),
        ("#F39C12", "안전표지"), ("#E74C3C", "적색표면"),
        ("#27AE60", "신호등"), ("#3498DB", "횡단보도"),
        ("#E67E22", "보호구역표지판"), ("#F1C40F", "옐로카펫"),
        ("#95A5A6", "펜스"),
    ]
    layer_items = "".join(
        f'<li style="margin:2px 0;"><span style="background:{c};width:10px;height:10px;'
        f'display:inline-block;border-radius:50%;margin-right:6px;vertical-align:middle;"></span>'
        f'<span style="vertical-align:middle;font-size:11px;">{n}</span></li>'
        for c, n in layer_colors
    )
    return f"""
    <div style="position:fixed;bottom:30px;right:30px;z-index:1000;
         background:white;padding:12px 16px;border-radius:10px;
         box-shadow:0 4px 12px rgba(0,0,0,.15);font-size:12px;
         font-family:'Noto Sans KR',sans-serif;border:1px solid #F5CBA7;max-height:380px;overflow-y:auto;">
      <div style="font-weight:700;color:#2C3E50;margin-bottom:4px;">안전등급</div>
      <ul style="list-style:none;padding:0;margin:0 0 6px 0;">{grade_items}</ul>
      <div style="font-weight:700;color:#2C3E50;margin-bottom:4px;border-top:1px solid #F5CBA7;padding-top:6px;">시설물 레이어</div>
      <ul style="list-style:none;padding:0;margin:0;">{layer_items}</ul>
    </div>
    """


def create_map(filtered_df, overlay_flags, pop_df, geo, selected_school="(전체)", city="성남시"):
    cfg = CITY_CONFIG[city]
    center = cfg["center"]
    zoom = cfg["zoom"]
    if selected_school != "(전체)":
        sel = filtered_df[filtered_df["시설물명"] == selected_school]
        if len(sel) > 0:
            center = [sel.iloc[0]["위도"], sel.iloc[0]["경도"]]
            zoom = 15
    m = folium.Map(location=center, zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles="https://mt0.google.com/vt/lyrs=r&hl=ko&x={x}&y={y}&z={z}",
        attr="Google", name="기본 지도", max_zoom=22,
    ).add_to(m)

    # Choropleth — 도시별 분기
    if geo and geo.get("features"):
        if city == "성남시":
            choropleth_data = pop_df[["구명", "동명", "어린이_비율"]].copy()
            choropleth_data["adm_nm"] = "경기도 성남시" + choropleth_data["구명"] + " " + choropleth_data["동명"]
        else:
            choropleth_data = pop_df[["동명", "어린이_비율"]].dropna(subset=["어린이_비율"]).copy()
            choropleth_data["adm_nm"] = "경기도 광명시 " + choropleth_data["동명"]
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

        # 행정동 경계선 + 동명 라벨 (별도 레이어)
        folium.GeoJson(
            geo,
            name="행정동 구분선",
            style_function=lambda _: {
                "fillOpacity": 0,
                "color": "#2C3E50",
                "weight": 2,
                "dashArray": "5,3",
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["adm_nm"],
                aliases=["행정동"],
                style="font-size:12px;font-weight:600;",
            ),
        ).add_to(m)

    for _, row in filtered_df.iterrows():
        grade_key = row["등급"]
        color = GRADE_COLORS.get(grade_key, "#999")
        grade_label = GRADE_LABELS.get(grade_key, grade_key)
        is_selected = (selected_school != "(전체)" and row["시설물명"] == selected_school)
        radius = 14 if is_selected else (9 if row["시설유형"] == "초등학교" else 6)
        acc_count = int(row.get("발생건수", 0)) if pd.notna(row.get("발생건수")) else 0
        child_ratio = row.get("어린이비율", 0) if pd.notna(row.get("어린이비율")) else 0
        tooltip_text = (
            f"{row['시설물명']} ({grade_label}) {row['활성_안전점수']:.1f}점"
            f" | {row['시설유형']} | 사고 {acc_count}건 | 어린이 {child_ratio:.1f}%"
        )
        folium.CircleMarker(
            location=[row["위도"], row["경도"]],
            radius=radius,
            color="#E74C3C" if is_selected else "#FFFFFF",
            weight=4 if is_selected else 2,
            fill=True,
            fill_color=color,
            fill_opacity=1.0 if is_selected else 0.9,
            popup=folium.Popup(make_popup(row, city=city), max_width=290),
            tooltip=tooltip_text,
        ).add_to(m)

    # 오버레이 (성남시 전용 — 광명시는 개별 시설 CSV 없음)
    if city != "성남시":
        pass  # 광명시는 오버레이 비활성
    elif overlay_flags.get("지킴이집"):
        gh = load_guardhouses()
        for _, r in gh.iterrows():
            if pd.notna(r["위도"]) and pd.notna(r["경도"]):
                folium.Marker(
                    [r["위도"], r["경도"]],
                    icon=folium.Icon(color="green", icon="home", prefix="fa"),
                    tooltip=r["안전시설명"],
                ).add_to(m)

    if city == "성남시" and overlay_flags.get("사고다발지"):
        acc = load_accidents()
        for _, r in acc.iterrows():
            if pd.notna(r["위도"]) and pd.notna(r["경도"]):
                folium.CircleMarker(
                    [r["위도"], r["경도"]],
                    radius=6, color="#E74C3C", fill=True,
                    fill_color="#E74C3C", fill_opacity=0.6,
                    tooltip=f"사고다발지: {r['사고지역위치명']}",
                ).add_to(m)

    if city == "성남시" and overlay_flags.get("CCTV"):
        _cctv = load_cctv().dropna(subset=["위도", "경도"])
        FastMarkerCluster(
            data=_cctv[["위도", "경도"]].values.tolist(),
            callback="""function(row) {
                var m = L.circleMarker(new L.LatLng(row[0], row[1]),
                    {radius:3, color:'#8E44AD', fillColor:'#8E44AD', fill:true, fillOpacity:0.4});
                m.bindTooltip('CCTV'); return m;
            }""",
        ).add_to(m)

    if city == "성남시" and overlay_flags.get("카메라"):
        _cam = load_cameras().dropna(subset=["위도", "경도"])
        FastMarkerCluster(
            data=_cam[["위도", "경도"]].values.tolist(),
            callback="""function(row) {
                var m = L.circleMarker(new L.LatLng(row[0], row[1]),
                    {radius:3, color:'#2980B9', fillColor:'#2980B9', fill:true, fillOpacity:0.4});
                m.bindTooltip('단속카메라'); return m;
            }""",
        ).add_to(m)

    if city == "성남시" and overlay_flags.get("표지판"):
        _signs = load_signs().dropna(subset=["위도", "경도"])
        FastMarkerCluster(
            data=_signs[["위도", "경도"]].values.tolist(),
            callback="""function(row) {
                var m = L.circleMarker(new L.LatLng(row[0], row[1]),
                    {radius:2, color:'#F39C12', fillColor:'#F39C12', fill:true, fillOpacity:0.3});
                m.bindTooltip('안전표지'); return m;
            }""",
        ).add_to(m)

    if city == "성남시" and overlay_flags.get("적색표면"):
        _rs = load_red_surface().dropna(subset=["위도", "경도"])
        FastMarkerCluster(
            data=_rs[["위도", "경도"]].values.tolist(),
            callback="""function(row) {
                var m = L.circleMarker(new L.LatLng(row[0], row[1]),
                    {radius:3, color:'#E74C3C', fillColor:'#E74C3C', fill:true, fillOpacity:0.5});
                m.bindTooltip('도로적색표면'); return m;
            }""",
        ).add_to(m)

    if city == "성남시" and overlay_flags.get("신호등"):
        _tl = load_traffic_lights().dropna(subset=["위도", "경도"])
        FastMarkerCluster(
            data=_tl[["위도", "경도"]].values.tolist(),
            callback="""function(row) {
                var m = L.circleMarker(new L.LatLng(row[0], row[1]),
                    {radius:3, color:'#27AE60', fillColor:'#27AE60', fill:true, fillOpacity:0.5});
                m.bindTooltip('신호등'); return m;
            }""",
        ).add_to(m)

    if city == "성남시" and overlay_flags.get("횡단보도"):
        _cw = load_crosswalks().dropna(subset=["위도", "경도"])
        FastMarkerCluster(
            data=_cw[["위도", "경도"]].values.tolist(),
            callback="""function(row) {
                var m = L.circleMarker(new L.LatLng(row[0], row[1]),
                    {radius:3, color:'#3498DB', fillColor:'#3498DB', fill:true, fillOpacity:0.5});
                m.bindTooltip('횡단보도'); return m;
            }""",
        ).add_to(m)

    if city == "성남시" and overlay_flags.get("보호구역표지판"):
        _zs = load_zone_signs().dropna(subset=["위도", "경도"])
        FastMarkerCluster(
            data=_zs[["위도", "경도"]].values.tolist(),
            callback="""function(row) {
                var m = L.circleMarker(new L.LatLng(row[0], row[1]),
                    {radius:3, color:'#E67E22', fillColor:'#E67E22', fill:true, fillOpacity:0.5});
                m.bindTooltip('보호구역표지판'); return m;
            }""",
        ).add_to(m)

    if city == "성남시" and overlay_flags.get("옐로카펫"):
        _yc = load_yellow_carpet().dropna(subset=["위도", "경도"])
        for _, r in _yc.iterrows():
            folium.CircleMarker(
                [r["위도"], r["경도"]], radius=5,
                color="#F1C40F", fill=True, fill_color="#F1C40F", fill_opacity=0.8,
                tooltip=f"옐로카펫: {r.get('시설물명', '')}",
            ).add_to(m)

    if city == "성남시" and overlay_flags.get("펜스"):
        _fn = load_fences().dropna(subset=["위도", "경도"])
        FastMarkerCluster(
            data=_fn[["위도", "경도"]].values.tolist(),
            callback="""function(row) {
                var m = L.circleMarker(new L.LatLng(row[0], row[1]),
                    {radius:3, color:'#95A5A6', fillColor:'#95A5A6', fill:true, fillOpacity:0.5});
                m.bindTooltip('무단횡단방지펜스'); return m;
            }""",
        ).add_to(m)

    # 지도 UX 플러그인
    MiniMap(tile_layer="OpenStreetMap", position="bottomright", width=120, height=90).add_to(m)
    Fullscreen(position="topleft").add_to(m)
    MeasureControl(position="topleft", primary_length_unit="meters", primary_area_unit="sqmeters").add_to(m)

    m.get_root().html.add_child(folium.Element(create_legend_html()))
    return m


# ──────────────────────────────────────────────
# 5. Sidebar
# ──────────────────────────────────────────────

# ── 도시 선택 (최상단) ──
selected_city = st.sidebar.radio("도시 선택", ["성남시", "광명시"], horizontal=True)

# ── 성남시 데이터 (항상 로드 — 모델 학습 + 비교용) ──
df_sn_raw = load_data()
df_sn = df_sn_raw.copy()

cv_df = load_cv_features()
df_sn = df_sn.merge(cv_df, on="시설물명", how="left")

for _fc in ["보호구역표지판", "옐로카펫", "무단횡단방지펜스"]:
    if _fc in df_sn.columns:
        df_sn[_fc] = df_sn[_fc].fillna(0)

_, _, _fac_risk = train_structure_model()
df_sn = df_sn.merge(_fac_risk, on="시설물명", how="left")
df_sn["structure_risk"] = df_sn["structure_risk"].fillna(df_sn["structure_risk"].median())

_integ_model, integ_feats, integ_auc, integ_coef = train_integrated_model()

# 개선 모델 결과 병합 (117개소)
_improved = load_improved_scores()
_imp_merge = _improved[["시설물명", "risk_prob", "risk_prob_calibrated",
                         "safety_score", "safety_grade"]].copy()
_imp_merge.columns = ["시설물명", "IM_risk_prob", "IM_사고확률",
                       "IM_안전점수", "IM_등급"]
df_sn = df_sn.merge(_imp_merge, on="시설물명", how="left")

# 어린이 비율(%) 컬럼 호환 (2nd dataset은 '어린이 비율(%)', 팀통합은 '어린이비율')
if "어린이 비율(%)" not in df_sn.columns and "어린이비율" in df_sn.columns:
    df_sn["어린이 비율(%)"] = df_sn["어린이비율"]

# 사고확률: 개선 모델 결과 사용 (117개소), 나머지는 inline 모델
_prob_valid = df_sn.dropna(subset=integ_feats)
if len(_prob_valid) > 0:
    _inline_prob = _integ_model.predict_proba(
        _prob_valid[integ_feats].fillna(0).values
    )[:, 1]
    df_sn.loc[_prob_valid.index, "_inline_사고확률"] = _inline_prob
# 개선 모델 결과 우선, 없으면 inline fallback
df_sn["사고확률"] = df_sn["IM_사고확률"].fillna(df_sn.get("_inline_사고확률", np.nan))

df_sn["_시설합계"] = df_sn[FACILITY_COLS].sum(axis=1)
df_sn["활성_안전점수"] = df_sn["최종안전점수_V6"]
df_sn["등급"] = df_sn["등급_V6"]
df_sn["안전등급"] = df_sn["등급_V6"].map(GRADE_LABELS)

# ── 광명시 안전점수 예측 ──
gm_raw = load_gwangmyung()
safety_model, model_features, model_r2 = train_safety_model()
gs_q1, gs_q2, gs_q3 = df_sn["최종안전점수_V6"].quantile([0.25, 0.5, 0.75]).values


def _classify_grade(score):
    if score >= gs_q3:
        return "A"
    if score >= gs_q2:
        return "B"
    if score >= gs_q1:
        return "C"
    return "D"


df_gm = gm_raw.copy()
for _fc in FACILITY_COLS:
    if _fc in df_gm.columns:
        df_gm[_fc] = df_gm[_fc].fillna(0)
_gm_child_median = df_gm["어린이비율"].median() if df_gm["어린이비율"].notna().any() else 10.0
_gm_inputs = []
for _, _gm_r in df_gm.iterrows():
    _gm_input = {f: (int(_gm_r[f]) if pd.notna(_gm_r.get(f)) else 0) for f in FACILITY_COLS}
    _gm_input["발생건수"] = int(_gm_r["발생건수"]) if pd.notna(_gm_r.get("발생건수")) else 0
    _gm_input["어린이비율"] = float(_gm_r["어린이비율"]) if pd.notna(_gm_r.get("어린이비율")) else _gm_child_median
    _gm_inputs.append(_gm_input)
_gm_scores = np.clip(safety_model.predict(pd.DataFrame(_gm_inputs)), 0, 100).tolist()
df_gm["활성_안전점수"] = _gm_scores
df_gm["등급"] = df_gm["활성_안전점수"].apply(_classify_grade)
df_gm["안전등급"] = df_gm["등급"].map(GRADE_LABELS)
df_gm["_시설합계"] = df_gm[FACILITY_COLS].sum(axis=1)

# ── 활성 데이터 선택 ──
if selected_city == "성남시":
    df = df_sn
else:
    df = df_gm

city_label = f"{selected_city} 어린이 보호구역"
st.sidebar.markdown(
    f"<h2 style='text-align:center;margin-bottom:0;'>스쿨존 안전 분석</h2>"
    f"<p style='text-align:center;opacity:0.6;font-size:13px;'>{city_label}</p>",
    unsafe_allow_html=True,
)

# ── 개별 시설 선택 (헤더 바로 아래) ──
school_list = ["(전체)"] + sorted(df["시설물명"].tolist())
selected_school = st.sidebar.selectbox("개별 시설 선택", school_list)
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
overlay_flags = {}
if selected_city == "성남시":
    st.sidebar.markdown(
        "<p style='font-weight:600;font-size:14px;margin-bottom:8px;'>시설물 레이어</p>",
        unsafe_allow_html=True,
    )
    ov_guardhouse = st.sidebar.checkbox("아동안전지킴이집", value=True)
    ov_accident = st.sidebar.checkbox("사고다발지", value=True)
    ov_cctv = st.sidebar.checkbox("생활안전 CCTV", value=False)
    ov_camera = st.sidebar.checkbox("무인교통단속카메라", value=False)
    ov_sign = st.sidebar.checkbox("도로안전표지", value=False)
    ov_red_surface = st.sidebar.checkbox("도로적색표면", value=False)
    ov_traffic_light = st.sidebar.checkbox("신호등", value=False)
    ov_crosswalk = st.sidebar.checkbox("횡단보도", value=False)
    ov_zone_sign = st.sidebar.checkbox("보호구역표지판", value=False)
    ov_yellow_carpet = st.sidebar.checkbox("옐로카펫", value=False)
    ov_fence = st.sidebar.checkbox("무단횡단방지펜스", value=False)
    overlay_flags = {
        "지킴이집": ov_guardhouse, "사고다발지": ov_accident,
        "CCTV": ov_cctv, "카메라": ov_camera, "표지판": ov_sign,
        "적색표면": ov_red_surface, "신호등": ov_traffic_light,
        "횡단보도": ov_crosswalk, "보호구역표지판": ov_zone_sign,
        "옐로카펫": ov_yellow_carpet, "펜스": ov_fence,
    }
else:
    st.sidebar.markdown(
        "<p style='font-size:12px;opacity:0.7;'>시설물 레이어는 성남시에서만 지원됩니다.</p>",
        unsafe_allow_html=True,
    )

# CSV 다운로드
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-weight:600;font-size:14px;margin-bottom:8px;'>데이터 내보내기</p>",
    unsafe_allow_html=True,
)
csv_cols = ["시설물명", "시설유형", "구", "안전등급", "활성_안전점수"]
if selected_city == "성남시":
    csv_cols += ["가산점_시설_V6", "가산점_보너스_V6", "감산점_합계_V6"]
csv_cols += FACILITY_COLS + ["발생건수", "어린이비율"]
if selected_city == "성남시":
    csv_cols += ["사고확률"]
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

if len(filtered_df) == 0:
    st.warning("선택한 필터 조건에 해당하는 시설이 없습니다. 사이드바에서 필터를 조정해 주세요.")
    st.stop()

# Header
_n_facilities = len(filtered_df)
_model_note = " (모델 추정)" if selected_city == "광명시" else ""
st.markdown(
    f'<div style="margin-bottom:8px;">'
    f'<span style="font-size:36px;font-weight:700;color:#2C3E50;">내 아이가 살기 좋은 동네</span>'
    f'<span style="font-size:14px;color:#34495E;margin-left:12px;">'
    f'{selected_city} 어린이 보호구역 {_n_facilities}개소 안전 분석 대시보드{_model_note}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

st.caption("점수는 확률 추정치이며, 보조 의사결정 도구로 사용하도록 권장합니다.")

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
total_accidents = int(filtered_df["발생건수"].fillna(0).sum())
k4.metric("사고건수 합계", f"{total_accidents}건")

# ── 핵심 인사이트 카드 ──
if len(filtered_df) > 0:
    # 1) 가장 위험한 구
    _gu_acc = filtered_df.groupby("구")["발생건수"].mean()
    _worst_gu = _gu_acc.idxmax()
    _worst_gu_val = _gu_acc.max()
    _best_gu = _gu_acc.idxmin()
    _best_gu_val = _gu_acc.min()
    # 2) 가장 부족한 시설 (D등급 vs A등급 차이 최대)
    _d_fac = filtered_df[filtered_df["등급"] == "D"][FACILITY_COLS].mean()
    _a_fac = filtered_df[filtered_df["등급"] == "A"][FACILITY_COLS].mean()
    if len(_d_fac) > 0 and len(_a_fac) > 0 and not _d_fac.isna().all():
        _gap = (_a_fac - _d_fac).sort_values(ascending=False)
        _top_gap = _gap.index[0] if len(_gap) > 0 else None
    else:
        _top_gap = None
    # 3) 사고 집중도
    _zero_acc = (filtered_df["발생건수"] == 0).sum()
    _zero_pct = _zero_acc / len(filtered_df) * 100

    _ins1 = (
        f'<b>{_worst_gu}</b> 평균 발생건수 {_worst_gu_val:.1f}건으로 가장 높고, '
        f'<b>{_best_gu}</b>는 {_best_gu_val:.1f}건으로 가장 낮습니다.'
    )
    _ins2 = (
        f'A등급 대비 D등급에 가장 부족한 시설은 <b>{_top_gap}</b>입니다.'
        if _top_gap else '등급별 시설 차이를 비교할 데이터가 부족합니다.'
    )
    _ins3 = (
        f'전체 {len(filtered_df)}개소 중 <b>{_zero_acc}개소({_zero_pct:.0f}%)</b>는 '
        f'사고 발생건수 0건(안전)입니다.'
    )

    st.markdown(
        '<div style="background:linear-gradient(135deg,#FEF9E7,#FDEBD0);'
        'padding:14px 20px;border-radius:10px;border-left:4px solid #E67E22;'
        'margin-bottom:16px;">'
        '<b style="color:#2C3E50;font-size:15px;">핵심 발견</b><br>'
        f'<span style="font-size:13px;color:#2C3E50;">'
        f'1. {_ins1}<br>2. {_ins2}<br>3. {_ins3}'
        f'</span></div>',
        unsafe_allow_html=True,
    )

# Tabs
tab_map, tab_analysis, tab_facility, tab_cv, tab_district, tab_sim, tab_method = st.tabs(
    ["지도", "상세분석", "시설점수", "도로환경 (CV)", "동네정보", "광명 시뮬레이션", "분석 방법론"]
)

# ============================
# Tab 1: 지도
# ============================
with tab_map:
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

    # ── D등급 시설 부족 우선순위 테이블 ──
    if len(d_grade) > 0:
        _a_avg = df[df["등급"] == "A"][FACILITY_COLS].mean()
        _priority_rows = []
        for _, _dr in d_grade.sort_values("활성_안전점수").iterrows():
            _gaps = {}
            for _fc in FACILITY_COLS:
                _need = max(0, round(_a_avg[_fc] - _dr[_fc], 1))
                if _need > 0:
                    _gaps[_fc] = _need
            _top3 = sorted(_gaps.items(), key=lambda x: -x[1])[:3]
            _priority_rows.append({
                "시설물명": _dr["시설물명"],
                "구": _dr["구"],
                "안전점수": round(_dr["활성_안전점수"], 1),
                "1순위 보강": f"{_top3[0][0]} (+{_top3[0][1]:.0f})" if len(_top3) > 0 else "-",
                "2순위 보강": f"{_top3[1][0]} (+{_top3[1][1]:.0f})" if len(_top3) > 1 else "-",
                "3순위 보강": f"{_top3[2][0]} (+{_top3[2][1]:.0f})" if len(_top3) > 2 else "-",
            })
        with st.expander(f"D등급 시설 보강 우선순위 ({len(d_grade)}개소)", expanded=False):
            st.caption("A등급 평균 대비 부족한 시설을 우선순위별로 표시합니다.")
            st.dataframe(
                pd.DataFrame(_priority_rows),
                use_container_width=True, hide_index=True,
            )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    if selected_city == "성남시":
        pop_df = load_population()
        geo = load_geojson()
    else:
        pop_df = load_gm_population()
        geo = load_gm_geojson()
    m = create_map(filtered_df, overlay_flags, pop_df, geo, selected_school, city=selected_city)
    if selected_city == "광명시":
        st.caption("시설물 레이어(지킴이집, 사고다발지 등)는 성남시에서만 지원됩니다.")
    st_folium(m, height=550, use_container_width=True, returned_objects=[])

# ============================
# Tab 2: 상세분석
# ============================
with tab_analysis:
    _sub_all, _sub_indiv = st.tabs(["전체 시설", "개별 시설"])

with _sub_all:
    # ── 점수 구조 시각화 ──
    if selected_city == "성남시" and "가산점_시설_V6" in filtered_df.columns:
        st.markdown("##### 점수 구조: 기본(50) + 가산점(시설+보너스) - 감산점")
        score_struct = pd.DataFrame({
            "항목": ["가산점(시설)", "가산점(보너스)", "감산점 합계"],
            "평균": [
                filtered_df["가산점_시설_V6"].mean(),
                filtered_df["가산점_보너스_V6"].mean(),
                -filtered_df["감산점_합계_V6"].mean(),
            ],
        })
        fig_struct = px.bar(
            score_struct, x="항목", y="평균",
            title="안전점수 구성 요소 평균",
            color="항목",
            color_discrete_map={
                "가산점(시설)": "#F39C12",
                "가산점(보너스)": "#F39C12",
                "감산점 합계": "#E74C3C",
            },
            text="평균",
        )
        fig_struct.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_struct.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False)
        st.plotly_chart(fig_struct, use_container_width=True)
    elif selected_city == "광명시":
        st.markdown("##### 안전점수 개요")
        st.caption("광명시 안전점수는 성남시 LinearRegression 모델로 추정한 값입니다.")

    fig_hist = px.histogram(
        filtered_df, x="활성_안전점수", nbins=20,
        title="안전점수 분포",
        labels={"활성_안전점수": "안전점수"},
        color_discrete_sequence=["#F39C12"],
    )
    fig_hist.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_title="시설 수", bargap=0.08)
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── 등급별 현황 ──
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 등급별 현황")
    grade_stats = filtered_df.groupby("등급").agg(
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
    type_stats = filtered_df.groupby("시설유형").agg(
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
    low_facilities = filtered_df[filtered_df["등급"] == "D"].sort_values("활성_안전점수")
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


with _sub_indiv:
    if selected_school != "(전체)":
        school_row = df[df["시설물명"] == selected_school].iloc[0]

        # ── 상세 카드 ──
        _gc = GRADE_COLORS.get(school_row["등급"], "#999")
        _gl = GRADE_LABELS.get(school_row["등급"], school_row["등급"])
        st.markdown(
            f"<div style='background:#FEF5E7;padding:12px 16px;border-radius:8px;"
            f"border-left:4px solid {_gc};'>"
            f"<b style='color:#2C3E50;'>{selected_school}</b> &nbsp; "
            f"<span style='background:{_gc};color:#fff;padding:2px 10px;"
            f"border-radius:20px;font-size:12px;'>{_gl}</span> &nbsp; "
            f"<span style='color:#2C3E50;'>안전점수: <b>{school_row['활성_안전점수']:.1f}</b></span> &nbsp; "
            f"<span style='color:#2C3E50;'>발생건수: <b>{int(school_row['발생건수']) if pd.notna(school_row['발생건수']) else 0}</b>건</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ── 로드뷰 이미지 ──
        _rv_name = selected_school.replace(" ", "_")
        _rv_path = DATA_DIR / "roadview" / f"{_rv_name}_북쪽.jpg"
        if _rv_path.exists():
            st.markdown("##### 로드뷰 (북쪽 방향)")
            st.image(str(_rv_path), use_container_width=True)
        else:
            # 공백 없는 원본 이름으로도 시도
            _rv_path2 = DATA_DIR / "roadview" / f"{selected_school}_북쪽.jpg"
            if _rv_path2.exists():
                st.markdown("##### 로드뷰 (북쪽 방향)")
                st.image(str(_rv_path2), use_container_width=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ── 레이더 차트 ──
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
            line=dict(color="#2C3E50", width=2),
        ))
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#F5CBA7"),
                angularaxis=dict(gridcolor="#F5CBA7"),
                bgcolor="#FAFCFF",
            ),
            title=f"{selected_school} 시설물 현황",
            height=420, showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── 정책 시뮬레이션 (성남시 전용) ──
        if selected_city != "성남시":
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            st.info("정책 시뮬레이션은 성남시에서만 지원됩니다.")
        else:
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            st.markdown("##### 정책 시뮬레이션: 시설물 추가 효과")
            st.caption("선택한 시설에 시설물 1개를 추가할 때 사고 발생 확률 변화량을 예측합니다.")

            _integ_model_pol, integ_feats_pol = _integ_model, integ_feats
            _sim_input = {}
            for _f in integ_feats_pol:
                if _f == "어린이 비율(%)":
                    _sim_input[_f] = school_row.get("어린이비율", 10.0)
                else:
                    _sim_input[_f] = school_row.get(_f, 0)
            pol_base = pd.DataFrame([_sim_input])[integ_feats_pol].fillna(0).values
            pol_base_prob = float(_integ_model_pol.predict_proba(pol_base)[0, 1])

            pol_results = []
            for i, feat in enumerate(integ_feats_pol):
                if feat in FACILITY_COLS:
                    pol_modified = pol_base.copy()
                    pol_modified[0, i] += 1
                    pol_new_prob = float(_integ_model_pol.predict_proba(pol_modified)[0, 1])
                    pol_delta = pol_new_prob - pol_base_prob
                    pol_results.append({
                        "시설물": feat,
                        "현재 수량": int(school_row[feat]),
                        "현재 사고확률": pol_base_prob,
                        "추가 후 사고확률": pol_new_prob,
                        "변화량 (%p)": pol_delta,
                    })

            pol_df = pd.DataFrame(pol_results).sort_values("변화량 (%p)")

            top3 = pol_df.head(3)
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#FDEBD0,#FEF9E7);'
                f'padding:14px 18px;border-radius:10px;border-left:4px solid #27AE60;">'
                f'<b style="color:#2C3E50;">{selected_school}</b> — '
                f'현재 사고확률: <b>{pol_base_prob:.1%}</b><br>'
                f'<span style="font-size:13px;color:#2C3E50;">'
                f'사고확률 감소 TOP 3: '
                + " / ".join(
                    f'<b>{r["시설물"]}</b> +1 → {r["변화량 (%p)"]:+.1%}p'
                    for _, r in top3.iterrows()
                )
                + '</span></div>',
                unsafe_allow_html=True,
            )

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
                title=f"{selected_school}: 시설물 +1개 추가 시 사고확률 변화",
                xaxis=dict(title="사고확률 변화 (%p)"),
                yaxis=dict(title=""),
            )
            st.plotly_chart(fig_pol, use_container_width=True)
    else:
        st.markdown(
            "<div style='background:#FEF5E7;padding:30px;border-radius:10px;"
            "text-align:center;color:#E67E22;'>"
            "<span style='font-size:16px;font-weight:600;'>사이드바에서 개별 시설을 선택해 주세요</span><br><br>"
            "<span style='font-size:13px;color:#566573;'>"
            "시설을 선택하면 시설물 레이더 차트, 상세 정보,<br>"
            "시설물 추가 효과 시뮬레이션을 확인할 수 있습니다.</span>"
            "</div>",
            unsafe_allow_html=True,
        )


# ============================
# Tab 3: 시설점수
# ============================
with tab_facility:
    st.markdown("### 시설물 보유 현황 및 분석")
    st.caption(f"{len(df)}개소 스쿨존의 9개 안전 시설물 보유 현황과 사고 관계를 분석합니다.")

    # ── (a) 시설 보유 현황 KPI ──
    _fac_totals = df[FACILITY_COLS].sum()
    _fac_total_sum = int(_fac_totals.sum())
    _fac_per_school = df[FACILITY_COLS].sum(axis=1)
    _fac_mean_per_school = _fac_per_school.mean()
    _fac_max_school = df.loc[_fac_per_school.idxmax(), "시설물명"]
    _fac_max_val = int(_fac_per_school.max())
    _fac_min_school = df.loc[_fac_per_school.idxmin(), "시설물명"]
    _fac_min_val = int(_fac_per_school.min())

    fk1, fk2, fk3, fk4 = st.columns(4)
    fk1.metric("전체 시설 합계", f"{_fac_total_sum:,}개")
    fk2.metric("교당 평균 시설 수", f"{_fac_mean_per_school:.1f}개")
    fk3.metric("시설 최다 보유", f"{_fac_max_school} ({_fac_max_val}개)")
    fk4.metric("시설 최소 보유", f"{_fac_min_school} ({_fac_min_val}개)")

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── (b) 구별 시설 보유 현황 (Stacked Bar) — 성남 3구 + 광명 ──
    st.markdown("##### 구별 시설 보유 현황 (성남 + 광명)")
    _gu_fac = df_sn.groupby("구")[FACILITY_COLS].sum().reset_index()
    _gm_fac = df_gm[FACILITY_COLS].sum().to_frame().T
    _gm_fac.insert(0, "구", "광명시")
    _gu_fac_all = pd.concat([_gu_fac, _gm_fac], ignore_index=True)
    _gu_fac_melt = _gu_fac_all.melt(id_vars="구", var_name="시설종류", value_name="수량")
    fig_gu_fac = px.bar(
        _gu_fac_melt, x="구", y="수량", color="시설종류",
        barmode="stack",
        title="구별 시설물 보유 현황 (9개 시설 합산)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_gu_fac.update_layout(**PLOTLY_LAYOUT, height=450)
    st.plotly_chart(fig_gu_fac, use_container_width=True)
    st.caption("※ 성남시 3구 + 광명시 시설물 보유 현황 비교")

    # ── (b-2) 구별 등급 분포 파이차트 (성남 + 광명) ──
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 구별 안전등급 분포 (성남 + 광명)")

    # 광명 예측 등급 — df_gm에 이미 '등급' 존재
    _gm_grade_df = pd.DataFrame({"구": "광명시", "등급": df_gm["등급"].tolist()})

    _gu_grade = df_sn.groupby(["구", "등급"]).size().reset_index(name="개소")
    _gm_gu_grade = _gm_grade_df.groupby(["구", "등급"]).size().reset_index(name="개소")
    _gu_grade_all = pd.concat([_gu_grade, _gm_gu_grade], ignore_index=True)
    _gu_list = sorted(df_sn["구"].dropna().unique().tolist()) + ["광명시"]
    _pie_cols = st.columns(len(_gu_list))
    for _pi, _gu_name in enumerate(_gu_list):
        with _pie_cols[_pi]:
            _gu_sub = _gu_grade_all[_gu_grade_all["구"] == _gu_name]
            _title = f"{_gu_name}" + (" (예측)" if _gu_name == "광명시" else "")
            fig_pie = px.pie(
                _gu_sub, values="개소", names="등급",
                title=_title,
                color="등급",
                color_discrete_map={g: GRADE_COLORS[g] for g in ["A", "B", "C", "D"]},
                category_orders={"등급": ["A", "B", "C", "D"]},
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+value")
            fig_pie.update_layout(
                **PLOTLY_LAYOUT, height=320, showlegend=True,
                legend=dict(orientation="h", y=-0.1),
                margin=dict(t=40, b=40, l=10, r=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── (c) 등급별 시설 보유 현황 (Grouped Bar) ──
    st.markdown("##### 등급별 시설 보유 현황")
    _grade_fac = df.groupby("등급")[FACILITY_COLS].mean().reindex(["A", "B", "C", "D"]).reset_index()
    _grade_fac_melt = _grade_fac.melt(id_vars="등급", var_name="시설종류", value_name="평균수량")
    fig_grade_fac = px.bar(
        _grade_fac_melt, x="시설종류", y="평균수량", color="등급",
        barmode="group",
        title="등급별 평균 시설 수 (A~D등급 비교)",
        color_discrete_map={g: GRADE_COLORS[g] for g in ["A", "B", "C", "D"]},
    )
    fig_grade_fac.update_layout(**PLOTLY_LAYOUT, height=450)
    st.plotly_chart(fig_grade_fac, use_container_width=True)

    # ── (c-2) 성남 vs 광명 교당 평균 시설 비교 ──
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 성남 vs 광명 교당 평균 시설 수")
    _sn_avg = df_sn[FACILITY_COLS].mean().rename("성남시")
    _gm_avg = df_gm[FACILITY_COLS].mean().rename("광명시")
    _cmp = pd.DataFrame({"성남시": _sn_avg, "광명시": _gm_avg}).reset_index()
    _cmp.columns = ["시설종류", "성남시", "광명시"]
    _cmp_melt = _cmp.melt(id_vars="시설종류", var_name="지역", value_name="교당 평균")
    fig_cmp = px.bar(
        _cmp_melt, x="시설종류", y="교당 평균", color="지역",
        barmode="group", title="성남 vs 광명 — 교당 평균 시설 수 비교",
        color_discrete_map={"성남시": "#27AE60", "광명시": "#E67E22"},
    )
    fig_cmp.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig_cmp, use_container_width=True)
    st.caption("※ 광명시 도로안전표지·생활안전CCTV·무인교통단속카메라·보호구역표지판은 데이터 미수집으로 0 표시")

    # 인사이트 카드: A등급 vs D등급 시설 격차
    _a_fac_avg = df[df["등급"] == "A"][FACILITY_COLS].mean()
    _d_fac_avg = df[df["등급"] == "D"][FACILITY_COLS].mean()
    if not _a_fac_avg.isna().all() and not _d_fac_avg.isna().all():
        _ad_gap = (_a_fac_avg - _d_fac_avg).sort_values(ascending=False)
        _top3_gap = _ad_gap.head(3)
        st.markdown(
            '<div style="background:linear-gradient(135deg,#FEF9E7,#FDEBD0);'
            'padding:14px 18px;border-radius:10px;border-left:4px solid #E67E22;margin-bottom:16px;">'
            '<b style="color:#2C3E50;">A등급 vs D등급 시설 격차 TOP 3</b><br>'
            '<span style="font-size:13px;color:#2C3E50;">'
            + " / ".join(
                f'<b>{f}</b>: A등급 평균 {_a_fac_avg[f]:.1f} vs D등급 {_d_fac_avg[f]:.1f} (차이 {v:+.1f})'
                for f, v in _top3_gap.items()
            )
            + '</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── (d) 시설 수 vs 사고 관계 (Scatter + Trendline) ──
    st.markdown("##### 시설 수 vs 사고 관계")
    st.caption("9개 시설 합계와 발생건수의 관계 — 시설이 많을수록 사고가 줄어드는가?")

    fig_scatter = px.scatter(
        df, x="_시설합계", y="발생건수",
        color="등급", size="사고심각도" if "사고심각도" in df.columns else None,
        color_discrete_map={g: GRADE_COLORS[g] for g in ["A", "B", "C", "D"]},
        trendline="ols",
        title="총 시설 수 vs 사고 발생건수",
        labels={"_시설합계": "총 시설 수 (9개 합)", "발생건수": "사고 발생건수"},
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 상관계수 표시
    _corr_fac_acc = df[["_시설합계", "발생건수"]].corr().iloc[0, 1]
    st.markdown(
        f'<div style="background:#FEF5E7;padding:10px 16px;border-radius:8px;'
        f'border-left:4px solid #F39C12;">'
        f'<span style="font-size:13px;color:#2C3E50;">'
        f'상관계수: <b>{_corr_fac_acc:.3f}</b> '
        f'({"음의 상관: 시설이 많을수록 사고가 적은 경향" if _corr_fac_acc < -0.1 else "양의 상관: 사고 후 시설을 설치하는 사후 대응 패턴" if _corr_fac_acc > 0.1 else "약한 상관"})'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    # ── (d-2) 시설 간 상관 히트맵 ──
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 시설 간 상관관계 히트맵")
    st.caption("9개 시설물 사이의 상관계수 — 어떤 시설이 함께 설치되는 경향이 있는가?")

    _fac_corr = df[FACILITY_COLS].corr()
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=_fac_corr.values,
        x=FACILITY_COLS, y=FACILITY_COLS,
        colorscale=[[0, "#E74C3C"], [0.5, "#FFFFFF"], [1, "#154360"]],
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in _fac_corr.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    fig_heatmap.update_layout(
        **PLOTLY_LAYOUT, height=500,
        title="9개 시설물 상관관계 매트릭스",
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── (e) 시설 갭 분석 테이블 ──
    st.markdown("##### 시설 갭 분석")

    if selected_school != "(전체)":
        _sel_row = df[df["시설물명"] == selected_school]
        if len(_sel_row) > 0:
            _sel_row = _sel_row.iloc[0]
            _all_avg = df[FACILITY_COLS].mean()
            _a_avg = df[df["등급"] == "A"][FACILITY_COLS].mean()
            _d_avg = df[df["등급"] == "D"][FACILITY_COLS].mean()

            _gap_data = []
            for _fc in FACILITY_COLS:
                _gap_data.append({
                    "시설물": _fc,
                    "전체 평균": round(_all_avg[_fc], 1),
                    "A등급 평균": round(_a_avg[_fc], 1),
                    "D등급 평균": round(_d_avg[_fc], 1),
                    f"{selected_school}": int(_sel_row[_fc]),
                    "A등급 대비 부족분": max(0, round(_a_avg[_fc] - _sel_row[_fc], 1)),
                })
            st.dataframe(
                pd.DataFrame(_gap_data),
                use_container_width=True, hide_index=True,
            )
    else:
        st.caption("전체 학교 시설 순위 (시설 합계 상위 20)")
        _rank_df = df[["시설물명", "시설유형", "구", "등급", "_시설합계"]].copy()
        _rank_df.columns = ["시설물명", "시설유형", "구", "등급", "시설합계"]
        _rank_df = _rank_df.nlargest(20, "시설합계").reset_index(drop=True)
        _rank_df.index = _rank_df.index + 1
        st.dataframe(_rank_df, use_container_width=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── (f) 개별 학교 시설 레이더 차트 ──
    st.markdown("##### 개별 시설 레이더 차트")

    if selected_school != "(전체)":
        _sel_r = df[df["시설물명"] == selected_school]
        if len(_sel_r) > 0:
            _sel_r = _sel_r.iloc[0]
            _a_avg_r = df[df["등급"] == "A"][FACILITY_COLS].mean()

            # 정규화 (max 기준 0~100)
            _radar_vals = []
            _radar_a_vals = []
            for _fc in FACILITY_COLS:
                _mx = df[_fc].max()
                _radar_vals.append(_sel_r[_fc] / _mx * 100 if _mx > 0 else 0)
                _radar_a_vals.append(_a_avg_r[_fc] / _mx * 100 if _mx > 0 else 0)

            _theta = FACILITY_COLS + [FACILITY_COLS[0]]

            fig_fac_radar = go.Figure()
            fig_fac_radar.add_trace(go.Scatterpolar(
                r=_radar_vals + [_radar_vals[0]],
                theta=_theta,
                fill="toself", name=selected_school,
                fillcolor="rgba(27,79,114,0.2)",
                line=dict(color="#2C3E50", width=2),
            ))
            fig_fac_radar.add_trace(go.Scatterpolar(
                r=_radar_a_vals + [_radar_a_vals[0]],
                theta=_theta,
                fill="toself", name="A등급 평균",
                fillcolor="rgba(39,174,96,0.1)",
                line=dict(color="#27AE60", width=1, dash="dash"),
            ))
            fig_fac_radar.update_layout(
                **PLOTLY_LAYOUT,
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="#F5CBA7"),
                    angularaxis=dict(gridcolor="#F5CBA7"),
                    bgcolor="#FAFCFF",
                ),
                title=f"{selected_school} 시설 현황 vs A등급 평균",
                height=450, showlegend=True,
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_fac_radar, use_container_width=True)
    else:
        st.markdown(
            "<div style='background:#FEF5E7;padding:20px;border-radius:8px;"
            "text-align:center;color:#E67E22;'>"
            "사이드바에서 개별 시설을 선택하면<br>시설 레이더 차트가 표시됩니다."
            "</div>",
            unsafe_allow_html=True,
        )



# ============================
# Tab 4: 도로환경 (CV)
# ============================
# 구조 모델 (CV탭 + 방법론탭에서 사용)
_struct_model, struct_auc, _fac_risk_cv = train_structure_model()

with tab_cv:
    st.markdown("### 캡스톤 연구: 2단계 사고 예측 모델 (개선)")
    st.caption(
        "핵심 메시지: 스쿨존 사고는 구조 + 정책(시설) + 노출(어린이)의 결합 결과이다. "
        "도로 구조 단독으로는 설명력이 부족하며, 시설 + 구조위험 통합이 핵심이다."
    )
    if selected_city == "광명시":
        st.info("아래 모델 분석은 성남시 데이터 기반입니다. 개별 시설 분석·로드뷰는 성남시 전용입니다.")

    # 성남시 데이터로 외부검증 AUC 계산 (모델 분석은 항상 성남시 기준)
    if True:
        from sklearn.metrics import roc_auc_score as _roc_auc
        _ext_valid = df_sn.dropna(subset=["structure_risk", "발생건수"])
        _ext_y = (_ext_valid["발생건수"] > 0).astype(int)
        extern_auc = float(_roc_auc(_ext_y, _ext_valid["structure_risk"])) if len(_ext_y.unique()) > 1 else 0.5

        # ── (a) 2단계 모델 AUC 비교 차트 ──
        st.markdown("##### 2단계 모델 AUC 비교")
        auc_data = pd.DataFrame({
            "단계": [
                "1단계\n구조 (CV 이미지)",
                "2단계\n외부검증 (구조→시설데이터)",
                "3단계\n통합 (구조+시설+인구)",
            ],
            "AUC": [struct_auc, extern_auc, integ_auc],
            "색상": ["#F39C12", "#E74C3C", "#27AE60"],
        })
        fig_auc = go.Figure()
        fig_auc.add_trace(go.Bar(
            x=auc_data["단계"], y=auc_data["AUC"],
            marker_color=auc_data["색상"].tolist(),
            text=[f"{v:.3f}" for v in auc_data["AUC"]],
            textposition="outside", textfont=dict(size=16, color="#2C3E50"),
        ))
        fig_auc.add_hline(y=0.5, line_dash="dash", line_color="#E74C3C",
                          annotation_text="무작위 기준선 (0.5)", annotation_position="top left")
        fig_auc.update_layout(
            **PLOTLY_LAYOUT, height=380,
            title="모델 성능 비교 (5-Fold CV AUC)",
            yaxis=dict(title="ROC-AUC", range=[0, 1]),
            xaxis=dict(title=""),
        )
        st.plotly_chart(fig_auc, use_container_width=True)

        st.markdown(
            '<div style="background:linear-gradient(135deg,#FEF9E7,#FDEBD0);padding:14px 18px;'
            'border-radius:10px;border-left:4px solid #27AE60;margin-bottom:16px;">'
            '<b style="color:#2C3E50;">핵심 발견:</b> '
            f'도로 구조만으로는 AUC {struct_auc:.3f}로 제한적이며, '
            f'구조 위험도를 시설 데이터에 직접 적용하면 AUC {extern_auc:.3f}로 오히려 하락합니다. '
            f'그러나 <b>구조 + 9개 시설 + 어린이비율</b>을 통합하면 AUC <b>{integ_auc:.3f}</b>으로 '
            '유의미한 예측력을 확보합니다. (이진 분류: 사고 미발생 / 발생)'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── (a-2) 5종 분류 모델 비교 테이블 ──
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown("##### 분류 모델 비교 (5종)")
        st.caption("115개소 데이터 기반 이진 분류 모델 비교 (사고 발생 여부)")
        _model_comp = pd.DataFrame({
            "모델": ["KNN (K=5)", "XGBoost", "Random Forest", "SVM (RBF)", "Logistic Regression"],
            "Val AUC": [0.857, 0.814, 0.800, 0.771, 0.757],
            "Test AUC": [0.925, "-", "-", "-", "-"],
            "Test Accuracy": ["88.9%", "-", "-", "-", "-"],
            "클래스1 F1": [0.857, 0.667, 0.500, 0.545, 0.769],
            "특징": [
                "AUC 최고, FP=0 (보수적)",
                "Recall 높음, 탐지 우수",
                "피처 중요도 해석용",
                "비선형, Recall 낮음",
                "Accuracy 최고, 계수 해석 가능",
            ],
        })
        st.dataframe(_model_comp, use_container_width=True, hide_index=True)
        st.markdown(
            '<div style="background:#EAFAF1;padding:10px 14px;border-radius:8px;'
            'font-size:13px;color:#2C3E50;">'
            '<b>최적 모델:</b> KNN (AUC=0.925) — 발생 예측 시 확실히 맞추지만 발생을 놓칠 수 있는 보수적 분류기. '
            '개선 2차 모델은 LR 기반 SMOTE + 확률 보정(Calibration) + 상호작용 피처 적용 (CV AUC=0.818).'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ── (b) 통합 모델 계수 해석 차트 ──
        st.markdown("##### 통합 모델 변수 계수 해석 (이진 분류)")
        coef_sorted = integ_coef.sort_values("계수")
        coef_colors = ["#E74C3C" if c > 0 else "#F39C12" for c in coef_sorted["계수"]]

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
            title="통합 모델 — 사고 발생 예측 변수 계수 (양수=위험 증가 / 음수=보호 효과)",
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
                '<div style="background:#FEF9E7;padding:12px 16px;border-radius:8px;'
                'border-left:4px solid #F39C12;">'
                '<b style="color:#2C3E50;">보호 효과 (-) 변수</b><br>'
                '<span style="font-size:13px;color:#2C3E50;">'
                + "<br>".join(f"{r['변수']}: {r['계수']:+.3f}" for _, r in neg_vars.iterrows())
                + '<br><br>예방적 시설의 사고 억제 효과</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # ── (c) 시설별 사고 확률 예측 (개선 모델) ──
        st.markdown("##### 시설별 사고 확률 예측 — 개선 통합 모델")
        st.caption("SMOTE + 확률 보정(Calibration) + 상호작용 피처 적용. 보정 확률 기준 상위 20개소.")

        df_for_prob = df_sn.dropna(subset=["사고확률"]).copy()
        if len(df_for_prob) > 0:
            _imp_with_score = df_for_prob.copy()
            # 개선 모델 안전점수가 있으면 표시
            _prob_cols = ["시설물명", "시설유형", "구", "등급",
                          "사고확률", "발생건수"]
            if "IM_안전점수" in _imp_with_score.columns:
                _prob_cols.insert(5, "IM_안전점수")
                _prob_cols.insert(6, "IM_등급")
            prob_display = _imp_with_score.nlargest(20, "사고확률")[
                [c for c in _prob_cols if c in _imp_with_score.columns]
            ].copy()
            _rename = {"사고확률": "사고확률(보정)", "IM_안전점수": "모델안전점수",
                       "IM_등급": "모델등급", "발생건수": "실제발생건수"}
            prob_display = prob_display.rename(columns=_rename)
            if "사고확률(보정)" in prob_display.columns:
                prob_display["사고확률(보정)"] = prob_display["사고확률(보정)"].map("{:.1%}".format)
            if "모델안전점수" in prob_display.columns:
                prob_display["모델안전점수"] = prob_display["모델안전점수"].round(1)
            st.dataframe(prob_display, use_container_width=True, hide_index=True)
            st.caption(f"사고확률 상위 20개소 (전체 {len(df_for_prob)}개소 분석)")

        st.markdown("---")

        # ── (d) 기존 유지: 등급별 CV 특성 + 레이더 ──
        st.markdown("##### 등급별 도로환경 특성 (CV)")
        cv_cols = ["CV_도로폭확률", "CV_분리장치확률", "CV_도로상대폭", "CV_보행공간비율", "CV_주정차밀도"]
        cv_labels = ["넓은 도로 확률", "분리장치 확률", "도로 상대폭", "보행공간 비율", "주정차 밀도"]
        df_cv = df_sn.dropna(subset=["CV_도로폭확률"])

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
                    "넓은 도로 확률": "#E74C3C", "분리장치 확률": "#F39C12",
                },
            )
            fig_cv_grade.update_layout(**PLOTLY_LAYOUT, height=380)
            st.plotly_chart(fig_cv_grade, use_container_width=True)

        with cv_col2:
            if selected_city != "성남시":
                st.markdown(
                    "<div style='background:#FEF9E7;padding:20px;border-radius:8px;"
                    "text-align:center;color:#E67E22;margin-top:40px;'>"
                    "개별 시설 CV 분석은 성남시에서만 지원됩니다."
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif selected_school != "(전체)" and selected_school in df_cv["시설물명"].values:
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
                    line=dict(color="#E67E22", width=2),
                ))
                fig_cv_radar.add_trace(go.Scatterpolar(
                    r=cv_avg_vals + [cv_avg_vals[0]],
                    theta=cv_theta,
                    fill="toself", name="전체 평균",
                    fillcolor="rgba(46,134,193,0.1)",
                    line=dict(color="#F39C12", width=1, dash="dash"),
                ))
                fig_cv_radar.update_layout(
                    **PLOTLY_LAYOUT,
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], gridcolor="#FDEBD0"),
                        angularaxis=dict(gridcolor="#FDEBD0"),
                        bgcolor="#FAFCFF",
                    ),
                    title=f"{selected_school} 도로환경 프로필",
                    height=380, showlegend=True,
                    legend=dict(x=0.01, y=0.99),
                )
                st.plotly_chart(fig_cv_radar, use_container_width=True)

                # ── (d-1) 로드뷰 + CV 게이지 오버레이 ──
                _cv_rv_name = selected_school.replace(" ", "_")
                _cv_rv_path = DATA_DIR / "roadview" / f"{_cv_rv_name}_북쪽.jpg"
                if not _cv_rv_path.exists():
                    _cv_rv_path = DATA_DIR / "roadview" / f"{selected_school}_북쪽.jpg"

                st.markdown("##### 로드뷰 + CV 분석 결과")
                _rv_col, _gauge_col = st.columns([3, 2])
                with _rv_col:
                    if _cv_rv_path.exists():
                        st.image(str(_cv_rv_path), caption=f"{selected_school} 북쪽 방향", use_container_width=True)
                    else:
                        st.info("로드뷰 이미지 없음")
                with _gauge_col:
                    _cv_items = [
                        ("넓은 도로", cv_row["CV_도로폭확률"], "#E74C3C", "높을수록 넓은 도로"),
                        ("차단시설", cv_row["CV_분리장치확률"], "#27AE60", "높을수록 안전"),
                        ("도로 비율", cv_row["CV_도로상대폭"], "#3498DB", "화면 내 도로 면적"),
                        ("보행공간", cv_row["CV_보행공간비율"], "#8E44AD", "높을수록 보행자 안전"),
                        ("주정차", min(cv_row["CV_주정차밀도"] / 5, 1.0), "#F39C12", "높을수록 시야 방해"),
                    ]
                    _gauge_html = ""
                    for _lbl, _val, _clr, _desc in _cv_items:
                        _pct = min(_val * 100, 100)
                        _gauge_html += (
                            f'<div style="margin-bottom:10px;">'
                            f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#2C3E50;">'
                            f'<span style="font-weight:600;">{_lbl}</span>'
                            f'<span>{_pct:.0f}%</span></div>'
                            f'<div style="background:#ECF0F1;border-radius:6px;height:14px;overflow:hidden;">'
                            f'<div style="width:{_pct:.0f}%;height:100%;background:{_clr};border-radius:6px;'
                            f'transition:width 0.3s;"></div></div>'
                            f'<div style="font-size:10px;color:#7F8C8D;margin-top:1px;">{_desc}</div>'
                            f'</div>'
                        )
                    st.markdown(
                        f'<div style="background:#FAFCFF;padding:14px 16px;border-radius:10px;'
                        f'border:1px solid #F5CBA7;">'
                        f'<div style="font-weight:700;color:#2C3E50;margin-bottom:10px;font-size:14px;">'
                        f'CV 분석 지표</div>{_gauge_html}</div>',
                        unsafe_allow_html=True,
                    )

                # ── (d-2) 유사 도로환경 학교 매칭 ──
                st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                st.markdown("##### 유사 도로환경 학교")
                st.caption("CV 5개 지표 기반 코사인 유사도 — 도로환경이 비슷한 학교를 비교합니다.")
                _cv_feats = cv_cols
                _cv_valid = df_cv.dropna(subset=_cv_feats).copy()
                if len(_cv_valid) > 1 and selected_school in _cv_valid["시설물명"].values:
                    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
                    _cv_mat = _cv_valid[_cv_feats].values
                    _sel_idx = _cv_valid[_cv_valid["시설물명"] == selected_school].index[0]
                    _sel_vec = _cv_valid.loc[_sel_idx, _cv_feats].values.reshape(1, -1)
                    _sims = _cos_sim(_sel_vec, _cv_mat)[0]
                    _cv_valid["유사도"] = _sims
                    _similar = _cv_valid[_cv_valid["시설물명"] != selected_school].nlargest(5, "유사도")

                    _sim_cols = st.columns(5)
                    for _si, (_, _sr) in enumerate(_similar.iterrows()):
                        with _sim_cols[_si]:
                            _s_rv = _sr["시설물명"].replace(" ", "_")
                            _s_path = DATA_DIR / "roadview" / f"{_s_rv}_북쪽.jpg"
                            if not _s_path.exists():
                                _s_path = DATA_DIR / "roadview" / f"{_sr['시설물명']}_북쪽.jpg"
                            if _s_path.exists():
                                st.image(str(_s_path), use_container_width=True)
                            _s_grade = _sr.get("등급", _sr.get("등급_V6", "?"))
                            _s_color = GRADE_COLORS.get(_s_grade, "#999")
                            st.markdown(
                                f'<div style="text-align:center;font-size:12px;">'
                                f'<b>{_sr["시설물명"]}</b><br>'
                                f'<span style="background:{_s_color};color:#fff;padding:1px 8px;'
                                f'border-radius:10px;font-size:11px;">{_s_grade}</span> '
                                f'유사도 {_sr["유사도"]:.0%}</div>',
                                unsafe_allow_html=True,
                            )

            else:
                st.markdown(
                    "<div style='background:#FEF9E7;padding:20px;border-radius:8px;"
                    "text-align:center;color:#E67E22;margin-top:40px;'>"
                    "사이드바에서 개별 시설을 선택하면<br>도로환경 레이더 차트와 로드뷰 이미지가 표시됩니다."
                    "</div>",
                    unsafe_allow_html=True,
                )

        # ── (e) A등급 vs D등급 로드뷰 비교 (성남시 전용) ──
        st.markdown("---")
        if selected_city != "성남시":
            st.info("A등급 vs D등급 로드뷰 비교는 성남시에서만 지원됩니다.")
        _a_schools = df_cv[df_cv["등급"] == "A"].copy()
        _d_schools = df_cv[df_cv["등급"] == "D"].copy()
        if selected_city == "성남시" and len(_a_schools) > 0 and len(_d_schools) > 0:
            st.markdown("##### A등급 vs D등급 도로환경 비교")
            st.caption("안전등급 최상위(A)와 최하위(D) 학교의 로드뷰 · CV 지표를 직접 비교합니다.")
            # 대표 선정: A등급 중 활성_안전점수 최고, D등급 중 최저
            _a_rep = _a_schools.sort_values("활성_안전점수", ascending=False).iloc[0]
            _d_rep = _d_schools.sort_values("활성_안전점수", ascending=True).iloc[0]

            _ad_col1, _ad_col2 = st.columns(2)

            for _ad_col, _ad_row, _ad_label, _ad_border, _ad_bg in [
                (_ad_col1, _a_rep, "A등급 (최상위)", "#27AE60", "#EAFAF1"),
                (_ad_col2, _d_rep, "D등급 (최하위)", "#E74C3C", "#FDEDEC"),
            ]:
                with _ad_col:
                    st.markdown(
                        f'<div style="background:{_ad_bg};padding:12px 14px;border-radius:10px;'
                        f'border:2px solid {_ad_border};margin-bottom:8px;">'
                        f'<div style="font-weight:700;color:{_ad_border};font-size:15px;'
                        f'text-align:center;margin-bottom:6px;">{_ad_label}</div>'
                        f'<div style="text-align:center;font-size:13px;color:#2C3E50;'
                        f'font-weight:600;">{_ad_row["시설물명"]}</div></div>',
                        unsafe_allow_html=True,
                    )
                    _ad_rv = _ad_row["시설물명"].replace(" ", "_")
                    _ad_path = DATA_DIR / "roadview" / f"{_ad_rv}_북쪽.jpg"
                    if not _ad_path.exists():
                        _ad_path = DATA_DIR / "roadview" / f"{_ad_row['시설물명']}_북쪽.jpg"
                    if _ad_path.exists():
                        st.image(str(_ad_path), use_container_width=True)
                    else:
                        st.info("로드뷰 이미지 없음")

                    # CV 게이지 비교 (소형)
                    _ad_items = [
                        ("넓은도로", _ad_row["CV_도로폭확률"], "#E74C3C"),
                        ("차단시설", _ad_row["CV_분리장치확률"], "#27AE60"),
                        ("도로비율", _ad_row["CV_도로상대폭"], "#3498DB"),
                        ("보행공간", _ad_row["CV_보행공간비율"], "#8E44AD"),
                        ("주정차", min(_ad_row["CV_주정차밀도"] / 5, 1.0), "#F39C12"),
                    ]
                    _ad_html = ""
                    for _al, _av, _ac in _ad_items:
                        _ap = min(_av * 100, 100)
                        _ad_html += (
                            f'<div style="margin-bottom:6px;">'
                            f'<div style="display:flex;justify-content:space-between;font-size:11px;">'
                            f'<span>{_al}</span><span>{_ap:.0f}%</span></div>'
                            f'<div style="background:#ECF0F1;border-radius:4px;height:10px;overflow:hidden;">'
                            f'<div style="width:{_ap:.0f}%;height:100%;background:{_ac};border-radius:4px;">'
                            f'</div></div></div>'
                        )
                    st.markdown(
                        f'<div style="padding:8px 10px;border-radius:8px;'
                        f'border:1px solid #D5D8DC;">{_ad_html}</div>',
                        unsafe_allow_html=True,
                    )

            # 지표 차이 요약
            st.markdown(
                '<div style="background:#F8F9F9;padding:10px 14px;border-radius:8px;'
                'margin-top:8px;font-size:13px;color:#2C3E50;">'
                '<b>핵심 차이:</b> A등급은 차단시설·보행공간이 높고, 주정차 밀도가 낮은 패턴을 보입니다. '
                'D등급은 도로폭이 넓어도 보행자 보호 시설이 부족합니다.</div>',
                unsafe_allow_html=True,
            )


    # ============================
    # Tab 5: 동네정보
    # ============================
with tab_district:
    if selected_city == "성남시":
        _dist_pop = load_population()
    else:
        _dist_pop = load_gm_population().dropna(subset=["어린이_비율"])
    nat_df = load_national_stats()

    # 어린이 비율
    pop_sorted = _dist_pop.sort_values("어린이_비율", ascending=True)
    _pop_height = max(300, len(pop_sorted) * 35)
    fig_pop = px.bar(
        pop_sorted, x="어린이_비율", y="동명", orientation="h",
        title=f"{selected_city} 행정동별 어린이(0~14세) 비율",
        labels={"어린이_비율": "어린이 비율 (%)", "동명": ""},
        color="어린이_비율",
        color_continuous_scale=[[0, "#FEF9E7"], [0.5, "#F39C12"], [1, "#E67E22"]],
    )
    fig_pop.update_layout(**PLOTLY_LAYOUT, height=_pop_height, coloraxis_showscale=False)
    st.plotly_chart(fig_pop, use_container_width=True)

    # 전국 추이
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=nat_df["발생년"], y=nat_df["사고건수"],
        mode="lines+markers", name="사고건수",
        line=dict(color="#2C3E50", width=3),
        marker=dict(size=9, color="#2C3E50"),
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
                    bordercolor="#F5CBA7", borderwidth=1),
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
            title="성남 인근 주요 국도 등하교 시간대 평균 교통량 (성남시 기준)",
            labels={"평균교통량": "평균 교통량 (대)", "호선명": "", "시간대": ""},
            color_discrete_map={"등교": "#E67E22", "하교": "#F1C40F"},
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
# Tab 6: 광명 시뮬레이션
# ============================
with tab_sim:
    st.markdown("### 광명시 스쿨존 시뮬레이션")
    st.caption(
        "성남시 모델을 광명시 51개소에 적용하여 예상 안전점수·등급을 산출합니다. "
        "광명시에 없는 시설(도로안전표지, CCTV, 단속카메라)은 0으로 처리됩니다."
    )

    # 이미 전역에서 계산된 df_gm 사용
    gm_result = df_gm[["시설물명", "시설유형", "위도", "경도", "활성_안전점수", "등급"]].copy()
    gm_result = gm_result.rename(columns={"활성_안전점수": "예상점수", "등급": "예상등급"})
    gm_result["발생건수"] = df_gm["발생건수"].fillna(0).astype(int)
    gm_result["어린이비율"] = df_gm["어린이비율"].fillna(10.0).round(1)

    # ── (a) KPI ──
    gm_k1, gm_k2, gm_k3, gm_k4 = st.columns(4)
    gm_k1.metric("광명 시설 수", f"{len(gm_result)}개소")
    gm_k2.metric("평균 예상점수", f"{gm_result['예상점수'].mean():.1f}")
    gm_safe_pct = (gm_result["예상등급"].isin(["A", "B"])).sum() / len(gm_result) * 100
    gm_k3.metric("안전(A+B) 비율", f"{gm_safe_pct:.0f}%")
    gm_k4.metric("모델 R²", f"{model_r2:.3f}")

    # ── (b) 광명 지도 ──
    gm_map = folium.Map(
        location=[df_gm["위도"].mean(), df_gm["경도"].mean()],
        zoom_start=13, tiles=None,
    )
    folium.TileLayer(
        tiles="https://mt0.google.com/vt/lyrs=r&hl=ko&x={x}&y={y}&z={z}",
        attr="Google", name="기본 지도", max_zoom=22,
    ).add_to(gm_map)
    for _, gm_r in gm_result.iterrows():
        gm_color = GRADE_COLORS.get(gm_r["예상등급"], "#999")
        folium.CircleMarker(
            [gm_r["위도"], gm_r["경도"]],
            radius=8, color="#fff", weight=2,
            fill=True, fill_color=gm_color, fill_opacity=0.9,
            tooltip=f"{gm_r['시설물명']} ({gm_r['시설유형']}) — {gm_r['예상등급']} ({gm_r['예상점수']}점)",
        ).add_to(gm_map)
    st_folium(gm_map, height=450, use_container_width=True, returned_objects=[])

    # ── (c) 등급 분포 + 예측 결과 테이블 ──
    gm_col1, gm_col2 = st.columns(2)
    with gm_col1:
        gm_grade_cnt = gm_result["예상등급"].value_counts().reindex(["A", "B", "C", "D"]).fillna(0).astype(int)
        fig_gm_pie = px.pie(
            names=gm_grade_cnt.index, values=gm_grade_cnt.values,
            title="광명시 예상 등급 분포",
            color=gm_grade_cnt.index,
            color_discrete_map=GRADE_COLORS,
        )
        fig_gm_pie.update_layout(**PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig_gm_pie, use_container_width=True)

    with gm_col2:
        st.markdown("##### 예측 결과 (점수 하위순)")
        gm_display = gm_result.sort_values("예상점수")[
            ["시설물명", "시설유형", "예상등급", "예상점수", "발생건수"]
        ].reset_index(drop=True)
        gm_display.index = gm_display.index + 1
        st.dataframe(gm_display, use_container_width=True, height=350)

    st.markdown("---")

    # ── (d) 성남 대비 비교 ──
    st.markdown("##### 광명 vs 성남 비교")
    gs_gavg_score = df_sn.groupby("등급_V6")["최종안전점수_V6"].mean().reindex(["A", "B", "C", "D"])
    gs_gavg_fac = df_sn.groupby("등급_V6")[FACILITY_COLS].mean().reindex(["A", "B", "C", "D"])

    gm_comp_col1, gm_comp_col2 = st.columns(2)
    with gm_comp_col1:
        gm_scomp = pd.DataFrame({
            "구분": ["광명 평균"] + [f"성남 {g}등급" for g in ["A", "B", "C", "D"]],
            "안전점수": [gm_result["예상점수"].mean()] + gs_gavg_score.tolist(),
        })
        fig_gm_sc = px.bar(
            gm_scomp, x="구분", y="안전점수",
            title="예상 안전점수: 광명 평균 vs 성남 등급별",
            color="구분",
            color_discrete_sequence=["#F39C12", "#154360", "#2471A3", "#85C1E9", "#E74C3C"],
            text="안전점수",
        )
        fig_gm_sc.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_gm_sc.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False)
        st.plotly_chart(fig_gm_sc, use_container_width=True)

    with gm_comp_col2:
        # 광명 평균 시설물 vs 성남 A/D등급
        gm_fac_avg = df_gm[FACILITY_COLS].mean()
        gm_fcomp = []
        for gs_f in FACILITY_COLS:
            gm_fcomp.append({"시설물": gs_f, "구분": "광명", "수량": round(float(gm_fac_avg[gs_f]), 1)})
            gm_fcomp.append({"시설물": gs_f, "구분": "성남 A등급", "수량": round(gs_gavg_fac.loc["A", gs_f], 1)})
            gm_fcomp.append({"시설물": gs_f, "구분": "성남 D등급", "수량": round(gs_gavg_fac.loc["D", gs_f], 1)})
        fig_gm_fc = px.bar(
            pd.DataFrame(gm_fcomp), x="시설물", y="수량",
            color="구분", barmode="group",
            title="시설물 평균: 광명 vs 성남 A/D등급",
            color_discrete_map={"광명": "#F39C12", "성남 A등급": "#154360", "성남 D등급": "#E74C3C"},
        )
        fig_gm_fc.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig_gm_fc, use_container_width=True)

    # ── (e) D등급 개선 제안 ──
    gm_d = gm_result[gm_result["예상등급"] == "D"]
    if len(gm_d) > 0:
        st.markdown("##### 광명시 D등급 예상 시설 — 우선 개선 대상")
        for _, gm_r in gm_d.sort_values("예상점수").iterrows():
            gm_row_data = df_gm[df_gm["시설물명"] == gm_r["시설물명"]].iloc[0]
            worst, worst_pct = None, 1.0
            for f in FACILITY_COLS:
                mx = df_gm[f].max()
                if mx > 0:
                    pct = gm_row_data[f] / mx
                    if pct < worst_pct:
                        worst_pct, worst = pct, f
            suggestion = f"{worst} 보강 필요 (현재 {int(gm_row_data[worst])}개)" if worst else "추가 분석 필요"
            st.markdown(
                f'<div class="suggestion-card">'
                f'<span class="school-name">{gm_r["시설물명"]}</span> '
                f'<span style="font-size:11px;color:#34495E;">({gm_r["시설유형"]})</span> &nbsp; '
                f'<span style="background:#E74C3C;color:#fff;padding:2px 10px;'
                f'border-radius:20px;font-size:11px;">D ({gm_r["예상점수"]}점)</span>'
                f'<div class="suggestion">개선 제안: {suggestion}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ============================
# Tab 7: 분석 방법론
# ============================
with tab_method:
    st.markdown("### 분석 방법론")
    st.caption("스쿨존 안전등급 분석에 사용된 데이터, 변수, 모델을 설명합니다.")

    # ── (a) 프로젝트 개요 ──
    st.markdown("##### 프로젝트 개요")
    st.markdown(
        '<div style="background:#FEF5E7;padding:16px 20px;border-radius:10px;'
        'border-left:4px solid #E67E22;margin-bottom:16px;">'
        '<span style="font-size:14px;color:#2C3E50;">'
        '<b style="color:#2C3E50;">목표:</b> 어린이 보호구역(스쿨존)의 '
        '안전등급을 데이터 기반으로 분석하여, 시설물 투자 우선순위를 제공합니다.<br><br>'
        '<b style="color:#2C3E50;">분석 대상:</b> 성남시 142개소 + 광명시 51개소<br>'
        '<b style="color:#2C3E50;">분석 기간:</b> 2018~2023년 사고 데이터 + 2024년 시설 현황<br>'
        '<b style="color:#2C3E50;">핵심 메시지:</b> 스쿨존 사고는 <b>도로 구조 + 정책(시설) + 노출(어린이)</b>의 결합 결과이며, '
        '시설물 투입이 사고 예방의 핵심이다.'
        '</span></div>',
        unsafe_allow_html=True,
    )

    # ── (a-1) 시스템 아키텍처 ──
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
    st.markdown("##### 시스템 아키텍처")
    _arch_path = DATA_DIR / "system_architecture.jpg"
    if _arch_path.exists():
        st.image(str(_arch_path), caption="데이터 수집 → AI 비전 분석 → ML 통합 분석 → 서비스 인터페이스", use_container_width=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── (b) 사용 변수 목록 테이블 ──
    st.markdown("##### 사용 변수 목록")

    _var_data = [
        # 시설 변수
        {"카테고리": "시설", "변수명": "도로적색표면", "설명": "보호구역 내 적색 도로 표면 개수", "출처": "공공데이터포털", "범위": "0~40+"},
        {"카테고리": "시설", "변수명": "신호등", "설명": "보호구역 내 교통 신호등 수", "출처": "공공데이터포털", "범위": "0~20+"},
        {"카테고리": "시설", "변수명": "횡단보도", "설명": "보호구역 내 횡단보도 수", "출처": "공공데이터포털", "범위": "0~15+"},
        {"카테고리": "시설", "변수명": "도로안전표지", "설명": "속도 제한, 주의 표지 등 안전표지 수", "출처": "공공데이터포털", "범위": "0~15+"},
        {"카테고리": "시설", "변수명": "생활안전CCTV", "설명": "보호구역 300m 내 생활안전 CCTV 수", "출처": "공공데이터포털", "범위": "0~50+"},
        {"카테고리": "시설", "변수명": "무인교통단속카메라", "설명": "보호구역 내 과속/신호 단속 카메라 수", "출처": "공공데이터포털", "범위": "0~10"},
        {"카테고리": "시설", "변수명": "보호구역표지판", "설명": "어린이 보호구역 안내 표지판 수", "출처": "공공데이터포털", "범위": "0~80+"},
        {"카테고리": "시설", "변수명": "옐로카펫", "설명": "횡단보도 앞 노란색 안전 구역 수", "출처": "공공데이터포털", "범위": "0~5"},
        {"카테고리": "시설", "변수명": "무단횡단방지펜스", "설명": "무단횡단 방지용 보행자 펜스 수", "출처": "공공데이터포털", "범위": "0~30+"},
        # 사고 변수
        {"카테고리": "사고", "변수명": "발생건수", "설명": "2018~2023 스쿨존 교통사고 건수", "출처": "도로교통공단", "범위": "0~38"},
        {"카테고리": "사고", "변수명": "사고심각도", "설명": "사망x10 + 중상x5 + 경상x3 + 부상x1", "출처": "산출", "범위": "0~200+"},
        # 인구 변수
        {"카테고리": "인구", "변수명": "어린이비율", "설명": "행정동 0~14세 어린이 인구 비율 (%)", "출처": "경기데이터드림", "범위": "5~25%"},
        # CV 변수
        {"카테고리": "CV (도로환경)", "변수명": "CV_도로폭확률", "설명": "CNN 예측 넓은 도로 확률", "출처": "카카오맵 로드뷰", "범위": "0~1"},
        {"카테고리": "CV (도로환경)", "변수명": "CV_분리장치확률", "설명": "CNN 예측 차도-보도 분리 장치 확률", "출처": "카카오맵 로드뷰", "범위": "0~1"},
        {"카테고리": "CV (도로환경)", "변수명": "CV_도로상대폭", "설명": "이미지 내 도로가 차지하는 비율", "출처": "카카오맵 로드뷰", "범위": "0~1"},
        {"카테고리": "CV (도로환경)", "변수명": "CV_보행공간비율", "설명": "이미지 내 보행 공간 비율", "출처": "카카오맵 로드뷰", "범위": "0~1"},
        {"카테고리": "CV (도로환경)", "변수명": "CV_주정차밀도", "설명": "이미지 내 주정차 차량 밀도", "출처": "카카오맵 로드뷰", "범위": "0~10+"},
    ]
    st.dataframe(pd.DataFrame(_var_data), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── (b-1) 스케일링 전후 비교 차트 ──
    st.markdown("##### 전처리: 스케일링 전후 Skewness 비교")
    st.caption("skewness > 1인 변수에 Log(x+1) 변환 → StandardScaler 적용. 나머지는 StandardScaler만 적용.")

    _sum_sn_path = DATA_DIR / "feature_summary_sn.csv"
    _sum_gm_path = DATA_DIR / "feature_summary_gm.csv"
    if _sum_sn_path.exists() and _sum_gm_path.exists():
        _sum_sn = pd.read_csv(_sum_sn_path, index_col=0)
        _sum_gm = pd.read_csv(_sum_gm_path, index_col=0)

        _sk_col1, _sk_col2 = st.columns(2)

        for _sk_col, _sk_df, _sk_title, _sk_clr_before, _sk_clr_after in [
            (_sk_col1, _sum_sn, "성남시 (142개소)", "#E67E22", "#27AE60"),
            (_sk_col2, _sum_gm, "광명시 (51개소)", "#E67E22", "#27AE60"),
        ]:
            with _sk_col:
                _sk_labels = _sk_df.index.tolist()
                fig_sk = go.Figure()
                fig_sk.add_trace(go.Bar(
                    name="변환 전", x=_sk_labels, y=_sk_df["skewness_전"],
                    marker_color=_sk_clr_before, opacity=0.7,
                ))
                fig_sk.add_trace(go.Bar(
                    name="변환 후", x=_sk_labels, y=_sk_df["skewness_후"],
                    marker_color=_sk_clr_after, opacity=0.9,
                ))
                fig_sk.add_hline(y=1.0, line_dash="dash", line_color="#E74C3C",
                                 annotation_text="skew=1 기준", annotation_position="top left",
                                 annotation_font_size=10)
                fig_sk.update_layout(
                    **PLOTLY_LAYOUT, height=350, barmode="group",
                    title=_sk_title,
                    xaxis=dict(title="", tickangle=-45, tickfont=dict(size=9)),
                    yaxis=dict(title="Skewness"),
                    legend=dict(x=0.01, y=0.99, font=dict(size=10)),
                    margin=dict(b=80),
                )
                st.plotly_chart(fig_sk, use_container_width=True)

        # 변환 방식 요약
        _log_vars = _sum_sn[_sum_sn["변환방식"] == "Log+Standard"].index.tolist()
        st.markdown(
            f'<div style="background:#EAFAF1;padding:10px 14px;border-radius:8px;'
            f'font-size:13px;color:#2C3E50;margin-bottom:16px;">'
            f'<b>Log(x+1) + StandardScaler 적용 변수:</b> {", ".join(_log_vars)}<br>'
            f'<b>StandardScaler만 적용:</b> 나머지 {len(_sum_sn) - len(_log_vars)}개 변수</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── (c) 안전점수 계산 방법 ──
    st.markdown("##### 안전점수 계산 방법")
    st.markdown(
        '<div style="background:#FEF9E7;padding:16px 20px;border-radius:10px;'
        'border-left:4px solid #F39C12;margin-bottom:12px;">'
        '<b style="color:#2C3E50;font-size:15px;">안전점수 산출 공식</b><br><br>'
        '<span style="font-size:14px;color:#2C3E50;">'
        '<code style="background:#F5CBA7;padding:6px 12px;border-radius:6px;font-size:14px;">'
        '안전점수 = 기본(50) + 가산점(시설) + 가산점(보너스) - 감산점(사고+환경)'
        '</code><br><br>'
        '<b>가산점(시설):</b> 9개 시설물 보유량 기반 점수 (많을수록 가산)<br>'
        '<b>가산점(보너스):</b> 어린이비율 등 환경적 보너스<br>'
        '<b>감산점:</b> 사고 발생건수, 사고심각도 기반 감점'
        '</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="background:#FEF5E7;padding:14px 18px;border-radius:10px;'
        'border-left:4px solid #154360;margin-bottom:16px;">'
        '<b style="color:#2C3E50;">등급 기준 (사분위수)</b><br>'
        '<span style="font-size:13px;color:#2C3E50;">'
        '<b style="color:#154360;">A등급 (우수)</b>: 상위 25% &nbsp;&nbsp;|&nbsp;&nbsp; '
        '<b style="color:#2471A3;">B등급 (양호)</b>: 25~50% &nbsp;&nbsp;|&nbsp;&nbsp; '
        '<b style="color:#85C1E9;">C등급 (보통)</b>: 50~75% &nbsp;&nbsp;|&nbsp;&nbsp; '
        '<b style="color:#E74C3C;">D등급 (주의)</b>: 하위 25%'
        '</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── (d) 2단계 사고 예측 모델 ──
    st.markdown("##### 2단계 사고 예측 모델 (개선)")
    st.markdown(
        '<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">'
        # 1단계
        '<div style="flex:1;min-width:200px;background:linear-gradient(135deg,#FEF9E7,#FCF3CF);'
        'padding:14px 16px;border-radius:10px;border-top:4px solid #F39C12;">'
        '<b style="color:#F39C12;font-size:15px;">1단계: 구조 모델</b><br>'
        '<span style="font-size:12px;color:#2C3E50;">'
        '카카오맵 로드뷰 이미지 분석<br>'
        '5개 CV 변수 (도로폭, 분리장치, 보행공간 등)<br>'
        'Binary 분류 (사고 부근 여부)<br>'
        '<b>Logistic Regression</b>'
        '</span></div>'
        # 화살표
        '<div style="display:flex;align-items:center;font-size:24px;color:#2C3E50;">&#10132;</div>'
        # 2단계
        '<div style="flex:1;min-width:200px;background:linear-gradient(135deg,#FDEBD0,#EAFAF1);'
        'padding:14px 16px;border-radius:10px;border-top:4px solid #27AE60;">'
        '<b style="color:#27AE60;font-size:15px;">2단계: 통합 모델 (개선)</b><br>'
        '<span style="font-size:12px;color:#2C3E50;">'
        'structure_risk + 9개 시설 + 어린이비율<br>'
        'SMOTE + 확률보정(Calibration)<br>'
        'Log변환 + 상호작용 피처 5개 추가<br>'
        '<b>이진 분류: 사고 미발생 / 발생</b>'
        '</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _m_col1, _m_col2 = st.columns(2)
    with _m_col1:
        st.markdown(
            '<div style="background:#FEF5E7;padding:12px 16px;border-radius:8px;'
            'border-left:4px solid #F39C12;">'
            '<b style="color:#2C3E50;">이진 분류 라벨링</b><br>'
            '<span style="font-size:13px;color:#2C3E50;">'
            '사고 0건 → <b>미발생</b> (110개소, 94.0%)<br>'
            '사고 1건+ → <b>발생</b> (7개소, 6.0%)<br><br>'
            '<b>개선사항:</b> log1p 변환, 상호작용 피처,<br>'
            'SMOTE 오버샘플링, CalibratedClassifierCV,<br>'
            'PR 커브 기반 최적 임계값 적용'
            '</span></div>',
            unsafe_allow_html=True,
        )
    with _m_col2:
        st.markdown(
            f'<div style="background:#FDEBD0;padding:12px 16px;border-radius:8px;'
            f'border-left:4px solid #27AE60;">'
            f'<b style="color:#2C3E50;">모델 성능</b><br>'
            f'<span style="font-size:13px;color:#2C3E50;">'
            f'1단계 구조 모델 AUC: <b>{struct_auc:.3f}</b><br>'
            f'2단계 통합 모델 CV AUC: <b>{integ_auc:.3f}</b><br>'
            f'개선 모델 CV AUC: <b>0.818</b> (SMOTE+Calibration)<br>'
            f'모델 비교 최적: <b>KNN AUC 0.925</b> (Test)<br>'
            f'Recall: 0.857 | F1: 0.86'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── (e) 광명 시뮬레이션 설명 ──
    st.markdown("##### 광명시 시뮬레이션 방법")
    st.markdown(
        f'<div style="background:#FEF9E7;padding:16px 20px;border-radius:10px;'
        f'border-left:4px solid #F39C12;margin-bottom:16px;">'
        f'<span style="font-size:14px;color:#2C3E50;">'
        f'<b style="color:#F39C12;">안전점수 예측 모델</b><br>'
        f'성남시 데이터로 학습한 <b>LinearRegression</b> 모델 (R² = {model_r2:.3f})을 '
        f'광명시 51개소에 적용<br>'
        f'입력: 9개 시설물 수량 + 발생건수 + 어린이비율<br>'
        f'출력: 예상 안전점수 → 사분위수 기반 등급 부여<br><br>'
        f'<b style="color:#F39C12;">참고</b><br>'
        f'광명시에 없는 시설 데이터(도로안전표지, 생활안전CCTV, 무인교통단속카메라)는 '
        f'0으로 처리되어 실제보다 낮게 예측될 수 있습니다.'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── (f) 데이터 출처 ──
    st.markdown("##### 데이터 출처")
    st.markdown(
        '<div style="background:#FEF5E7;padding:16px 20px;border-radius:10px;">'
        '<table style="width:100%;font-size:13px;color:#2C3E50;border-collapse:collapse;">'
        '<tr style="background:#F5CBA7;font-weight:600;color:#2C3E50;">'
        '<td style="padding:8px 12px;">출처</td>'
        '<td style="padding:8px 12px;">데이터 내용</td>'
        '<td style="padding:8px 12px;">기간</td></tr>'
        '<tr><td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">공공데이터포털</td>'
        '<td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">스쿨존 목록, 9개 시설물 현황</td>'
        '<td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">2024</td></tr>'
        '<tr><td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">도로교통공단</td>'
        '<td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">어린이보호구역 교통사고 통계</td>'
        '<td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">2018~2023</td></tr>'
        '<tr><td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">경기데이터드림</td>'
        '<td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">행정동별 연령별 인구 (어린이비율)</td>'
        '<td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">2024</td></tr>'
        '<tr><td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">광명시</td>'
        '<td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">광명시 어린이보호구역 시설물·사고 데이터</td>'
        '<td style="padding:6px 12px;border-bottom:1px solid #F5CBA7;">2024</td></tr>'
        '<tr><td style="padding:6px 12px;">카카오맵 로드뷰</td>'
        '<td style="padding:6px 12px;">스쿨존 도로환경 이미지 (CV 분석용)</td>'
        '<td style="padding:6px 12px;">2023~2024</td></tr>'
        '</table></div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 7. Footer
# ──────────────────────────────────────────────
st.markdown(
    '<div class="footer-text">'
    "데이터 출처: 공공데이터포털, 도로교통공단, 경기데이터드림, 성남시, 광명시"
    "</div>",
    unsafe_allow_html=True,
)

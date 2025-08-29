#######################
# Import libraries

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

#######################
# Page configuration
st.set_page_config(
    page_title="출생 통계 대시보드",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("default")

#######################
# CSS styling
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}
[data-testid="stMetric"] {
    background-color: #2a2a2a;
    border-radius: 12px;
    text-align: center;
    padding: 14px 0;
    border: 1px solid #3a3a3a;
}
[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}
[data-testid="stMetricDeltaIcon-Up"],
[data-testid="stMetricDeltaIcon-Down"]{
    position: relative;
    left: 38%;
    transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

#######################
# 시도코드 ↔ 시도명
REGION_MAP = {
    11: "서울특별시", 21: "부산광역시", 22: "대구광역시", 23: "인천광역시",
    24: "광주광역시", 25: "대전광역시", 26: "울산광역시", 29: "세종특별자치시",
    31: "경기도", 32: "강원도", 33: "충청북도", 34: "충청남도",
    35: "전라북도", 36: "전라남도", 37: "경상북도", 38: "경상남도", 39: "제주특별자치도",
}
REGION_MAP_REV = {v: k for k, v in REGION_MAP.items()}

#######################
# 변수 코드 → 라벨 매핑
AGE_MAP = {
    1:"0~14세", 2:"15~19세", 3:"20~24세", 4:"25~29세", 5:"30~34세",
    6:"35~39세", 7:"40~44세", 8:"45~49세", 9:"50세 이상", 99:"미상"
}
MULTI_ORDER_MAP = {1:"첫째", 2:"둘째", 3:"셋째", 4:"넷째", 9:"미상"}
MULTI_TYPE_MAP  = {1:"단태아", 2:"쌍태아", 3:"삼태아이상", 9:"미상"}
MOTHER_TOTAL_CHILD_MAP = {
    1:"1명", 2:"2명", 3:"3명", 4:"4명", 5:"5명", 6:"6명", 7:"7명", 8:"8명 이상", 99:"미상"
}
NATIONALITY_MAP = {1:"출생한국인", 2:"귀화한국인", 3:"외국", 9:"미상"}

#######################
# Load data  (인코딩 이슈 안전 처리)
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    encodings = ["cp949", "euc-kr", "utf-8-sig", "utf-8", "ISO-8859-1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="cp949", encoding_errors="ignore")

df_reshaped = load_data("chd_2023.csv")

#######################
# Plotly 공통 레이아웃(다크/투명 배경) 헬퍼
def apply_dark_layout(fig: go.Figure, title=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6e6e6"),
        margin=dict(l=10, r=10, t=50 if title else 30, b=10),
        title=dict(text=title, x=0.02, xanchor="left") if title else None,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, linecolor="rgba(255,255,255,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, linecolor="rgba(255,255,255,0.15)")
    return fig

#######################
# Sidebar
with st.sidebar:
    st.header("출생 통계 대시보드")
    st.caption("필터 선택에 따라 본문 시각화가 업데이트됩니다.")

    def to_int_series(s, dropna=True):
        x = pd.to_numeric(s, errors="coerce").astype("Int64")
        return x.dropna().astype(int) if dropna else x

    # 옵션
    years = sorted(to_int_series(df_reshaped["연도"]).unique().tolist())
    present_codes = sorted(to_int_series(df_reshaped["출생자주소지_행정구역시도코드"]).unique().tolist())
    present_names = [REGION_MAP[c] for c in present_codes if c in REGION_MAP]
    months = list(range(1, 13))

    # 위젯
    selected_year = st.selectbox("연도 선택", options=["전체"] + years, index=0)

    selected_regions_name = st.multiselect(
        "시도 선택 (다중)", options=present_names, default=present_names, help="출생자 주소지 시도 기준"
    )

    gender_map = {1: "남", 2: "여"}
    selected_genders = st.multiselect("성별", options=list(gender_map.values()), default=list(gender_map.values()))
    selected_months = st.multiselect("출생월", options=months, default=months)

    multiple_map = MULTI_TYPE_MAP.copy()
    selected_multiple = st.selectbox("다태아 분류", options=["전체"] + list(multiple_map.values()), index=0)

    marital_map = {1: "혼인 중 출생", 2: "혼인 외 출생"}
    selected_marital = st.selectbox("혼인 상태(출생 유형)", options=["전체"] + list(marital_map.values()), index=0)

    wt = pd.to_numeric(df_reshaped["출생아체중"], errors="coerce")
    wt_min, wt_max = float(wt.min()), float(wt.max())
    selected_wt = st.slider("출생 체중(kg) 범위",
        min_value=round(max(wt_min, 0.0), 2), max_value=round(wt_max, 2),
        value=(round(max(wt_min, 0.0), 2), round(wt_max, 2)), step=0.05)

    viz_theme = st.selectbox("시각화 컬러 스케일",
        options=["blues", "viridis", "plasma", "magma", "inferno", "greys"], index=0)
    exclude_na = st.checkbox("결측치 제외", value=True)

    # 필터 적용
    _df = df_reshaped.copy()

    if selected_year != "전체":
        _df = _df[to_int_series(_df["연도"], dropna=False) == int(selected_year)]

    if selected_regions_name:
        selected_codes = [REGION_MAP_REV[n] for n in selected_regions_name if n in REGION_MAP_REV]
        _df = _df[to_int_series(_df["출생자주소지_행정구역시도코드"], dropna=False).isin(selected_codes)]

    if selected_genders:
        rev_gender = {v: k for k, v in gender_map.items()}
        _df = _df[to_int_series(_df["성별코드"], dropna=False).isin([rev_gender[g] for g in selected_genders])]

    if selected_months:
        _df = _df[to_int_series(_df["출생월"], dropna=False).isin(selected_months)]

    if selected_multiple != "전체":
        rev_multiple = {v: k for k, v in multiple_map.items()}
        _df = _df[to_int_series(_df["다태아분류코드"], dropna=False) == rev_multiple[selected_multiple]]

    if selected_marital != "전체":
        rev_marital = {v: k for k, v in marital_map.items()}
        _df = _df[to_int_series(_df["결혼중외의자녀여부코드"], dropna=False) == rev_marital[selected_marital]]

    # 체중 필터
    _df["출생아체중"] = pd.to_numeric(_df["출생아체중"], errors="coerce")
    _df = _df[_df["출생아체중"].between(selected_wt[0], selected_wt[1], inclusive="both")]

    # 결측치 제거 옵션
    if exclude_na:
        _df = _df.dropna(subset=["연도","출생자주소지_행정구역시도코드","성별코드","출생월","출생아체중"])

    # ✅ 부모동거기간 999(미상) 제외
    if "부모동거기간" in _df.columns:
        _df["부모동거기간"] = pd.to_numeric(_df["부모동거기간"], errors="coerce")
        _df = _df[_df["부모동거기간"] != 999]

    # ===== 코드 → 라벨 컬럼 추가 =====
    _df["시도코드_int"] = to_int_series(_df["출생자주소지_행정구역시도코드"], dropna=False)
    _df["시도명"] = _df["시도코드_int"].map(REGION_MAP)

    _df["부연령코드_int"] = to_int_series(_df["부연령_5세단위코드"], dropna=False)
    _df["모연령코드_int"] = to_int_series(_df["모연령_5세단위코드"], dropna=False)
    _df["부연령_라벨"] = _df["부연령코드_int"].map(AGE_MAP)
    _df["모연령_라벨"] = _df["모연령코드_int"].map(AGE_MAP)

    _df["다태아출산순위"] = to_int_series(_df["다태아출산순위코드"], dropna=False).map(MULTI_ORDER_MAP)
    _df["다태아분류"] = to_int_series(_df["다태아분류코드"], dropna=False).map(MULTI_TYPE_MAP)
    _df["모총출생아수"] = to_int_series(_df["모총출생아수코드"], dropna=False).map(MOTHER_TOTAL_CHILD_MAP)
    _df["부_국적구분"] = to_int_series(_df["부_국적구분코드"], dropna=False).map(NATIONALITY_MAP)
    _df["모_국적구분"] = to_int_series(_df["모_국적구분코드"], dropna=False).map(NATIONALITY_MAP)

    st.session_state["filtered_df"] = _df
    st.session_state["viz_theme"] = viz_theme

    st.markdown("---")
    st.metric("선택된 데이터 개수", f"{len(_df):,}")

#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

#######################
# col[0] : 요약 KPI + 도넛
with col[0]:
    st.subheader("📌 출생 통계 요약")
    df_filtered = st.session_state.get("filtered_df", df_reshaped)

    total_births = len(df_filtered)
    avg_weight = pd.to_numeric(df_filtered["출생아체중"], errors="coerce").mean()
    multiple_ratio = (pd.to_numeric(df_filtered["다태아분류코드"], errors="coerce").fillna(1).astype(int) > 1).mean() * 100
    marital_code = pd.to_numeric(df_filtered["결혼중외의자녀여부코드"], errors="coerce")
    marital_ratio = (marital_code == 2).mean() * 100  # 2=혼인 외(가정)

    k1, k2 = st.columns(2)
    k3, k4 = st.columns(2)
    k1.metric("총 출생아 수", f"{total_births:,}")
    k2.metric("평균 출생 체중", f"{avg_weight:.2f} kg")
    k3.metric("다태아 비율", f"{multiple_ratio:.1f}%")
    k4.metric("혼인 외 출생 비율", f"{marital_ratio:.1f}%")

    st.markdown("---")

    # 성별 도넛
    gender_map = {1: "남", 2: "여"}
    gender_counts = (
        pd.to_numeric(df_filtered["성별코드"], errors="coerce")
        .map(gender_map).value_counts(dropna=False)
        .rename_axis("성별").reset_index(name="인원").dropna(subset=["성별"])
    )
    fig_gender = px.pie(gender_counts, values="인원", names="성별", hole=0.5)
    fig_gender = apply_dark_layout(fig_gender, "성별 분포")
    st.plotly_chart(fig_gender, use_container_width=True)

    # 다태아 분포 (라벨 사용)
    multi_counts = (
        df_filtered["다태아분류"].value_counts(dropna=False)
        .rename_axis("구분").reset_index(name="인원").dropna(subset=["구분"])
    )
    fig_multi = px.pie(multi_counts, values="인원", names="구분", hole=0.5)
    fig_multi = apply_dark_layout(fig_multi, "다태아 분포")
    st.plotly_chart(fig_multi, use_container_width=True)

#######################
# col[1] : 지역/시계열/히트맵
with col[1]:
    st.subheader("📘 지역 및 시계열 분석")
    df_filtered = st.session_state.get("filtered_df", df_reshaped)
    theme = st.session_state.get("viz_theme", "blues")

    # 지역별 출생아 수 (시도명)
    st.markdown("#### 지역별 출생아 분포 (시도 단위)")
    region_counts = (
        df_filtered.groupby("시도명").size().reset_index(name="출생아수")
        .dropna(subset=["시도명"]).sort_values("출생아수", ascending=False)
    )
    fig_region = px.bar(region_counts, x="시도명", y="출생아수", color="출생아수", color_continuous_scale=theme)
    fig_region = apply_dark_layout(fig_region, "시도별 출생아 수")
    st.plotly_chart(fig_region, use_container_width=True)

    st.markdown("---")

    # 월별 출생 추이
    st.markdown("#### 월별 출생 추이")
    monthly_counts = (
        df_filtered.groupby("출생월").size().reset_index(name="출생아수").sort_values("출생월")
    )
    fig_month = px.line(monthly_counts, x="출생월", y="출생아수", markers=True)
    fig_month = apply_dark_layout(fig_month, "월별 출생아 수 추이")
    st.plotly_chart(fig_month, use_container_width=True)

    st.markdown("---")

    # 부모 연령대 히트맵 (라벨 축)
    st.markdown("#### 부모 연령대별 출생 분포")
    age_heatmap = (
        df_filtered.groupby(["부연령_라벨", "모연령_라벨"]).size().reset_index(name="출생아수")
        .dropna(subset=["부연령_라벨","모연령_라벨"])
    )
    age_order = ["0~14세","15~19세","20~24세","25~29세","30~34세","35~39세","40~44세","45~49세","50세 이상","미상"]
    age_heatmap["부연령_라벨"] = pd.Categorical(age_heatmap["부연령_라벨"], categories=age_order, ordered=True)
    age_heatmap["모연령_라벨"] = pd.Categorical(age_heatmap["모연령_라벨"], categories=age_order, ordered=True)

    fig_heat = px.density_heatmap(
        age_heatmap, x="부연령_라벨", y="모연령_라벨", z="출생아수", color_continuous_scale=theme
    )
    fig_heat = apply_dark_layout(fig_heat, "부모 연령대별 출생 분포")
    fig_heat.update_traces(hovertemplate="부연령 %{x}<br>모연령 %{y}<br>출생아수 %{z}<extra></extra>")
    st.plotly_chart(fig_heat, use_container_width=True)

#######################
# col[2] : 순위/인사이트/설명
with col[2]:
    st.subheader("📊 세부 분석 & 인사이트")
    df_filtered = st.session_state.get("filtered_df", df_reshaped)
    theme = st.session_state.get("viz_theme", "blues")

    st.markdown("#### 출생아 수 Top 10 지역")
    top_regions = (
        df_filtered.groupby("시도명").size().reset_index(name="출생아수")
        .dropna(subset=["시도명"]).sort_values("출생아수", ascending=False).head(10)
    )
    fig_top = px.bar(top_regions, x="출생아수", y="시도명", orientation="h",
                     color="출생아수", color_continuous_scale=theme)
    fig_top = apply_dark_layout(fig_top, "출생아 수 상위 10개 지역")
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")

    st.markdown("#### 지역별 평균 출생 체중")
    region_weight = (
        df_filtered.dropna(subset=["시도명"]).groupby("시도명")["출생아체중"]
        .mean().reset_index().sort_values("출생아체중", ascending=False)
    )
    fig_w = px.bar(region_weight, x="시도명", y="출생아체중",
                   color="출생아체중", color_continuous_scale=theme)
    fig_w = apply_dark_layout(fig_w, "지역별 평균 출생 체중(kg)")
    st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("---")

    st.markdown("#### 주요 인사이트")
    c1, c2 = st.columns(2)
    with c1:
        if "임신주수" in df_filtered.columns:
            st.metric("평균 임신 주수", f"{pd.to_numeric(df_filtered['임신주수'], errors='coerce').mean():.1f} 주")
        else:
            st.metric("평균 임신 주수", "데이터 없음")
    with c2:
        # 평균 계산 시 혹시 남아있을지 모를 999 방어적으로 재제거
        if "부모동거기간" in df_filtered.columns:
            pdk = pd.to_numeric(df_filtered["부모동거기간"], errors="coerce").replace(999, pd.NA)
            st.metric("평균 부모 동거기간", f"{pdk.dropna().mean():.1f} 년")
        else:
            st.metric("평균 부모 동거기간", "데이터 없음")

    st.markdown("---")
    st.markdown("#### ℹ️ 데이터 설명")
    st.info(
        """
        - 주요 변수/라벨:
            - `시도명`: 시도코드 → 시도명으로 변환
            - 연령: `부연령_라벨`, `모연령_라벨` (AGE_MAP 적용)
            - `다태아출산순위`, `다태아분류`, `모총출생아수`
            - `부_국적구분`, `모_국적구분`
            - `부모동거기간`의 미상값(999)은 분석에서 제외
        - 차트 배경/글자색은 다크 테마로 설정하여 가독성을 높였습니다.
        """
    )

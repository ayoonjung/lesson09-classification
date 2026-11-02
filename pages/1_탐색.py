import pandas as pd
import plotly.express as px
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/ayoonjung/data/refs/heads/main/stroke.csv"

st.set_page_config(page_title="탐색 - 뇌졸중 예측 실습실", page_icon="🔍", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔍 탐색")
st.caption("데이터를 여러 방향에서 살펴봐요.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결이나 주소를 확인해 주세요. ({e})")
    st.stop()

df = df.copy()
df["뇌졸중 여부"] = df["stroke"].map({0: "겪지 않음", 1: "겪음"})
GROUP_ORDER = ["겪지 않음", "겪음"]

# ---------- 1. 히스토그램 두 개 ----------
st.subheader("1. 나이와 평균 혈당의 분포")
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x="age", nbins=30, title="나이 분포")
    fig.update_layout(xaxis_title="나이(세)", yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig)

with col2:
    fig = px.histogram(df, x="avg_glucose_level", nbins=40, title="평균 혈당 분포")
    fig.update_layout(xaxis_title="평균 혈당(avg_glucose_level)", yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig)

# ---------- 2. 상자그림 두 개 + 평균 표 ----------
st.subheader("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
col1, col2 = st.columns(2)

with col1:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": GROUP_ORDER},
        title="나이 비교",
    )
    fig.update_layout(xaxis_title="뇌졸중 여부", yaxis_title="나이(세)", showlegend=False)
    st.plotly_chart(fig)

with col2:
    fig = px.box(
        df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        category_orders={"뇌졸중 여부": GROUP_ORDER},
        title="평균 혈당 비교",
    )
    fig.update_layout(xaxis_title="뇌졸중 여부", yaxis_title="평균 혈당", showlegend=False)
    st.plotly_chart(fig)

mean_table = (
    df.groupby("뇌졸중 여부")
    .agg(
        사람_수=("stroke", "size"),
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
    )
    .reindex(GROUP_ORDER)
    .round(2)
    .reset_index()
)
mean_table.columns = ["뇌졸중 여부", "사람 수", "평균 나이(세)", "평균 혈당"]
st.write("**두 그룹의 평균값**")
st.dataframe(mean_table, hide_index=True)

# ---------- 3. 고혈압·심장병별 뇌졸중 비율 ----------
st.subheader("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")


def rate_bar(column: str, name: str):
    rate = (
        df.groupby(column)["stroke"]
        .mean()
        .mul(100)
        .rename("뇌졸중 비율(%)")
        .reset_index()
    )
    rate[name] = rate[column].map({0: "없음", 1: "있음"})
    rate["표시"] = rate["뇌졸중 비율(%)"].map(lambda v: f"{v:.1f}%")
    fig = px.bar(
        rate,
        x=name,
        y="뇌졸중 비율(%)",
        text="표시",
        category_orders={name: ["없음", "있음"]},
        title=f"{name}에 따른 뇌졸중 비율",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title=name, yaxis_title="뇌졸중 비율(%)")
    return fig


col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(rate_bar("hypertension", "고혈압"))
with col2:
    st.plotly_chart(rate_bar("heart_disease", "심장병"))

# ---------- 4. bmi 빈 값인 사람들 ----------
st.subheader("4. 체질량지수(bmi)가 빈 사람들")
missing = df[df["bmi"].isna()]
n_missing = len(missing)
n_missing_stroke = int((missing["stroke"] == 1).sum())
n_all = len(df)
n_all_stroke = int((df["stroke"] == 1).sum())

bmi_table = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람", "전체 사람"],
        "사람 수": [n_missing, n_all],
        "뇌졸중을 겪은 사람 수": [n_missing_stroke, n_all_stroke],
        "뇌졸중 비율": [
            f"{n_missing_stroke / n_missing:.1%}" if n_missing else "해당 없음",
            f"{n_all_stroke / n_all:.1%}",
        ],
    }
)
st.dataframe(bmi_table, hide_index=True)

# ---------- 5. 흡연 상태별 사람 수 ----------
st.subheader("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태(smoking_status)")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, hide_index=True)

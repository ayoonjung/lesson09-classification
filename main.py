import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
APP_ICON = "🧠"
DATA_URL = "https://raw.githubusercontent.com/ayoonjung/data/refs/heads/main/stroke.csv"

RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "데이터_소개_기록.json"

# 브라우저 탭 제목과 아이콘 (다른 st 명령보다 가장 먼저 와야 함)
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

# 큰 숫자 카드 글자 크기 키우기
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 2.8rem; font-weight: 700; }
    [data-testid="stMetricLabel"] p { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 데이터 불러오기 ----------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


# ---------- 내_기록 폴더의 json 읽기/쓰기 ----------
def load_record() -> dict:
    """저장된 기록을 읽어 온다. 파일이 없거나 깨졌으면 빈 기록을 돌려준다."""
    try:
        with open(RECORD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        meanings = data.get("우리말 뜻", {})
        source = data.get("데이터 출처", "")
        if not isinstance(meanings, dict) or not isinstance(source, str):
            raise ValueError("형식이 맞지 않는 기록 파일")
        return {"우리말 뜻": meanings, "데이터 출처": source}
    except (FileNotFoundError, ValueError, OSError):
        return {"우리말 뜻": {}, "데이터 출처": ""}


def save_record(meanings: dict, source: str) -> None:
    RECORD_DIR.mkdir(exist_ok=True)
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"우리말 뜻": meanings, "데이터 출처": source},
            f,
            ensure_ascii=False,
            indent=2,
        )


# ---------- 열마다 값의 종류를 글로 설명 ----------
def describe_values(s: pd.Series) -> str:
    values = s.dropna().unique()
    n = len(values)
    if n == 0:
        return "값 없음"
    if pd.api.types.is_numeric_dtype(s) and n > 5:
        return f"숫자 {n:,}가지 (최소 {s.min():g} ~ 최대 {s.max():g})"
    shown = ", ".join(str(v) for v in sorted(values, key=str)[:8])
    if n > 8:
        shown += ", ..."
    return f"{n}가지: {shown}"


# ---------- 화면 맨 위 ----------
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption("첫 화면: 이 데이터가 무엇인지 알아보기")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요. 인터넷 연결이나 주소를 확인해 주세요. ({e})")
    st.stop()

# 세션이 처음 시작될 때 한 번만 저장된 기록을 불러온다 (새로고침하면 다시 불러옴)
if "base_table" not in st.session_state:
    record = load_record()
    st.session_state["base_table"] = pd.DataFrame(
        {
            "열 이름": df.columns,
            "우리말 뜻": [record["우리말 뜻"].get(c, "") for c in df.columns],
            "값의 종류": [describe_values(df[c]) for c in df.columns],
            "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
        }
    )
    st.session_state["source_text"] = record["데이터 출처"]

# ---------- 1. 큰 숫자 카드 네 개 ----------
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total if total else 0

c1, c2, c3, c4 = st.columns(4)
with c1:
    with st.container(border=True):
        st.metric("전체 사람 수", f"{total:,}명")
with c2:
    with st.container(border=True):
        st.metric("열 개수", f"{n_cols}개")
with c3:
    with st.container(border=True):
        st.metric("stroke가 1인 사람 수", f"{n_stroke:,}명")
with c4:
    with st.container(border=True):
        st.metric("stroke가 1인 비율", f"{ratio:.1%}")

# ---------- 2. 열 설명 표 ----------
st.subheader("열 설명 표")
st.write("**우리말 뜻** 칸은 비어 있어요. 교재를 보고 칸을 눌러 직접 적은 뒤 저장 버튼을 눌러 주세요.")

with st.form("meaning_form"):
    edited = st.data_editor(
        st.session_state["base_table"],
        key="meaning_editor",
        hide_index=True,
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn("열 이름"),
            "우리말 뜻": st.column_config.TextColumn("우리말 뜻 (직접 쓰기)"),
            "값의 종류": st.column_config.TextColumn("값의 종류"),
            "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d"),
        },
    )
    save_meaning = st.form_submit_button("우리말 뜻 저장")

if save_meaning:
    meanings = {
        row["열 이름"]: (row["우리말 뜻"] or "").strip()
        for _, row in edited.iterrows()
    }
    try:
        save_record(meanings, st.session_state["source_text"])
        st.success("우리말 뜻을 내_기록 폴더에 저장했어요.")
    except OSError as e:
        st.error(f"저장하지 못했어요: {e}")

# ---------- 3. 처음 다섯 줄 ----------
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5))

# ---------- 4. 데이터 출처 ----------
st.subheader("데이터 출처")
with st.form("source_form"):
    st.text_area(
        "교재에 나온 출처를 적어 주세요.",
        key="source_text",
        height=120,
    )
    save_source = st.form_submit_button("출처 저장")

if save_source:
    # 표에서 이미 저장해 둔 우리말 뜻은 그대로 두고, 현재 표 내용(저장된 값)을 함께 보존
    current = load_record()["우리말 뜻"]
    try:
        save_record(current, st.session_state["source_text"])
        st.success("데이터 출처를 내_기록 폴더에 저장했어요.")
    except OSError as e:
        st.error(f"저장하지 못했어요: {e}")

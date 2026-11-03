import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
st.set_page_config(
    page_title="채점표 - 뇌졸중 예측 실습실", page_icon="📝", layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/ayoonjung/data/refs/heads/main/stroke.csv"
)

# '분류 모델' 페이지에서 처음 골라져 있는 네 가지 속성을 그대로 쓴다.
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LOGI_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"

NOT_AVAILABLE = "계산할 수 없음"
MAX_ROWS = 20  # 목록에 보여 줄 최대 줄 수


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("📝 채점표")
st.caption(
    "테스트 데이터로 모델을 채점합니다. 뇌졸중(stroke=1)을 양성으로 둡니다."
)

try:
    raw_df = load_data()
except Exception as e:  # 주소 오류, 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ---------------------------------------------------------------
# 1) 데이터 나누기: 번호 순 정렬 -> 열 명씩 묶어 앞 세 명은 테스트용
# ---------------------------------------------------------------
df = raw_df.sort_values("id").reset_index(drop=True)
is_test = (df.index % 10) < 3

X_train = df.loc[~is_test, FEATURES].copy()
y_train = df.loc[~is_test, "stroke"].copy()
X_test = df.loc[is_test, FEATURES].copy()
y_test = df.loc[is_test, "stroke"].copy()
test_info = df.loc[is_test, ["id"] + FEATURES].copy()  # 목록에 보여 줄 정보

st.caption(
    "번호 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했습니다. "
    f"테스트용 {len(X_test):,}명, 훈련용 {len(X_train):,}명입니다. "
    "입력 속성은 나이·평균 혈당·고혈압·심장병입니다."
)

# ---------------------------------------------------------------
# 2) 두 모델 학습 ('분류 모델' 페이지와 같은 방법)
# ---------------------------------------------------------------
scaler = StandardScaler().fit(X_train)  # 크기 맞추기는 훈련용으로만
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

logi = LogisticRegression(max_iter=1000, random_state=42)
logi.fit(X_train_scaled, y_train)

tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

base = DummyClassifier(strategy="most_frequent")
base.fit(X_train, y_train)

# 테스트 데이터에 대한 예측 (사람마다 같은 번호표를 유지하려고 인덱스를 붙인다)
predictions = {
    LOGI_NAME: pd.Series(logi.predict(X_test_scaled), index=X_test.index),
    TREE_NAME: pd.Series(tree.predict(X_test), index=X_test.index),
    BASE_NAME: pd.Series(base.predict(X_test), index=X_test.index),
}


def count_cells(actual: pd.Series, predicted: pd.Series):
    """혼동행렬의 네 칸 TP, FP, FN, TN을 센다. (뇌졸중 = 양성)"""
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    return tp, fp, fn, tn


cells = {name: count_cells(y_test, pred) for name, pred in predictions.items()}

# ---------------------------------------------------------------
# 3) 혼동행렬 (행: 실제, 열: 예측)
# ---------------------------------------------------------------
st.header("혼동행렬")
st.caption("행은 실제, 열은 예측입니다. 오른쪽 끝은 행의 합계, 맨 아래는 열의 합계입니다.")


def confusion_table(tp: int, fp: int, fn: int, tn: int) -> pd.DataFrame:
    total = tp + fp + fn + tn
    return pd.DataFrame(
        {
            "예측: 뇌졸중": [f"TP: {tp:,}명", f"FP: {fp:,}명", f"{tp + fp:,}명"],
            "예측: 아님": [f"FN: {fn:,}명", f"TN: {tn:,}명", f"{fn + tn:,}명"],
            "행 합계": [f"{tp + fn:,}명", f"{fp + tn:,}명", f"{total:,}명"],
        },
        index=["실제: 뇌졸중", "실제: 아님", "열 합계"],
    )


col_l, col_r = st.columns(2)
with col_l:
    st.subheader(LOGI_NAME)
    st.dataframe(confusion_table(*cells[LOGI_NAME]), width="stretch")
with col_r:
    st.subheader(TREE_NAME)
    st.dataframe(confusion_table(*cells[TREE_NAME]), width="stretch")

st.caption(
    "TP: 뇌졸중인 사람을 뇌졸중이라 예측 / FP: 아닌 사람을 뇌졸중이라 예측 / "
    "FN: 뇌졸중인 사람을 아니라고 예측 / TN: 아닌 사람을 아니라고 예측"
)

# ---------------------------------------------------------------
# 4) 정확도·재현율·정밀도·F1 한 표
# ---------------------------------------------------------------
st.header("네 가지 지표")

n_test = len(y_test)
n_test_stroke = int((y_test == 1).sum())
st.write(f"테스트용 **{n_test:,}명** 가운데 뇌졸중인 사람은 **{n_test_stroke:,}명**입니다.")


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def four_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else None

    # 재현율: 분모는 실제 뇌졸중인 사람 수
    actual_pos = tp + fn
    recall = tp / actual_pos if actual_pos else None

    # 정밀도: 분모는 뇌졸중이라 예측한 사람 수 (0명이면 계산할 수 없음)
    predicted_pos = tp + fp
    precision = tp / predicted_pos if predicted_pos else None

    # F1 = 2 x 정밀도 x 재현율 / (정밀도 + 재현율)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    elif precision is not None and recall is not None:
        f1 = 0.0  # 정밀도와 재현율이 모두 0이면 F1도 0으로 적는다.

    def show(v):
        return percent(v) if v is not None else NOT_AVAILABLE

    return {
        "정확도": show(accuracy),
        "재현율": show(recall),
        "정밀도": show(precision),
        "F1": show(f1),
    }


metric_rows = []
for name in (LOGI_NAME, TREE_NAME, BASE_NAME):
    row = {"모델": name}
    row.update(four_metrics(*cells[name]))
    metric_rows.append(row)

st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch")
st.caption(
    "정확도 = (TP+TN) ÷ 전체 / 재현율 = TP ÷ (TP+FN), 분모는 실제 뇌졸중인 사람 수 / "
    "정밀도 = TP ÷ (TP+FP), 분모는 뇌졸중이라 예측한 사람 수 / "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율)"
)

# ---------------------------------------------------------------
# 5) 놓친 환자와 헛짚은 사람 목록
# ---------------------------------------------------------------
st.header("놓친 환자와 헛짚은 사람")

chosen = st.radio(
    "어느 모델의 목록을 볼까요?",
    [LOGI_NAME, TREE_NAME],
    horizontal=True,
)
chosen_pred = predictions[chosen]

missed_mask = (y_test == 1) & (chosen_pred == 0)  # 실제 뇌졸중인데 아니라고 예측
wrong_mask = (y_test == 0) & (chosen_pred == 1)  # 아닌데 뇌졸중이라고 예측


def make_list(mask: pd.Series) -> pd.DataFrame:
    """번호, 나이, 평균 혈당, 고혈압 여부, 심장병 여부를 나이가 많은 순서로 정렬한 표."""
    sub = test_info[mask.to_numpy()].sort_values("age", ascending=False)
    return pd.DataFrame(
        {
            "번호": sub["id"].to_list(),
            "나이": sub["age"].to_list(),
            "평균 혈당": sub["avg_glucose_level"].to_list(),
            "고혈압 여부": ["있음" if v == 1 else "없음" for v in sub["hypertension"]],
            "심장병 여부": ["있음" if v == 1 else "없음" for v in sub["heart_disease"]],
        }
    )


def show_list(title_text: str, mask: pd.Series):
    full = make_list(mask)
    total = len(full)
    st.subheader(f"{title_text} (전체 {total:,}명)")
    if total == 0:
        st.info("해당하는 사람이 없습니다.")
        return
    if total > MAX_ROWS:
        st.caption(f"나이가 많은 순서로 앞 {MAX_ROWS}줄만 보여 줍니다.")
    else:
        st.caption("나이가 많은 순서로 모두 보여 줍니다.")
    st.dataframe(full.head(MAX_ROWS), hide_index=True, width="stretch")


list_l, list_r = st.columns(2)
with list_l:
    show_list("놓친 환자: 실제 뇌졸중인데 아니라고 예측", missed_mask)
with list_r:
    show_list("헛짚은 사람: 아닌데 뇌졸중이라고 예측", wrong_mask)

_, chosen_fp, chosen_fn, _ = cells[chosen]
st.write(
    f"{chosen}의 목록은 놓친 환자 {int(missed_mask.sum()):,}명, "
    f"헛짚은 사람 {int(wrong_mask.sum()):,}명이며, "
    f"혼동행렬의 FN {chosen_fn:,}명, FP {chosen_fp:,}명 칸과 같습니다."
)

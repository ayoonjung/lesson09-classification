import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
st.set_page_config(
    page_title="성능 높이기 - 뇌졸중 예측 실습실", page_icon="⚖️", layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/ayoonjung/data/refs/heads/main/stroke.csv"
)

# 열 이름 -> 우리말 이름 (목록과 그림의 축 이름에 우리말을 보여 준다)
FEATURES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = ("hypertension", "heart_disease")  # 0/1 값을 가지는 속성
DEFAULT_PICKED = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

LOGI_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"

BEFORE = "가중치를 맞추기 전"
AFTER = "가중치를 맞춘 뒤"
NOT_AVAILABLE = "계산할 수 없음"

COLOR_NEG = "#1f77b4"  # 실제 아님: 파랑
COLOR_POS = "#d62728"  # 실제 뇌졸중: 빨강
COLOR_BEFORE = "#2ca02c"  # 맞추기 전 0.5 경계선: 초록
COLOR_AFTER = "#ff7f0e"  # 맞춘 뒤 0.5 경계선: 주황
LABELS = {0: "아님", 1: "뇌졸중"}


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("⚖️ 성능 높이기")
st.caption(
    "뇌졸중을 겪은 사람이 아주 드뭅니다. 양쪽의 가중치를 맞추기 전과 맞춘 뒤의 결과를 나란히 비교합니다. "
    "뇌졸중(stroke=1)을 양성으로 둡니다."
)

try:
    raw_df = load_data()
except Exception as e:  # 주소 오류, 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ---------------------------------------------------------------
# 1) 입력으로 사용할 속성 고르기 ('분류 모델' 페이지와 같은 방법)
# ---------------------------------------------------------------
picked = st.multiselect(
    "입력으로 사용할 속성을 고르세요.",
    options=list(FEATURES.keys()),
    default=DEFAULT_PICKED,
    format_func=lambda col: FEATURES[col],
)
selected = [col for col in FEATURES if col in picked]  # 항상 같은 순서로

if len(selected) < 2:
    st.info("속성을 두 개 이상 골라 주세요. 두 개보다 적으면 그림을 그릴 수 없습니다.")
    st.stop()

# ---------------------------------------------------------------
# 2) 데이터 나누기 ('채점표' 페이지와 같은 방법)
# ---------------------------------------------------------------
df = raw_df.sort_values("id").reset_index(drop=True)
is_test = (df.index % 10) < 3

X_train = df.loc[~is_test, selected].copy()
y_train = df.loc[~is_test, "stroke"].copy()
X_test = df.loc[is_test, selected].copy()
y_test = df.loc[is_test, "stroke"].copy()

st.caption(
    "번호 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했습니다. "
    f"테스트용 {len(X_test):,}명, 훈련용 {len(X_train):,}명입니다."
)

# 체질량지수는 고른 경우에만, 훈련용의 중앙값으로 빈 값을 채운다.
if "bmi" in selected:
    bmi_median = float(X_train["bmi"].median())
    n_filled = int(X_train["bmi"].isna().sum() + X_test["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)
    st.caption(
        f"체질량지수의 빈 값 {n_filled:,}명을 훈련용의 중앙값 {bmi_median:.2f}로 채웠습니다."
    )

# ---------------------------------------------------------------
# 3) 모델을 가중치 맞추기 전·후로 각각 학습 (크기 맞추기는 훈련용으로만)
# ---------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 가중치 맞추기: 드문 쪽(뇌졸중)의 한 명이 더 무겁게 계산되도록 양쪽 합이 같아지게 한다.
n_train = len(y_train)
n_pos = int((y_train == 1).sum())
n_neg = n_train - n_pos
weight_pos = n_train / (2 * n_pos)
weight_neg = n_train / (2 * n_neg)


def fit_logistic(class_weight):
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
    return model.fit(X_train_scaled, y_train)


def fit_tree(class_weight):
    model = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, random_state=42, class_weight=class_weight
    )
    return model.fit(X_train, y_train)


logi_before = fit_logistic(None)
logi_after = fit_logistic("balanced")
tree_before = fit_tree(None)
tree_after = fit_tree("balanced")

pred_logi_before = pd.Series(logi_before.predict(X_test_scaled), index=y_test.index)
pred_logi_after = pd.Series(logi_after.predict(X_test_scaled), index=y_test.index)
pred_tree_before = pd.Series(tree_before.predict(X_test), index=y_test.index)
pred_tree_after = pd.Series(tree_after.predict(X_test), index=y_test.index)


def count_cells(actual: pd.Series, predicted: pd.Series):
    """혼동행렬의 네 칸 TP, FP, FN, TN을 센다. (뇌졸중 = 양성)"""
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    return tp, fp, fn, tn


def four_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else None
    recall = tp / (tp + fn) if (tp + fn) else None  # 분모: 실제 뇌졸중인 사람 수
    precision = tp / (tp + fp) if (tp + fp) else None  # 분모: 뇌졸중이라 예측한 사람 수
    f1 = None
    if precision is not None and recall is not None:
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0  # 정밀도와 재현율이 모두 0이면 F1도 0으로 적는다.

    def show(v):
        return f"{v * 100:.2f}%" if v is not None else NOT_AVAILABLE

    return {
        "정확도": show(accuracy),
        "재현율": show(recall),
        "정밀도": show(precision),
        "F1": show(f1),
    }


def before_after_table(before: tuple, after: tuple) -> pd.DataFrame:
    """네 칸과 네 지표를 전·후로 나란히 놓은 표."""
    names = ["TP (찾아낸 환자)", "FN (놓친 환자)", "FP (헛짚은 사람)", "TN (아닌 사람을 아니라고 예측)"]
    order = [0, 2, 1, 3]  # TP, FN, FP, TN 순서 (before/after는 TP, FP, FN, TN 순)
    rows = {"항목": [], BEFORE: [], AFTER: []}
    for name, idx in zip(names, order):
        rows["항목"].append(name)
        rows[BEFORE].append(f"{before[idx]:,}명")
        rows[AFTER].append(f"{after[idx]:,}명")
    metrics_before = four_metrics(*before)
    metrics_after = four_metrics(*after)
    for key in ("정확도", "재현율", "정밀도", "F1"):
        rows["항목"].append(key)
        rows[BEFORE].append(metrics_before[key])
        rows[AFTER].append(metrics_after[key])
    return pd.DataFrame(rows)


def change_line(before: tuple, after: tuple) -> str:
    """찾아낸 환자(TP)·놓친 환자(FN)·헛짚은 사람(FP) 수가 몇 명에서 몇 명으로 바뀌었는지 한 줄."""
    return (
        f"찾아낸 환자 {before[0]:,}명에서 {after[0]:,}명으로, "
        f"놓친 환자 {before[2]:,}명에서 {after[2]:,}명으로, "
        f"헛짚은 사람 {before[1]:,}명에서 {after[1]:,}명으로 바뀌었습니다."
    )


# ---------------------------------------------------------------
# 4) 전·후 비교 표 (두 모델을 나란히)
# ---------------------------------------------------------------
st.header("가중치를 맞추기 전과 맞춘 뒤")
st.write(
    f"훈련용 {n_train:,}명 가운데 뇌졸중인 사람은 {n_pos:,}명뿐입니다. "
    f"가중치를 맞추면 뇌졸중인 사람 한 명을 {weight_pos:.2f}명, 아닌 사람 한 명을 {weight_neg:.2f}명처럼 계산해서 "
    "양쪽의 무게 합이 같아집니다. 채점은 둘 다 같은 테스트 데이터로 합니다."
)

logi_cells = (count_cells(y_test, pred_logi_before), count_cells(y_test, pred_logi_after))
tree_cells = (count_cells(y_test, pred_tree_before), count_cells(y_test, pred_tree_after))

col_l, col_r = st.columns(2)
with col_l:
    st.subheader(LOGI_NAME)
    st.dataframe(before_after_table(*logi_cells), hide_index=True, width="stretch")
    st.write(change_line(*logi_cells))
with col_r:
    st.subheader(TREE_NAME)
    st.dataframe(before_after_table(*tree_cells), hide_index=True, width="stretch")
    st.write(change_line(*tree_cells))

st.caption(
    "TP: 뇌졸중인 사람을 뇌졸중이라 예측 / FN: 뇌졸중인 사람을 아니라고 예측 / "
    "FP: 아닌 사람을 뇌졸중이라 예측 / TN: 아닌 사람을 아니라고 예측"
)

# ---------------------------------------------------------------
# 5) 산점도에 쓸 공통 도구
# ---------------------------------------------------------------
medians = X_test.median()  # 그림에 쓰지 않은 속성은 테스트 데이터의 중앙값에 세워 둔다.


def axis_range(col: str):
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    if col in BINARY:
        return -0.3, 1.3
    span = hi - lo
    pad = span * 0.05 if span > 0 else 0.5
    return lo - pad, hi + pad


def make_points(values: dict) -> pd.DataFrame:
    """values에 있는 속성은 주어진 값으로, 나머지는 테스트 데이터의 중앙값으로 채운 표."""
    n = len(next(iter(values.values())))
    data = {
        col: values[col] if col in values else [float(medians[col])] * n
        for col in selected
    }
    return pd.DataFrame(data, columns=selected)


def decision(model, points: pd.DataFrame):
    """로지스틱 회귀의 결정 값. 0이면 뇌졸중일 확률이 정확히 0.5이다."""
    return model.decision_function(scaler.transform(points))


def jitter(n: int, mult: int) -> list:
    """0/1 값 속성의 점이 겹치지 않도록 그림에서만 살짝 흩뜨리는 값."""
    return [((i * mult) % 101 / 100 - 0.5) * 0.24 for i in range(n)]


def fixed_text(axes: list) -> str:
    rest = [col for col in selected if col not in axes]
    if not rest:
        return ""
    return ", ".join(f"{FEATURES[col]} = {round(float(medians[col]), 2):g}" for col in rest)


def outside_text(label: str, model, center_values: dict, shape: str) -> str:
    center = decision(model, make_points(center_values))[0]
    if center > 0:
        side = "뇌졸중일 확률이 0.5보다 높게 나옵니다(이 모델은 그림 안 어디에서나 '뇌졸중'이라고 답합니다)"
    else:
        side = "뇌졸중일 확률이 0.5보다 낮게 나옵니다(이 모델은 그림 안 어디에서도 '뇌졸중'이라고 답하지 않습니다)"
    return f"{label}의 0.5 {shape}은 이 그림 밖에 있습니다. 그림 안의 모든 자리에서 {side}."


# ---------------------------------------------------------------
# 6) 두 속성으로 보는 산점도 + 전·후 0.5 경계선
# ---------------------------------------------------------------
st.header("두 속성으로 보는 산점도")
st.caption(
    f"경계선은 {LOGI_NAME}만 그립니다. 확률 0.5로 가르는 하나의 선(평면)이 있는 모델이기 때문입니다."
)

c1, c2 = st.columns(2)
with c1:
    x_feat = st.selectbox(
        "가로축", selected, index=0, format_func=lambda col: FEATURES[col]
    )
y_options = [col for col in selected if col != x_feat]
with c2:
    y_feat = st.selectbox(
        "세로축", y_options, index=0, format_func=lambda col: FEATURES[col]
    )
x_name, y_name = FEATURES[x_feat], FEATURES[y_feat]
x_lo, x_hi = axis_range(x_feat)
y_lo, y_hi = axis_range(y_feat)


def line_coefficients(model):
    """a + bx*x + by*y = 0 인 직선의 계수 (나머지 속성은 중앙값에 고정)."""
    d = decision(model, make_points({x_feat: [0.0, 1.0, 0.0], y_feat: [0.0, 0.0, 1.0]}))
    a = float(d[0])
    return a, float(d[1] - d[0]), float(d[2] - d[0])


def line_in_box(a, bx, by, x0, x1, y0, y1):
    """직선이 그림 상자 안을 지나는 두 끝점을 돌려준다. 지나지 않으면 None."""
    tol_x, tol_y = 1e-9 * (x1 - x0), 1e-9 * (y1 - y0)
    found = []
    if abs(by) > 1e-12:
        for x in (x0, x1):
            y = -(a + bx * x) / by
            if y0 - tol_y <= y <= y1 + tol_y:
                found.append((x, y))
    if abs(bx) > 1e-12:
        for y in (y0, y1):
            x = -(a + by * y) / bx
            if x0 - tol_x <= x <= x1 + tol_x:
                found.append((x, y))
    unique = []
    for p in found:
        if all(abs(p[0] - q[0]) > tol_x or abs(p[1] - q[1]) > tol_y for q in unique):
            unique.append(p)
    if len(unique) < 2:
        return None
    p0 = unique[0]
    p1 = max(unique, key=lambda q: (q[0] - p0[0]) ** 2 + (q[1] - p0[1]) ** 2)
    return p0, p1


segments = {
    BEFORE: line_in_box(*line_coefficients(logi_before), x_lo, x_hi, y_lo, y_hi),
    AFTER: line_in_box(*line_coefficients(logi_after), x_lo, x_hi, y_lo, y_hi),
}

plot_df = pd.DataFrame(
    {
        "x0": X_test[x_feat].to_numpy(),
        "y0": X_test[y_feat].to_numpy(),
        "stroke": y_test.to_numpy(),
    }
)
plot_df["x"] = plot_df["x0"]
plot_df["y"] = plot_df["y0"]
if x_feat in BINARY:
    plot_df["x"] = plot_df["x0"] + jitter(len(plot_df), 37)
if y_feat in BINARY:
    plot_df["y"] = plot_df["y0"] + jitter(len(plot_df), 53)

fig2d = go.Figure()
for value in (0, 1):  # 드문 뇌졸중 점이 위에 오도록 나중에 그린다
    sub = plot_df[plot_df["stroke"] == value]
    fig2d.add_trace(
        go.Scatter(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            marker=dict(
                size=7,
                color=COLOR_POS if value == 1 else COLOR_NEG,
                opacity=0.75,
                line=dict(width=0.5, color="white"),
            ),
            customdata=sub[["x0", "y0"]],
            hovertemplate=(
                f"{x_name}: %{{customdata[0]}}<br>{y_name}: %{{customdata[1]}}"
                f"<extra>실제 {LABELS[value]}</extra>"
            ),
            name=f"실제 {LABELS[value]}",
        )
    )

for label, color, dash in ((BEFORE, COLOR_BEFORE, "dash"), (AFTER, COLOR_AFTER, "solid")):
    seg = segments[label]
    if seg is not None:
        (lx0, ly0), (lx1, ly1) = seg
        fig2d.add_trace(
            go.Scatter(
                x=[lx0, lx1],
                y=[ly0, ly1],
                mode="lines",
                line=dict(color=color, width=4, dash=dash),
                name=f"{label}의 0.5 경계선",
                hoverinfo="skip",
            )
        )

fig2d.update_layout(
    title=f"테스트 데이터: {x_name} 과(와) {y_name}",
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    height=560,
)
fig2d.update_xaxes(title=x_name, range=[x_lo, x_hi])
fig2d.update_yaxes(title=y_name, range=[y_lo, y_hi])
st.plotly_chart(fig2d, width="stretch")

rest_text = fixed_text([x_feat, y_feat])
if rest_text:
    st.write(f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: {rest_text}")
else:
    st.write("고른 속성이 두 개뿐이라, 두 축 말고는 세워 둘 속성이 없습니다.")

center2d = {x_feat: [(x_lo + x_hi) / 2], y_feat: [(y_lo + y_hi) / 2]}
for label, model in ((BEFORE, logi_before), (AFTER, logi_after)):
    if segments[label] is None:
        st.write(outside_text(label, model, center2d, "경계선"))
    else:
        st.write(f"{label}의 0.5 경계선은 그림 안에 그려져 있습니다.")

if x_feat in BINARY or y_feat in BINARY:
    st.caption(
        "고혈압·심장병은 0(없음)과 1(있음)뿐이라, 점이 겹치지 않도록 그림에서만 점을 살짝 흩뜨려 찍었습니다."
    )

# ---------------------------------------------------------------
# 7) 속성을 셋 이상 고른 경우: 3차원 산점도 + 전·후 0.5 평면
# ---------------------------------------------------------------
st.header("세 속성으로 보는 3차원 산점도")

if len(selected) < 3:
    st.info("속성을 셋 이상 고르면 3차원 산점도가 나타납니다.")
    st.stop()

st.caption("마우스로 끌어서 돌려 보고, 휠로 확대·축소할 수 있습니다.")

d1, d2, d3 = st.columns(3)
with d1:
    ax1 = st.selectbox(
        "3차원 첫째 축", selected, index=0, format_func=lambda col: FEATURES[col]
    )
opts2 = [col for col in selected if col != ax1]
with d2:
    ax2 = st.selectbox(
        "3차원 둘째 축", opts2, index=0, format_func=lambda col: FEATURES[col]
    )
opts3 = [col for col in selected if col not in (ax1, ax2)]
with d3:
    ax3 = st.selectbox(
        "3차원 셋째 축", opts3, index=0, format_func=lambda col: FEATURES[col]
    )
axes3 = [ax1, ax2, ax3]
ranges3 = [axis_range(col) for col in axes3]
lo3 = tuple(r[0] for r in ranges3)
hi3 = tuple(r[1] for r in ranges3)


def plane_coefficients(model):
    """a + bx*x + by*y + bz*z = 0 인 평면의 계수 (나머지 속성은 중앙값에 고정)."""
    d = decision(
        model,
        make_points(
            {
                ax1: [0.0, 1.0, 0.0, 0.0],
                ax2: [0.0, 0.0, 1.0, 0.0],
                ax3: [0.0, 0.0, 0.0, 1.0],
            }
        ),
    )
    a = float(d[0])
    return a, (float(d[1] - d[0]), float(d[2] - d[0]), float(d[3] - d[0]))


def plane_polygon(a, b, lo, hi):
    """평면이 3차원 상자를 자르는 단면의 꼭짓점을 순서대로 돌려준다. 지나지 않으면 None."""
    if max(abs(v) for v in b) < 1e-12:
        return None
    corners = [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    edges = []
    for i, p in enumerate(corners):
        for j in range(i + 1, len(corners)):
            q = corners[j]
            if sum(1 for k in range(3) if p[k] != q[k]) == 1:  # 한 좌표만 다르면 상자의 모서리
                edges.append((p, q))

    def f(p):
        return a + b[0] * p[0] + b[1] * p[1] + b[2] * p[2]

    scale = abs(a) + sum(abs(b[k]) * max(abs(lo[k]), abs(hi[k])) for k in range(3)) + 1e-12
    tol = 1e-9 * scale
    found = []
    for p, q in edges:
        fp, fq = f(p), f(q)
        if abs(fp) <= tol:
            found.append(p)
        elif fp * fq < 0:
            t = fp / (fp - fq)
            found.append(tuple(p[k] + t * (q[k] - p[k]) for k in range(3)))
    unique = []
    for p in found:
        if all(max(abs(p[k] - q[k]) for k in range(3)) > 1e-9 * (1 + max(abs(v) for v in p)) for q in unique):
            unique.append(p)
    if len(unique) < 3:
        return None

    # 평면 위에서 중심을 기준으로 각도 순으로 정렬
    cx = [sum(p[k] for p in unique) / len(unique) for k in range(3)]
    n = b
    helper = (1.0, 0.0, 0.0) if abs(n[0]) < max(abs(n[1]), abs(n[2])) else (0.0, 1.0, 0.0)
    u = (
        n[1] * helper[2] - n[2] * helper[1],
        n[2] * helper[0] - n[0] * helper[2],
        n[0] * helper[1] - n[1] * helper[0],
    )
    v = (
        n[1] * u[2] - n[2] * u[1],
        n[2] * u[0] - n[0] * u[2],
        n[0] * u[1] - n[1] * u[0],
    )

    def angle(p):
        d = [p[k] - cx[k] for k in range(3)]
        return math.atan2(
            sum(d[k] * v[k] for k in range(3)), sum(d[k] * u[k] for k in range(3))
        )

    return sorted(unique, key=angle)


planes = {}
for label, model in ((BEFORE, logi_before), (AFTER, logi_after)):
    a3, b3 = plane_coefficients(model)
    planes[label] = plane_polygon(a3, b3, lo3, hi3)

# 점 (0/1 값 속성은 그림에서만 살짝 흩뜨린다)
pts3 = pd.DataFrame(
    {
        "a": X_test[ax1].to_numpy(),
        "b": X_test[ax2].to_numpy(),
        "c": X_test[ax3].to_numpy(),
        "stroke": y_test.to_numpy(),
    }
)
pts3["pa"], pts3["pb"], pts3["pc"] = pts3["a"], pts3["b"], pts3["c"]
for col_name, feat, mult in (("pa", ax1, 37), ("pb", ax2, 53), ("pc", ax3, 71)):
    if feat in BINARY:
        pts3[col_name] = pts3[col_name] + jitter(len(pts3), mult)

fig3d = go.Figure()
for value in (0, 1):
    sub = pts3[pts3["stroke"] == value]
    fig3d.add_trace(
        go.Scatter3d(
            x=sub["pa"],
            y=sub["pb"],
            z=sub["pc"],
            mode="markers",
            marker=dict(
                size=3,
                color=COLOR_POS if value == 1 else COLOR_NEG,
                opacity=0.8,
            ),
            customdata=sub[["a", "b", "c"]],
            hovertemplate=(
                f"{FEATURES[ax1]}: %{{customdata[0]}}<br>{FEATURES[ax2]}: %{{customdata[1]}}"
                f"<br>{FEATURES[ax3]}: %{{customdata[2]}}<extra>실제 {LABELS[value]}</extra>"
            ),
            name=f"실제 {LABELS[value]}",
        )
    )

for label, color in ((BEFORE, COLOR_BEFORE), (AFTER, COLOR_AFTER)):
    poly = planes[label]
    if poly is None:
        continue
    m = len(poly)
    fig3d.add_trace(
        go.Mesh3d(
            x=[p[0] for p in poly],
            y=[p[1] for p in poly],
            z=[p[2] for p in poly],
            i=[0] * (m - 2),
            j=list(range(1, m - 1)),
            k=list(range(2, m)),
            color=color,
            opacity=0.45,
            name=f"{label}의 0.5 평면",
            showlegend=True,
            hoverinfo="skip",
        )
    )

fig3d.update_layout(
    title="테스트 데이터와 0.5 평면",
    height=680,
    legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0),
    scene=dict(
        xaxis=dict(title=FEATURES[ax1], range=[lo3[0], hi3[0]]),
        yaxis=dict(title=FEATURES[ax2], range=[lo3[1], hi3[1]]),
        zaxis=dict(title=FEATURES[ax3], range=[lo3[2], hi3[2]]),
    ),
)
st.plotly_chart(fig3d, width="stretch")

rest3 = fixed_text(axes3)
if rest3:
    st.write(f"세 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: {rest3}")
else:
    st.write("고른 속성이 세 개뿐이라, 세 축 말고는 세워 둘 속성이 없습니다.")

center3 = {
    ax1: [(lo3[0] + hi3[0]) / 2],
    ax2: [(lo3[1] + hi3[1]) / 2],
    ax3: [(lo3[2] + hi3[2]) / 2],
}
for label, model in ((BEFORE, logi_before), (AFTER, logi_after)):
    if planes[label] is None:
        st.write(outside_text(label, model, center3, "평면"))
    else:
        st.write(f"{label}의 0.5 평면은 그림 안에 그려져 있습니다.")

if any(col in BINARY for col in axes3):
    st.caption(
        "고혈압·심장병은 0(없음)과 1(있음)뿐이라, 점이 겹치지 않도록 그림에서만 점을 살짝 흩뜨려 찍었습니다."
    )

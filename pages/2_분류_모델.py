import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------
# 기본 설정
# ---------------------------------------------------------------
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실", page_icon="🧪", layout="wide"
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
BASE_NAME = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

LABELS = {0: "아님", 1: "뇌졸중"}  # stroke가 1이면 뇌졸중(양성), 0이면 아님
COLOR_NEG = "#1f77b4"  # 아님: 파랑
COLOR_POS = "#d62728"  # 뇌졸중: 빨강


@st.cache_data(show_spinner="데이터를 불러오는 중입니다...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧪 분류 모델")
st.caption(
    "stroke가 1이면 '뇌졸중', 0이면 '아님'입니다. 뇌졸중을 양성으로 둡니다."
)

try:
    raw_df = load_data()
except Exception as e:  # 주소 오류, 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

# ---------------------------------------------------------------
# 1) 입력으로 사용할 속성 고르기
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
# 2) 데이터 나누기: 번호 순 정렬 -> 열 명씩 묶어 앞 세 명은 테스트용
# ---------------------------------------------------------------
df = raw_df.sort_values("id").reset_index(drop=True)
is_test = (df.index % 10) < 3

X_train = df.loc[~is_test, selected].copy()
y_train = df.loc[~is_test, "stroke"].copy()
X_test = df.loc[is_test, selected].copy()
y_test = df.loc[is_test, "stroke"].copy()

st.caption(
    f"번호 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정했습니다. "
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
# 3) 학습과 크기 맞추기 (훈련용 데이터로만)
# ---------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 로지스틱 회귀: 크기를 맞춘 값으로 학습
logi = LogisticRegression(max_iter=1000, random_state=42)
logi.fit(X_train_scaled, y_train)

# 의사결정트리: 질문은 세 번까지, 마지막 마디에 사람이 다섯 명 미만이면 나누지 않음, 난수 고정
# (트리는 크기를 맞추지 않은 값으로 학습해도 같은 질문을 던지므로, 가지 그림에서 원래 값을 그대로 보여 준다.)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
tree.fit(X_train, y_train)

# 기준 모델: 입력을 보지 않고 훈련용에서 많은 쪽으로만 답함
base = DummyClassifier(strategy="most_frequent")
base.fit(X_train, y_train)
base_answer = LABELS[int(base.predict(X_train.head(1))[0])]

# ---------------------------------------------------------------
# 4) 정확도 카드
# ---------------------------------------------------------------
st.header("정확도")


def accuracy_card(title: str, train_acc: float, test_acc: float, note: str = ""):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.metric("테스트 데이터 정확도", f"{test_acc * 100:.2f}%")
        left, right = st.columns(2)
        left.caption(f"훈련 데이터 {train_acc * 100:.2f}%")
        right.caption(f"테스트 데이터 {test_acc * 100:.2f}%")
        if note:
            st.caption(note)


card1, card2, card3 = st.columns(3)
with card1:
    accuracy_card(
        LOGI_NAME,
        logi.score(X_train_scaled, y_train),
        logi.score(X_test_scaled, y_test),
    )
with card2:
    accuracy_card(
        TREE_NAME,
        tree.score(X_train, y_train),
        tree.score(X_test, y_test),
    )
with card3:
    accuracy_card(
        BASE_NAME,
        base.score(X_train, y_train),
        base.score(X_test, y_test),
        note=f"언제나 '{base_answer}'이라고 답합니다.",
    )

# ---------------------------------------------------------------
# 5) 산점도: 테스트 데이터 + 로지스틱 회귀 0.5 경계선 + 의사결정트리 칸
# ---------------------------------------------------------------
st.header("두 속성으로 보는 산점도")

col_x, col_y = st.columns(2)
with col_x:
    x_feat = st.selectbox(
        "가로축", selected, index=0, format_func=lambda col: FEATURES[col]
    )
y_options = [col for col in selected if col != x_feat]
with col_y:
    y_feat = st.selectbox(
        "세로축", y_options, index=0, format_func=lambda col: FEATURES[col]
    )
x_name, y_name = FEATURES[x_feat], FEATURES[y_feat]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 둔다.
medians = X_test.median()
fixed = {col: float(medians[col]) for col in selected if col not in (x_feat, y_feat)}


def axis_range(col: str):
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    if col in BINARY:
        return -0.3, 1.3
    span = hi - lo
    pad = span * 0.05 if span > 0 else 0.5
    return lo - pad, hi + pad


x_lo, x_hi = axis_range(x_feat)
y_lo, y_hi = axis_range(y_feat)


def make_points(xs: list, ys: list) -> pd.DataFrame:
    """가로·세로 값은 주어진 대로, 나머지 속성은 중앙값으로 채운 표를 만든다."""
    n = len(xs)
    data = {}
    for col in selected:
        if col == x_feat:
            data[col] = xs
        elif col == y_feat:
            data[col] = ys
        else:
            data[col] = [fixed[col]] * n
    return pd.DataFrame(data, columns=selected)


# --- 의사결정트리가 나눈 칸 (격자의 모든 자리에 대해 답을 계산) ---
N_GRID = 120
grid_x = [x_lo + (x_hi - x_lo) * i / (N_GRID - 1) for i in range(N_GRID)]
grid_y = [y_lo + (y_hi - y_lo) * i / (N_GRID - 1) for i in range(N_GRID)]
grid = make_points(grid_x * N_GRID, [gy for gy in grid_y for _ in range(N_GRID)])
grid_pred = tree.predict(grid).tolist()
grid_z = [grid_pred[i * N_GRID:(i + 1) * N_GRID] for i in range(N_GRID)]

# --- 로지스틱 회귀가 확률 0.5로 가르는 선: a + bx*x + by*y = 0 ---
def logi_decision(points: pd.DataFrame):
    return logi.decision_function(scaler.transform(points))


d3 = logi_decision(make_points([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))
coef_a = float(d3[0])
coef_bx = float(d3[1] - d3[0])
coef_by = float(d3[2] - d3[0])


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


segment = line_in_box(coef_a, coef_bx, coef_by, x_lo, x_hi, y_lo, y_hi)

# --- 그림 그리기 ---
def jitter(n: int, mult: int) -> list:
    """0/1 값 속성의 점이 겹치지 않도록 그림에서만 살짝 흩뜨리는 값."""
    return [((i * mult) % 101 / 100 - 0.5) * 0.24 for i in range(n)]


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

fig = go.Figure()

# 의사결정트리가 나눈 칸 (옅은 색)
fig.add_trace(
    go.Heatmap(
        x=grid_x,
        y=grid_y,
        z=grid_z,
        zmin=0,
        zmax=1,
        colorscale=[[0, COLOR_NEG], [1, COLOR_POS]],
        showscale=False,
        opacity=0.18,
        hoverinfo="skip",
    )
)
# 칸의 색을 설명하는 범례 항목
for value in (0, 1):
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                symbol="square",
                size=13,
                color=COLOR_POS if value == 1 else COLOR_NEG,
                opacity=0.3,
            ),
            name=f"의사결정트리가 '{LABELS[value]}'이라고 답하는 칸",
        )
    )

# 테스트 데이터의 점 (실제 뇌졸중 여부를 색으로)
for value in (0, 1):
    sub = plot_df[plot_df["stroke"] == value]
    fig.add_trace(
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

# 로지스틱 회귀가 0.5로 가르는 선
if segment is not None:
    (lx0, ly0), (lx1, ly1) = segment
    fig.add_trace(
        go.Scatter(
            x=[lx0, lx1],
            y=[ly0, ly1],
            mode="lines",
            line=dict(color="black", width=3, dash="dash"),
            name="로지스틱 회귀의 0.5 경계선",
            hoverinfo="skip",
        )
    )

fig.update_layout(
    title=f"테스트 데이터: {x_name} 과(와) {y_name}",
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    height=560,
)
fig.update_xaxes(title=x_name, range=[x_lo, x_hi])
fig.update_yaxes(title=y_name, range=[y_lo, y_hi])
st.plotly_chart(fig, width="stretch")

# --- 그림 설명 글 ---
if fixed:
    fixed_text = ", ".join(f"{FEATURES[col]} = {round(v, 2):g}" for col, v in fixed.items())
    st.write(
        f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: {fixed_text}"
    )
else:
    st.write("고른 속성이 두 개뿐이라, 두 축 말고는 세워 둘 속성이 없습니다.")

if segment is not None:
    st.write(
        f"검은 점선은 {LOGI_NAME}이 뇌졸중일 확률을 0.5로 가르는 선입니다."
    )
else:
    center = logi_decision(
        make_points([(x_lo + x_hi) / 2], [(y_lo + y_hi) / 2])
    )[0]
    if center > 0:
        side = "뇌졸중일 확률이 0.5보다 높게 나옵니다"
    else:
        side = (
            "뇌졸중일 확률이 0.5보다 낮게 나오므로, "
            "이 그림 안 어디에서도 이 모델은 '뇌졸중'이라고 답하지 않습니다"
        )
    st.write(
        f"{LOGI_NAME}이 확률 0.5로 가르는 선은 이 그림 밖에 있습니다. 그림 안의 모든 자리에서 {side}."
    )

if x_feat in BINARY or y_feat in BINARY:
    st.caption(
        "고혈압·심장병은 0(없음)과 1(있음)뿐이라, 점이 겹치지 않도록 그림에서만 점을 살짝 흩뜨려 찍었습니다."
    )

# ---------------------------------------------------------------
# 6) 의사결정트리가 던진 질문: 가지 그림
# ---------------------------------------------------------------
st.header("의사결정트리가 던진 질문")

t = tree.tree_

# 마디마다 훈련용 사람 수와 그중 실제 뇌졸중인 사람 수
path = pd.DataFrame(tree.decision_path(X_train).toarray())
node_total = path.sum()
node_positive = path[(y_train == 1).to_numpy()].sum()

LEAF_FILL = {0: "#b9d6ee", 1: "#f4b6b6"}  # 답에 따라 색을 다르게
FONT = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"

dot_lines = [
    "digraph Tree {",
    "graph [rankdir=TB, nodesep=0.3, ranksep=0.5];",
    f'node [shape=box, style="rounded,filled", fontname="{FONT}", fontsize=12, '
    'width=2.6, height=0.9, margin="0.15,0.1"];',
    f'edge [fontname="{FONT}", fontsize=12];',
]

leaf_answers = []
asked_features = []

for node in range(t.node_count):
    total = int(node_total[node])
    positive = int(node_positive[node])
    ratio = positive / total * 100 if total else 0.0
    stats = f"훈련용 {total:,}명\\n뇌졸중 {positive:,}명 ({ratio:.1f}%)"

    left, right = int(t.children_left[node]), int(t.children_right[node])
    if left == -1:  # 더 묻지 않고 답을 내는 마디
        answer = int(t.value[node][0].argmax())
        leaf_answers.append(answer)
        dot_lines.append(
            f'{node} [label="답: {LABELS[answer]}\\n{stats}", fillcolor="{LEAF_FILL[answer]}"];'
        )
    else:  # 질문을 던지는 마디
        col = selected[int(t.feature[node])]
        name = FEATURES[col]
        if name not in asked_features:
            asked_features.append(name)
        if col in BINARY:
            question = f"{name} 없음?"
        else:
            question = f"{name} ≤ {round(float(t.threshold[node]), 2):g} ?"
        dot_lines.append(
            f'{node} [label="질문: {question}\\n{stats}", fillcolor="#f2f2f2"];'
        )
        dot_lines.append(f'{node} -> {left} [label="예"];')
        dot_lines.append(f'{node} -> {right} [label="아니요"];')

dot_lines.append("}")
st.graphviz_chart("\n".join(dot_lines), width="stretch")

n_leaf = len(leaf_answers)
n_no = sum(1 for a in leaf_answers if a == 0)
st.write(
    f"답을 내는 마디(칸)는 모두 **{n_leaf}칸**이고, 그중 **{n_no}칸**이 '아님'이라고 답합니다."
)
if asked_features:
    st.write(
        f"고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 것은 "
        f"**{', '.join(asked_features)}**입니다."
    )
else:
    st.write(
        f"고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 것은 없습니다. 질문 없이 바로 답을 냈습니다."
    )

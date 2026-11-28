"""Графики для расширенного конспекта лекции 4 (переобучение, регуляризация, кодирование категорий).

Запуск из корня проекта:
    .venv/bin/python my_smth/lectures/scripts/lecture_04_plots.py
Данные: diabetes (встроен в sklearn — тот же, что в Week3_GD_Features/Regularization_and_Encoding_Base.ipynb)
        и синтетические примеры (генерируем сами).
Картинки: my_smth/lectures/img/lecture_04/
"""
import warnings
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # рисуем сразу в файлы, без окон
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, TargetEncoder

warnings.filterwarnings("ignore")           # Lasso иногда предупреждает о сходимости — для картинок не важно
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "img" / "lecture_04"
OUT.mkdir(parents=True, exist_ok=True)

# Та же палитра и стиль, что в графиках лекции 3
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096", "#6b46c1"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 11,
})


def save(fig, name):
    """Сохраняем картинку на белом фоне и закрываем, чтобы не копить память."""
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("saved", name)


# ============================================================ 1. Недо- и переобучение
def under_over():
    """Полиномы степени 1, 3 и 12 на данных из кубической зависимости + их веса (как слайды 6–7)."""
    rng = np.random.default_rng(1)
    true_f = lambda x: 1.5 * x ** 3 - 2 * x + 1          # «загаданная» зависимость — полином 3-й степени
    x_tr = np.sort(rng.uniform(-1, 1, 18))               # мало обучающих точек — легко переобучиться
    y_tr = true_f(x_tr) + rng.normal(0, 0.25, x_tr.size)
    x_te = rng.uniform(-1, 1, 300)                       # много новых точек для честной проверки
    y_te = true_f(x_te) + rng.normal(0, 0.25, x_te.size)
    grid = np.linspace(-1, 1, 500)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), gridspec_kw={"height_ratios": [1.3, 1]})
    for j, (deg, title, c) in enumerate([(1, "степень 1 — недообучение", BLUE),
                                         (3, "степень 3 — как надо", GREEN),
                                         (12, "степень 12 — переобучение", RED)]):
        poly = PolynomialFeatures(deg, include_bias=False)   # x, x², ..., x^deg
        model = LinearRegression().fit(poly.fit_transform(x_tr[:, None]), y_tr)
        pred = lambda x: model.predict(poly.transform(x[:, None]))
        mse_tr = mean_squared_error(y_tr, pred(x_tr))
        mse_te = mean_squared_error(y_te, pred(x_te))

        ax = axes[0, j]
        ax.plot(grid, true_f(grid), "--", c=GRAY, lw=1.5, label="истинная зависимость")
        ax.plot(grid, pred(grid), c=c, lw=2.5, label="модель")
        ax.scatter(x_tr, y_tr, c="black", s=25, zorder=3, label="обучающие точки")
        ax.set(ylim=(-1.2, 3.5), title=f"{title}\nMSE train = {mse_tr:.3f}   test = {mse_te:.3f}")
        if j == 0:
            ax.legend(loc="upper left", fontsize=9)

        ax = axes[1, j]
        w = np.r_[model.intercept_, model.coef_]           # w0, w1, ..., w_deg
        ax.bar(range(len(w)), w, color=c)
        ax.set_xticks(range(len(w)))                       # целые номера: w0, w1, ...
        ax.set(xlabel="номер веса (степень x)", ylabel="значение веса",
               title=f"веса: от {w.min():.1f} до {w.max():.1f}")
        print(f"deg {deg}: MSE train {mse_tr:.3f}, test {mse_te:.3f}, |w|max {abs(w).max():.1f}")
    fig.tight_layout()
    save(fig, "01_under_over.png")


# ============================================================ 2. Почти одинаковые признаки → веса скачут
def collinear_weights():
    """Два почти одинаковых признака: на разных подвыборках OLS даёт совершенно разные веса, Ridge — стабильные."""
    rng = np.random.default_rng(7)
    n = 60
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.02, n)                     # x2 почти копия x1 (корреляция ≈ 0.9998)
    y = 3 * x1 + rng.normal(0, 1, n)                     # на самом деле важно только x1, «сумма весов» = 3
    X = np.c_[x1, x2]
    ols, rdg = [], []
    for _ in range(200):                                 # 200 раз берём случайную подвыборку с повторами
        idx = rng.integers(0, n, n)
        ols.append(LinearRegression().fit(X[idx], y[idx]).coef_)
        rdg.append(Ridge(alpha=1).fit(X[idx], y[idx]).coef_)
    ols, rdg = np.array(ols), np.array(rdg)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    t = np.linspace(-40, 40, 2)
    ax.plot(t, 3 - t, ":", c=GRAY, label="$w_1 + w_2 = 3$ — все такие веса дают почти одинаковый прогноз")
    ax.scatter(ols[:, 0], ols[:, 1], s=14, alpha=.5, c=RED, label="LinearRegression (без регуляризации)")
    ax.scatter(rdg[:, 0], rdg[:, 1], s=70, c=BLUE, edgecolor="white", zorder=4, label="Ridge, α = 1")
    ax.annotate("Ridge: все 200 точек\nлегли почти в одну", rdg.mean(0), xytext=(8, 22), color=BLUE, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.set(xlabel="$w_1$", ylabel="$w_2$", xlim=(-35, 40), ylim=(-35, 40),
           title="200 обучений на разных подвыборках:\nбез регуляризации веса «гуляют» вдоль линии, Ridge держит их у (1.5, 1.5)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    save(fig, "02_collinear_weights.png")
    print("OLS w1 range", ols[:, 0].min().round(1), ols[:, 0].max().round(1),
          "sum mean", ols.sum(1).mean().round(2), "Ridge mean", rdg.mean(0).round(2))


# ============================================================ 3. Геометрия L1 и L2
def constrained_min(center, A, kind, C):
    """Минимум квадратичной ошибки (w-center)ᵀA(w-center) при ||w|| <= C перебором по границе.
    Если center уже внутри области — ответ сам center (ограничение не мешает)."""
    norm = (np.abs(center).sum() if kind == "l1" else np.sqrt((center ** 2).sum()))
    if norm <= C:
        return center
    t = np.linspace(0, 2 * np.pi, 200_001)
    if kind == "l2":
        pts = C * np.c_[np.cos(t), np.sin(t)]          # окружность радиуса C
    else:
        u = np.c_[np.cos(t), np.sin(t)]
        pts = C * u / np.abs(u).sum(1, keepdims=True)  # ромб |w1| + |w2| = C
    d = pts - center
    q = np.einsum("ij,jk,ik->i", d, A, d)
    return pts[q.argmin()]


def geometry():
    A = np.array([[1.0, 0.3], [0.3, 0.7]])            # форма «эллипсов» ошибки (признаки немного связаны)
    cases = [  # (центр = решение без регуляризации, подпись)
        (np.array([0.5, 2.3]), "один вес маленький"),
        (np.array([1.6, 2.2]), "оба веса большие"),
    ]
    C = 1.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    g = np.linspace(-1.6, 3.2, 400)
    W1, W2 = np.meshgrid(g, g)
    for row, kind in enumerate(["l1", "l2"]):
        for col, (center, label) in enumerate(cases):
            ax = axes[row, col]
            D1, D2 = W1 - center[0], W2 - center[1]
            Q = A[0, 0] * D1 ** 2 + 2 * A[0, 1] * D1 * D2 + A[1, 1] * D2 ** 2   # уровень ошибки в каждой точке
            ax.contour(W1, W2, Q, levels=[.05, .2, .5, 1, 1.7, 2.6, 3.7, 5], colors=GRAY, linewidths=1)
            if kind == "l1":
                ax.add_patch(Polygon([[C, 0], [0, C], [-C, 0], [0, -C]], closed=True,
                                     fc=ORANGE, alpha=.25, ec=ORANGE, lw=2))
            else:
                ax.add_patch(Circle((0, 0), C, fc=PURPLE, alpha=.2, ec=PURPLE, lw=2))
            w = constrained_min(center, A, kind, C)
            # линия уровня, которая касается области, — та, на которой лежит решение
            d = w - center
            ax.contour(W1, W2, Q, levels=[d @ A @ d], colors=RED, linewidths=2)
            ax.scatter(*center, c=BLUE, s=80, zorder=4)
            ax.annotate("решение без\nрегуляризации", center, xytext=(center[0] + .1, center[1] + .25), color=BLUE)
            ax.scatter(*w, c=RED, s=90, zorder=5, edgecolor="white")
            ax.annotate(f"$w^*$ = ({w[0]:.2f}, {w[1]:.2f})", w, xytext=(-1.5, -1.35),
                        color=RED, fontsize=12, weight="bold", arrowprops=dict(arrowstyle="->", color=RED),
                        bbox=dict(fc="white", ec="none", alpha=.85))  # подпись в углу + стрелка, чтобы не закрывать линии
            ax.axhline(0, c="black", lw=.8); ax.axvline(0, c="black", lw=.8)
            ax.set(xlim=(-1.6, 3.2), ylim=(-1.6, 3.2), aspect="equal", xlabel="$w_1$", ylabel="$w_2$",
                   title=f"{'L1: ромб |w₁|+|w₂| ≤ C' if kind == 'l1' else 'L2: круг w₁²+w₂² ≤ C²'} — {label}")
            print(kind, label, "->", w.round(3))
    fig.tight_layout()
    save(fig, "03_l1_l2_geometry.png")


# ============================================================ 4. Почему L1 зануляет: одномерный пример
def soft_threshold():
    """Q(w) = (w - w0)² + α·R(w). Решения: L1 — «сдвиг к нулю на α/2 и стоп в нуле», L2 — «деление на (1+α)»."""
    alpha = 1.0
    w0 = np.linspace(-2.5, 2.5, 501)
    lasso = np.sign(w0) * np.maximum(np.abs(w0) - alpha / 2, 0)   # формула L1-решения (выведена в конспекте)
    ridge = w0 / (1 + alpha)                                       # формула L2-решения

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    ax = axes[0]
    ax.plot(w0, w0, ":", c=GRAY, label="без регуляризации: $w^* = w_0$")
    ax.plot(w0, ridge, c=PURPLE, lw=2.5, label=r"L2: $w^* = w_0/(1+\alpha)$ — сжимает, но не до нуля")
    ax.plot(w0, lasso, c=ORANGE, lw=2.5, label=r"L1: $w^* = \mathrm{sign}(w_0)\,\max(|w_0| - \alpha/2,\ 0)$")
    ax.axvspan(-alpha / 2, alpha / 2, color=ORANGE, alpha=.12)
    ax.text(0, -1.9, "все $|w_0| \\leq \\alpha/2$\nL1 превращает в ровно 0", ha="center", color=ORANGE, fontsize=10)
    ax.set(xlabel="$w_0$ — вес без регуляризации", ylabel="$w^*$ — вес с регуляризацией",
           title="α = 1: что делает каждая регуляризация с весом")
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    w = np.linspace(-0.6, 0.8, 600)
    w0_ = 0.3
    for f, c, lab in [((w - w0_) ** 2 + alpha * np.abs(w), ORANGE, r"$(w-0.3)^2 + |w|$  (L1)"),
                      ((w - w0_) ** 2 + alpha * w ** 2, PURPLE, r"$(w-0.3)^2 + w^2$  (L2)")]:
        ax.plot(w, f, c=c, lw=2.5, label=lab)
        ax.scatter(w[f.argmin()], f.min(), c=c, s=80, zorder=4, edgecolor="white")
    ax.annotate("минимум ровно в 0:\nизлом модуля", (0, .09), xytext=(-.62, .12), color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.annotate("минимум в 0.15", (.15, .045), xytext=(.45, .08), color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE))
    ax.set(xlabel="w", ylabel="Q(w)", ylim=(0, 1), title="$w_0 = 0.3$, α = 1: где минимум у каждой функции")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "04_soft_threshold.png")


# ============================================================ 5–6. Семинар: diabetes
def diabetes_data():
    """Тот же датасет и то же разбиение, что в ноутбуке семинара (random_state=121)."""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=121)

    def add_pairwise(df):
        df = df.copy()
        for c1, c2 in combinations_with_replacement(X.columns, 2):   # все попарные произведения и квадраты
            df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
        return df
    return X, Xtr, Xte, ytr, yte, add_pairwise(Xtr), add_pairwise(Xte)


def alpha_curves():
    X, Xtr, Xte, ytr, yte, Btr, Bte = diabetes_data()
    alphas = np.logspace(-4, 1.5, 40)                    # 40 значений α от 0.0001 до ~30, равномерно в логарифме
    lr = LinearRegression().fit(Btr, ytr)
    base_te = mean_squared_error(yte, lr.predict(Bte))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, (name, Model, c) in zip(axes, [("Lasso (L1)", lambda a: Lasso(alpha=a, max_iter=100_000), ORANGE),
                                           ("Ridge (L2)", lambda a: Ridge(alpha=a), PURPLE)]):
        tr, te, zeros = [], [], []
        for a in alphas:
            m = Model(a).fit(Btr, ytr)
            tr.append(mean_squared_error(ytr, m.predict(Btr)))
            te.append(mean_squared_error(yte, m.predict(Bte)))
            zeros.append((np.abs(m.coef_) < 1e-8).sum())
        best = int(np.argmin(te))
        ax.plot(alphas, tr, c=c, lw=2, ls="--", label="train MSE")
        ax.plot(alphas, te, c=c, lw=2.5, label="test MSE")
        ax.axhline(base_te, c=RED, ls=":", label=f"без регуляризации: test {base_te:.0f}")
        ax.scatter(alphas[best], te[best], c=GREEN, s=90, zorder=5, label=f"лучшее α ≈ {alphas[best]:.3g}: test {te[best]:.0f}")
        ax.set_xscale("log")
        ax.set(xlabel="α (логарифмическая шкала)", ylabel="MSE", ylim=(2000, 7300), title=f"{name}: 65 признаков")
        ax.text(1.3e-4, 7000, "← переобучение", color=RED)
        ax.text(30, 7000, "недообучение →", color=BLUE, ha="right")
        if name.startswith("Lasso"):
            ax2 = ax.twinx()                              # вторая ось справа — число нулевых весов
            ax2.plot(alphas, zeros, c=GRAY, lw=1.5)
            ax2.set_ylabel("нулевых весов из 65 (серая линия)", color=GRAY)
            ax2.set_ylim(0, 70); ax2.grid(False)
        ax.legend(loc="lower right", fontsize=9)
        print(name, "best alpha", alphas[best].round(4), "test", round(te[best], 1))
    fig.tight_layout()
    save(fig, "05_alpha_curves.png")


def paths():
    """Как каждый из 10 исходных весов меняется с ростом α (путь регуляризации)."""
    X, Xtr, Xte, ytr, yte, *_ = diabetes_data()
    alphas = np.logspace(-3, 2, 80)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    colors = plt.cm.tab10(np.arange(10))
    for ax, (name, Model) in zip(axes, [("Lasso: веса по очереди становятся ровно 0", lambda a: Lasso(alpha=a, max_iter=100_000)),
                                        ("Ridge: веса плавно уменьшаются, но не до 0", lambda a: Ridge(alpha=a))]):
        W = np.array([Model(a).fit(Xtr, ytr).coef_ for a in alphas])     # строка = веса при одном α
        for j, col in enumerate(X.columns):
            ax.plot(alphas, W[:, j], c=colors[j], lw=2, label=col)
        ax.axhline(0, c="black", lw=.8)
        ax.set_xscale("log")
        ax.set(xlabel="α (логарифмическая шкала)", title=name)
    axes[0].set_ylabel("вес признака")
    axes[1].legend(ncol=2, fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, "06_paths.png")


# ============================================================ 7. Target encoding: сглаживание
def te_smoothing():
    # Таблица со слайда 23: Москва 2 из 5 вернули, Тверь 4 из 5, Клин 0 из 2; всего 6 из 12 → 0.5
    cities = {"Москва": (0.4, 5), "Тверь": (0.8, 5), "Клин": (0.0, 2)}
    gm = 0.5
    a = np.linspace(0, 30, 301)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ax = axes[0]
    for (name, (m, n)), c in zip(cities.items(), [BLUE, GREEN, RED]):
        ax.plot(a, (m * n + gm * a) / (n + a), c=c, lw=2.5, label=f"{name}: среднее {m}, объектов {n}")
    ax.axhline(gm, c=GRAY, ls=":", label="глобальное среднее 0.5")
    ax.axvline(2, c="black", lw=.8, ls="--")
    ax.text(2.4, .05, "α = 2", fontsize=10)
    ax.set(xlabel="α — сила сглаживания", ylabel="код категории", ylim=(-.03, .85),
           title="Чем больше α, тем сильнее коды тянутся к 0.5.\nКлин (2 объекта) тянется быстрее всех")
    ax.legend(fontsize=9)

    ax = axes[1]
    n = np.arange(1, 201)
    for alpha, c in [(0, GRAY), (5, BLUE), (20, PURPLE)]:
        ax.plot(n, (0.0 * n + gm * alpha) / (n + alpha), c=c, lw=2.5, label=f"α = {alpha}")
    ax.set_xscale("log")
    ax.set(xlabel="сколько объектов в категории (лог. шкала)", ylabel="код категории",
           title="Категория, где таргет всегда 0. Мало объектов → код ≈ 0.5,\nмного объектов → код → 0 (доверяем данным)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "07_te_smoothing.png")


# ============================================================ 8. Target encoding: утечка на редких категориях
def te_leak():
    """Категория из 600 «городов», которая НИКАК не связана с таргетом. Наивный счётчик всё равно «помогает» на train."""
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(0, 1, n)
    city = rng.integers(0, 600, n).astype(str)          # в среднем 5 объектов на город, город на y не влияет
    y = 2 * x + rng.normal(0, 2, n)
    df = pd.DataFrame({"x": x, "city": city, "y": y})
    tr, te = train_test_split(df, test_size=0.3, random_state=0)

    def fit_eval(f_tr, f_te):
        m = LinearRegression().fit(f_tr, tr.y)
        return r2_score(tr.y, m.predict(f_tr)), r2_score(te.y, m.predict(f_te))

    res = {}
    res["только x\n(без города)"] = fit_eval(tr[["x"]], te[["x"]])

    gm = tr.y.mean()
    stats = tr.groupby("city").y.agg(["mean", "count"])
    for name, alpha in [("наивный счётчик\n(α = 0)", 0), ("сглаживание\nα = 20", 20)]:
        code = (stats["mean"] * stats["count"] + alpha * gm) / (stats["count"] + alpha)
        enc_tr = tr.city.map(code)                       # train кодируем средними, посчитанными по нему же — утечка
        enc_te = te.city.map(code).fillna(gm)            # незнакомый город → глобальное среднее
        res[name] = fit_eval(np.c_[tr.x, enc_tr], np.c_[te.x, enc_te])

    enc = TargetEncoder(target_type="continuous", cv=5, random_state=0)
    enc_tr = enc.fit_transform(tr[["city"]], tr.y).ravel()   # out-of-fold: код объекта считается по ДРУГИМ фолдам
    enc_te = enc.transform(te[["city"]]).ravel()
    res["по кросс-валидации\n(TargetEncoder, 5 фолдов)"] = fit_eval(np.c_[tr.x, enc_tr], np.c_[te.x, enc_te])

    fig, ax = plt.subplots(figsize=(12, 5.2))
    names = list(res)
    pos = np.arange(len(names))
    ax.bar(pos - .2, [res[k][0] for k in names], .4, color=GRAY, label="R² train")
    ax.bar(pos + .2, [res[k][1] for k in names], .4, color=BLUE, label="R² test")
    for i, k in enumerate(names):
        ax.text(i - .2, res[k][0] + .01, f"{res[k][0]:.2f}", ha="center")
        ax.text(i + .2, res[k][1] + .01, f"{res[k][1]:.2f}", ha="center", color=BLUE, weight="bold")
    ax.set_xticks(pos, names)
    ax.set(ylim=(0, 1), ylabel="R²",
           title="Город — чистый шум (600 городов, ≈5 объектов на город).\nНаивный счётчик «улучшает» train и портит test")
    ax.legend()
    fig.tight_layout()
    save(fig, "08_te_leak.png")
    for k, v in res.items():
        print(k.replace("\n", " "), [round(t, 3) for t in v])


# ============================================================ 9. Схема кодирования по фолдам
def kfold_scheme():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axis("off")
    ax.set(xlim=(0, 13), ylim=(-.2, 4.4))
    for step in range(3):                                # три шага: кодируем фолд 3, потом 2, потом 1
        y0 = 3.2 - step * 1.25
        target = 2 - step                                # какой фолд сейчас кодируем
        ax.text(0, y0 + .32, f"шаг {step + 1}", fontsize=12, weight="bold")
        for k in range(3):
            is_t = k == target
            ax.add_patch(Rectangle((1.3 + k * 2.6, y0), 2.4, .65, fc=ORANGE if is_t else GREEN,
                                   alpha=.85 if is_t else .45, ec="white", lw=2))
            ax.text(2.5 + k * 2.6, y0 + .32, f"фолд {k + 1}\n{'кодируем' if is_t else 'считаем средние'}",
                    ha="center", va="center", fontsize=10)
    ax.text(9.5, 3.52, "Средние считаем по зелёным фолдам\nи ими кодируем оранжевый.", fontsize=11, va="center")
    ax.text(9.5, 2.27, "У объекта код никогда не считается\nпо его собственному таргету.", fontsize=11, va="center")
    ax.text(9.5, 1.02, "В итоге закодирован ВЕСЬ train,\nничего не выброшено.", fontsize=11, va="center")
    save(fig, "09_kfold_scheme.png")


if __name__ == "__main__":
    under_over()
    collinear_weights()
    geometry()
    soft_threshold()
    alpha_curves()
    paths()
    te_smoothing()
    te_leak()
    kfold_scheme()

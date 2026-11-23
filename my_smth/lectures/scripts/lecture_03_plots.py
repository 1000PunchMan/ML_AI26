"""Графики для расширенного конспекта лекции 3.

Запуск из корня проекта:
    .venv/bin/python my_smth/lectures/scripts/lecture_03_plots.py
Данные: my_smth/lectures/data/imports-85.data (UCI Automobile — тот же, что в Week2_Regression/LinReg_pipeline.ipynb)
Картинки: my_smth/lectures/img/lecture_03/
"""
import warnings
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import impute
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "img" / "lecture_03"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096", "#6b46c1"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 11,
})
rng = np.random.default_rng(3)


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("saved", name)


# ====================================================== ЧАСТЬ 1. Правдоподобие
def coin_likelihood():
    p = np.linspace(0, 1, 400)
    L = comb(10, 7) * p ** 7 * (1 - p) ** 3
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    ax.plot(p, L, c=BLUE, lw=2.5)
    for pv, c in [(.1, GRAY), (.5, ORANGE), (.7, RED)]:
        v = comb(10, 7) * pv ** 7 * (1 - pv) ** 3
        ax.scatter([pv], [v], c=c, s=70, zorder=3)
        ax.annotate(f"p={pv}: {v:.3f}", (pv, v), xytext=(pv + .04, v - .004 if pv > .5 else v + .01 if pv < .5 else v - .012),
                    color=c, fontsize=10)
    ax.axvline(.7, c=RED, ls=":", lw=1)
    ax.set(title="Правдоподобие L(p) = C(10,7)·p⁷·(1−p)³\n«насколько вероятно увидеть 7 орлов из 10 при данном p»",
           xlabel="p — вероятность орла (параметр)", ylabel="L(p)")
    ax = axes[1]
    ax.plot(p[1:-1], -np.log(L[1:-1]), c=PURPLE, lw=2.5)
    ax.axvline(.7, c=RED, ls=":", lw=1)
    ax.set_ylim(0, 12)
    ax.set(title="−log L(p): максимум превратился в минимум\nи находится в той же точке p = 0.7",
           xlabel="p", ylabel="−log L(p)")
    save(fig, "01_coin_likelihood.png")


def regression_as_distribution():
    x = rng.uniform(0, 10, 120)
    y = 1 + 0.8 * x + rng.normal(0, 1, 120)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(x, y, s=14, c=GRAY, alpha=.6)
    xs = np.linspace(0, 10, 50)
    ax.plot(xs, 1 + .8 * xs, c=BLUE, lw=2.5, label="$\\langle w, x\\rangle$ — среднее y при данном x")
    for x0 in [2, 5, 8]:
        yy = np.linspace(-3.2, 3.2, 100)
        dens = stats.norm.pdf(yy, 0, 1)
        ax.fill_betweenx(1 + .8 * x0 + yy, x0, x0 + dens * 2.2, color=ORANGE, alpha=.45)
        ax.plot([x0, x0], [1 + .8 * x0 - 3.2, 1 + .8 * x0 + 3.2], c=ORANGE, lw=1)
    ax.plot([], [], c=ORANGE, lw=6, alpha=.45, label="$y \\mid x \\sim \\mathcal{N}(\\langle w,x\\rangle,\\ \\sigma^2)$ — колокол шума")
    ax.set(title="Вероятностный взгляд: y = ⟨w, x⟩ + ε,  ε ~ N(0, σ²)\n"
                 "в каждой точке x целевая переменная «разбросана» колоколом вокруг прямой",
           xlabel="x", ylabel="y")
    ax.legend(loc="upper left")
    save(fig, "02_regression_as_distribution.png")


def normal_vs_laplace():
    z = np.linspace(-5, 5, 600)
    b = 1 / np.sqrt(2)  # Лаплас с той же дисперсией 1
    n_pdf, l_pdf = stats.norm.pdf(z), stats.laplace.pdf(z, scale=b)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), gridspec_kw={"width_ratios": [1.2, 1, 1]})
    ax = axes[0]
    ax.plot(z, n_pdf, c=BLUE, lw=2.5, label="Нормальное: $\\propto e^{-z^2/2\\sigma^2}$")
    ax.plot(z, l_pdf, c=RED, lw=2.5, label="Лапласа: $\\propto e^{-|z|/b}$")
    ax.set(title="Одна и та же дисперсия, разная форма", xlabel="шум ε (в σ)", ylabel="плотность")
    ax.legend(fontsize=9)
    ax = axes[1]
    ax.semilogy(z, n_pdf, c=BLUE, lw=2.5)
    ax.semilogy(z, l_pdf, c=RED, lw=2.5)
    ax.set_ylim(1e-7, 1)
    ax.set(title="То же в лог-шкале: «тяжёлый хвост» Лапласа", xlabel="шум ε (в σ)")
    ax.annotate("у Лапласа далёкие\nзначения встречаются\nнамного чаще", xy=(4, l_pdf[np.argmin(abs(z - 4))]),
                xytext=(.3, 1e-6), arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax = axes[2]
    ks = np.array([1, 2, 3, 4])
    pn = 2 * stats.norm.sf(ks)
    pl = np.exp(-ks * np.sqrt(2))
    w = .38
    ax.bar(ks - w / 2, pn * 100, w, color=BLUE, label="нормальное")
    ax.bar(ks + w / 2, pl * 100, w, color=RED, label="Лапласа")
    for k, a, c in zip(ks, pn, pl):
        ax.text(k - w / 2, a * 100 * 1.15, f"{a * 100:.2g}%", ha="center", fontsize=8.5, color=BLUE)
        ax.text(k + w / 2, c * 100 * 1.15, f"{c * 100:.2g}%", ha="center", fontsize=8.5, color=RED)
    ax.set_yscale("log")
    ax.set_xticks(ks, [f"> {k}σ" for k in ks])
    ax.set(title="Доля точек дальше kσ от среднего", ylabel="%, лог-шкала")
    ax.legend(fontsize=9)
    save(fig, "03_normal_vs_laplace.png")


def mle_to_loss():
    z = np.linspace(-3, 3, 400)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    d = 1.0
    huber = np.where(np.abs(z) <= d, .5 * z ** 2, d * (np.abs(z) - .5 * d))
    for ax, (pdf, nll, name, col) in zip(axes, [
        (np.exp(-z ** 2 / 2), z ** 2 / 2, "Нормальный шум → MSE", BLUE),
        (np.exp(-np.abs(z)), np.abs(z), "Шум Лапласа → MAE", RED),
        (np.exp(-huber), huber, "«Нормальный в центре + Лаплас\nна хвостах» → Huber", GREEN)]):
        ax.plot(z, pdf / pdf.max(), c=col, lw=2.5, label="плотность шума (норм.)")
        ax2 = ax.twinx()
        ax2.plot(z, nll, c="k", lw=2, ls="--", label="−log плотности = loss")
        ax2.set_ylim(0, 4.6)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax.set_ylim(0, 1.15)
        ax.set(title=name, xlabel="ошибка z = y − ⟨w,x⟩")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    fig.suptitle("Метод максимального правдоподобия: функция потерь = −log(плотность шума)", y=1.04)
    save(fig, "04_mle_to_loss.png")


def four_situations():
    n = 120
    x = rng.uniform(0, 10, n)
    cases = [
        ("Нормальный шум, σ = const\n→ MSE подходит идеально", 1 + x + rng.normal(0, 1, n), BLUE),
        ("Гетероскедастичность:\nразброс растёт с x", 1 + x + rng.normal(0, .15 + .35 * x, n), ORANGE),
        ("Выбросы → шум не нормальный\n→ MAE / Huber", None, RED),
        ("Зависимость нелинейная\n→ нужны новые признаки", 1 + .15 * (x - 5) ** 2 * 2 + rng.normal(0, 1, n), PURPLE),
    ]
    y_out = 1 + x + rng.normal(0, 1, n)
    idx = rng.choice(n, 8, replace=False)
    y_out[idx] += rng.choice([-1, 1], 8) * rng.uniform(7, 12, 8)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, (t, y, c) in zip(axes, cases):
        y = y_out if y is None else y
        ax.scatter(x, y, s=12, c=c, alpha=.65)
        w1, w0 = np.polyfit(x, y, 1)
        ax.plot([0, 10], [w0, w0 + 10 * w1], c="k", lw=1.8)
        ax.set_title(t, fontsize=10.5)
    save(fig, "05_four_situations.png")


def outlier_vs_cluster():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharex=True, sharey=True)
    base = rng.normal([3, 3], .7, size=(150, 2))
    ax = axes[0]
    ax.scatter(*base.T, c=GRAY, s=14, alpha=.6)
    out = np.array([[7.5, 1], [1, 7.8], [8, 7.5], [6.8, 6.4], [.3, .4]])
    ax.scatter(*out.T, c=RED, marker="X", s=110, label="выбросы")
    ax.set_title("Выбросы: далеко от всех\nи НЕ похожи друг на друга")
    ax.legend(loc="upper right", fontsize=9)
    ax = axes[1]
    ax.scatter(*base.T, c=GRAY, s=14, alpha=.6)
    cl = rng.normal([7, 7], .35, size=(15, 2))
    ax.scatter(*cl.T, c=GREEN, s=22, label="маленький кластер (элитное жильё)")
    ax.set_title("Не выбросы: маленькая, но плотная\nгруппа похожих объектов")
    ax.legend(loc="lower right", fontsize=9)
    ax = axes[2]
    x = np.linspace(.5, 8.5, 60)
    ax.scatter(x, .8 * x + .5 + rng.normal(0, .35, 60), c=BLUE, s=16, label="шум: небольшие отклонения от закономерности")
    ax.plot(x, .8 * x + .5, c="k", lw=1.5)
    ax.set_title("Шум: точки чуть выше/ниже\nистинной зависимости")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "06_outlier_cluster_noise.png")


def huber_constant():
    y = np.array([1.5, 2.0, 2.0, 2.5, 10.0])
    deltas = np.linspace(.05, 9, 120)
    best = []
    a_grid = np.linspace(1, 5, 4001)
    for d in deltas:
        z = y[None, :] - a_grid[:, None]
        L = np.where(np.abs(z) <= d, .5 * z ** 2, d * (np.abs(z) - .5 * d)).sum(1)
        best.append(a_grid[np.argmin(L)])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(deltas, best, c=GREEN, lw=2.5, label="оптимальная константа Huber")
    ax.axhline(np.median(y), c=RED, ls="--", label=f"медиана = {np.median(y)} (MAE)")
    ax.axhline(y.mean(), c=BLUE, ls="--", label=f"среднее = {y.mean()} (MSE)")
    ax.set(title="Цены [1.5, 2, 2, 2.5, 10]: порог δ у Huber двигает прогноз от медианы к среднему",
           xlabel="порог δ (где квадрат переходит в модуль)", ylabel="лучшая константа")
    ax.legend(fontsize=9)
    save(fig, "07_huber_constant.png")


# ====================================================== ЧАСТЬ 2. Метрики
def r2_picture():
    x = np.array([1, 2, 3, 4, 5, 6], float)
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.3, 11.7])
    w1, w0 = np.polyfit(x, y, 1)
    pred = w0 + w1 * x
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, p, t, c in [(axes[0], np.full_like(y, y.mean()), f"Константа ȳ = {y.mean():.1f}:  SS_tot = Σ(y − ȳ)²", ORANGE),
                        (axes[1], pred, "Модель:  SS_res = Σ(y − a(x))²", BLUE)]:
        ax.scatter(x, y, c="k", zorder=3)
        ax.plot(x, p, c=c, lw=2.5)
        for xi, yi, pi in zip(x, y, p):
            ax.add_patch(plt.Rectangle((xi, min(yi, pi)), abs(yi - pi) * .35, abs(yi - pi), color=c, alpha=.25))
            ax.plot([xi, xi], [yi, pi], c=c, lw=1.5)
        ss = ((y - p) ** 2).sum()
        ax.set_title(f"{t} = {ss:.2f}")
    sst, ssr = ((y - y.mean()) ** 2).sum(), ((y - pred) ** 2).sum()
    fig.suptitle(f"R² = 1 − SS_res / SS_tot = 1 − {ssr:.2f} / {sst:.2f} = {1 - ssr / sst:.3f}   "
                 "(квадратики — это квадраты ошибок)", y=1.03)
    save(fig, "08_r2_picture.png")


def r2_vs_features():
    n, ks = 80, np.arange(0, 61, 3)
    tr, te, adj = [], [], []
    for k in ks:
        a, b, c = [], [], []
        for _ in range(30):
            X = rng.normal(size=(2 * n, 3 + k))
            y = X[:, 0] + .5 * X[:, 1] - .5 * X[:, 2] + rng.normal(0, 1, 2 * n)
            m = LinearRegression().fit(X[:n], y[:n])
            r = r2_score(y[:n], m.predict(X[:n]))
            a.append(r); b.append(r2_score(y[n:], m.predict(X[n:])))
            p = 3 + k
            c.append(1 - (1 - r) * (n - 1) / (n - p - 1))
        tr.append(np.mean(a)); te.append(np.mean(b)); adj.append(np.mean(c))
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(ks, tr, "-o", c=BLUE, label="R² на train — только растёт")
    ax.plot(ks, adj, "-s", c=GREEN, label="adjusted R² на train — штраф за признаки")
    ax.plot(ks, te, "-^", c=RED, label="R² на test — падает (переобучение)")
    ax.set(title=f"Добавляем бесполезные (случайные) признаки к 3 полезным, n_train = {n}",
           xlabel="число мусорных признаков", ylabel="R²")
    ax.legend(fontsize=9)
    save(fig, "09_r2_vs_features.png")


def asymmetric_metrics():
    y = 100.0
    a = np.linspace(20, 300, 500)
    msle = (np.log1p(a) - np.log1p(y)) ** 2
    yy = np.linspace(5, 60, 500)
    mape = np.abs(yy - 20) / yy  # прогноз фиксирован = 20, меняется факт
    smape = np.abs(y - a) / ((np.abs(y) + np.abs(a)) / 2)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    ax = axes[1]
    ax.plot(yy, mape, c=ORANGE, lw=2.5)
    ax.axvline(20, c="k", lw=.8)
    for yv in [10, 30]:
        v = abs(yv - 20) / yv
        ax.scatter([yv], [v], c=BLUE if yv < 20 else RED, s=60, zorder=3)
        ax.annotate(f"факт {yv}: {v:.0%}", (yv, v), xytext=(yv + 2, v + .08), fontsize=9)
    ax.set(title="MAPE (одно слагаемое), прогноз a = 20", xlabel="факт y")
    ax.text(32, 1.6, "ошибка ±10 в обе стороны,\nно перепрогноз (факт 10) = 100%,\nа недопрогноз (факт 30) = 33%:\nделим на факт", fontsize=9, color=ORANGE)
    for ax, L, t, c in [(axes[0], msle, "MSLE (одно слагаемое)", PURPLE),
                        (axes[2], smape, "SMAPE (одно слагаемое)", GREEN)]:
        ax.plot(a, L, c=c, lw=2.5)
        ax.axvline(y, c="k", lw=.8)
        for ap in [50, 150]:
            v = L[np.argmin(abs(a - ap))]
            ax.scatter([ap], [v], c=RED if ap < y else BLUE, s=60, zorder=3)
            ax.annotate(f"a={ap}: {v:.3f}", (ap, v), xytext=(ap + 8, v + .03 * L.max()), fontsize=9)
        ax.set(title=f"{t}, факт y = 100", xlabel="прогноз a")
    axes[0].text(170, msle.max() * .55, "занижение на 50\nштрафуется почти\nв 3 раза сильнее\nзавышения на 50",
                 fontsize=9, color=PURPLE)
    axes[2].text(165, 1.12, "растёт всё медленнее:\nогромный прогноз почти\nне штрафуется (≤ 2)",
                 fontsize=9, color=GREEN)
    save(fig, "10_asymmetric_metrics.png")


# ====================================================== ЧАСТЬ 3. Семинар
COLUMNS = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors',
           'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height',
           'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore',
           'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']


def cars():
    X = pd.read_csv(ROOT / "data" / "imports-85.data", header=None, na_values=["?"], names=COLUMNS)
    X = X[X.price.notna()].reset_index(drop=True)
    y = X.pop("price")
    m = (~X.dtypes.apply(pd.api.types.is_numeric_dtype)).values
    Xr = pd.DataFrame(impute.SimpleImputer(strategy="mean").fit_transform(X[X.columns[~m]]), columns=X.columns[~m])
    Xn = pd.concat([Xr, X[X.columns[m]].fillna("NA")], axis=1)
    D = pd.get_dummies(Xn, drop_first=True).astype(float)
    Xs = MinMaxScaler().fit_transform(D)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=.25, random_state=123)
    const = DummyRegressor().fit(Xtr, ytr)
    lr = LinearRegression().fit(Xtr, ytr)
    r = {k: (r2_score(ytr, mdl.predict(Xtr)), r2_score(yte, mdl.predict(Xte)))
         for k, mdl in [("константа\n(DummyRegressor)", const), ("LinearRegression", lr)]}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1, 1.1, 1]})
    ax = axes[0]
    names = list(r)
    w = .35
    for i, (lab, c) in enumerate([("train", BLUE), ("test", RED)]):
        vals = [r[k][i] for k in names]
        bars = ax.bar(np.arange(2) + (i - .5) * w, vals, w, color=c, label=lab)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + (.02 if v >= 0 else -.08), f"{v:.3f}", ha="center")
    ax.axhline(0, c="k", lw=.8)
    ax.set_xticks(range(2), names)
    ax.set(title="R²: модель ≫ константы,\nно train 0.976 vs test 0.772 → переобучение", ylim=(-.45, 1.15))
    ax.legend()

    ax = axes[1]
    ax.scatter(yte, lr.predict(Xte), c=BLUE, s=30, alpha=.75, label="test")
    lim = [0, 50000]
    ax.plot(lim, lim, c="k", lw=1, ls="--", label="идеал: прогноз = факт")
    ax.set(xlim=lim, ylim=lim, xlabel="настоящая цена, $", ylabel="прогноз, $",
           title="Факт vs прогноз на test:\nдалеко от диагонали — большие ошибки")
    ax.legend(fontsize=9)

    ax = axes[2]
    coefs = np.sort(np.abs(lr.coef_))
    ax.bar(range(len(coefs)), coefs, color=[RED if c > 10000 else GRAY for c in coefs])
    ax.set_yscale("symlog", linthresh=10)
    ax.set(title=f"|веса| модели (65 признаков, ранг X = {np.linalg.matrix_rank(Xs)})\n"
                 "от 0 до 43 000 — симптом переобучения", xlabel="признаки (отсортированы)", ylabel="|w|")
    save(fig, "11_cars_results.png")


def feature_engineering_toy():
    n = 300
    x1, x2 = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n)
    y = 2 * x1 + 3 * x2 ** 2 + rng.normal(0, 2, n)
    X = np.c_[x1, x2]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, random_state=0)
    res = {}
    lr = LinearRegression().fit(Xtr, ytr)
    res["линейно\n(x1, x2)"] = r2_score(yte, lr.predict(Xte))
    edges = np.linspace(-3, 3, 5)
    def bins(Z):
        idx = np.clip(np.digitize(Z[:, 1], edges[1:-1]), 0, 3)
        return np.c_[Z[:, 0], np.eye(4)[idx][:, 1:]]
    lb = LinearRegression().fit(bins(Xtr), ytr)
    res["x2 → 4 бина\n+ one-hot"] = r2_score(yte, lb.predict(bins(Xte)))
    pipe = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), LinearRegression()).fit(Xtr, ytr)
    res["Polynomial\nFeatures(2)"] = r2_score(yte, pipe.predict(Xte))

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.scatter(x2, y - 2 * x1, s=12, c=GRAY, alpha=.6, label="y − 2·x1 (вклад x2)")
    g = np.linspace(-3, 3, 300)
    G = np.c_[np.zeros_like(g), g]
    ax.plot(g, lr.predict(G) - lr.intercept_ + lr.intercept_, c=RED, lw=2.5, label="линейно по x2")
    ax.plot(g, lb.predict(bins(G)), c=ORANGE, lw=2.5, label="бины x2")
    ax.plot(g, pipe.predict(G), c=GREEN, lw=2.5, label="полиномы степени 2")
    ax.set(title="Истина: y = 2·x1 + 3·x2² + шум. Как модели видят x2", xlabel="x2", ylabel="вклад x2")
    ax.legend(fontsize=9)
    ax = axes[1]
    bars = ax.bar(list(res), list(res.values()), color=[RED, ORANGE, GREEN])
    for b_, v in zip(bars, res.values()):
        ax.text(b_.get_x() + b_.get_width() / 2, v + .02, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set(title="R² на test — модель всё та же LinearRegression,\nменяются только признаки", ylim=(0, 1.1))
    save(fig, "12_feature_engineering.png")


def pipeline_diagram():
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.4), gridspec_kw={"height_ratios": [1, 1.25]})
    ax = axes[0]
    ax.axis("off")
    ax.set_title("ПЛОХО. Утечка (data leakage): преобразования «подсмотрели» в test", loc="left", color=RED)
    boxes = [(0, "ВСЕ данные\n(train + test)", "#fff5f5"), (2.6, "fit scaler / imputer /\nквантили на ВСЁМ", "#fed7d7"),
             (5.6, "split на\ntrain / test", "#edf2f7"), (8.2, "fit модели\nна train", "#edf2f7"),
             (10.8, "оценка на test\n(завышена!)", "#fed7d7")]
    for x0, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x0, 0), 2.1, 1, boxstyle="round,pad=.05", fc=c, ec="#2d3748"))
        ax.text(x0 + 1.05, .5, t, ha="center", va="center", fontsize=10)
    for x0 in [2.1, 4.7, 7.7, 10.3]:
        ax.annotate("", xy=(x0 + .45, .5), xytext=(x0 + .05, .5), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.set_xlim(-.2, 13.2); ax.set_ylim(-.2, 1.3)

    ax = axes[1]
    ax.axis("off")
    ax.set_title("ХОРОШО. Pipeline: всё, что «обучается», учится только на train", loc="left", color=GREEN)
    ax.add_patch(FancyBboxPatch((0, .35), 1.8, 1, boxstyle="round,pad=.05", fc="#edf2f7", ec="#2d3748"))
    ax.text(.9, .85, "split\ntrain / test", ha="center", va="center", fontsize=10)
    ax.add_patch(FancyBboxPatch((2.4, 0), 8.2, 1.7, boxstyle="round,pad=.08", fc="#f0fff4", ec=GREEN, lw=2))
    ax.text(6.5, 1.5, "Pipeline  (pipe.fit(X_train) / pipe.predict(X_test))", ha="center", color=GREEN, fontsize=10)
    steps = [(2.7, "SimpleImputer"), (4.7, "OneHotEncoder"), (6.7, "MinMaxScaler"), (8.7, "LinearRegression")]
    for x0, t in steps:
        ax.add_patch(FancyBboxPatch((x0, .35), 1.7, .8, boxstyle="round,pad=.04", fc="white", ec="#2d3748"))
        ax.text(x0 + .85, .75, t, ha="center", va="center", fontsize=9.5)
    for x0 in [4.4, 6.4, 8.4]:
        ax.annotate("", xy=(x0 + .3, .75), xytext=(x0 + .02, .75), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(2.65, .85), xytext=(1.85, .85), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.add_patch(FancyBboxPatch((11.1, .35), 2, 1, boxstyle="round,pad=.05", fc="#c6f6d5", ec="#2d3748"))
    ax.text(12.1, .85, "честная оценка\nна test", ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(11.05, .85), xytext=(10.65, .85), arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.text(6.5, -.42, "при fit каждый шаг вычисляет свои параметры (средние, категории, min/max, веса) ТОЛЬКО по train;\n"
                       "при predict — лишь применяет их к test", ha="center", fontsize=9.5, color=GRAY)
    ax.set_xlim(-.2, 13.4); ax.set_ylim(-.75, 1.9)
    save(fig, "13_pipeline_leakage.png")


if __name__ == "__main__":
    coin_likelihood()
    regression_as_distribution()
    normal_vs_laplace()
    mle_to_loss()
    four_situations()
    outlier_vs_cluster()
    huber_constant()
    r2_picture()
    r2_vs_features()
    asymmetric_metrics()
    cars()
    feature_engineering_toy()
    pipeline_diagram()

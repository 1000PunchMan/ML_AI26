"""Графики для расширенного конспекта лекции 2.

Запуск из корня проекта:
    .venv/bin/python my_smth/lectures/scripts/lecture_02_plots.py
Данные: my_smth/lectures/data/CustomerChurn.csv (Telco churn, тот же, что в AdvancedEDA_Pro.ipynb)
Картинки: my_smth/lectures/img/lecture_02/
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "img" / "lecture_02"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096", "#6b46c1"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 11,
})
rng = np.random.default_rng(7)


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("saved", name)


def load_churn():
    df = pd.read_csv(ROOT / "data" / "CustomerChurn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", "0"))
    df["y"] = (df["Churn"] == "Yes").astype(int)
    return df


# ====================================================== ЧАСТЬ 1. EDA (churn)
def churn_numeric(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, title in zip(axes[:2], ["tenure", "MonthlyCharges"],
                              ["tenure — сколько месяцев клиент с нами", "MonthlyCharges — платёж в месяц"]):
        for lab, col_, name in [(0, BLUE, "остался"), (1, RED, "ушёл (churn)")]:
            v = df.loc[df.y == lab, col]
            ax.hist(v, bins=30, density=True, alpha=.55, color=col_, label=f"{name}, медиана {v.median():.0f}")
        ax.set(title=title, ylabel="плотность")
        ax.legend(fontsize=9)
    ax = axes[2]
    data = [df.loc[df.y == 0, "TotalCharges"], df.loc[df.y == 1, "TotalCharges"]]
    bp = ax.boxplot(data, tick_labels=["остался", "ушёл"], patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for p, c in zip(bp["boxes"], [BLUE, RED]):
        p.set_facecolor(c); p.set_alpha(.55)
    ax.set_title("TotalCharges — всего заплатил\nушедшие платят больше в месяц, но меньше в сумме!")
    save(fig, "01_churn_numeric.png")


def churn_categorical(df):
    mean = df.y.mean()
    cols = [("Contract", "Тип контракта"), ("InternetService", "Интернет"),
            ("PaymentMethod", "Способ оплаты"), ("gender", "Пол")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    for ax, (c, t) in zip(axes, cols):
        r = df.groupby(c).y.mean().sort_values(ascending=False)
        labels = [s.replace(" (automatic)", "\n(auto)").replace(" ", "\n", 1) if c == "PaymentMethod" else s
                  for s in r.index]
        bars = ax.bar(labels, r.values, color=[RED if v > mean + .03 else BLUE for v in r.values], alpha=.8)
        for b, v in zip(bars, r.values):
            ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.0%}", ha="center", fontsize=9)
        ax.axhline(mean, ls="--", c="k", lw=1.2, label=f"средний отток {mean:.1%}")
        ax.set_title(t)
        ax.tick_params(axis="x", labelsize=8.5)
    axes[0].set_ylabel("доля ушедших")
    axes[0].legend(fontsize=9)
    fig.suptitle("Доля оттока внутри каждой категории vs средняя (пунктир). Пол почти не влияет, контракт — сильно",
                 y=1.03)
    save(fig, "02_churn_categorical.png")


def churn_bins_pairs(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    edges = [-1, 6, 12, 24, 36, 48, 60, 72]
    labels = ["0–6", "7–12", "13–24", "25–36", "37–48", "49–60", "61–72"]
    r = df.groupby(pd.cut(df.tenure, edges, labels=labels), observed=True).y.mean()
    ax = axes[0]
    bars = ax.bar(r.index.astype(str), r.values, color=RED, alpha=.75)
    for b, v in zip(bars, r.values):
        ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.0%}", ha="center")
    ax.set(title="Бинаризация tenure (pd.cut): чёткая монотонная зависимость",
           xlabel="месяцев с компанией", ylabel="доля ушедших")

    ax = axes[1]
    pv = pd.pivot_table(df, index="Contract", columns="InternetService", values="y")
    im = ax.imshow(pv.values, cmap="Reds", vmin=0, vmax=.6)
    ax.set_xticks(range(pv.shape[1]), pv.columns)
    ax.set_yticks(range(pv.shape[0]), pv.index)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax.text(j, i, f"{pv.values[i, j]:.0%}", ha="center", va="center",
                    color="white" if pv.values[i, j] > .4 else "black", fontweight="bold")
    ax.grid(False)
    ax.set_title("Двухфакторный анализ: отток по паре признаков\n"
                 "Month-to-month + Fiber = 55% → кандидат в новый признак")
    fig.colorbar(im, ax=ax, fraction=.046)
    save(fig, "03_tenure_bins_pairs.png")


def anscombe():
    x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], float)
    ys = [np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
          np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
          np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])]
    x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], float)
    y4 = np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])
    sets = [(x, ys[0]), (x, ys[1]), (x, ys[2]), (x4, y4)]
    notes = ["похоже на линию", "парабола!", "линия + 1 выброс", "вертикаль + 1 выброс"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    for i, (ax, (a, b)) in enumerate(zip(axes, sets)):
        ax.scatter(a, b, c=BLUE, s=45, zorder=3)
        w1, w0 = np.polyfit(a, b, 1)
        xs = np.linspace(3, 20, 10)
        ax.plot(xs, w0 + w1 * xs, c=ORANGE, lw=2)
        r = np.corrcoef(a, b)[0, 1]
        ax.set_title(f"{['I', 'II', 'III', 'IV'][i]}: r = {r:.2f}\n{notes[i]}")
        ax.set_xlim(2, 20)
    fig.suptitle("Квартет Анскомба: одинаковые среднее, дисперсия, корреляция и прямая — а данные совсем разные",
                 y=1.05)
    save(fig, "04_anscombe.png")


def pearson_vs_spearman():
    n = 150
    x = rng.uniform(0, 5, n)
    cases = [(x, 2 * x + rng.normal(0, 1, n), "линейная"),
             (x, np.exp(1.1 * x) + rng.normal(0, 3, n), "монотонная, но не линейная"),
             (x - 2.5, (x - 2.5) ** 2 + rng.normal(0, .4, n), "немонотонная (парабола)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, (a, b, t) in zip(axes, cases):
        ax.scatter(a, b, c=BLUE, s=16, alpha=.7)
        r = np.corrcoef(a, b)[0, 1]
        rho = spearmanr(a, b).statistic
        ax.set_title(f"{t}\nПирсон r = {r:.2f}   |   Спирмен ρ = {rho:.2f}")
    axes[2].text(.5, .85, "оба ≈ 0, хотя связь сильная!", transform=axes[2].transAxes,
                 ha="center", color=RED)
    fig.suptitle("Пирсон ловит линейную связь, Спирмен — монотонную. Немонотонную не ловит никто из них", y=1.04)
    save(fig, "05_pearson_spearman.png")


def chi2_table():
    O = np.array([[40, 30], [32, 48]], float)
    n = O.sum()
    E = O.sum(1, keepdims=True) @ O.sum(0, keepdims=True) / n
    chi2 = ((O - E) ** 2 / E).sum()
    V = np.sqrt(chi2 / (n * (min(O.shape) - 1)))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    rows, cols = ["курит", "не курит"], ["давление есть", "давления нет"]
    for ax, M, t, cmap in [(axes[0], O, "Наблюдаемые O (observed)", "Blues"),
                           (axes[1], E, "Ожидаемые E при независимости\nE = (итог строки × итог столбца) / n",
                            "Greens")]:
        ax.imshow(M, cmap=cmap, vmin=0, vmax=60)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1], cols); ax.set_yticks([0, 1], rows)
        ax.grid(False); ax.set_title(t)
    ax = axes[2]
    ax.axis("off")
    terms = (O - E) ** 2 / E
    txt = "χ² = Σ (O − E)² / E\n\n" + "\n".join(
        f"({O[i, j]:.0f} − {E[i, j]:.1f})² / {E[i, j]:.1f} = {terms[i, j]:.3f}" for i in range(2) for j in range(2))
    txt += f"\n\nχ² ≈ {chi2:.2f}   (n = {n:.0f})\nV Крамера = √(χ² / (n·(min(r,c)−1))) ≈ {V:.2f}"
    ax.text(0, .5, txt, va="center", fontsize=11, family="DejaVu Sans Mono")
    save(fig, "06_chi2.png")


def phik_matrix(df):
    import phik  # noqa: F401
    cols = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService",
            "PaymentMethod", "SeniorCitizen", "gender", "Churn"]
    M = df[cols].phik_matrix(interval_cols=["tenure", "MonthlyCharges", "TotalCharges"])
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(M.values, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white" if M.values[i, j] > .6 else "black")
    ax.grid(False)
    ax.set_title("φK (phik) — одна шкала 0…1 для ЛЮБЫХ пар признаков\n(числовые + категориальные)")
    fig.colorbar(im, ax=ax, fraction=.046)
    save(fig, "07_phik.png")


def causation():
    months = np.arange(1, 13)
    temp = 22 + 9 * np.sin((months - 4) / 12 * 2 * np.pi)
    ice = 50 + 9 * (temp - 13) + rng.normal(0, 8, 12)
    sharks = 2 + .45 * (temp - 13) + rng.normal(0, .6, 12)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.3))
    ax = axes[0]
    ax.plot(months, ice, "-o", c=ORANGE, label="продажи мороженого")
    ax2 = ax.twinx()
    ax2.plot(months, sharks, "-s", c=BLUE, label="нападения акул")
    ax.set(xlabel="месяц", ylabel="мороженое, тыс. шт.", title="Две величины растут и падают вместе…")
    ax2.set_ylabel("нападения акул")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax = axes[1]
    ax.scatter(ice, sharks, c=PURPLE, s=50)
    r = spearmanr(ice, sharks).statistic
    ax.set(xlabel="мороженое", ylabel="акулы",
           title=f"…корреляция Спирмена ρ = {r:.2f}\nно причина общая — жара (третий фактор)")
    save(fig, "08_causation.png")


# ====================================================== ЧАСТЬ 2. Линейная регрессия
def losses():
    z = np.linspace(-3, 3, 400)  # z = y − a: >0 недопрогноз, <0 перепрогноз
    d = 1.0
    huber = np.where(np.abs(z) <= d, .5 * z ** 2, d * (np.abs(z) - .5 * d))

    def pinball(z, t):
        return np.where(z >= 0, t * z, (t - 1) * z)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    ax = axes[0]
    ax.plot(z, z ** 2, c=RED, lw=2.5, label="MSE: $z^2$")
    ax.plot(z, np.abs(z), c=BLUE, lw=2.5, label="MAE: $|z|$")
    ax.plot(z, huber, c=GREEN, lw=2.5, ls="--", label="Huber (δ=1): квадрат в центре, модуль по краям")
    ax.set_ylim(0, 5)
    ax.annotate("у MAE излом в 0:\nнет производной", xy=(0.05, 0.05), xytext=(0.5, 2.3),
                arrowprops=dict(arrowstyle="->", color=BLUE), fontsize=9, color=BLUE)
    ax.set(title="Симметричные функции потерь", xlabel="ошибка $z = y - a(x)$", ylabel="штраф")
    ax.legend(fontsize=8.5, loc="upper center")

    ax = axes[1]
    for t, c in [(.5, BLUE), (.9, RED), (.1, GREEN)]:
        ax.plot(z, pinball(z, t), c=c, lw=2.5, label=f"τ = {t}" + (" (= MAE/2)" if t == .5 else ""))
    ax.axvline(0, c=GRAY, lw=.8)
    ax.text(.9, 1.7, "недопрогноз\n(y > a)", ha="center", color=GRAY)
    ax.text(-.9, 1.7, "перепрогноз\n(y < a)", ha="center", color=GRAY)
    ax.text(1.9, 2.25, "τ=0.9: недопрогноз\nштрафуется в 9 раз\nсильнее → прогноз\nсдвигается ВВЕРХ",
            color=RED, fontsize=9, ha="center")
    ax.set_ylim(0, 3)
    ax.set(title="Квантильная (pinball) функция потерь — несимметричная",
           xlabel="ошибка $z = y - a(x)$", ylabel="штраф")
    ax.legend(loc="upper center", fontsize=9)
    save(fig, "09_losses.png")


def constant_prediction():
    y = np.array([1.5, 2.0, 2.0, 2.5, 10.0])
    a = np.linspace(0, 11, 600)
    mse = ((y[None, :] - a[:, None]) ** 2).mean(1)
    mae = np.abs(y[None, :] - a[:, None]).mean(1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    ax = axes[0]
    ax.scatter(y, np.zeros_like(y), s=90, c=[BLUE] * 4 + [RED], zorder=3)
    ax.axvline(y.mean(), c=RED, lw=2, label=f"среднее = {y.mean():.1f}  (оптимум MSE)")
    ax.axvline(np.median(y), c=GREEN, lw=2, ls="--", label=f"медиана = {np.median(y):.1f}  (оптимум MAE)")
    ax.annotate("опечатка: 10 вместо 1.0?", xy=(10, 0), xytext=(6, .03), arrowprops=dict(arrowstyle="->"))
    ax.set_yticks([])
    ax.set(title="5 квартир, цены в млн ₽", xlabel="цена", ylim=(-.05, .06))
    ax.legend(fontsize=9, loc="upper left")
    for ax, L, name, opt, col in [(axes[1], mse, "MSE(a)", y.mean(), RED),
                                  (axes[2], mae, "MAE(a)", np.median(y), GREEN)]:
        ax.plot(a, L, c=col, lw=2.5)
        ax.axvline(opt, c=col, ls=":", lw=1.5)
        ax.scatter([opt], [L[np.argmin(np.abs(a - opt))]], c=col, s=80, zorder=3)
        ax.set(title=f"{name} при константном прогнозе a\nминимум в a = {opt:.1f}", xlabel="константа a",
               ylabel=name)
    axes[1].text(6.5, mse.min() + 25, "гладкая парабола", color=GRAY)
    axes[2].text(5.5, mae.min() + 1.3, "ломаная из\nкусков прямых", color=GRAY)
    save(fig, "10_constant_prediction.png")


def outliers_fit():
    x = rng.uniform(0, 10, 50)
    y = 1.5 * x + 2 + rng.normal(0, 1.2, 50)
    xo = np.array([8.5, 9, 9.5, 9.8])
    yo = np.array([-8, -10, -7, -9])
    X = np.concatenate([x, xo])[:, None]
    Y = np.concatenate([y, yo])
    xs = np.linspace(0, 10, 50)[:, None]
    models = [("MSE (LinearRegression)", LinearRegression(), RED, "-"),
              ("MAE (QuantileRegressor τ=0.5)", QuantileRegressor(quantile=.5, alpha=0), BLUE, "-"),
              ("Huber (HuberRegressor)", HuberRegressor(), GREEN, "--")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, y, c=GRAY, alpha=.7, label="обычные точки")
    ax.scatter(xo, yo, c=RED, marker="X", s=120, label="4 выброса")
    for name, m, c, ls in models:
        m.fit(X, Y)
        ax.plot(xs, m.predict(xs), c=c, lw=2.5, ls=ls, label=f"{name}: наклон {m.coef_[0]:.2f}")
    ax.plot(xs, 1.5 * xs + 2, c="k", lw=1, ls=":", label="истинная зависимость: наклон 1.50")
    ax.set(title="MSE «перетягивается» к выбросам, MAE и Huber — устойчивы", xlabel="x", ylabel="y")
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "11_outliers_fit.png")


def quantile_regression():
    x = rng.uniform(0, 10, 400)
    y = 10 + 2 * x + rng.normal(0, .4 + .45 * x, 400)  # разброс растёт с x
    xs = np.linspace(0, 10, 50)[:, None]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, y, s=10, c=GRAY, alpha=.5)
    preds = {}
    for t, c in [(.1, GREEN), (.5, BLUE), (.9, RED)]:
        m = QuantileRegressor(quantile=t, alpha=0).fit(x[:, None], y)
        preds[t] = m.predict(xs)
        cover = (y <= m.predict(x[:, None])).mean()
        ax.plot(xs, preds[t], c=c, lw=2.5, label=f"τ = {t}: ниже линии {cover:.0%} точек")
    ax.fill_between(xs[:, 0], preds[.1], preds[.9], color=ORANGE, alpha=.15,
                    label="интервал 10%…90% → покрывает ~80% точек")
    ax.set(title="Квантильная регрессия: прогноз не числом, а интервалом", xlabel="x", ylabel="y")
    ax.legend(fontsize=9, loc="upper left")
    save(fig, "12_quantile_regression.png")


def nonmonotonic():
    d = rng.uniform(0, 100, 300)
    price = 12 - 4 * np.exp(-d / 4) + 3 * np.exp(-((d - 25) / 15) ** 2) - 0.07 * d + rng.normal(0, .6, 300)
    xs = np.linspace(0, 100, 300)
    lin = LinearRegression().fit(d[:, None], price)
    edges = [0, 10, 30, 50, 100]
    def onehot(v):
        idx = np.clip(np.digitize(v, edges[1:-1]), 0, len(edges) - 2)
        return np.eye(len(edges) - 1)[idx][:, 1:]  # drop first → нет ловушки
    binm = LinearRegression().fit(onehot(d), price)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.scatter(d, price, s=12, c=GRAY, alpha=.6)
    ax.plot(xs, lin.predict(xs[:, None]), c=RED, lw=2.5, label="линрег на сыром признаке: одна прямая")
    ax.plot(xs, binm.predict(onehot(xs)), c=GREEN, lw=3, label="линрег на бинах 0–10 / 10–30 / 30–50 / 50+ (one-hot)")
    for e in edges[1:-1]:
        ax.axvline(e, c=GREEN, ls=":", lw=1)
    ax.annotate("у МКАД шумно\nи грязно", xy=(2, 8.3), xytext=(8, 6.4), arrowprops=dict(arrowstyle="->"),
                fontsize=9)
    ax.annotate("лучшее место", xy=(24, 12.2), xytext=(40, 13.3), arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.set(title="Немонотонная зависимость: цена vs расстояние от МКАД",
           xlabel="км от МКАД", ylabel="цена, млн ₽")
    ax.legend(fontsize=9, loc="lower left")
    save(fig, "13_nonmonotonic.png")


def multicollinearity():
    n = 60
    def weights(corr, reps=300):
        W = []
        for _ in range(reps):
            x1 = rng.normal(0, 1, n)
            x2 = corr * x1 + np.sqrt(1 - corr ** 2) * rng.normal(0, 1, n)
            y = 1 * x1 + 1 * x2 + rng.normal(0, 1, n)
            X = np.c_[np.ones(n), x1, x2]
            W.append(np.linalg.solve(X.T @ X, X.T @ y)[1:])
        X = np.c_[np.ones(n), x1, x2]
        return np.array(W), np.linalg.cond(X.T @ X)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for ax, corr, t in [(axes[0], 0.0, "признаки независимы"), (axes[1], 0.99, "corr(x1, x2) = 0.99")]:
        W, cond = weights(corr)
        ax.scatter(W[:, 0], W[:, 1], s=10, alpha=.5, c=BLUE if corr == 0 else RED)
        ax.scatter([1], [1], marker="*", s=250, c="k", zorder=3, label="истина: w1 = w2 = 1")
        ax.set(title=f"{t}\nчисло обусловленности XᵀX ≈ {cond:.0f}", xlabel="найденный $w_1$",
               ylabel="найденный $w_2$")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("300 разных выборок → веса по формуле $(X^TX)^{-1}X^Ty$. "
                 "При почти линейной зависимости веса «скачут», хотя их сумма ≈ 2", y=1.03)
    save(fig, "14_multicollinearity.png")


if __name__ == "__main__":
    df = load_churn()
    churn_numeric(df)
    churn_categorical(df)
    churn_bins_pairs(df)
    anscombe()
    pearson_vs_spearman()
    chi2_table()
    phik_matrix(df)
    causation()
    losses()
    constant_prediction()
    outliers_fit()
    quantile_regression()
    nonmonotonic()
    multicollinearity()

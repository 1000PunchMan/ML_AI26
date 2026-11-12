"""Графики для расширенного конспекта лекции 1.

Запуск из корня проекта:
    .venv/bin/python my_smth/lectures/scripts/lecture_01_plots.py
Картинки сохраняются в my_smth/lectures/img/lecture_01/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent.parent / "img" / "lecture_01"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GRAY = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#718096"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 11,
})
rng = np.random.default_rng(42)


def save(fig, name):
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- 1. Типы задач
def task_types():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Классификация
    ax = axes[0]
    a = rng.normal([30, 40], [6, 10], size=(60, 2))
    b = rng.normal([50, 90], [8, 15], size=(60, 2))
    ax.scatter(*a.T, c=RED, label="не вернул", alpha=.75)
    ax.scatter(*b.T, c=GREEN, label="вернул", alpha=.75)
    xs = np.linspace(15, 75, 10)
    ax.plot(xs, 125 - 1.4 * xs, "--", c="k", lw=1.5, label="граница классов")
    ax.set(title="Классификация\nответ — один из классов", xlabel="возраст", ylabel="доход, тыс.")
    ax.set_ylim(0, 140)
    ax.legend(loc="upper left", fontsize=9)

    # Регрессия
    ax = axes[1]
    x = rng.uniform(30, 150, 80)
    y = 0.12 * x + 1.5 + rng.normal(0, 2, 80)
    ax.scatter(x, y, c=BLUE, alpha=.7)
    xs = np.linspace(30, 150, 10)
    ax.plot(xs, 0.12 * xs + 1.5, c=ORANGE, lw=2.5, label="прогноз модели")
    ax.set(title="Регрессия\nответ — любое число", xlabel="площадь, м²", ylabel="цена, млн ₽")
    ax.legend(loc="upper left", fontsize=9)

    # Кластеризация
    ax = axes[2]
    centers = [(2, 2), (7, 3), (4.5, 7)]
    for c in centers:
        p = rng.normal(c, 0.8, size=(40, 2))
        ax.scatter(*p.T, c=GRAY, alpha=.7)
    for c, col in zip(centers, [BLUE, ORANGE, GREEN]):
        ax.add_patch(plt.Circle(c, 1.9, fill=False, ec=col, lw=2, ls="--"))
    ax.set(title="Кластеризация\nответов нет, ищем группы", xlabel="признак 1", ylabel="признак 2")
    ax.set_aspect("equal", adjustable="datalim")
    save(fig, "01_task_types.png")


# --------------------------------------------- 2. Линейная регрессия и остатки
def linreg_residuals():
    x = np.array([35, 48, 55, 62, 70, 81, 90, 104, 118, 130], float)
    y = 0.11 * x + 1.0 + np.array([.9, -1.2, .4, 1.5, -.8, .3, -1.6, 1.1, -.5, .7])
    w1, w0 = np.polyfit(x, y, 1)
    pred = w0 + w1 * x

    fig, ax = plt.subplots(figsize=(8, 5))
    for xi, yi, pi in zip(x, y, pred):
        ax.plot([xi, xi], [yi, pi], c=RED, lw=1.5, alpha=.8)
    ax.scatter(x, y, c=BLUE, s=55, zorder=3, label="реальные дома $(x_i,\\ y_i)$")
    xs = np.linspace(30, 135, 10)
    ax.plot(xs, w0 + w1 * xs, c=ORANGE, lw=2.5,
            label=f"$a(x) = {w0:.2f} + {w1:.3f}\\,x$")
    ax.plot([], [], c=RED, label="ошибка $y_i - a(x_i)$")
    i = 6
    ax.annotate("модель недооценила\nили переоценила\nэтот дом", xy=(x[i], (y[i] + pred[i]) / 2),
                xytext=(x[i] + 8, y[i] - 2.5), arrowprops=dict(arrowstyle="->", color=GRAY),
                fontsize=9, color=GRAY)
    ax.set(title="Линейная регрессия: прямая + ошибки на каждом объекте",
           xlabel="площадь $x$, м²", ylabel="цена $y$, млн ₽")
    ax.legend(loc="upper left")
    save(fig, "02_linreg_residuals.png")
    return x, y


# --------------------------------------------- 3. Ландшафт функции потерь
def loss_landscape(x, y):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (а) MSE как функция одного веса w1 при фиксированном оптимальном w0
    w1_opt, w0_opt = np.polyfit(x, y, 1)
    w1s = np.linspace(w1_opt - 0.08, w1_opt + 0.08, 200)
    mse = [np.mean((y - (w0_opt + w * x)) ** 2) for w in w1s]
    ax = axes[0]
    ax.plot(w1s, mse, c=BLUE, lw=2.5)
    ax.scatter([w1_opt], [min(mse)], c=RED, s=80, zorder=3)
    ax.annotate(f"минимум: $w_1^* \\approx {w1_opt:.3f}$", xy=(w1_opt, min(mse)),
                xytext=(w1_opt + 0.015, max(mse) * 0.45), arrowprops=dict(arrowstyle="->"))
    for w, lab in [(w1_opt - 0.06, "наклон мал →\nошибка большая"),
                   (w1_opt + 0.06, "наклон велик →\nошибка большая")]:
        m = np.mean((y - (w0_opt + w * x)) ** 2)
        ax.scatter([w], [m], c=GRAY, s=40)
        ax.annotate(lab, xy=(w, m), xytext=(w + (-0.006 if w > w1_opt else 0.006), m),
                    ha="right" if w > w1_opt else "left", va="center", fontsize=9, color=GRAY)
    ax.set(title="MSE как функция веса $w_1$\n(обучение = поиск дна)",
           xlabel="$w_1$", ylabel="MSE$(w)$")

    # (б) MSE(w0, w1) — линии уровня
    ax = axes[1]
    W0, W1 = np.meshgrid(np.linspace(w0_opt - 6, w0_opt + 6, 200),
                         np.linspace(w1_opt - 0.06, w1_opt + 0.06, 200))
    Z = ((y[None, None, :] - (W0[..., None] + W1[..., None] * x[None, None, :])) ** 2).mean(-1)
    cs = ax.contourf(W0, W1, np.log(Z), levels=25, cmap="Blues_r")
    ax.contour(W0, W1, np.log(Z), levels=12, colors="white", linewidths=.5, alpha=.6)
    ax.scatter([w0_opt], [w1_opt], c=RED, s=90, marker="*", zorder=3, label="$w^*$ — минимум")
    ax.set(title="MSE$(w_0, w_1)$ — «чаша» сверху\n(темнее = меньше ошибка)",
           xlabel="$w_0$ (сдвиг)", ylabel="$w_1$ (наклон)")
    ax.grid(False)
    ax.legend(loc="upper right")
    save(fig, "03_loss_landscape.png")


# --------------------------------------------- 4. Зачем квадрат
def why_square():
    true = np.array([10, 10])
    models = {"Модель A\n(почти точная)": np.array([9.5, 10.5]),
              "Модель B\n(ужасная)": np.array([0, 20])}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    names = list(models)
    raw = [np.mean(true - p) for p in models.values()]
    absd = [np.mean(np.abs(true - p)) for p in models.values()]
    sq = [np.mean((true - p) ** 2) for p in models.values()]
    for ax, vals, title in zip(axes, [raw, absd, sq],
                               ["Просто среднее $(y-a)$", "Среднее $|y-a|$ (MAE)",
                                "Среднее $(y-a)^2$ (MSE)"]):
        bars = ax.bar(names, vals, color=[GREEN, RED], width=.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:g}", ha="center", fontweight="bold")
        ax.set_title(title)
        ax.set_ylim(0, max(max(vals) * 1.25, 3))
    axes[0].text(0.5, 0.6, "обе модели = 0 ?!\nошибки +10 и −10\nвзаимно уничтожились",
                 transform=axes[0].transAxes, ha="center", color=RED, fontsize=10)
    fig.suptitle("Два дома по 10 млн. A: 9.5 и 10.5; B: 0 и 20", y=1.03)
    save(fig, "04_why_square.png")


# --------------------------------------------- 5. Цикл ML-проекта
def project_cycle():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    steps = [("Сбор и\nхранение данных", "#ebf4ff"), ("EDA и\nочистка", "#ebf8ff"),
             ("Обучение +\nвалидация", "#fefcbf"), ("Препрод\n(A/B-тест)", "#fef5e7"),
             ("Прод\n(сервис, API)", "#f0fff4"), ("Мониторинг", "#fff5f5")]
    w, h, gap = 1.7, 1.0, 0.45
    for i, (t, col) in enumerate(steps):
        x0 = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x0, 0), w, h, boxstyle="round,pad=0.05",
                                    fc=col, ec="#2d3748", lw=1.5 if i != 2 else 3))
        ax.text(x0 + w / 2, h / 2, t, ha="center", va="center", fontsize=11)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x0 + w + gap - 0.05, h / 2), xytext=(x0 + w + 0.05, h / 2),
                        arrowprops=dict(arrowstyle="->", lw=1.8))
    x_last = (len(steps) - 1) * (w + gap) + w / 2
    ax.annotate("", xy=(w / 2, -0.1), xytext=(x_last, -0.1),
                arrowprops=dict(arrowstyle="->", lw=1.8, color=RED,
                                connectionstyle="arc3,rad=-0.12"))
    ax.text(x_last / 2 + w / 4, -1.15, "качество упало / data drift → переобучаем, чиним данные",
            ha="center", color=RED, fontsize=10)
    ax.text(2 * (w + gap) + w / 2, 1.25, "fit / predict —\nлишь этот кубик",
            ha="center", fontsize=10, color=GRAY)
    ax.set_xlim(-0.2, len(steps) * (w + gap))
    ax.set_ylim(-1.4, 1.8)
    save(fig, "05_project_cycle.png")


# --------------------------------------------- 6. Среднее vs медиана при ошибке
def mean_vs_median():
    ages = rng.normal(48, 13, 437).clip(19, 79).round()
    errors = np.array([510, 480, 620, 590, 505])
    dirty = np.concatenate([ages, errors])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.hist(ages, bins=30, color=BLUE, alpha=.7, label="нормальные значения")
    ax.axvline(ages.mean(), c=GREEN, lw=2, label=f"среднее без ошибок = {ages.mean():.1f}")
    ax.axvline(dirty.mean(), c=RED, lw=2, ls="--", label=f"среднее с 5 ошибками = {dirty.mean():.1f}")
    ax.axvline(np.median(dirty), c=ORANGE, lw=2, ls=":",
               label=f"медиана с ошибками = {np.median(dirty):.1f}")
    ax.set(title="Возраст: 437 нормальных значений + 5 опечаток (510, 620…)",
           xlabel="возраст", ylabel="кол-во пациентов")
    ax.legend(fontsize=9)

    ax = axes[1]
    labels = ["среднее", "медиана"]
    shift = [dirty.mean() - ages.mean(), np.median(dirty) - np.median(ages)]
    bars = ax.bar(labels, shift, color=[RED, ORANGE], width=.5)
    for b, v in zip(bars, shift):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"+{v:.1f} лет", ha="center", fontweight="bold")
    ax.set(title="Насколько 5 ошибок\nиз 442 сдвинули оценку", ylabel="сдвиг, лет")
    save(fig, "06_mean_vs_median.png")


# --------------------------------------------- 7. Анатомия боксплота
def boxplot_anatomy():
    data = np.concatenate([rng.normal(50, 10, 300), [84, 87, 15, 12]])
    q1, med, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    lo_w = data[data >= q1 - 1.5 * iqr].min()
    hi_w = data[data <= q3 + 1.5 * iqr].max()

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.boxplot(data, orientation="horizontal", widths=.5, patch_artist=True,
               boxprops=dict(facecolor="#bee3f8"), medianprops=dict(color=ORANGE, lw=2.5),
               flierprops=dict(marker="*", markersize=11, markerfacecolor=RED, markeredgecolor=RED))
    ann = [(q1, "$Q_1$ (25% левее)", 1.55), (med, "медиана", 1.85), (q3, "$Q_3$ (75% левее)", 1.55),
           (lo_w, "левый ус", 1.55), (hi_w, "правый ус", 1.55)]
    for xv, t, yt in ann:
        ax.annotate(t, xy=(xv, 1.27), xytext=(xv, yt), ha="center", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=GRAY))
    ax.annotate("", xy=(q1, 0.62), xytext=(q3, 0.62), arrowprops=dict(arrowstyle="<->"))
    ax.text((q1 + q3) / 2, 0.5, "IQR = $Q_3 - Q_1$", ha="center", fontsize=9)
    ax.text(hi_w + 3, 0.72, "★ за усами (> 1.5·IQR от ящика)\n= НЕ обязательно ошибка!\n"
            "Ошибки ищем здравым смыслом", fontsize=9, color=RED)
    ax.set_yticks([])
    ax.set_ylim(0.3, 2.05)
    ax.set_xlabel("значение признака")
    ax.set_title("Как читать «ящик с усами»")
    save(fig, "07_boxplot_anatomy.png")


# --------------------------------------------- 8. Масштабирование и веса (diabetes)
def scaling_weights():
    X, y = load_diabetes(return_X_y=True, as_frame=True, scaled=False)
    raw = LinearRegression().fit(X, y)
    Xs = StandardScaler().fit_transform(X)
    sc = LinearRegression().fit(Xs, y)
    r2_raw, r2_sc = r2_score(y, raw.predict(X)), r2_score(y, sc.predict(Xs))

    order = np.argsort(np.abs(sc.coef_))
    names = X.columns[order]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(names, raw.coef_[order], color=GRAY)
    axes[0].set(title=f"Веса ДО масштабирования (R² = {r2_raw:.3f})\n"
                      "s5 ≈ 68, sex ≈ −23, s1 ≈ −1 — сравнивать нельзя:\nв весе «сидит» масштаб признака",
                xlabel="вес $w_j$")
    cols = [GREEN if c > 0 else RED for c in sc.coef_[order]]
    axes[1].barh(names, sc.coef_[order], color=cols)
    axes[1].set(title=f"Веса ПОСЛЕ StandardScaler (R² = {r2_sc:.3f} — то же!)\n"
                      "теперь |w| = важность признака",
                xlabel="вес $w_j$  (зелёный: ↑признак → ↑y, красный: ↑признак → ↓y)")
    save(fig, "08_scaling_weights.png")
    return X, y


# --------------------------------------------- 9. Корреляции
def correlations(X, y):
    df = X.copy()
    df["target"] = y
    C = df.corr().values
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(df.columns)), df.columns, rotation=45)
    ax.set_yticks(range(len(df.columns)), df.columns)
    for i in range(len(C)):
        for j in range(len(C)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(C[i, j]) > .6 else "black")
    ax.grid(False)
    ax.set_title("Матрица корреляций diabetes\n(последняя строка — связь признаков с target)")
    fig.colorbar(im, ax=ax, fraction=.046)

    ax = axes[1]
    xq = rng.uniform(-3, 3, 200)
    yq = xq ** 2 + rng.normal(0, .4, 200)
    r = np.corrcoef(xq, yq)[0, 1]
    ax.scatter(xq, yq, c=BLUE, alpha=.6, s=18)
    ax.set(title=f"Ловушка: $y = x^2$, корреляция = {r:.2f}\n"
                 "но зависимость идеальная (просто нелинейная)",
           xlabel="x", ylabel="y")
    save(fig, "09_correlations.png")


# --------------------------------------------- 10. Разброс метрики от разбиения + CV
def split_variance(X, y):
    scores = []
    for seed in range(300):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=seed)
        scores.append(r2_score(yte, LinearRegression().fit(Xtr, ytr).predict(Xte)))
    cv = cross_val_score(LinearRegression(), X, y, cv=5, scoring="r2")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    ax.hist(scores, bins=30, color=BLUE, alpha=.75)
    ax.axvline(np.mean(scores), c=ORANGE, lw=2.5, label=f"среднее = {np.mean(scores):.3f}")
    ax.set(title=f"R² на test при 300 разных random_state\n"
                 f"от {min(scores):.2f} до {max(scores):.2f} — «повезло/не повезло» с разбиением",
           xlabel="R² на test", ylabel="сколько раз")
    ax.legend()

    ax = axes[1]
    k = 5
    for fold in range(k):
        for blk in range(k):
            col = ORANGE if blk == fold else "#bee3f8"
            ax.add_patch(plt.Rectangle((blk, k - 1 - fold), .95, .8, fc=col, ec="white"))
        ax.text(k + 0.15, k - 1 - fold + .4, f"R² = {cv[fold]:.2f}", va="center", fontsize=10)
    ax.text(k + 0.15, -0.6, f"среднее = {cv.mean():.3f}", fontweight="bold", color=ORANGE)
    ax.set_xlim(0, k + 1.6)
    ax.set_ylim(-1, k)
    ax.axis("off")
    ax.set_title("5-fold кросс-валидация на diabetes\nголубой — обучаем, оранжевый — проверяем")
    save(fig, "10_split_variance_cv.png")


# --------------------------------------------- 11. Дубликаты
def duplicates():
    x = rng.uniform(0, 10, 40)
    y = 2 * x + 3 + rng.normal(0, 2, 40)
    bad = (9.0, 4.0)  # неудачный объект
    xd = np.concatenate([x, [bad[0]] * 30])
    yd = np.concatenate([y, [bad[1]] * 30])
    w_clean = np.polyfit(np.append(x, bad[0]), np.append(y, bad[1]), 1)
    w_dup = np.polyfit(xd, yd, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, c=BLUE, alpha=.7, label="обычные объекты")
    ax.scatter(*bad, c=RED, s=160, marker="X", zorder=3, label="один объект, скопирован 30 раз")
    xs = np.linspace(0, 10, 10)
    ax.plot(xs, np.polyval(w_clean, xs), c=GREEN, lw=2.5, label="модель на данных без дублей")
    ax.plot(xs, np.polyval(w_dup, xs), c=RED, lw=2.5, ls="--", label="модель с 30 дублями")
    ax.set(title="Дубликат = объект с увеличенным весом:\nмодель «перетягивается» к нему",
           xlabel="x", ylabel="y")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "11_duplicates.png")


if __name__ == "__main__":
    task_types()
    x, y = linreg_residuals()
    loss_landscape(x, y)
    why_square()
    project_cycle()
    mean_vs_median()
    boxplot_anatomy()
    Xd, yd = scaling_weights()
    correlations(Xd, yd)
    split_variance(Xd, yd)
    duplicates()

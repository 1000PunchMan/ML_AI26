"""Лекция 3, MSLE: почему метрика несимметрична (график 14)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # рисуем сразу в файл, без окна
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "img" / "lecture_03"
BLUE, RED, GREEN, GRAY = "#2563eb", "#dc2626", "#15803d", "#6b7280"

y = 100                               # правильный ответ


def sle(a, y):
    """Одно слагаемое MSLE: квадрат разности логарифмов (log1p(x) = ln(x+1))."""
    return (np.log1p(a) - np.log1p(y)) ** 2


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

# --- Слева: причина. Равные шаги по x дают неравные шаги по ln ---
x = np.linspace(5, 220, 500)
ax1.plot(x, np.log1p(x), color=GRAY, lw=2.5)
y0 = np.log1p(y)
for a, c in [(50, RED), (150, BLUE)]:
    ya = np.log1p(a)
    ax1.plot([a, a], [1.6, ya], ls=":", color=c)                 # проекция на ось x
    ax1.annotate("", xy=(a, ya), xytext=(a, y0),                 # высота = разность логарифмов
                 arrowprops=dict(arrowstyle="<->", color=c, lw=2.5))
    ax1.scatter([a], [ya], color=c, zorder=5, s=55)
ax1.axhline(y0, color=GREEN, lw=1.2, ls="--")
ax1.scatter([y], [y0], color=GREEN, zorder=5, s=70)
ax1.text(102, y0 - .25, "факт 100", color=GREEN, fontsize=11)
ax1.text(45, y0 - .33, "−0.68", color=RED, fontsize=13, ha="right")
ax1.text(155, y0 + .13, "+0.40", color=BLUE, fontsize=13)
ax1.text(50, 1.68, " прогноз 50", color=RED, fontsize=10)
ax1.text(150, 1.68, " прогноз 150", color=BLUE, fontsize=10)
ax1.set(xlabel="прогноз $a$", ylabel=r"$\ln(a+1)$", ylim=(1.6, 5.8),
        title="Причина: логарифм слева круче, чем справа")

# --- Справа: следствие. Штраф MSLE как функция прогноза ---
a = np.linspace(20, 220, 600)
ax2.plot(a, sle(a, y), color=GRAY, lw=2.5)
ax2.axvline(y, color=GREEN, lw=1.2, ls="--")
ax2.text(102, 1.5, "факт 100", color=GREEN, fontsize=11)
for p, c, dx, dy, t in [(50, RED, 6, .10, "50: штраф 0.47"),
                        (150, BLUE, 4, -.10, "150: штраф 0.16"),
                        (200, BLUE, -30, .12, "200: штраф 0.47")]:
    ax2.scatter([p], [sle(p, y)], color=c, zorder=5, s=60)
    ax2.text(p + dx, sle(p, y) + dy, t, color=c, fontsize=11)
# 50 и 200 стоят на одной высоте: в 2 раза меньше и в 2 раза больше
ax2.plot([50, 200], [sle(50, y), sle(200, y)], color=RED, lw=1, ls=":")
ax2.set(xlabel="прогноз $a$", ylabel="штраф (одно слагаемое MSLE)", ylim=(-.05, 1.7),
        title="Следствие: 50 и 150 — разный штраф, 50 и 200 — одинаковый")
for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "14_msle_asymmetry.png", dpi=130)

# --- Числа для конспекта ---
for p in (50, 150, 200):
    print(p, round(np.log1p(p) - np.log1p(y), 4), round(sle(p, y), 4))
print("широкий разброс, ошибка 10%:", sle(2.2e6, 2e6), sle(110e6, 100e6))
print("ошибка 1 млн при 2 и 100 млн:", sle(3e6, 2e6), sle(101e6, 100e6))
print("узкий разброс, ошибка 1 млн при 8,10,12 млн:", [round(sle(v + 1e6, v), 5) for v in (8e6, 10e6, 12e6)])
v = np.array([10, 20, 30, 40, 100])
print("оптимальная константа: MSE", v.mean(), "MSLE", np.expm1(np.log1p(v).mean()))

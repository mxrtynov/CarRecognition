import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data import get_train_set, get_unknown_set, MAX_X1, MAX_X2, FEATURE_MAX
from MinDistanceClassifier import MinDistanceClassifier
from PerceptronClassifier import PerceptronClassifier
from complexity import report as complexity_report


BG = "#0f1218"
BG_PANEL = "#161a22"
BG_CARD = "#1a1f29"
BG_INPUT = "#1f2530"
BORDER = "#262c38"

FG = "#e8ecf4"
FG_DIM = "#8b93a8"
FG_MUTED = "#5a6076"

ACCENT = "#7aa2f7"
ACCENT2 = "#bb9af7"
SUCCESS = "#9ece6a"
DANGER = "#f7768e"
WARN = "#e0af68"

FONT = ("Segoe UI", 10)
FONT_S = ("Segoe UI", 9)
FONT_XS = ("Segoe UI", 8)
FONT_B = ("Segoe UI", 10, "bold")
FONT_H1 = ("Segoe UI", 15, "bold")
FONT_H3 = ("Segoe UI", 10, "bold")
FONT_MONO = ("Consolas", 10)
FONT_VALUE = ("Segoe UI", 26, "bold")


def apply_theme(root):
    """Настройка ttk-стилей."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=BG, foreground=FG, font=FONT)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=FG, font=FONT)

    style.configure("TButton", background=BG_INPUT, foreground=FG,
                    font=FONT_B, borderwidth=0, padding=(14, 9),
                    focusthickness=0, relief="flat")
    style.map("TButton",
              background=[("active", BG_CARD), ("pressed", BG_CARD)])

    style.configure("Accent.TButton", background=ACCENT, foreground="#0f1218",
                    font=FONT_B, borderwidth=0, padding=(14, 9),
                    focusthickness=0, relief="flat")
    style.map("Accent.TButton",
              background=[("active", ACCENT2), ("pressed", ACCENT2)])

    style.configure("Flat.TButton", background=BG_PANEL, foreground=FG_DIM,
                    font=FONT_S, borderwidth=0, padding=(10, 5),
                    focusthickness=0, relief="flat")
    style.map("Flat.TButton",
              background=[("active", BG_CARD)],
              foreground=[("active", FG)])

    style.configure("Horizontal.TScale",
                    background=BG_PANEL, troughcolor=BG_INPUT,
                    borderwidth=0, lightcolor=ACCENT, darkcolor=ACCENT)

    style.configure("TNotebook", background=BG, borderwidth=0,
                    tabmargins=(8, 6, 0, 0))
    style.configure("TNotebook.Tab",
                    background=BG, foreground=FG_MUTED,
                    font=FONT_B, padding=(20, 10), borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", BG_CARD), ("active", BG_CARD)],
              foreground=[("selected", ACCENT), ("active", FG)])


CLASS_NAMES = {1: "Экономный", 2: "Спортивный"}
CLASS_COLORS = {1: ACCENT, 2: DANGER}

# Нормированные значения X3, X4, X5 для «своего автомобиля».
# На графике визуализируются только X1 и X2; остальные зафиксированы.
X3_DEFAULT = 0.35
X4_DEFAULT = 0.55
X5_DEFAULT = 0.55


class LabUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ЛР №7 — Распознавание автомобилей")
        self.root.geometry("1380x840")
        self.root.configure(bg=BG)
        self.root.minsize(1220, 740)

        self.train_set = get_train_set()
        self.unknown_set = get_unknown_set()
        self.min_dist = MinDistanceClassifier().fit(self.train_set)
        self.perceptron = PerceptronClassifier().fit(self.train_set)

        self.x1 = tk.DoubleVar(value=9.0)
        self.x2 = tk.DoubleVar(value=200.0)

        apply_theme(self.root)
        self._build()
        self._update_all()

    # ---------- построение интерфейса ----------

    def _build(self):
        self._build_header()

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        self.right = tk.Frame(body, bg=BG_PANEL, width=380)
        self.right.pack(side="right", fill="y")
        self.right.pack_propagate(False)

        self.left = tk.Frame(body, bg=BG)
        self.left.pack(side="left", fill="both", expand=True)

        self._build_main_notebook()
        self._build_side()

    def _build_header(self):
        header = tk.Frame(self.root, bg=BG, height=72)
        header.pack(fill="x")
        header.pack_propagate(False)

        inner = tk.Frame(header, bg=BG)
        inner.pack(fill="both", expand=True, padx=28, pady=16)

        tk.Label(inner, text="Распознавание автомобилей", bg=BG, fg=FG,
                 font=FONT_H1).pack(side="left")

        tk.Label(inner, text="5 признаков · 2 класса",
                 bg=BG, fg=FG_DIM, font=FONT).pack(
            side="left", padx=(16, 0), pady=(4, 0))

        ttk.Button(inner, text="Сбросить", style="Flat.TButton",
                   command=self._reset).pack(side="right", pady=(4, 0))

    def _build_main_notebook(self):
        self.tabs = ttk.Notebook(self.left)
        self.tabs.pack(fill="both", expand=True,
                       padx=(24, 12), pady=(0, 16))

        self.tab_plot = tk.Frame(self.tabs, bg=BG_CARD)
        self.tab_objects = tk.Frame(self.tabs, bg=BG_CARD)
        self.tab_cx = tk.Frame(self.tabs, bg=BG_CARD)

        self.tabs.add(self.tab_plot, text="Признаковое пространство")
        self.tabs.add(self.tab_objects, text="Объекты")
        self.tabs.add(self.tab_cx, text="Сложность")

        self._build_plot_tab()
        self._build_objects_tab()
        self._build_cx_tab()

    def _build_plot_tab(self):
        head = tk.Frame(self.tab_plot, bg=BG_CARD)
        head.pack(fill="x", padx=22, pady=(16, 4))
        tk.Label(head, text="Признаковое пространство (X1 × X2)",
                 bg=BG_CARD, fg=FG, font=FONT_H3).pack(side="left")
        tk.Label(head, text="клик по полю — задать автомобиль",
                 bg=BG_CARD, fg=FG_DIM, font=FONT_S).pack(side="right")

        plot_wrap = tk.Frame(self.tab_plot, bg=BG_CARD)
        plot_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.fig = Figure(figsize=(7, 5.5), dpi=100, facecolor=BG_CARD)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_wrap)
        self.canvas.get_tk_widget().configure(bg=BG_CARD,
                                              highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

    def _build_objects_tab(self):
        head = tk.Frame(self.tab_objects, bg=BG_CARD)
        head.pack(fill="x", padx=22, pady=(16, 8))
        tk.Label(head, text="Распознавание заданных объектов",
                 bg=BG_CARD, fg=FG, font=FONT_H3).pack(side="left")
        tk.Label(head, text="5 автомобилей · 5 признаков · оба метода",
                 bg=BG_CARD, fg=FG_DIM, font=FONT_S).pack(side="right")

        self.rec_text = tk.Text(self.tab_objects, bg=BG_CARD, fg=FG,
                                font=FONT_MONO, relief="flat",
                                highlightthickness=0, wrap="none",
                                padx=22, pady=0, insertwidth=0,
                                spacing1=2, spacing3=2)
        self.rec_text.pack(fill="both", expand=True, padx=0, pady=(0, 16))
        self.rec_text.configure(state="disabled")

    def _build_cx_tab(self):
        head = tk.Frame(self.tab_cx, bg=BG_CARD)
        head.pack(fill="x", padx=22, pady=(16, 8))
        tk.Label(head, text="Вычислительная сложность",
                 bg=BG_CARD, fg=FG, font=FONT_H3).pack(side="left")

        bar = tk.Frame(self.tab_cx, bg=BG_CARD)
        bar.pack(fill="x", padx=22, pady=(0, 8))
        ttk.Button(bar, text="▶  Замерить", style="Accent.TButton",
                   command=self._run_cx).pack(side="left")
        ttk.Button(bar, text="▶  Масштабирование", style="Accent.TButton",
                   command=self._run_scaling).pack(side="left", padx=(8, 0))

        self.cx_text = tk.Text(self.tab_cx, bg=BG_CARD, fg=FG,
                               font=FONT_MONO, relief="flat",
                               highlightthickness=0, wrap="word",
                               padx=22, pady=0, insertwidth=0,
                               spacing1=2, spacing3=2)
        self.cx_text.pack(fill="both", expand=True, padx=0, pady=(0, 16))
        self.cx_text.configure(state="disabled")

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor(BG_CARD)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=FG_MUTED, labelsize=9, length=0)
        ax.set_xlabel("Расход топлива, л/100 км (X1)", color=FG_DIM,
                      fontsize=10, labelpad=8)
        ax.set_ylabel("Мощность, л.с. (X2)", color=FG_DIM,
                      fontsize=10, labelpad=8)
        ax.grid(True, color=BORDER, linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    def _build_side(self):
        pad = tk.Frame(self.right, bg=BG_PANEL)
        pad.pack(fill="both", expand=True, padx=24, pady=24)

        tk.Label(pad, text="СВОЙ АВТОМОБИЛЬ", bg=BG_PANEL, fg=FG_MUTED,
                 font=FONT_XS).pack(anchor="w")
        tk.Frame(pad, bg=BG_PANEL, height=10).pack()

        self._slider(pad, "Расход топлива (X1)", "л/100 км",
                     self.x1, 0, MAX_X1, 0.1, "{:.1f}")
        tk.Frame(pad, bg=BG_PANEL, height=14).pack()
        self._slider(pad, "Мощность (X2)", "л.с.",
                     self.x2, 0, MAX_X2, 5, "{:.0f}")

        tk.Frame(pad, bg=BG_PANEL, height=10).pack()
        tk.Label(pad,
                 text=f"X3 = {X3_DEFAULT:.2f}   X4 = {X4_DEFAULT:.2f}   X5 = {X5_DEFAULT:.2f}",
                 bg=BG_PANEL, fg=FG_MUTED, font=FONT_XS).pack(anchor="w")

        tk.Frame(pad, bg=BG_PANEL, height=24).pack()

        tk.Label(pad, text="ПРЕДСКАЗАНИЕ", bg=BG_PANEL, fg=FG_MUTED,
                 font=FONT_XS).pack(anchor="w")

        self.verdict = tk.Label(pad, text="—", bg=BG_PANEL, fg=ACCENT,
                                font=FONT_VALUE, anchor="w")
        self.verdict.pack(anchor="w", pady=(4, 0))

        self.verdict_sub = tk.Label(pad, text="", bg=BG_PANEL, fg=FG_DIM,
                                    font=FONT_S, anchor="w", justify="left")
        self.verdict_sub.pack(anchor="w", pady=(2, 0))

        tk.Frame(pad, bg=BG_PANEL, height=28).pack()

        tk.Label(pad, text="РЕШАЮЩИЕ ФУНКЦИИ", bg=BG_PANEL, fg=FG_MUTED,
                 font=FONT_XS).pack(anchor="w")

        self.md_line = tk.Label(pad, text="", bg=BG_PANEL, fg=FG,
                                font=FONT_MONO, anchor="w", justify="left")
        self.md_line.pack(anchor="w", pady=(8, 0))

        self.pc_line = tk.Label(pad, text="", bg=BG_PANEL, fg=FG,
                                font=FONT_MONO, anchor="w", justify="left")
        self.pc_line.pack(anchor="w", pady=(12, 0))

        bottom = tk.Frame(pad, bg=BG_PANEL)
        bottom.pack(side="bottom", fill="x")
        self.models_info = tk.Label(bottom, text="", bg=BG_PANEL,
                                    fg=FG_MUTED, font=FONT_XS,
                                    anchor="w", justify="left")
        self.models_info.pack(anchor="w")

    def _slider(self, parent, label, unit, var, frm, to, step, fmt):
        block = tk.Frame(parent, bg=BG_PANEL)
        block.pack(fill="x")

        top = tk.Frame(block, bg=BG_PANEL)
        top.pack(fill="x")
        tk.Label(top, text=label, bg=BG_PANEL, fg=FG_DIM, font=FONT_S,
                 anchor="w").pack(side="left")
        tk.Label(top, text=unit, bg=BG_PANEL, fg=FG_MUTED, font=FONT_XS,
                 anchor="w").pack(side="left", padx=(6, 0), pady=(2, 0))

        value_lbl = tk.Label(top, text=fmt.format(var.get()), bg=BG_PANEL,
                             fg=FG, font=FONT_B)
        value_lbl.pack(side="right")

        def on_change(v):
            value_lbl.configure(text=fmt.format(float(v)))
            self._update_all()

        ttk.Scale(block, from_=frm, to=to, orient="horizontal",
                  variable=var, command=on_change,
                  style="Horizontal.TScale").pack(fill="x", pady=(6, 0))

    # ---------- события ----------

    def _on_plot_click(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        x1 = max(0.0, min(MAX_X1, event.xdata * MAX_X1))
        x2 = max(0.0, min(MAX_X2, event.ydata * MAX_X2))
        self.x1.set(round(x1, 1))
        self.x2.set(round(x2, 0))
        self._update_all()

    def _reset(self):
        self.x1.set(9.0)
        self.x2.set(200.0)
        self._update_all()

    # ---------- обновление ----------

    def _update_all(self):
        self._refresh_plot()
        self._refresh_recognition()
        self._refresh_custom()
        self._refresh_models_info()

    def _current_features(self):
        """Нормированный вектор из 5 признаков для «своего автомобиля»."""
        x1n = self.x1.get() / MAX_X1
        x2n = self.x2.get() / MAX_X2
        return (x1n, x2n, X3_DEFAULT, X4_DEFAULT, X5_DEFAULT)

    def _refresh_plot(self):
        ax = self.ax
        ax.clear()
        self._style_axes()

        # Точки обучающего множества — проекция на (X1, X2).
        shown = set()
        for sample in self.train_set:
            *features, c = sample
            x1, x2 = features[0], features[1]
            label = CLASS_NAMES[c] if c not in shown else None
            shown.add(c)
            ax.scatter(x1, x2, c=CLASS_COLORS[c], s=70, marker="o",
                       edgecolors="white", linewidths=0.8,
                       label=label, zorder=5)

        # Прототипы классов — тоже проекция на (X1, X2).
        for i, proto in enumerate(self.min_dist.prototypes):
            p1, p2 = proto[0], proto[1]
            ax.scatter(p1, p2, c=WARN, s=220, marker="*",
                       edgecolors=BG_CARD, linewidths=1.0, zorder=6)
            ax.annotate(f"P{i+1}", (p1, p2), textcoords="offset points",
                        xytext=(10, 10), color=WARN, fontsize=9,
                        fontweight="bold")

        # Границы — срезы 5-мерных гиперплоскостей при фиксированных X3..X5.
        self._draw_slice(ax)
        self._draw_md_slice(ax)

        # Объекты для распознавания.
        for sample in self.unknown_set:
            x1, x2 = sample[1], sample[2]
            ax.scatter(x1, x2, facecolors="none", edgecolors=FG_MUTED,
                       s=110, marker="s", linewidths=1.6, zorder=7)

        # «Свой автомобиль».
        features = self._current_features()
        md_class = self.min_dist.predict(*features)
        ax.scatter(features[0], features[1], c=CLASS_COLORS[md_class],
                   s=280, marker="D",
                   edgecolors="white", linewidths=1.6, zorder=9,
                   label="Свой автомобиль")

        ax.legend(loc="upper left", facecolor=BG_CARD, edgecolor="none",
                  labelcolor=FG, fontsize=9, framealpha=0.0)

        self.canvas.draw_idle()

    def _draw_slice(self, ax):
        """Срез гиперплоскости перцептрона при X3..Xd = фиксированные."""
        w = self.perceptron.w
        if len(w) < 4:
            return
        fixed = (X3_DEFAULT, X4_DEFAULT, X5_DEFAULT)
        w1, w2 = w[0], w[1]
        # Вклад зафиксированных признаков + смещение.
        C = sum(w[2 + j] * fixed[j] for j in range(len(fixed))) + w[-1]
        if abs(w2) < 1e-9:
            return
        x1a, x1b = 0.0, 1.0
        x2a = -(w1 * x1a + C) / w2
        x2b = -(w1 * x1b + C) / w2
        ax.plot([x1a, x1b], [x2a, x2b], color=SUCCESS, linewidth=1.8,
                linestyle="--", label="Граница (восприятие)", zorder=4)

    def _draw_md_slice(self, ax):
        """Срез границы метода минимального расстояния при X3..Xd = фиксированные."""
        if len(self.min_dist.prototypes) != 2:
            return
        p = self.min_dist.prototypes[0]
        q = self.min_dist.prototypes[1]
        if len(p) < 3:
            return
        fixed = (X3_DEFAULT, X4_DEFAULT, X5_DEFAULT)
        A = 2 * (p[0] - q[0])
        B = 2 * (p[1] - q[1])
        # Вклад зафиксированных признаков.
        C = 2 * sum((p[2 + j] - q[2 + j]) * fixed[j]
                    for j in range(len(fixed)))
        C += sum(qi * qi - pi * pi for pi, qi in zip(p, q))
        if abs(B) < 1e-9:
            return
        x1a, x1b = 0.0, 1.0
        x2a = -(A * x1a + C) / B
        x2b = -(A * x1b + C) / B
        ax.plot([x1a, x1b], [x2a, x2b], color=ACCENT, linewidth=1.6,
                linestyle=":", label="Граница (мин. расст.)", zorder=3)

    def _refresh_custom(self):
        features = self._current_features()

        md_vals = self.min_dist.decision_values(*features)
        best = max(range(len(md_vals)), key=lambda i: md_vals[i])
        md_class = self.min_dist.classes[best]

        pc_val = self.perceptron.decision_value(*features)
        pc_class = self.perceptron.predict(*features)

        self.verdict.configure(text=CLASS_NAMES[md_class],
                               fg=CLASS_COLORS[md_class])

        agree = "оба метода согласны" if md_class == pc_class else \
                f"расхождение · восприятие: {CLASS_NAMES[pc_class]}"
        self.verdict_sub.configure(text=agree)

        self.md_line.configure(
            text=f"мин. расстояние\n"
                 f"  D1 = {md_vals[0]:+.4f}\n"
                 f"  D2 = {md_vals[1]:+.4f}")

        self.pc_line.configure(
            text=f"восприятие\n"
                 f"  D12 = {pc_val:+.4f}")

    def _refresh_recognition(self):
        lines = []
        for sample in self.unknown_set:
            name = sample[0]
            features = sample[1:]

            md_vals = self.min_dist.decision_values(*features)
            best = max(range(len(md_vals)), key=lambda i: md_vals[i])
            md_class = self.min_dist.classes[best]
            pc_class = self.perceptron.predict(*features)
            pc_val = self.perceptron.decision_value(*features)

            x1, x2, x3, x4, x5 = features
            lines.append(f"■  {name}")
            lines.append(f"     X1 = {x1:.3f}   X2 = {x2:.3f}   "
                         f"X3 = {x3:.3f}   X4 = {x4:.3f}   X5 = {x5:.3f}")
            lines.append(f"     мин. расстояние     "
                         f"D1 = {md_vals[0]:+.3f}   D2 = {md_vals[1]:+.3f}"
                         f"   →   {CLASS_NAMES[md_class]}")
            lines.append(f"     восприятие          "
                         f"D12 = {pc_val:+.3f}"
                         f"{'':>15}→   {CLASS_NAMES[pc_class]}")
            mark = "✓  согласовано" if md_class == pc_class else \
                   "✗  расходятся"
            lines.append(f"     {mark}")
            lines.append("")

        self.rec_text.configure(state="normal")
        self.rec_text.delete("1.0", "end")
        self.rec_text.insert("1.0", "\n".join(lines).rstrip())
        self.rec_text.configure(state="disabled")

    def _refresh_models_info(self):
        lines = ["прототипы (X1; X2; X3; X4; X5)"]
        for i, (proto, c) in enumerate(
                zip(self.min_dist.prototypes, self.min_dist.classes)):
            vals = "; ".join(f"{p:.3f}" for p in proto)
            lines.append(f"  P{i+1}  {CLASS_NAMES[c]:<10}  ({vals})")
        lines.append("")
        lines.append("восприятие")
        w = self.perceptron.w
        w_terms = "   ".join(f"W{j+1}={wi:+.3f}" for j, wi in enumerate(w[:-1]))
        lines.append(f"  {w_terms}   W0={w[-1]:+.3f}")
        lines.append(f"  эпох {self.perceptron._epochs}, "
                     f"коррекций {self.perceptron._updates}, "
                     f"{'сошлось' if self.perceptron._converged else 'не сошлось'}")
        self.models_info.configure(text="\n".join(lines))

    def _run_cx(self):
        text = complexity_report(self.train_set)
        self.cx_text.configure(state="normal")
        self.cx_text.delete("1.0", "end")
        self.cx_text.insert("1.0", text)
        self.cx_text.configure(state="disabled")

    def _run_scaling(self):
        from complexity import scaling_report
        text = scaling_report()
        self.cx_text.configure(state="normal")
        self.cx_text.delete("1.0", "end")
        self.cx_text.insert("1.0", text)
        self.cx_text.configure(state="disabled")
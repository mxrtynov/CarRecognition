import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from data import get_train_set, get_unknown_set, MAX_X1, MAX_X2, normalize
from classifier import MinDistanceClassifier, PerceptronClassifier
from complexity import report as complexity_report
from ui_theme import (
    apply_theme,
    BG, BG_PANEL, BG_CARD, BG_INPUT, BORDER,
    FG, FG_DIM, FG_MUTED,
    ACCENT, ACCENT2, SUCCESS, DANGER, WARN,
    FONT, FONT_S, FONT_XS, FONT_B, FONT_H1, FONT_H3, FONT_MONO,
    FONT_VALUE,
)

CLASS_NAMES = {1: "Экономный", 2: "Спортивный"}
CLASS_COLORS = {1: ACCENT, 2: DANGER}


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

    # ---------- layout ----------

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

        tk.Label(inner, text="расход топлива  ×  мощность",
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
        tk.Label(head, text="Признаковое пространство", bg=BG_CARD, fg=FG,
                 font=FONT_H3).pack(side="left")
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
        tk.Label(head, text="Распознавание заданных объектов", bg=BG_CARD,
                 fg=FG, font=FONT_H3).pack(side="left")
        tk.Label(head, text="5 автомобилей · оба метода",
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
        tk.Label(head, text="Вычислительная сложность", bg=BG_CARD,
                 fg=FG, font=FONT_H3).pack(side="left")

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
        ax.set_xlabel("Расход топлива, л/100 км", color=FG_DIM,
                      fontsize=10, labelpad=8)
        ax.set_ylabel("Мощность, л.с.", color=FG_DIM,
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

        self._slider(pad, "Расход топлива", "л/100 км",
                     self.x1, 0, MAX_X1, 0.1, "{:.1f}")
        tk.Frame(pad, bg=BG_PANEL, height=14).pack()
        self._slider(pad, "Мощность", "л.с.",
                     self.x2, 0, MAX_X2, 5, "{:.0f}")

        tk.Frame(pad, bg=BG_PANEL, height=28).pack()

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

    # ---------- events ----------

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

    # ---------- refresh ----------

    def _update_all(self):
        self._refresh_plot()
        self._refresh_recognition()
        self._refresh_custom()
        self._refresh_models_info()

    def _refresh_plot(self):
        ax = self.ax
        ax.clear()
        self._style_axes()

        shown = set()
        for x1, x2, c in self.train_set:
            label = CLASS_NAMES[c] if c not in shown else None
            shown.add(c)
            ax.scatter(x1, x2, c=CLASS_COLORS[c], s=70, marker="o",
                       edgecolors="white", linewidths=0.8,
                       label=label, zorder=5)

        for i, (p1, p2) in enumerate(self.min_dist.prototypes):
            ax.scatter(p1, p2, c=WARN, s=220, marker="*",
                       edgecolors=BG_CARD, linewidths=1.0, zorder=6)
            ax.annotate(f"P{i+1}", (p1, p2), textcoords="offset points",
                        xytext=(10, 10), color=WARN, fontsize=9,
                        fontweight="bold")

        x1a, x2a, x1b, x2b = self.perceptron.boundary_points()
        ax.plot([x1a, x1b], [x2a, x2b], color=SUCCESS, linewidth=1.8,
                linestyle="--", label="Граница (восприятие)", zorder=4)

        if len(self.min_dist.prototypes) == 2:
            p1 = self.min_dist.prototypes[0]
            p2 = self.min_dist.prototypes[1]
            A = 2 * (p1[0] - p2[0])
            B = 2 * (p1[1] - p2[1])
            C = (p1[0] ** 2 + p1[1] ** 2) - (p2[0] ** 2 + p2[1] ** 2)
            pts = []
            if abs(B) > 1e-9:
                for xv in [0.0, 1.0]:
                    pts.append((xv, (C - A * xv) / B))
            elif abs(A) > 1e-9:
                xv = C / A
                pts = [(xv, 0.0), (xv, 1.0)]
            if len(pts) == 2:
                ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
                        color=ACCENT, linewidth=1.6, linestyle=":",
                        label="Граница (мин. расст.)", zorder=3)

        for _, ux1, ux2 in self.unknown_set:
            ax.scatter(ux1, ux2, facecolors="none", edgecolors=FG_MUTED,
                       s=110, marker="s", linewidths=1.6, zorder=7)

        x1, x2 = normalize(self.x1.get(), self.x2.get())
        md_class = self.min_dist.predict(x1, x2)
        ax.scatter(x1, x2, c=CLASS_COLORS[md_class], s=280, marker="D",
                   edgecolors="white", linewidths=1.6, zorder=9,
                   label="Свой автомобиль")

        ax.legend(loc="upper left", facecolor=BG_CARD, edgecolor="none",
                  labelcolor=FG, fontsize=9, framealpha=0.0)

        self.canvas.draw_idle()

    def _refresh_custom(self):
        x1r, x2r = self.x1.get(), self.x2.get()
        x1, x2 = normalize(x1r, x2r)

        md_vals = self.min_dist.decision_values(x1, x2)
        best = max(range(len(md_vals)), key=lambda i: md_vals[i])
        md_class = self.min_dist.classes[best]

        pc_val = self.perceptron.decision_value(x1, x2)
        pc_class = self.perceptron.predict(x1, x2)

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
        for name, x1, x2 in self.unknown_set:
            md_vals = self.min_dist.decision_values(x1, x2)
            best = max(range(len(md_vals)), key=lambda i: md_vals[i])
            md_class = self.min_dist.classes[best]
            pc_class = self.perceptron.predict(x1, x2)
            pc_val = self.perceptron.decision_value(x1, x2)

            lines.append(f"■  {name}")
            lines.append(f"     X1 = {x1:.3f}    X2 = {x2:.3f}")
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
        lines = ["прототипы"]
        for i, ((p1, p2), c) in enumerate(
                zip(self.min_dist.prototypes, self.min_dist.classes)):
            lines.append(f"  P{i+1}  {CLASS_NAMES[c]:<10}  "
                         f"({p1:.3f}; {p2:.3f})")
        lines.append("")
        lines.append("восприятие")
        w1, w2, w0 = self.perceptron.w
        lines.append(f"  W1 = {w1:+.3f}   W2 = {w2:+.3f}   W0 = {w0:+.3f}")
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
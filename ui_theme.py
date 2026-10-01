import tkinter as tk
from tkinter import ttk

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
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", background=BG, foreground=FG, font=FONT)
    style.configure("TFrame", background=BG)
    style.configure("Panel.TFrame", background=BG_PANEL)
    style.configure("Card.TFrame", background=BG_CARD)

    style.configure("TLabel", background=BG, foreground=FG, font=FONT)
    style.configure("Card.TLabel", background=BG_CARD, foreground=FG)
    style.configure("Dim.TLabel", background=BG_CARD, foreground=FG_DIM,
                    font=FONT_S)

    style.configure("TButton", background=BG_INPUT, foreground=FG,
                    font=FONT_B, borderwidth=0, padding=(14, 9),
                    focusthickness=0, relief="flat")
    style.map("TButton",
              background=[("active", BG_CARD), ("pressed", BG_CARD)])

    style.configure("Accent.TButton", background=ACCENT, foreground="#0f1218",
                    font=FONT_B, borderwidth=0, padding=(14, 9),
                    focusthickness=0, relief="flat")
    style.map("Accent.TButton",
              background=[("active", ACCENT2), ("pressed", ACCENT2)],
              foreground=[("active", "#0f1218")])

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
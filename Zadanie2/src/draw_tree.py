"""Etap 6: rysunek drzewa decyzyjnego jednej osoby -> tree.png.

Drzewo jest uczone tak samo jak w predict.py: na wszystkich 90 ocenach osoby
z train.csv, z parametrami z results/best_tree.json. Osoba 459 to jedna z 79 osób,
u których dostrojone drzewo się dzieli (największy zysk Giniego wśród nich).

Użycie:
    uv run python src/draw_tree.py                       # osoba 459, parametry z dostrajania
    uv run python src/draw_tree.py --user 624
    uv run python src/draw_tree.py --params etap2        # większe drzewo z Etapu 2 (ilustracja)
    uv run python src/draw_tree.py --out results/tree_etap2.png --params etap2
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from data import ROOT, load_all
from matrix import NUMERIC, load_matrix, user_data
from predict import load_params
from tree import N_CLASSES, DecisionTree

DEFAULT_USER = 459
OUT = ROOT / "tree.png"
ETAP2 = {"max_depth": 5, "min_samples_split": 10, "min_samples_leaf": 5, "min_gain": 0.0}

# kolory (paleta z dataviz: powierzchnia, tekst, siatka, slot 1)
SURFACE, INK, MUTED, BORDER, GRID = "#fcfcfb", "#0b0b0b", "#5f5e58", "#c3c2b7", "#e1e0d9"
BLUE, BAR = "#2a78d6", "#c3c2b7"

BOX_W, BOX_H = 0.9, 0.62      # rozmiar węzła (odstęp między liśćmi = 1)
LEVEL = 1.05                  # odstęp między poziomami


# ---------------------------------------------------------------- opis węzłów

def fmt_number(x):
    if abs(x) >= 1e6:
        return f"{x / 1e6:.4g} mln"
    return f"{x:.10g}"


def question(tree, node):
    """Pytanie w węźle i etykiety gałęzi (lewa = warunek X <= próg spełniony).

    Kolumna 0/1 "genre=Drama <= 0.5" znaczy "film NIE jest dramatem", więc
    rysujemy ją jako pytanie "genre = Drama?" z lewą gałęzią "nie".
    """
    name = tree.feature_names_[node.feature]
    if name not in NUMERIC:
        return f"{name.replace('=', ' = ', 1)}?", "nie", "tak"
    return f"{name} ≤ {fmt_number(node.threshold)}?", "tak", "nie"


# ---------------------------------------------------------------- układ

def layout(tree):
    """Pozycje (x, y) węzłów: liście kolejno od lewej, rodzic nad środkiem dzieci."""
    pos, next_leaf = {}, [0]

    def place(node):
        if node.is_leaf:
            x = next_leaf[0]
            next_leaf[0] += 1
        else:
            x = (place(node.left) + place(node.right)) / 2
        pos[id(node)] = (x, -node.depth * LEVEL)
        return x

    place(tree.root)
    return pos


def draw_node(ax, tree, node, x, y):
    w, h = BOX_W, BOX_H
    left, bottom = x - w / 2, y - h / 2
    ax.add_patch(FancyBboxPatch((left, bottom), w, h, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="white", ec=BLUE if node.is_leaf else BORDER,
                                lw=2 if node.is_leaf else 1, zorder=2))
    title = f"ocena {node.prediction}" if node.is_leaf else question(tree, node)[0]
    ax.text(x, y + h * 0.33, title, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=INK, zorder=3)
    ax.text(x, y + h * 0.13, f"n = {node.n}, Gini = {node.gini:.3f}", ha="center",
            va="center", fontsize=8, color=MUTED, zorder=3)

    # mini-histogram ocen 0-5 w węźle; niebieski słupek = ocena, którą węzeł przewiduje
    bar_w = w * 0.8 / N_CLASSES
    base, top = bottom + h * 0.16, bottom + h * 0.47
    peak = max(node.counts.max(), 1)
    for r in range(N_CLASSES):
        bx = left + w * 0.1 + r * bar_w
        bh = (top - base) * node.counts[r] / peak
        ax.add_patch(Rectangle((bx + bar_w * 0.12, base), bar_w * 0.76, bh,
                               fc=BLUE if r == node.prediction else BAR, lw=0, zorder=3))
        ax.text(bx + bar_w / 2, base - h * 0.03, str(r), ha="center", va="top",
                fontsize=6.5, color=MUTED, zorder=3)
    ax.plot([left + w * 0.1, left + w * 0.9], [base, base], color=GRID, lw=0.8, zorder=3)


def draw(tree, user, params, label, out):
    pos = layout(tree)
    n_leaves, depth = tree.n_leaves(), tree.depth()
    header, footer = 1.2, 0.45          # wysokość nagłówka i stopki w calach
    fig_w = max(6.5, 2.2 * n_leaves)
    fig_h = max(3.6, 1.55 * (depth + 1)) + header + footer
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for node in tree.nodes():
        if node.is_leaf:
            continue
        x, y = pos[id(node)]
        _, yes_left, yes_right = question(tree, node)
        for child, text in ((node.left, yes_left), (node.right, yes_right)):
            cx, cy = pos[id(child)]
            ax.plot([x, cx], [y - BOX_H / 2, cy + BOX_H / 2], color=BORDER, lw=1.2, zorder=1)
            ax.text((x + cx) / 2, (y - BOX_H / 2 + cy + BOX_H / 2) / 2, text, ha="center",
                    va="center", fontsize=8.5, color=INK, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"))
    for node in tree.nodes():
        draw_node(ax, tree, node, *pos[id(node)])

    params_txt = ", ".join(f"{k}={'brak' if v is None else f'{v:g}'}" for k, v in params.items())
    left_in = 0.25 / fig_w                # lewy margines tekstu (0.25 cala)
    fig.text(left_in, 1 - 0.18 / fig_h, f"Drzewo decyzyjne dla osoby {user}", ha="left",
             va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(left_in, 1 - 0.5 / fig_h, f"{label}\n{params_txt}\nUczone na 90 ocenach osoby "
             f"z train.csv; głębokość {depth}, liście {n_leaves}", ha="left", va="top",
             fontsize=8.5, color=MUTED, linespacing=1.4)
    fig.text(left_in, 0.15 / fig_h, "Słupki: rozkład ocen 0–5 w węźle, niebieski = ocena "
             "przewidywana przez węzeł. Liście mają niebieską ramkę.", ha="left", va="bottom",
             fontsize=7.5, color=MUTED)

    xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(-depth * LEVEL - 0.55, 0.55)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=1 - header / fig_h, bottom=footer / fig_h)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------- uruchomienie

def main():
    parser = argparse.ArgumentParser(description="Rysunek drzewa jednej osoby (Etap 6).")
    parser.add_argument("--user", type=int, default=DEFAULT_USER)
    parser.add_argument("--params", choices=["best", "etap2"], default="best",
                        help="best = z results/best_tree.json (jak w submission_tree.csv), "
                             "etap2 = większe drzewo z Etapu 2, do ilustracji")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out

    movies, train, task = load_all()
    X_train, y_train, _, _ = user_data(load_matrix(), train, task, args.user)
    if args.params == "best":
        params, label = load_params(), "Parametry z dostrajania (jak w submission_tree.csv)"
    else:
        params, label = ETAP2, "Parametry z Etapu 2 (ilustracja, nie użyte w submission)"
    tree = DecisionTree(**params).fit(X_train, y_train)

    print(f"Osoba {args.user}, {label.lower()}: {params}\n")
    print(tree.to_text())
    if tree.root.is_leaf:
        print("\nUwaga: drzewo tej osoby to jeden liść (brak podziału).")
    draw(tree, args.user, params, label, out)
    print(f"\nZapisano {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()

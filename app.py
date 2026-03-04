#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:08:37 2026

@author: hugoplouhinec
"""

import io
import json
import re
import zipfile
import tempfile
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import streamlit as st
from PIL import Image


# =========================
# ======= PALETTE =========
# =========================
PALETTE_MASTER = [
    ("white",(255,255,255)),("light-gray",(210,210,210)),("gray",(160,160,160)),
    ("dark-gray",(90,90,90)),("black",(0,0,0)),
    ("skin-light",(255,224,189)),("skin",(242,210,161)),("skin-dark",(198,134,66)),
    ("brown",(100,45,20)),("dark-brown",(60,30,15)),
    ("red",(220,40,50)),("dark-red",(150,0,0)),("orange-red",(255,100,40)),
    ("pink",(255,170,200)),("magenta",(255,0,200)),
    ("yellow",(255,215,0)),("gold",(240,190,70)),("orange",(255,140,0)),
    ("peach",(255,200,140)),("yellow-light",(255,240,180)),
    ("green",(40,170,80)),("dark-green",(25,95,70)),("lime",(150,255,120)),
    ("olive",(100,120,40)),("mint",(130,200,160)),
    ("blue",(0,100,220)),("dark-blue",(0,50,130)),("light-blue",(130,200,255)),
    ("teal",(0,180,190)),("navy",(10,30,80))
]

name_en_to_fr = {
    "white":"blanc","light-gray":"gris clair","gray":"gris","dark-gray":"gris foncé","black":"noir",
    "skin-light":"chair clair","skin":"chair","skin-dark":"chair foncé","brown":"marron","dark-brown":"marron foncé",
    "red":"rouge","dark-red":"rouge foncé","orange-red":"rouge orangé","pink":"rose","magenta":"magenta",
    "yellow":"jaune","gold":"doré","orange":"orange","peach":"pêche","yellow-light":"jaune clair",
    "green":"vert","dark-green":"vert foncé","lime":"vert clair","olive":"olive","mint":"vert menthe",
    "blue":"bleu","dark-blue":"bleu foncé","light-blue":"bleu clair","teal":"bleu-vert","navy":"bleu marine"
}

LETTERS = list("ABCDE")  # on prend les N_KEEP premières


# =========================
# ========= UTILS =========
# =========================
def to_float(rgb):
    return np.array(rgb)/255.0

def brightness(rgb):
    r,g,b = [v/255 for v in rgb]
    return 0.299*r + 0.587*g + 0.114*b

def nearest_color(rgb, palette):
    r,g,b = rgb
    return min(
        palette,
        key=lambda item: (r-item[1][0])**2 + (g-item[1][1])**2 + (b-item[1][2])**2
    )

def latex_escape(s: str) -> str:
    return (s.replace("&", r"\&")
             .replace("%", r"\%")
             .replace("_", r"\_")
             .replace("#", r"\#")
             .replace("$", r"\$")
             .replace("{", r"\{")
             .replace("}", r"\}")
            )

def parse_values_fr(raw: str) -> List[str]:
    """
    Accepte : valeurs séparées par virgules, points-virgules, espaces, retours ligne.
    Conserve les virgules décimales (FR). Ex : "6; 12; 17,5"
    """
    tokens = re.split(r"[;\n\r\t ]+", raw.strip())
    tokens = [t.strip() for t in tokens if t.strip() != ""]
    # autoriser aussi "6,12,15" (virgules de séparation), MAIS attention décimales FR.
    # Donc on ne split pas sur "," ; c'est à l'utilisateur de séparer via ; ou espace.
    return tokens

def fr_to_float(s: str) -> float:
    return float(s.strip().replace(",", "."))

def decimals_count_fr(s: str) -> int:
    s = s.strip()
    return len(s.split(",")[1]) if "," in s else 0

def float_to_fr(x: float, nd: int) -> str:
    if nd <= 0:
        return str(int(round(x)))
    return f"{x:.{nd}f}".replace(".", ",")

def make_prop2_values(values_prop1: List[str], seed: int) -> List[str]:
    import random
    rng = random.Random(seed)
    floats = [fr_to_float(v) for v in values_prop1]
    lo, hi = min(floats), max(floats)
    out = []
    for v in values_prop1:
        nd = decimals_count_fr(v)
        r = rng.uniform(lo, hi)
        r = round(r, nd)
        out.append(float_to_fr(r, nd))
    return out

def is_very_light(rgb, white_threshold: float) -> bool:
    r,g,b = [v/255 for v in rgb]
    return 0.299*r + 0.587*g + 0.114*b > white_threshold


# =========================
# ====== CORE LOGIC =======
# =========================
def process_one_image(
    im_in: Image.Image,
    grid_len: int,
    block: int,
    n_keep: int,
    white_threshold: float,
    seed: int,
    invert_y_blocks: bool,
    values_prop1: List[str],
    out_dir: Path
) -> None:
    """
    Écrit dans out_dir :
    - Pixel_art_grille_vierge.png
    - Pixel_art_solution.png
    - Pixel_art_correction.png
    - Pixel_art_legende.png
    - pixel_art.csv
    - pixel_art_blocs_{block}x{block}.csv/.json/.txt
    - pixel_art_blocs_{block}x{block}.tex
    - pixel_art_blocs_{block}x{block}_jeu.tex
    """

    GRID_W, GRID_H = grid_len, grid_len
    assert GRID_W % block == 0 and GRID_H % block == 0

    # ------- sanitize / RGB -------
    im = im_in.convert("RGBA")
    bg = Image.new("RGBA", im.size, (255,255,255,255))
    im = Image.alpha_composite(bg, im).convert("RGB")

    # ------- crop square center -------
    w,h = im.size
    m = min(w,h)
    L = (w-m)//2
    T = (h-m)//2
    im = im.crop((L,T,L+m,T+m))

    # ------- resize to grid -------
    im_small = im.resize((GRID_W, GRID_H), Image.NEAREST)

    name_by_idx = {i:n for i,(n,_) in enumerate(PALETTE_MASTER)}
    rgb_by_idx  = {i:c for i,(_,c) in enumerate(PALETTE_MASTER)}

    mapped_idxs = []
    for y in range(GRID_H):
        row = []
        for x in range(GRID_W):
            rgb = im_small.getpixel((x,y))
            if is_very_light(rgb, white_threshold):
                idx = next(i for i,(n,_) in enumerate(PALETTE_MASTER) if n=="white")
            else:
                idx = PALETTE_MASTER.index(nearest_color(rgb, PALETTE_MASTER))
            row.append(idx)
        mapped_idxs.append(row)

    flat   = [idx for row in mapped_idxs for idx in row]
    counts = Counter(flat)

    top = [idx for idx,_ in counts.most_common(n_keep)]
    if len(top) < n_keep:
        extra = [i for i in range(len(PALETTE_MASTER)) if i not in top]
        top += extra[:(n_keep-len(top))]

    idx2letter = {idx: LETTERS[i] for i,idx in enumerate(top)}
    letter2rgb = {idx2letter[idx]: rgb_by_idx[idx] for idx in top}

    def nearest_top_letter(idx):
        c = rgb_by_idx[idx]
        best = min(top, key=lambda j: sum((c[k]-rgb_by_idx[j][k])**2 for k in range(3)))
        return idx2letter[best]

    letters = []
    colors  = np.zeros((GRID_H, GRID_W, 3), dtype=float)
    for y in range(GRID_H):
        rowL = []
        for x in range(GRID_W):
            idx = mapped_idxs[y][x]
            Ltr = idx2letter[idx] if idx in idx2letter else nearest_top_letter(idx)
            rowL.append(Ltr)
            colors[y,x] = to_float(letter2rgb[Ltr])
        letters.append(rowL)

    # ------- outputs names -------
    CSV_OUT    = out_dir / "pixel_art.csv"
    PNG_STUD   = out_dir / "Pixel_art_grille_vierge.png"
    PNG_SOL    = out_dir / "Pixel_art_solution.png"
    PNG_CORR   = out_dir / "Pixel_art_correction.png"
    PNG_LEGEND = out_dir / "Pixel_art_legende.png"

    BLOC_CSV     = out_dir / f"pixel_art_blocs_{block}x{block}.csv"
    BLOC_TXT     = out_dir / f"pixel_art_blocs_{block}x{block}.txt"
    BLOC_JSON    = out_dir / f"pixel_art_blocs_{block}x{block}.json"
    BLOC_TEX_RAW = out_dir / f"pixel_art_blocs_{block}x{block}.tex"
    BLOC_TEX_JEU = out_dir / f"pixel_art_blocs_{block}x{block}_jeu.tex"

    pd.DataFrame(letters).to_csv(CSV_OUT, header=False, index=False, encoding="utf-8")

    # ------- blocs data -------
    def letter_to_color_fr(L):
        rgb = tuple(int(v) for v in letter2rgb[L])
        name_en = next(n for i,(n,c) in enumerate(PALETTE_MASTER) if tuple(c)==rgb)
        return name_en_to_fr.get(name_en, name_en)

    bloc_rows = GRID_H // block
    bloc_cols = GRID_W // block

    blocs: Dict[Tuple[int,int], List[dict]] = {}
    rows_csv = []

    for br in range(bloc_rows):         # br=0 : rangée du haut dans l'image
        for bc in range(bloc_cols):     # bc=0 : colonne de gauche

            bloc_ligne = (bloc_rows - br) if invert_y_blocks else (br + 1)
            bloc_col   = bc + 1
            key = (bloc_ligne, bloc_col)

            items = []
            card = 0

            # carte 1 = bas-gauche ; puis droite ; puis on monte
            for dy in range(block-1, -1, -1):     # bas -> haut
                for dx in range(block):           # gauche -> droite
                    y = br*block + dy
                    x = bc*block + dx

                    cx = dx + 1
                    cy = block - dy

                    Ltr = letters[y][x]
                    colfr = letter_to_color_fr(Ltr)

                    card += 1
                    items.append({"carte": card, "coord": f"({cx};{cy})", "couleur": colfr})
                    rows_csv.append({
                        "bloc_ligne": bloc_ligne,
                        "bloc_colonne": bloc_col,
                        "carte": card,
                        "coord": f"({cx};{cy})",
                        "couleur": colfr
                    })

            blocs[key] = items

    pd.DataFrame(rows_csv).to_csv(BLOC_CSV, index=False, encoding="utf-8")
    with open(BLOC_JSON, "w", encoding="utf-8") as f:
        json.dump({f"{k[0]}-{k[1]}": v for k,v in blocs.items()}, f, ensure_ascii=False, indent=2)
    with open(BLOC_TXT, "w", encoding="utf-8") as f:
        for (bl,bc),items in sorted(blocs.items(), key=lambda kv:(kv[0][0], kv[0][1])):
            f.write(f"Bloc {block}×{block} (ligne {bl}, colonne {bc})\n")
            for it in items:
                f.write(f"  {it['carte']:>2}  {it['coord']} ; {it['couleur']}\n")
            f.write("\n")

    # ------- "mauvaise couleur = suivante (cycle) dans la liste globale" -------
    top_by_freq = [idx for idx, _ in counts.most_common(n_keep)]
    pixelart_colors_fr = [letter_to_color_fr(idx2letter[idx]) for idx in top_by_freq]
    next_wrong_color = {
        c: pixelart_colors_fr[(i + 1) % len(pixelart_colors_fr)]
        for i, c in enumerate(pixelart_colors_fr)
    }

    # ------- LaTeX RAW -------
    def write_latex_raw(path_out: Path) -> None:
        with open(path_out, "w", encoding="utf-8") as f:
            f.write(r"\documentclass[11pt,a4paper]{article}" + "\n")
            f.write(r"\usepackage[margin=1.5cm]{geometry}" + "\n")
            f.write(r"\usepackage[T1]{fontenc}" + "\n")
            f.write(r"\usepackage[utf8]{inputenc}" + "\n")
            f.write(r"\usepackage[french]{babel}" + "\n")
            f.write(r"\usepackage{multicol}" + "\n")
            f.write(r"\usepackage{array}" + "\n")
            f.write(r"\usepackage{xcolor}" + "\n")
            f.write(r"\usepackage{graphicx}" + "\n")
            f.write(r"\setlength{\tabcolsep}{6pt}" + "\n")
            f.write(r"\setlength{\columnsep}{0.8cm}" + "\n")
            f.write(r"\setlength{\parindent}{0pt}" + "\n\n")

            f.write(r"\begin{document}" + "\n\n")
            f.write(r"\begin{center}" + "\n")
            f.write(r"{\Large\bfseries Pixel Art — Blocs $" + f"{block}" + r"\times" + f"{block}" + r"$}\\[2mm]" + "\n")
            f.write(r"{\small Chaque tableau correspond à un bloc.}" + "\n")
            f.write(r"\end{center}" + "\n\n")

            f.write(r"\begin{multicols}{2}" + "\n")
            f.write(r"\raggedcolumns" + "\n\n")

            for (bl, bc) in sorted(blocs.keys(), key=lambda k: (k[0], k[1])):
                items = blocs[(bl, bc)]
                f.write(r"\subsection*{Bloc " + f"{block}" + r"$\times$" + f"{block}" + r" — ligne %d, colonne %d}" % (bl, bc) + "\n")
                f.write(r"\begin{center}" + "\n")
                f.write(r"\scalebox{0.75}{" + "\n")
                f.write(r"\renewcommand{\arraystretch}{1.15}" + "\n")
                f.write(r"\begin{tabular}{|c|c|l|l|}" + "\n")
                f.write(r"\hline" + "\n")
                f.write(r"\textbf{Numéro carte} & \textbf{Coordonnées} & \textbf{Proposition 1} & \textbf{Proposition 2} \\" + "\n")
                f.write(r"\hline" + "\n")
                for it in items:
                    i = it["carte"]
                    coord = latex_escape(it["coord"])
                    col   = latex_escape(it["couleur"])
                    f.write(f"{i} & ${coord}$ & {col} & {col} \\\\\n")
                    f.write(r"\hline" + "\n")
                f.write(r"\end{tabular}" + "\n")
                f.write(r"}" + "\n")
                f.write(r"\end{center}" + "\n")
                f.write(r"\vspace{3mm}" + "\n\n")

            f.write(r"\end{multicols}" + "\n")
            f.write(r"\end{document}" + "\n")

    write_latex_raw(BLOC_TEX_RAW)

    # ------- LaTeX JEU -------
    def write_latex_game(path_out: Path, values_prop1: List[str], seed: int) -> None:
        import random
        values_prop2 = make_prop2_values(values_prop1, seed=seed)
        rng = random.Random(seed)

        with open(path_out, "w", encoding="utf-8") as f:
            f.write(r"\documentclass[11pt,a4paper]{article}" + "\n")
            f.write(r"\usepackage[margin=1.5cm]{geometry}" + "\n")
            f.write(r"\usepackage[T1]{fontenc}" + "\n")
            f.write(r"\usepackage[utf8]{inputenc}" + "\n")
            f.write(r"\usepackage[french]{babel}" + "\n")
            f.write(r"\usepackage{multicol}" + "\n")
            f.write(r"\usepackage{array}" + "\n")
            f.write(r"\usepackage{xcolor}" + "\n")
            f.write(r"\usepackage{graphicx}" + "\n")
            f.write(r"\setlength{\tabcolsep}{6pt}" + "\n")
            f.write(r"\setlength{\columnsep}{0.8cm}" + "\n")
            f.write(r"\setlength{\parindent}{0pt}" + "\n\n")

            f.write(r"\begin{document}" + "\n\n")
            f.write(r"\begin{center}" + "\n")
            f.write(r"{\Large\bfseries Pixel Art — Blocs $" + f"{block}" + r"\times" + f"{block}" + r"$ (jeu)}\\[2mm]" + "\n")
            f.write(r"{\small Pour chaque ligne : une proposition est correcte.}" + "\n")
            f.write(r"\end{center}" + "\n\n")

            f.write(r"\begin{multicols}{2}" + "\n")
            f.write(r"\raggedcolumns" + "\n\n")

            for (bl, bc) in sorted(blocs.keys(), key=lambda k: (k[0], k[1])):
                items = blocs[(bl, bc)]

                f.write(r"\subsection*{Bloc " + f"{block}" + r"$\times$" + f"{block}" + r" — ligne %d, colonne %d}" % (bl, bc) + "\n")
                f.write(r"\begin{center}" + "\n")
                f.write(r"\scalebox{0.75}{" + "\n")
                f.write(r"\renewcommand{\arraystretch}{1.15}" + "\n")
                f.write(r"\begin{tabular}{|c|c|l|l|}" + "\n")
                f.write(r"\hline" + "\n")
                f.write(r"\textbf{Numéro carte} & \textbf{Coordonnées} & \textbf{Proposition 1} & \textbf{Proposition 2} \\" + "\n")
                f.write(r"\hline" + "\n")

                for it in items:
                    i = it["carte"]
                    coord = latex_escape(it["coord"])
                    true_color = it["couleur"]

                    v1 = values_prop1[i-1]
                    v2 = values_prop2[i-1]

                    prop2_color = next_wrong_color.get(true_color, true_color)

                    prop1_cell = latex_escape(f"{v1} {true_color}")
                    prop2_cell = latex_escape(f"{v2} {prop2_color}")

                    if rng.random() < 0.5:
                        prop1_cell, prop2_cell = prop2_cell, prop1_cell

                    f.write(f"{i} & ${coord}$ & {prop1_cell} & {prop2_cell} \\\\\n")
                    f.write(r"\hline" + "\n")

                f.write(r"\end{tabular}" + "\n")
                f.write(r"}" + "\n")
                f.write(r"\end{center}" + "\n")
                f.write(r"\vspace{3mm}" + "\n\n")

            f.write(r"\end{multicols}" + "\n")
            f.write(r"\end{document}" + "\n")

    write_latex_game(BLOC_TEX_JEU, values_prop1, seed=seed)

    # ------- images -------
    # 1) grille vierge
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.imshow(np.ones((GRID_H, GRID_W, 3)), interpolation="nearest")
    for x in range(GRID_W+1):
        plt.plot([x-0.5,x-0.5],[-0.5,GRID_H-0.5],color="0.6",linewidth=0.6)
    for y in range(GRID_H+1):
        plt.plot([-0.5,GRID_W-0.5],[y-0.5,y-0.5],color="0.6",linewidth=0.6)
    plt.xlim(-0.5, GRID_W-0.5)
    plt.ylim(GRID_H-0.5, -0.5)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PNG_STUD, dpi=220)
    plt.close()

    # 2) solution
    plt.figure(figsize=(8,8))
    plt.imshow(colors, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PNG_SOL, dpi=220)
    plt.close()

    # 3) correction (couleurs + lettres + grille)
    FONT_SIZE = 5.5
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.imshow(colors, interpolation="nearest", zorder=0)

    for y in range(GRID_H):
        for x in range(GRID_W):
            Ltr = letters[y][x]
            rgb = letter2rgb[Ltr]
            txt_col = "white" if brightness(rgb) < 0.45 else "black"
            stroke  = "black" if txt_col == "white" else "white"
            ax.text(
                x, y, Ltr,
                ha="center", va="center",
                fontsize=FONT_SIZE,
                color=txt_col, alpha=0.95, zorder=10,
                path_effects=[pe.withStroke(linewidth=1.2, foreground=stroke)]
            )

    THIN, THICK = 0.5, 1.8
    COL_T, COL_B = "0.7", "0.25"
    for x in range(GRID_W+1):
        lw  = THICK if x % block == 0 else THIN
        col = COL_B if x % block == 0 else COL_T
        plt.plot([x-0.5,x-0.5],[-0.5,GRID_H-0.5],color=col,linewidth=lw,zorder=6)
    for y in range(GRID_H+1):
        lw  = THICK if y % block == 0 else THIN
        col = COL_B if y % block == 0 else COL_T
        plt.plot([-0.5,GRID_W-0.5],[y-0.5,y-0.5],color=col,linewidth=lw,zorder=6)

    plt.xlim(-0.5, GRID_W-0.5)
    plt.ylim(GRID_H-0.5, -0.5)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PNG_CORR, dpi=220)
    plt.close()

    # 4) légende
    top_pairs = [(idx, counts[idx]) for idx,_ in counts.most_common(n_keep)]
    plt.figure(figsize=(max(6, n_keep*1.2), 1.6))
    for i,(idx,_) in enumerate(top_pairs):
        rgb  = rgb_by_idx[idx]
        name = name_by_idx[idx]
        Ltr  = idx2letter[idx]
        plt.fill_between([i,i+1],0,1,color=to_float(rgb))
        txt_col = "white" if brightness(rgb) < 0.45 else "black"
        plt.text(i+0.5,0.5,f"{Ltr}  {name}\nRGB{rgb}",
                 ha="center",va="center",fontsize=9,color=txt_col)
    plt.xlim(0,len(top_pairs))
    plt.ylim(0,1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PNG_LEGEND, dpi=220)
    plt.close()


def zip_directory(dir_path: Path) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in dir_path.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(dir_path))
    bio.seek(0)
    return bio.read()


# =========================
# ========== UI ===========
# =========================
st.set_page_config(page_title="Pixel Art (blocs + LaTeX)", layout="wide")
st.title("Pixel Art → grilles, correction, tableaux LaTeX (blocs)")

with st.sidebar:
    st.header("Entrées")
    files = st.file_uploader("1) Dépose tes images (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True)

    grid_len = st.number_input("2) Taille de la grille (carrée) : longueur", min_value=4, max_value=80, value=12, step=1)

    block = st.number_input("3) Taille d'un bloc", min_value=1, max_value=40, value=3, step=1)
    n_keep = st.number_input("4) Nombre de couleurs retenues", min_value=2, max_value=5, value=4, step=1)

    white_threshold = st.slider("Seuil 'blanc' (au-dessus → blanc)", min_value=0.80, max_value=0.99, value=0.93, step=0.01)

    seed = st.number_input("Seed (reproductibilité)", min_value=0, max_value=10_000, value=42, step=1)
    invert_y_blocks = st.checkbox("Bloc ligne 1 en bas", value=True)

    needed_values = int(block) * int(block)
    st.markdown(f"**5) Bonnes valeurs** : il faut **{needed_values}** valeurs (bloc {block}×{block}).")
    values_raw = st.text_area("Sépare avec ; ou espace (garde la virgule décimale si besoin)", value="6; 12; 15; 65; 40; 8; 54; 72; 17,5", height=120)

    run = st.button("Générer", type="primary")

# validations
errors = []
if run:
    if not files:
        errors.append("Dépose au moins une image.")
    if grid_len % block != 0:
        errors.append("La taille de bloc doit être un diviseur de la longueur de grille.")
    values = parse_values_fr(values_raw)
    if len(values) != needed_values:
        errors.append(f"Tu as donné {len(values)} valeur(s) ; il en faut {needed_values}.")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # output
    with tempfile.TemporaryDirectory() as td:
        out_root = Path(td) / "exports"
        out_root.mkdir(parents=True, exist_ok=True)

        cols = st.columns(2)
        with cols[0]:
            st.subheader("Aperçu")
        with cols[1]:
            st.subheader("Téléchargement")

        previews = []

        for uf in files:
            im = Image.open(uf)
            stem = Path(uf.name).stem
            out_dir = out_root / stem
            out_dir.mkdir(parents=True, exist_ok=True)

            process_one_image(
                im_in=im,
                grid_len=int(grid_len),
                block=int(block),
                n_keep=int(n_keep),
                white_threshold=float(white_threshold),
                seed=int(seed),
                invert_y_blocks=bool(invert_y_blocks),
                values_prop1=values,
                out_dir=out_dir
            )

            corr_path = out_dir / "Pixel_art_correction.png"
            grid_path = out_dir / "Pixel_art_grille_vierge.png"
            previews.append((stem, corr_path, grid_path))

        # previews
        with cols[0]:
            for stem, corr_path, grid_path in previews:
                st.markdown(f"### {stem}")
                st.image(str(grid_path), caption="Grille vierge", use_container_width=True)
                st.image(str(corr_path), caption="Correction", use_container_width=True)

        # zip
        zip_bytes = zip_directory(out_root)
        with cols[1]:
            st.download_button(
                label="Télécharger tout (ZIP)",
                data=zip_bytes,
                file_name="pixel_art_exports.zip",
                mime="application/zip",
                use_container_width=True
            )
            st.info("Dans le ZIP : images (grille vierge / solution / correction / légende), CSV/JSON/TXT, LaTeX brut + LaTeX jeu.")
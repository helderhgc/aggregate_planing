"""
Gera as figuras do artigo SBPO 2026:
  fig_comparacao.pdf  – Fig 2: gráfico de barras custo total por estratégia
  fig_dashboard.png   – Fig 1: screenshot do dashboard (gerado via Playwright/selenium)
                        -- se Playwright não estiver disponível, gera um mockup vetorial
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─── Carregar dados do cenário-padrão ─────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "..", "data", "default_scenario.json")) as f:
    d = json.load(f)

from solvers import solve_chase, solve_level, solve_mixed, solve_lp, solve_transportation

costs = d["costs"].copy()
costs.setdefault("lost_sales", 0.0)

KW = dict(
    periods=d["periods"],
    demand=d["demand"],
    costs=costs,
    capacity=d["capacity"],
    initial_workforce=float(d["initial_workforce"]),
    initial_inventory=float(d["initial_inventory"]),
    productivity=float(d["productivity"]),
    integer_workforce=False,
    integer_production=False,
    shortage_policy="backorders",
)

demand = d["demand"]
periods = list(range(1, len(demand) + 1))
period_labels = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                 "Jul", "Ago", "Set", "Out", "Nov", "Dez"][:len(demand)]

r_chase  = solve_chase(**KW)
r_level  = solve_level(**KW)
r_mixed  = solve_mixed(**KW)
r_lp     = solve_lp(**KW)
r_trans  = solve_transportation(**KW)

OUT = os.path.dirname(os.path.abspath(__file__))

# ─── PALETA ──────────────────────────────────────────────────────────────────
COLORS = {
    "Chase":      "#4C72B0",
    "Level":      "#DD8452",
    "Mixed":      "#55A868",
    "LP":         "#C44E52",
    "Transport":  "#8172B2",
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 – Comparação de estratégias (2 painéis: barras de custo + perfil)
# ═══════════════════════════════════════════════════════════════════════════════
strategies = ["Acomp.\nDemanda", "Prod.\nNivelada", "Estratégia\nMista",
              "PL\n(Ótimo)", "Método\nTransporte"]
costs_total = [r["grand_total"] for r in [r_chase, r_level, r_mixed, r_lp, r_trans]]
bar_colors  = list(COLORS.values())

fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
fig2.subplots_adjust(wspace=0.35)

# ── Painel esquerdo: barras de custo total ────────────────────────────────────
ax = axes[0]
bars = ax.bar(strategies, [c / 1e6 for c in costs_total],
              color=bar_colors, edgecolor="white", linewidth=0.8, zorder=3)

# Destaque da barra ótima
bars[3].set_edgecolor("#8B0000")
bars[3].set_linewidth(2.0)

# Rótulos sobre as barras
for bar, val in zip(bars, costs_total):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"R${val/1e6:.2f}M",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_ylabel("Custo Total (R$ milhões)", fontsize=10)
ax.set_title("(a) Custo Total por Estratégia", fontsize=11, fontweight="bold")
ax.set_ylim(0, max(costs_total) / 1e6 * 1.15)
ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="x", labelsize=9)

# Anotação do ótimo
ax.annotate("Ótimo Global", xy=(3, costs_total[3] / 1e6),
            xytext=(3.5, costs_total[3] / 1e6 + 0.15),
            arrowprops=dict(arrowstyle="->", color="#8B0000"),
            fontsize=8.5, color="#8B0000")

# ── Painel direito: radar / perfil normalizado ────────────────────────────────
# Dimensões: Custo, Contratações, Demissões, Estoque médio, Horas extras
dim_labels = ["Custo\nTotal", "Contrat.", "Demissões", "Est.\nMédio", "Horas\nExtras"]
raw_data = {
    "Chase":     [r_chase["grand_total"],  sum(r_chase["hires"]),  sum(r_chase["fires"]),
                  np.mean(r_chase["inventory"]),  sum(r_chase.get("overtime", [0]*len(demand)))],
    "Level":     [r_level["grand_total"],  sum(r_level["hires"]),  sum(r_level["fires"]),
                  np.mean(r_level["inventory"]),  sum(r_level.get("overtime", [0]*len(demand)))],
    "Mixed":     [r_mixed["grand_total"],  sum(r_mixed["hires"]),  sum(r_mixed["fires"]),
                  np.mean(r_mixed["inventory"]),  sum(r_mixed.get("overtime", [0]*len(demand)))],
    "LP":        [r_lp["grand_total"],     sum(r_lp["hires"]),     sum(r_lp["fires"]),
                  np.mean(r_lp["inventory"]),     sum(r_lp.get("overtime", [0]*len(demand)))],
    "Transport": [r_trans["grand_total"],  sum(r_trans["hires"]),  sum(r_trans["fires"]),
                  np.mean(r_trans["inventory"]),  sum(r_trans.get("overtime", [0]*len(demand)))],
}

# Normalização 0-1 por dimensão (0 = melhor)
rd = np.array(list(raw_data.values()), dtype=float)
rd_min = rd.min(axis=0)
rd_max = rd.max(axis=0)
rd_range = np.where(rd_max - rd_min == 0, 1, rd_max - rd_min)
rd_norm = (rd - rd_min) / rd_range  # 0=melhor, 1=pior

ax2 = axes[1]
x = np.arange(len(dim_labels))
width = 0.15
offsets = np.linspace(-2 * width, 2 * width, 5)
strat_names = ["Chase", "Level", "Mixed", "LP", "Transport"]
display_names = ["Acomp. Demanda", "Prod. Nivelada", "Est. Mista", "PL (Ótimo)", "M. Transporte"]

for i, (key, off, dname) in enumerate(zip(strat_names, offsets, display_names)):
    ax2.bar(x + off, rd_norm[i], width * 0.9, label=dname,
            color=bar_colors[i], edgecolor="white", linewidth=0.5, zorder=3)

ax2.set_xticks(x)
ax2.set_xticklabels(dim_labels, fontsize=9)
ax2.set_ylabel("Desempenho Normalizado (0=melhor)", fontsize=9)
ax2.set_title("(b) Perfil Comparativo Normalizado", fontsize=11, fontweight="bold")
ax2.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
ax2.set_ylim(0, 1.25)

fig2.suptitle("Comparação das Estratégias de Planejamento Agregado da Produção",
              fontsize=12, fontweight="bold", y=1.01)

fig2_path = os.path.join(OUT, "fig_comparacao.pdf")
fig2.savefig(fig2_path, bbox_inches="tight", dpi=200)
print(f"Salvo: {fig2_path}")
plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 – Mockup vetorial do dashboard (para substituição por screenshot real)
# ═══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(14, 7.5))
fig1.patch.set_facecolor("#F0F2F6")

# Layout: sidebar (20%) | main (80%)
ax_side = fig1.add_axes([0.0, 0.0, 0.20, 1.0])
ax_main = fig1.add_axes([0.21, 0.0, 0.79, 1.0])
for ax in [ax_side, ax_main]:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

# ── Sidebar ──────────────────────────────────────────────────────────────────
ax_side.set_facecolor("#FFFFFF")
ax_side.add_patch(mpatches.FancyBboxPatch((0,0), 1, 1, boxstyle="square",
                  fc="#FFFFFF", ec="#CCCCCC", lw=1.5))

def sb_label(ax, y, text, size=8, bold=False, color="black"):
    ax.text(0.06, y, text, transform=ax.transAxes, fontsize=size,
            va="center", color=color,
            fontweight="bold" if bold else "normal")

def sb_widget(ax, y, h=0.038):
    ax.add_patch(mpatches.FancyBboxPatch((0.06, y - h/2), 0.88, h,
                 boxstyle="round,pad=0.01", fc="#F0F2F6", ec="#CCCCCC", lw=0.8,
                 transform=ax.transAxes))

sb_label(ax_side, 0.97, "⚙  Aggregate Planning", size=9.5, bold=True, color="#1F2D3D")
sb_label(ax_side, 0.92, "Cenário", size=8, bold=True)
sb_widget(ax_side, 0.88)
ax_side.text(0.08, 0.88, "Alta Sazonalidade  ▾", transform=ax_side.transAxes,
             fontsize=7.5, va="center", color="#444")

sb_label(ax_side, 0.83, "Demanda por período", size=7.5, bold=True)
for i, (lbl, val) in enumerate(zip(period_labels[:6], demand[:6])):
    y = 0.79 - i * 0.038
    sb_widget(ax_side, y, h=0.032)
    ax_side.text(0.08, y, f"{lbl}:", transform=ax_side.transAxes,
                 fontsize=6.5, va="center", color="#333")
    ax_side.text(0.7, y, str(int(val)), transform=ax_side.transAxes,
                 fontsize=6.5, va="center", ha="right", color="#000")

sb_label(ax_side, 0.54, "Estratégia", size=7.5, bold=True)
sb_widget(ax_side, 0.50)
ax_side.text(0.08, 0.50, "LP (Programação Linear)  ▾",
             transform=ax_side.transAxes, fontsize=6.5, va="center", color="#444")

sb_label(ax_side, 0.45, "Política de Escassez", size=7.5, bold=True)
for i, opt in enumerate(["● Backorders", "○ Sem Escassez", "○ Lost Sales"]):
    ax_side.text(0.08, 0.415 - i * 0.036, opt, transform=ax_side.transAxes,
                 fontsize=6.5, va="center",
                 color="#1a73e8" if i == 0 else "#666")

sb_label(ax_side, 0.32, "Variáveis Inteiras", size=7.5, bold=True)
ax_side.text(0.08, 0.295, "☐ Mão-de-obra inteira",
             transform=ax_side.transAxes, fontsize=6.5, va="center", color="#555")
ax_side.text(0.08, 0.260, "☐ Produção inteira",
             transform=ax_side.transAxes, fontsize=6.5, va="center", color="#555")

# Botão
ax_side.add_patch(mpatches.FancyBboxPatch((0.06, 0.185), 0.88, 0.045,
                  boxstyle="round,pad=0.01", fc="#1a73e8", ec="none",
                  transform=ax_side.transAxes))
ax_side.text(0.50, 0.207, "▶  Calcular", transform=ax_side.transAxes,
             ha="center", va="center", fontsize=8, color="white", fontweight="bold")

# ── Main area ────────────────────────────────────────────────────────────────
ax_main.set_facecolor("#F0F2F6")

# Título
ax_main.text(0.02, 0.96, "Aggregate Planning Dashboard", transform=ax_main.transAxes,
             fontsize=13, fontweight="bold", color="#1F2D3D")
ax_main.text(0.02, 0.91, "Universidade Federal Fluminense  ·  Eng. de Produção",
             transform=ax_main.transAxes, fontsize=8, color="#666")

# KPI cards
kpi_vals = [f"R$ 7,421,250", "5", "0", "2.1", "0"]
kpi_lbls = ["Custo Total", "Contrat.", "Demissões", "Est. Médio", "H. Extras"]
kpi_cols = ["#1a73e8", "#34a853", "#ea4335", "#fbbc04", "#673ab7"]
for i, (lbl, val, col) in enumerate(zip(kpi_lbls, kpi_vals, kpi_cols)):
    x0 = 0.02 + i * 0.195
    ax_main.add_patch(mpatches.FancyBboxPatch((x0, 0.79), 0.18, 0.09,
                      boxstyle="round,pad=0.01", fc="white", ec="#DDDDDD", lw=1,
                      transform=ax_main.transAxes))
    ax_main.text(x0 + 0.09, 0.845, val, transform=ax_main.transAxes,
                 ha="center", va="center", fontsize=9, fontweight="bold", color=col)
    ax_main.text(x0 + 0.09, 0.807, lbl, transform=ax_main.transAxes,
                 ha="center", va="center", fontsize=7, color="#666")

# Abas
tabs = ["Demanda", "Plano", "Prod & Estoque", "Mão-de-Obra",
        "Custos", "Comparação", "Teoria"]
for i, tab in enumerate(tabs):
    xpos = 0.02 + i * 0.136
    is_active = (i == 2)
    ax_main.add_patch(mpatches.FancyBboxPatch((xpos, 0.73), 0.13, 0.038,
                      boxstyle="round,pad=0.005",
                      fc="#1a73e8" if is_active else "white",
                      ec="#1a73e8" if is_active else "#CCCCCC", lw=0.8,
                      transform=ax_main.transAxes))
    ax_main.text(xpos + 0.065, 0.749, tab, transform=ax_main.transAxes,
                 ha="center", va="center", fontsize=6.5,
                 color="white" if is_active else "#444")

# Gráfico de produção empilhada (simulado)
import matplotlib.transforms as mtransforms
ax_chart = fig1.add_axes([0.21 + 0.01, 0.05, 0.60, 0.63])
ax_chart.set_facecolor("white")

months = period_labels
prod_rt = [r_lp["production"][t] for t in range(len(demand))]
prod_ot = [r_lp.get("overtime", [0]*len(demand))[t] for t in range(len(demand))]
prod_sub= [r_lp.get("subcontracting", [0]*len(demand))[t] for t in range(len(demand))]

x_pos = np.arange(len(months))
ax_chart.bar(x_pos, prod_rt, label="Tempo Regular", color="#4C72B0", alpha=0.85)
ax_chart.bar(x_pos, prod_ot, bottom=prod_rt, label="Horas Extras", color="#DD8452", alpha=0.85)
ax_chart.bar(x_pos, prod_sub, bottom=[a+b for a,b in zip(prod_rt, prod_ot)],
             label="Subcontr.", color="#55A868", alpha=0.85)
ax_chart.plot(x_pos, demand, "o--", color="#C44E52", linewidth=2,
              markersize=5, label="Demanda", zorder=5)
ax_chart.set_xticks(x_pos); ax_chart.set_xticklabels(months, fontsize=8.5)
ax_chart.set_ylabel("Unidades", fontsize=9)
ax_chart.set_title("Produção & Estoque — PL (Ótimo)", fontsize=10, fontweight="bold")
ax_chart.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax_chart.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax_chart.set_axisbelow(True)

# Mini tabela de KPIs à direita
ax_tbl = fig1.add_axes([0.83, 0.05, 0.17, 0.63])
ax_tbl.set_facecolor("white")
ax_tbl.axis("off")
ax_tbl.text(0.5, 0.97, "Plano Detalhado", ha="center", va="top",
            fontsize=8.5, fontweight="bold", color="#1F2D3D")
col_hdr = ["t", "Prod", "W", "Inv"]
col_w = [0.1, 0.32, 0.28, 0.3]
ys = np.linspace(0.90, 0.10, len(demand) + 1)
for j, (hdr, w) in enumerate(zip(col_hdr, [sum(col_w[:k+1]) - col_w[k]/2 for k in range(4)])):
    ax_tbl.text(w, ys[0], hdr, ha="center", va="center",
                fontsize=7, fontweight="bold", color="#1a73e8")
for i in range(len(demand)):
    bg = "#F8F9FA" if i % 2 == 0 else "white"
    ax_tbl.add_patch(mpatches.Rectangle((0, ys[i+1]-0.025), 1, 0.05,
                     fc=bg, ec="none", transform=ax_tbl.transAxes))
    row = [str(i+1),
           f"{r_lp['production'][i]:.0f}",
           f"{r_lp['workforce'][i]:.0f}",
           f"{r_lp['inventory'][i]:.0f}"]
    for j, (val, w) in enumerate(zip(row, [sum(col_w[:k+1]) - col_w[k]/2 for k in range(4)])):
        ax_tbl.text(w, ys[i+1], val, ha="center", va="center", fontsize=6.5, color="#333")

fig1_path = os.path.join(OUT, "fig_dashboard.pdf")
fig1.savefig(fig1_path, bbox_inches="tight", dpi=200)
print(f"Salvo: {fig1_path}")
plt.close(fig1)

print("Figuras geradas com sucesso.")

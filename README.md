# 📦 Aggregate Planning Dashboard

**An interactive tool to support the teaching of Aggregate Planning in Production Planning and Control (PPC) courses.**

The app is designed for classroom use and self-study. Students set up an aggregate planning scenario, apply the main strategies covered in the course and compare the results. The plan is recalculated whenever a parameter changes, so students can see right away how each decision affects production, inventory, workforce and costs.

---

## 🎓 Learning objectives

With this tool, students can:

- **Understand the trade-offs of aggregate planning**, such as changing the workforce, building inventory, using overtime or subcontracting.
- **Compare classic strategies** on the same data and see why each one fits some demand patterns better than others.
- **Compare heuristic plans with optimal ones.** Intuitive plans (Chase, Level, Mixed) can be set against the optimal plans from Linear Programming and the Transportation Method.
- **Run sensitivity analysis** by changing costs, capacities and shortage policies.
- **Interpret shadow prices** as the marginal cost of meeting one more unit of demand in each period.
- **Build their own plan** in Trial-and-Error mode and compare its cost with the optimal plan.

## 🧭 Available strategies

| Strategy | Core idea |
|---|---|
| **Chase Demand** | Production follows demand in every period. The workforce is adjusted through hiring and layoffs. |
| **Level Production** | Production runs at a constant rate equal to average demand. Inventory and backorders absorb demand variation. |
| **Mixed / Hybrid** | The workforce is only partly adjusted. The remaining gaps are covered with overtime and subcontracting. |
| **Linear Programming (LP)** | Optimization model that minimizes total cost. Supports integer variables (MILP) and reports shadow prices. |
| **Transportation Method** | Classic transportation-tableau formulation, with regular-time, overtime and subcontracting sources in each period. |
| **Trial-and-Error** | Students enter their own plan, and the tool computes its costs and flags any violations. |

## ✨ Features

- **Preset scenarios** (high seasonality, stable demand, demand spike) and **demand series upload** from CSV or Excel, with any number of periods.
- **Configurable parameters:** initial conditions, costs (regular time, hiring, layoffs, holding, backorders, overtime, subcontracting), capacity limits and integer variables.
- **Shortage policies:** backorders, no shortages or lost sales.
- **Period-by-period results:** detailed plan table, production and inventory charts, workforce dynamics and cost breakdown.
- **Strategy comparison:** total cost and a normalized profile (radar chart) for all strategies, using the same parameters.
- **Built-in theory guide**, which can be downloaded as HTML and used offline as course material.

## 🚀 Getting started

Requirements: Python 3.11 or later.

```bash
git clone https://github.com/helderhgc/aggregate_planing.git
cd aggregate_planing
pip install -r requirements.txt
streamlit run app.py
```

The app opens in the browser at `http://localhost:8501`.

The repository also includes a **Dev Container** configuration (`.devcontainer/`), so the project can be opened in GitHub Codespaces without installing anything locally.

### Demand file format

The file must have two columns, **Period** and **Demand**. In Excel, the data goes on the first sheet. A template is available at [`data/demand_template.xlsx`](data/demand_template.xlsx).

```csv
Period,Demand
Jan,420
Feb,390
Mar,450
```

## 🗂️ Project structure

```
app.py                      Streamlit interface (interactive dashboard)
solvers.py                  Implementation of the planning strategies
theory.html                 Theory guide (shown in the app and available for download)
data/default_scenario.json  Default scenario and preset scenarios
data/demand_template.xlsx   Spreadsheet template for importing demand
```

## 👨‍🏫 Authors

- **Helder Costa**, heldergc@id.uff.br
- **Leonardo Costa**, leo_costa@id.uff.br

Universidade Federal Fluminense (UFF), Brazil

## 📄 License

Released under the **GNU GPL v3** license. See [LICENSE](LICENSE).

---

### Sobre (Português)

O *Aggregate Planning Dashboard* é uma ferramenta interativa de apoio ao ensino de Planejamento Agregado na disciplina de Planejamento e Controle da Produção (PCP). O estudante monta um cenário de planejamento, aplica as estratégias clássicas (acompanhamento da demanda, produção nivelada, estratégia mista, Programação Linear, Método dos Transportes e tentativa e erro) e compara custos, força de trabalho, estoques e preços-sombra em tempo real. Para executar: `streamlit run app.py`.

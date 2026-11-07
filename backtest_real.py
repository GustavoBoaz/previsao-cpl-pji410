"""
PJI410 - Grupo 9 - Frente 1 (Modelos): backtest temporal na base real.

Treina SOMENTE com dados ate mai/2026, preve o CPL de jun e jul/2026
sem ver o resultado e compara com o realizado.
Baseline: media historica de CPL do canal (o que a agencia usa hoje).

Uso:  python backtest_real.py   (base_cpl_anonimizada.csv na mesma pasta)
Saidas na pasta resultados/:
  metricas_modelos.csv, erro_por_grupo.csv, cobertura_intervalo.csv,
  predicoes_teste.csv, filtros_aplicados.csv, fig_erro_modelos.png,
  modelo_final.joblib (usado pelo app Streamlit)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

SEED = 42
FIM_TREINO = "2026-05"
MESES_TESTE = ["2026-06", "2026-07"]
OUT = Path("resultados")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. leitura
df = pd.read_csv("base_cpl_anonimizada.csv")

# ---------------------------------------------------------------- 2. filtros (registrar no relatorio)
log = [("Base completa", len(df))]
df = df[df["moeda"] == "BRL"];                       log.append(("Somente BRL (remove 3 anunciantes EUA)", len(df)))
df = df[df["cpl"].notna()];                          log.append(("CPL calculavel (conversoes > 0)", len(df)))
df = df[df["tipo_conversao"] != "sem_conversao"];    log.append(("Remove tipo_conversao = sem_conversao", len(df)))
df = df[df["conversoes"] >= 3];                      log.append(("Conversoes >= 3 no mes", len(df)))
df = df[df["investimento"] >= 100];                  log.append(("Investimento >= R$ 100 no mes", len(df)))
pd.DataFrame(log, columns=["etapa", "linhas"]).to_csv(OUT / "filtros_aplicados.csv", index=False)

# ---------------------------------------------------------------- 3. variaveis
m = pd.to_datetime(df["mes"]).dt.month
df["mes_sin"] = np.sin(2 * np.pi * m / 12)
df["mes_cos"] = np.cos(2 * np.pi * m / 12)
df["log_invest"] = np.log1p(df["investimento"])
df["y"] = np.log(df["cpl"])                          # alvo em log (CPL tem cauda longa)

CAT = ["canal", "setor", "etapa_funil", "tipo_conversao", "objetivo"]
NUM = ["mes_sin", "mes_cos", "log_invest"]

treino = df[df["mes"] <= FIM_TREINO].sort_values("mes").reset_index(drop=True)
teste = df[df["mes"].isin(MESES_TESTE)].reset_index(drop=True)
print(f"Treino: {len(treino)} linhas ({treino.mes.min()} a {treino.mes.max()})")
print(f"Teste : {len(teste)} linhas ({', '.join(MESES_TESTE)})")

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ("num", StandardScaler(), NUM),
])

modelos = {
    "k vizinhos mais próximos": (KNeighborsRegressor(),
        {"m__n_neighbors": [5, 10, 20, 40], "m__weights": ["uniform", "distance"]}),
    "Árvore de regressão": (DecisionTreeRegressor(random_state=SEED),
        {"m__max_depth": [3, 5, 8, None], "m__min_samples_leaf": [5, 10, 20, 40]}),
    "Floresta aleatória": (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
        {"m__max_depth": [5, 10, None], "m__min_samples_leaf": [5, 10, 20]}),
    "Regressão por vetores de suporte": (SVR(),
        {"m__C": [0.3, 1, 3, 10], "m__epsilon": [0.1, 0.3]}),
}

# TimeSeriesSplit: cada dobra valida em meses POSTERIORES aos de treino (sem vazar o futuro).
cv = TimeSeriesSplit(n_splits=5)


def metricas(y_real, y_prev):
    return {"MAE": mean_absolute_error(y_real, y_prev),
            "RMSE": np.sqrt(mean_squared_error(y_real, y_prev)),
            "R2": r2_score(y_real, y_prev),
            "MAE_mediano": np.median(np.abs(y_real - y_prev))}


resultados, preds = [], {}

# ---------------------------------------------------------------- 4. baseline
media_canal = treino.groupby("canal")["cpl"].mean()
p_base_tr = treino["canal"].map(media_canal)
p_base_te = teste["canal"].map(media_canal)
r = metricas(teste["cpl"], p_base_te)
r.update({"modelo": "Média histórica do canal (baseline)",
          "R2_treino": r2_score(treino["cpl"], p_base_tr), "params": "-"})
resultados.append(r); preds[r["modelo"]] = p_base_te.values

# referencia extra (nao e o baseline oficial): media do grupo setor+canal+tipo_conversao
chave = ["setor", "canal", "tipo_conversao"]
med_grupo = treino.groupby(chave)["cpl"].mean().rename("mg")
p_grp_te = teste.join(med_grupo, on=chave)["mg"].fillna(p_base_te)
r = metricas(teste["cpl"], p_grp_te)
r.update({"modelo": "Média do grupo setor+canal+conversão (referência)",
          "R2_treino": np.nan, "params": "-"})
resultados.append(r); preds[r["modelo"]] = p_grp_te.values

# ---------------------------------------------------------------- 5. modelos
melhores = {}
for nome, (est, grid) in modelos.items():
    pipe = Pipeline([("pre", pre), ("m", est)])
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(treino[CAT + NUM], treino["y"])
    best = gs.best_estimator_
    melhores[nome] = best
    p_tr = np.exp(best.predict(treino[CAT + NUM]))
    p_te = np.exp(best.predict(teste[CAT + NUM]))
    r = metricas(teste["cpl"], p_te)
    r.update({"modelo": nome, "R2_treino": r2_score(treino["cpl"], p_tr),
              "params": str({k.replace('m__', ''): v for k, v in gs.best_params_.items()})})
    resultados.append(r); preds[nome] = p_te
    print(f"{nome:35s} MAE={r['MAE']:.2f}  {r['params']}")

res = pd.DataFrame(resultados)[["modelo", "MAE", "RMSE", "MAE_mediano", "R2_treino", "R2", "params"]]
res = res.rename(columns={"R2": "R2_teste"})
mae_base = res.loc[0, "MAE"]
res["reducao_MAE_vs_baseline_%"] = (1 - res["MAE"] / mae_base) * 100
res.round(3).to_csv(OUT / "metricas_modelos.csv", index=False)
print("\n", res.round(2).to_string(index=False))

# ---------------------------------------------------------------- 6. erro por grupo (com n)
nome_melhor = res.iloc[2:].sort_values("MAE").iloc[0]["modelo"]
teste["cpl_prev_baseline"] = preds["Média histórica do canal (baseline)"]
teste["cpl_prev_modelo"] = preds[nome_melhor]
teste["erro_abs_baseline"] = (teste["cpl"] - teste["cpl_prev_baseline"]).abs()
teste["erro_abs_modelo"] = (teste["cpl"] - teste["cpl_prev_modelo"]).abs()
teste.to_csv(OUT / "predicoes_teste.csv", index=False)

grupos = []
for dim in ["setor", "canal", "objetivo", "tipo_conversao", "etapa_funil"]:
    g = teste.groupby(dim).agg(n=("cpl", "size"), cpl_real_medio=("cpl", "mean"),
                               MAE_baseline=("erro_abs_baseline", "mean"),
                               MAE_modelo=("erro_abs_modelo", "mean")).reset_index()
    g = g.rename(columns={dim: "grupo"}); g.insert(0, "dimensao", dim)
    g["modelo_melhor_que_baseline"] = g["MAE_modelo"] < g["MAE_baseline"]
    grupos.append(g)
pd.concat(grupos).round(2).to_csv(OUT / "erro_por_grupo.csv", index=False)

# ---------------------------------------------------------------- 7. intervalo (conservador / otimista)
# Faixa empirica: quantis do residuo (em log) no treino, aplicados a previsao.
# otimista = CPL menor (p25), conservador = CPL maior (p75). Mede-se a cobertura real no teste.
best = melhores[nome_melhor]
res_log = treino["y"] - best.predict(treino[CAT + NUM])
q25, q75 = np.quantile(res_log, [0.25, 0.75])
q10, q90 = np.quantile(res_log, [0.10, 0.90])
pl = np.log(teste["cpl_prev_modelo"])
cob = pd.DataFrame([
    {"faixa": "p25-p75 (otimista-conservador)", "cobertura_esperada": 0.50,
     "cobertura_real_teste": ((teste["cpl"] >= np.exp(pl + q25)) & (teste["cpl"] <= np.exp(pl + q75))).mean()},
    {"faixa": "p10-p90", "cobertura_esperada": 0.80,
     "cobertura_real_teste": ((teste["cpl"] >= np.exp(pl + q10)) & (teste["cpl"] <= np.exp(pl + q90))).mean()},
])
cob.round(3).to_csv(OUT / "cobertura_intervalo.csv", index=False)
print("\nCobertura do intervalo no teste:\n", cob.round(3).to_string(index=False))

# ---------------------------------------------------------------- 8. figura
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(res)); w = 0.38
ax.bar(x - w/2, res["MAE"], w, label="MAE", color="#4a6fa5")
ax.bar(x + w/2, res["RMSE"], w, label="RMSE", color="#b8c7de")
ax.axhline(mae_base, ls="--", c="grey", lw=1)
for i, v in enumerate(res["MAE"]):
    ax.text(i - w/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
rot = [s.replace(" (baseline)", "\n(baseline)").replace(" (referência)", "\n(referência)")
        .replace("Regressão por vetores de suporte", "Vetores de\nsuporte") for s in res["modelo"]]
ax.set_xticks(x); ax.set_xticklabels(rot, fontsize=7.5)
ax.set_ylabel("Erro no teste jun-jul/2026 (R$)"); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "fig_erro_modelos.png", dpi=200)

# ---------------------------------------------------------------- 9. modelo final para o app
# Re-treina com toda a base filtrada (inclui jun-set/2026) para uso no prototipo.
final = melhores[nome_melhor]
final.fit(df[CAT + NUM], df["y"])
res_final = df["y"] - final.predict(df[CAT + NUM])
mae_grupo_bt = (teste.groupby(["setor", "canal", "tipo_conversao"])
                .agg(n=("cpl", "size"), mae=("erro_abs_modelo", "mean")).reset_index())
joblib.dump({"modelo": final, "nome": nome_melhor, "CAT": CAT, "NUM": NUM,
             "q": dict(zip(["p10", "p25", "p75", "p90"], np.quantile(res_final, [.1, .25, .75, .9]))),
             "mae_global_bt": float(res.set_index("modelo").loc[nome_melhor, "MAE"]),
             "mae_baseline_bt": float(mae_base),
             "mae_grupo_bt": mae_grupo_bt}, OUT / "modelo_final.joblib")
print(f"\nMelhor modelo: {nome_melhor}. Arquivos em {OUT}/")

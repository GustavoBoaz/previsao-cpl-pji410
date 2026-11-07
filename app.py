"""
PJI410 - Grupo 9 - Frente 2 (Prototipo): tela de previsao de CPL.

Rodar:  pip install streamlit scikit-learn pandas matplotlib joblib
        python backtest_real.py      # gera resultados/modelo_final.joblib
        streamlit run app.py

Cada bloco da tela atende um dos nove requisitos da entrevista (Apendice B, Quadro B.1).
"""
import time
from datetime import date
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Previsão de CPL", page_icon="📈", layout="wide")

ORIGEM = "Meta Ads e Google Ads via Windsor.ai"
CRITERIO = "campanhas do mesmo setor, canal e tipo de conversão"
MIN_CAMPANHAS = 10
MESES_ANTIGO = 6


@st.cache_data
def carregar_base():
    df = pd.read_csv("base_cpl_anonimizada.csv")
    # mesmos filtros do backtest
    df = df[(df.moeda == "BRL") & df.cpl.notna() & (df.tipo_conversao != "sem_conversao")
            & (df.conversoes >= 3) & (df.investimento >= 100)].copy()
    df["data"] = pd.to_datetime(df["mes"])
    return df


@st.cache_resource
def carregar_modelo():
    return joblib.load("resultados/modelo_final.joblib")


df = carregar_base()
pack = carregar_modelo()
modelo, CAT, NUM, q = pack["modelo"], pack["CAT"], pack["NUM"], pack["q"]

# ------------------------------------------------------------ cronometro (criterio de aceitacao)
with st.sidebar:
    st.subheader("Avaliação de usabilidade")
    st.caption("Cronometre a conferência da previsão para comparar com o processo em planilha.")
    if st.button("Iniciar conferência"):
        st.session_state["t0"] = time.time()
    if st.button("Concluir conferência") and "t0" in st.session_state:
        st.session_state["tempo"] = time.time() - st.session_state.pop("t0")
    if "t0" in st.session_state:
        st.info("Cronômetro em andamento…")
    if "tempo" in st.session_state:
        st.success(f"Tempo de conferência: {st.session_state['tempo']:.0f} s")

# ------------------------------------------------------------ 8. filtros
st.title("Previsão de custo por lead")
st.caption("Protótipo do sistema de apoio à decisão. Toda previsão mostra de onde vem e quanto erra.")

c1, c2, c3, c4, c5 = st.columns(5)
setor = c1.selectbox("Segmento (setor)", sorted(df.setor.unique()))
d1 = df[df.setor == setor]
canal = c2.selectbox("Canal", sorted(d1.canal.unique()))
d2 = d1[d1.canal == canal]
tipo = c3.selectbox("Tipo de conversão", sorted(d2.tipo_conversao.unique()))
d3 = d2[d2.tipo_conversao == tipo]
objetivo = c4.selectbox("Objetivo", sorted(d3.objetivo.unique()))
mercado = c5.selectbox("Mercado", ["BR"], help="A base só tem BR e EUA; o EUA (USD) ficou fora do modelo. "
                                               "Região mais fina não existe na base (limitação registrada).")

c6, c7 = st.columns(2)
etapa_padrao = d3[d3.objetivo == objetivo].etapa_funil.mode()
etapas = sorted(df.etapa_funil.unique())
etapa = c6.selectbox("Etapa de funil", etapas,
                     index=etapas.index(etapa_padrao.iloc[0]) if len(etapa_padrao) else 0)
invest = c7.number_input("Investimento mensal previsto (R$)", min_value=100.0, value=2000.0, step=100.0)
mes_prev = c7.selectbox("Mês da previsão", list(range(1, 13)), index=date.today().month % 12,
                        format_func=lambda m: ["jan", "fev", "mar", "abr", "mai", "jun", "jul",
                                               "ago", "set", "out", "nov", "dez"][m - 1])

# ------------------------------------------------------------ previsao + 6. intervalo
X = pd.DataFrame([{"canal": canal, "setor": setor, "etapa_funil": etapa, "tipo_conversao": tipo,
                   "objetivo": objetivo, "mes_sin": np.sin(2 * np.pi * mes_prev / 12),
                   "mes_cos": np.cos(2 * np.pi * mes_prev / 12), "log_invest": np.log1p(invest)}])
y_log = modelo.predict(X[CAT + NUM])[0]
esperado = np.exp(y_log)
otimista = np.exp(y_log + q["p25"])       # CPL menor
conservador = np.exp(y_log + q["p75"])    # CPL maior

ref = d3  # campanhas que sustentam a previsao (criterio: setor + canal + tipo de conversao)

st.divider()
k1, k2, k3 = st.columns(3)
k1.metric("Otimista", f"R$ {otimista:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
k2.metric("Esperado", f"R$ {esperado:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
k3.metric("Conservador", f"R$ {conservador:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
leads = invest / esperado
st.write(f"Com R$ {invest:,.0f} no mês, isso equivale a cerca de **{invest/conservador:.0f} a "
         f"{invest/otimista:.0f} leads** (esperado: {leads:.0f}).".replace(",", "."))
st.caption(f"Valores previstos pelo modelo ({pack['nome'].lower()}), não observados. "
           "A faixa vai do quartil 25 ao 75 dos erros do modelo.")

# erro do modelo no backtest para o grupo
mg = pack["mae_grupo_bt"]
linha = mg[(mg.setor == setor) & (mg.canal == canal) & (mg.tipo_conversao == tipo)]
if len(linha) and linha.n.iloc[0] >= 5:
    st.info(f"No backtest (treino até mai/2026, teste jun–jul/2026) o modelo errou em média "
            f"**R$ {linha.mae.iloc[0]:.2f}** neste grupo (n = {int(linha.n.iloc[0])} linhas de teste).")
else:
    n_txt = f"apenas {int(linha.n.iloc[0])}" if len(linha) else "nenhuma"
    st.info(f"Este grupo teve {n_txt} linha(s) no teste do backtest, pouco para medir o erro do grupo. "
            f"Erro médio geral do modelo: **R$ {pack['mae_global_bt']:.2f}** "
            f"(média histórica do canal: R$ {pack['mae_baseline_bt']:.2f}).")

# ------------------------------------------------------------ 9. avisos
n_camp = ref.campanha.nunique()
ultimo = ref.data.max()
meses_desde = (date.today().year - ultimo.year) * 12 + date.today().month - ultimo.month
q1, q3 = ref.cpl.quantile([.25, .75])
lim = q3 + 1.5 * (q3 - q1)
outl = ref[ref.cpl > lim]
if n_camp < MIN_CAMPANHAS:
    st.warning(f"Amostra pequena: só {n_camp} campanhas sustentam esta previsão (mínimo recomendado: {MIN_CAMPANHAS}).")
if meses_desde > MESES_ANTIGO:
    st.warning(f"Dado antigo: o registro mais recente deste grupo é de {ultimo:%m/%Y}, há {meses_desde} meses.")
if len(outl):
    st.warning(f"Outlier: {len(outl)} registro(s) com CPL acima de R$ {lim:.2f} (regra 1,5×IQR). "
               f"Eles aparecem no gráfico e puxam a média para cima.")

# ------------------------------------------------------------ 1-5. rastreabilidade + 7. dispersao
st.divider()
esq, dir_ = st.columns([1, 1.3])
with esq:
    st.subheader("De onde vem este número")
    st.markdown(f"""
- **Origem dos dados:** {ORIGEM}
- **Período analisado:** {ref.data.min():%m/%Y} a {ref.data.max():%m/%Y}
- **Base da previsão:** {n_camp} campanhas ({len(ref)} registros campanha × mês)
- **Critério de seleção:** {CRITERIO}
- **Como a conversão foi medida:** só eventos do tipo *{tipo.replace('_', ' ')}*; campanhas que
  contam outro evento não entram como referência
- **Mercado:** {mercado} (valores em R$)
""")
    st.caption(f"Mediana do CPL no grupo: R$ {ref.cpl.median():.2f} · média: R$ {ref.cpl.mean():.2f}")

with dir_:
    st.subheader("Dispersão do CPL nas campanhas usadas")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.boxplot(ref.cpl, vert=False, widths=.5, showfliers=True,
               flierprops=dict(marker="o", markerfacecolor="#c0392b", markersize=4, alpha=.6))
    jitter = np.random.default_rng(0).normal(1, .05, len(ref))
    ax.scatter(ref.cpl, jitter, s=8, alpha=.25, color="#4a6fa5")
    ax.axvspan(otimista, conservador, color="#f5b041", alpha=.25, label="faixa prevista")
    ax.axvline(esperado, color="#d35400", lw=2, label="esperado")
    ax.set_yticks([]); ax.set_xlabel("CPL realizado (R$)")
    if ref.cpl.max() / max(ref.cpl.min(), .01) > 30:
        ax.set_xscale("log"); ax.set_xlabel("CPL realizado (R$, escala log)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right", "left"]].set_visible(False)
    st.pyplot(fig)

with st.expander("Ver as campanhas de referência"):
    st.dataframe(ref[["mes", "anunciante", "campanha", "objetivo", "etapa_funil",
                      "investimento", "conversoes", "cpl"]].sort_values("mes", ascending=False),
                 hide_index=True, width="stretch")

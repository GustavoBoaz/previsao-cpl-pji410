# Frentes 1 e 2 · PJI410 Grupo 9

1. `pip install streamlit scikit-learn pandas matplotlib joblib`
2. `python backtest_real.py`  → gera a pasta `resultados/` (métricas, erro por grupo, figura, modelo)
3. `streamlit run app.py`     → abre a tela de previsão no navegador

`base_cpl_anonimizada.csv` precisa estar na mesma pasta.

## Filtros aplicados (registrar no relatório)
Base 2.831 → só BRL 2.676 → CPL calculável 2.330 → conversões ≥ 3: 2.157 → investimento ≥ R$ 100: 2.008.
Treino até mai/2026: 1.590 linhas · Teste jun–jul/2026: 205 linhas.
Ajuste de hiperparâmetros com TimeSeriesSplit (5 dobras) só dentro do treino.

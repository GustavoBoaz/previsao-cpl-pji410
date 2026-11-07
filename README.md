# previsao-cpl-pji410

Frentes 1 e 2 (Boaz) · PJI410 Grupo 9 — previsão de CPL (custo por lead) com backtest de modelos e app Streamlit.

## Estrutura

| Arquivo | Descrição |
|---|---|
| `base_cpl_anonimizada.csv` | Base de dados anonimizada (precisa estar na raiz do projeto) |
| `backtest_real.py` | Treina/avalia os modelos e gera a pasta `resultados/` |
| `app.py` | Tela de previsão (Streamlit) |
| `resultados/` | Métricas, erro por grupo, figura e `modelo_final.joblib` |
| `requirements.txt` | Dependências do projeto |

## Pré-requisitos

- Python 3.9 ou superior
- Git

## 1. Clonar o repositório

```bash
git clone https://github.com/GustavoBoaz/previsao-cpl-pji410.git
cd previsao-cpl-pji410
```

## 2. Criar e ativar o ambiente virtual (venv)

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> Se o PowerShell bloquear o script, rode antes: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Windows (cmd)**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

Com o venv ativo, o terminal mostra `(venv)` no início da linha. Para sair, use `deactivate`.

## 3. Instalar as dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Rodar a aplicação

**(Opcional) Regerar os resultados e o modelo** — o repositório já traz `resultados/modelo_final.joblib`, então este passo só é necessário se quiser refazer o backtest:

```bash
python backtest_real.py
```

Gera a pasta `resultados/` (métricas, erro por grupo, figura e modelo).

**Abrir o app de previsão:**

```bash
streamlit run app.py
```

O navegador abre automaticamente em http://localhost:8501. Para encerrar, pressione `Ctrl+C` no terminal.

> Rode os comandos sempre a partir da raiz do projeto, pois os scripts leem `base_cpl_anonimizada.csv` e `resultados/` por caminho relativo.

## Filtros aplicados (registrar no relatório)

Base 2.831 → só BRL 2.676 → CPL calculável 2.330 → conversões ≥ 3: 2.157 → investimento ≥ R$ 100: 2.008.

Treino até mai/2026: 1.590 linhas · Teste jun–jul/2026: 205 linhas.

Ajuste de hiperparâmetros com TimeSeriesSplit (5 dobras) só dentro do treino.

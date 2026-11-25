# Capital de risco de modelo (Pilar 2) · abordagem interna

App Streamlit que calcula o capital econômico para risco de modelo a partir do desvio medido de cada modelo em
indicadores internos de performance:

Capital_i = Exposição_i × D_i(α, h) × (1 + κ(nota_i))

- **Crédito (behavior de PD), por GH:** d = BadRate do GH − PD do GH; viés só se a PD sair do intervalo de
  Jeffreys de 95%; volatilidade líquida do ruído binomial, só se o teste de sobredispersão (χ² a 95%) rejeitar o ruído;
  PD* = PD + max(0, viés + z_α σ √h); D = Σ w_g max{K(PD*) − K(PD); (PD* − PD)·LGD}.
- **Mercado N2, por família de instrumento:** excesso do desvio do IPV sobre a tolerância da fonte,
  e = sinal(d)·max(0, |d| − τ); viés só se o teste t rejeitar média zero; D_f = quantil bilateral de |e|;
  D = Σ w_f D_f pelo valor justo; horizonte de 1 ano (IPV mensal).
- **Mercado N3:** sensibilidade a premissas alternativas razoavelmente possíveis (CPC 46, 93(h)(ii)) ou diferença
  para o modelo alternativo, levada ao quantil α: D = max(s, a) × z_α / z_c; horizonte de 1 ano.
- **Nota de risco do modelo κ:** capturada pela validação independente (matriz A a I), 0 na nota A e 100% sem
  validação ou vencida; multiplica o desvio por (1 + κ).
- **Tier:** define a validade (12 ou 36 meses) e, em crédito, o horizonte h do desvio (1 ou 3 anos).

## Como rodar

```
conda activate capitalrm
pip install -r requirements.txt
# para as fórmulas do memorial em LaTeX: MiKTeX ou TeX Live com pdflatex no PATH
streamlit run app.py
```

## Abas

- **Inventário:** modelos, notas, tiers, datas, exposições e indicadores de desvio (editável).
- **Resultado:** capital por família e por modelo, com a régua padronizada anterior como referência.
- **Desvio por modelo:** cálculo do desvio por família; upload da série de calibragem por GH (ID, Data, GH, PD do GH (%),
  Contratos, Bads, Saldo (R$ mi)) e de IPV por família (ID, Data, Família de instrumento,
  Desvio IPV (%), Tolerância (%), Valor justo (R$ mi)), que entram direto no cálculo.
- **Matriz e nota de risco κ:** nota de risco do modelo κ por nota de validação.
- **Premissa e RAS:** premissa prudencial do escopo e cobertura de validação do inventário total.
- **Cenários:** capital se todos, ou só os pendentes, fossem validados com cada nota.
- **Memorial de cálculo:** paper em PDF com racional, execução e referências.

## Arquivos

- `app.py`: interface.
- `interno.py`: inventário, séries, cálculo e cenários da abordagem interna.
- `rmint.py`: motor da abordagem interna (desvios, fórmula IRB por classe, nota de risco κ).
- `rmcap.py`, `calculo.py`: matriz de notas, validade, RAS e régua padronizada de referência.
- `memorial.py`: memorial de cálculo em PDF (reportlab).
- `formulas.py`: fórmulas do memorial compostas pelo pdflatex (Times, mathptmx) e embutidas em vetor no PDF;
  sem LaTeX instalado, usa o mathtext do matplotlib.
- `packages.txt`: pacotes do sistema para o Streamlit Cloud (LaTeX mínimo para as fórmulas).
- `dados/`: séries de exemplo de crédito e de IPV.
- `fonts/`: Liberation Serif (licença SIL OFL 1.1).

Inventário, séries e saldos ilustrativos.

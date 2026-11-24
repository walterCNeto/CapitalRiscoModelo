# Capital de risco de modelo (Pilar 2) · abordagem interna

App Streamlit que calcula o capital econômico para risco de modelo a partir do desvio medido de cada modelo em
indicadores internos de performance:

Capital_i = Exposição_i × D_i(α, h) × (1 + MC(nota_i))

- **Crédito (behavior de PD):** razão taxa de default observada / PD prevista; m = exp(μ + z_α σ √h);
  D = max{K(m·PD) − K(PD) (fórmula IRB da classe); (m − 1)·PD·LGD (provisão)}.
- **Mercado N2:** desvio do IPV (marcação contra fonte independente), quantil bilateral de |d| com d ~ N(μ, σ² h).
- **Mercado N3:** sensibilidade a premissas alternativas razoavelmente possíveis (CPC 46, 93(h)(ii)) ou diferença
  para o modelo alternativo, levada ao quantil α: D = max(s, a) × z_α / z_c × √h.
- **Margem de conservadorismo:** pela nota da validação (matriz A a I), 0 na nota A e 100% sem validação ou vencida.
- **Tier:** define a validade (12 ou 36 meses) e o horizonte h do desvio (1 ou 3 anos).

## Como rodar

```
conda activate capitalrm
pip install -r requirements.txt
streamlit run app.py
```

## Abas

- **Inventário:** modelos, notas, tiers, datas, exposições e indicadores de desvio (editável).
- **Resultado:** capital por família e por modelo, com a régua padronizada anterior como referência.
- **Desvio por modelo:** cálculo do desvio por família; upload das séries de crédito (ID, Data, PD prevista (%),
  Taxa de default (%)) e de IPV (ID, Data, Desvio IPV (%)) para preencher o inventário.
- **Matriz e margem:** margem de conservadorismo por nota.
- **Premissa e RAS:** premissa prudencial do escopo e cobertura de validação do inventário total.
- **Cenários:** capital se todos, ou só os pendentes, fossem validados com cada nota.
- **Memorial de cálculo:** paper em PDF com racional, execução e referências.

## Arquivos

- `app.py`: interface.
- `interno.py`: inventário, séries, cálculo e cenários da abordagem interna.
- `rmint.py`: motor da abordagem interna (desvios, fórmula IRB por classe, margem).
- `rmcap.py`, `calculo.py`: matriz de notas, validade, RAS e régua padronizada de referência.
- `memorial.py`: memorial de cálculo em PDF (reportlab).
- `formulas.py`: fórmulas do memorial em LaTeX (mathtext, fonte STIX) convertidas em vetor.
- `dados/`: séries de exemplo de crédito e de IPV.
- `fonts/`: Liberation Serif (licença SIL OFL 1.1).

Inventário, séries e saldos ilustrativos.

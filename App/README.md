# Capital de risco de modelo (app Streamlit)

App que calcula o add-on de Pilar 2 para risco de modelo pela régua padronizada de FPR:
FPR da faixa da nota × λ tier × λ nível × exposição = RWA; capital = 8% × RWA.

## Como rodar

```
pip install -r requirements.txt
streamlit run app.py
```

## Abas

- **Inventário**: tabela editável (risco, tier, nível de valor justo, nota, data da última validação, exposição).
  Linhas podem ser incluídas e excluídas. Nota sem data, ou data fora da validade (12 meses tier G, 36 meses tier M),
  é tratada como sem validação.
- **Resultado**: capital, RWA, FPR médio, cobertura de validação, resumo por risco, cálculo modelo a modelo e
  download em Excel.
- **Régua e matrizes de FPR**: régua de sete faixas e as matrizes nine box de FPR para crédito (tier G e M) e
  mercado (tier G e M, nível 3 e 2).
- **Cenários**: capital se todos os modelos, ou só os pendentes, fossem validados com cada nota.
- **Racional dos FPRs**: relatório com a derivação dos FPRs (P, S, dois canais, ω, cauda Beta-PERT,
  arredondamento, λ), com download em HTML.

## Arquivos

- `app.py`: interface.
- `calculo.py`: ligação entre o inventário editável e o motor; relatório e exportação.
- `rmcap.py`: motor da metodologia (calibração, régua, capital).
- `.streamlit/config.toml`: tema visual.

Parâmetros na barra lateral: data-base, λ tier M e λ nível 2. A régua vigente é a aprovada (derivada da calibração
padrão); a salvaguarda de ordenação é verificada e sinalizada quando os λ são alterados.
Inventário e saldos ilustrativos.

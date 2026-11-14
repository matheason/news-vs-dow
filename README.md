# News vs. Dow

Can daily world-news headlines predict whether the Dow Jones Industrial Average goes up or down — and if so, is the effect statistically and economically real?

Built for the Veritas AI high-school hackathon (finance track).
**This project is a prototype and is not financial advice.**

## Setup (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Put the Kaggle files `Combined_News_DJIA.csv` and `DJIA_table.csv` in `data/`, then open `notebooks/main.ipynb`.

## Layout
- `notebooks/main.ipynb` — main analysis (saved with outputs)
- `src/` — reusable functions
- `data/` — dataset files
- `SOURCES.md` — all data, libraries, models and AI tools used

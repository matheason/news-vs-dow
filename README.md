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
## Data (not included in this repo)
The dataset files are **not** committed to GitHub. To reproduce the results:

1. Download the dataset **Daily News for Stock Market Prediction** (by Aaron7sun) from Kaggle:
   https://www.kaggle.com/datasets/aaron7sun/stocknews
2. Unzip it and copy these two files into the `data/` folder at the project root:
   ```
   news-vs-dow/
   └── data/
       ├── Combined_News_DJIA.csv
       └── upload_DJIA_table.csv   (DJIA_table.csv also works)
   ```
   The third file in the download, `RedditNews.csv`, is not used.
3. Open `notebooks/main.ipynb` and run all cells.

You don't need the data just to read the results: the notebook is saved with all of its outputs.

## Layout
- `notebooks/main.ipynb` — main analysis (saved with outputs)
- `src/` — reusable functions
- `data/` — dataset files
- `SOURCES.md` — all data, libraries, models and AI tools used

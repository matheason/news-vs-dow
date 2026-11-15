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

**FinBERT scores (stage 3).** The first run scores all 49,718 headlines with FinBERT and caches them to `data/finbert_scores.csv`; later runs just read that file. The cache is not in the repo (like the dataset). Two ways to create it:
- **Locally on CPU:** just run `main.ipynb`. It took about 53 minutes on a 16-thread laptop CPU.
- **On Google Colab (GPU, a few minutes):** this is how the saved results were produced. Upload `notebooks/finbert_colab.ipynb` to Colab, switch the runtime to a T4 GPU, upload `Combined_News_DJIA.csv`, `src/data_utils.py` and `src/finbert.py` when asked, then put the downloaded `finbert_scores.csv` in `data/`.

Both ways use the same code (`src/finbert.py`). CPU and GPU scores differ only around the 6th decimal place, and every result in the notebook is the same with either one.

You don't need the data just to read the results: the notebook is saved with all of its outputs.

## Layout
- `notebooks/main.ipynb` — main analysis (saved with outputs)
- `notebooks/finbert_colab.ipynb` — optional: compute FinBERT scores on a Colab GPU
- `src/` — reusable functions
- `data/` — dataset files
- `SOURCES.md` — all data, libraries, models and AI tools used

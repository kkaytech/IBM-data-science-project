# SpaceX Falcon 9 Landing Prediction

Predicts whether SpaceX's Falcon 9 first stage will land successfully —
the key factor behind SpaceX's $62M launch cost vs competitors' $165M+.
Built using exploratory data analysis, feature engineering, and one-hot
encoding to prepare data for classification modelling.


## Why This Matters

SpaceX reuses the Falcon 9 first stage to dramatically cut costs. Predicting
whether a landing will succeed helps estimate launch pricing and informs
competitive analysis in the commercial space industry.


## What I Did

- Analysed 80+ feature dataset of real SpaceX launch records
- Identified key relationships between flight number, payload mass, orbit
  type, launch site, and landing success
- Engineered features: extracted year from date for trend analysis,
  created season-style groupings
- Applied OneHotEncoding to categorical columns (Orbit, LaunchSite,
  LandingPad, Serial) → expanded to 80 feature columns
- Cast all features to float64, ready for ML model input


## Key Findings

| Observation | Insight |
|---|---|
| Flight number vs success | Higher flight numbers = higher success rate (experience effect) |
| Payload mass vs site | VAFB site never launched payloads > 10,000 kg |
| Best orbit types | ES-L1, GEO, HEO, SSO → 100% success rate |
| Yearly trend | Success rate rose steadily from 2013 to 2020 |
| LEO orbit | Strong correlation between flight number and success |
| GTO orbit | No clear pattern — both outcomes equally likely |


## Feature Engineering Pipeline
Raw Launch Data (90 rows × 17 cols)
│
├── Select 12 key features
├── One-Hot Encode: Orbit, LaunchSite, LandingPad, Serial
└── Cast all to float64
│
features_one_hot (90 rows × 80 cols)
→ Ready for classification model

## Tech Stack

`Python` · `pandas` · `NumPy` · `matplotlib` · `seaborn`


## Data Source

[IBM Skills Network / SpaceX Launch Dataset](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv)
— sourced from SpaceX public launch records.


## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/spacex-landing-prediction
pip install pandas numpy matplotlib seaborn
jupyter notebook spacex_eda.ipynb
```

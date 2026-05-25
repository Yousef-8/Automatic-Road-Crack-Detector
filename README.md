## Model Weights

The trained weights (`raw_best.pt`, `prep_best.pt`) are not included
in this repository. Download them from:

https://www.kaggle.com/datasets/youseffuadahmed/road-crack-yolo-final-weights

Place them in the `models/` folder of the project before running the app:

road-crack-detector/
└── models/
    ├── raw_best.pt
    └── prep_best.pt

## Steps to run his project:

# 1. Extract the zip — it creates road-crack-detector/
# 2. Place the downlaoded model weights:
#    road-crack-detector/models/raw_best.pt
#    road-crack-detector/models/prep_best.pt

# 3. Create and activate virtualenv
python -m venv .venv
.venv\Scripts\activate        # for Windows
# source .venv/bin/activate   # for Mac

# 4. Install dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Run
streamlit run app.py

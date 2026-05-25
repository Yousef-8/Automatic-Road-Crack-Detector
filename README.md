## Model Weights

The trained weights (`raw_best.pt`, `prep_best.pt`) are not included
in this repository. Download them from:

https://www.kaggle.com/datasets/youseffuadahmed/road-crack-yolo-final-weights

Place the 2 files  in the `models/` folder of the project before running the app:

So it should be like:
road-crack-detector/ models/raw_best.pt

and

road-crack-detector/ models/prep_best.pt

## Steps to run this project:

 1. Extract the zip — it creates road-crack-detector/
 2. Place the downloaded model weights:
    
    road-crack-detector/models/raw_best.pt
    and
    road-crack-detector/models/prep_best.pt

4. Create and activate virtualenv
python -m venv .venv
.venv\Scripts\activate        # for Windows
 source .venv/bin/activate   # for Mac

 5. Install dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

 6. Run
streamlit run app.py

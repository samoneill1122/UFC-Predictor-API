import uvicorn
import joblib
import pandas as pd
import os
import hashlib
from fastapi import FastAPI, Depends, HTTPException, Header, status
from pydantic import BaseModel
# from methods import predict_fight

model = joblib.load("./models/random_forest_model.joblib")
fighter_name_to_id = joblib.load("./fighters_data/fighter_name_to_id.joblib")
fighters_accum_stats = joblib.load("./fighters_data/fighters_accum_stats.joblib")

app = FastAPI()

def verify_api_key(x_api_key: str = Header(...)):
    print(hashlib.sha256(x_api_key.encode("utf-8")).hexdigest())
    print(os.environ["UFC_PREDICTOR_API_KEY"])
    if hashlib.sha256(x_api_key.encode("utf-8")).hexdigest() != os.environ["UFC_PREDICTOR_API_KEY"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )

# Define request schema
class PredictRequest(BaseModel):
    fighter_b: str
    fighter_r: str

class PredictCardRequest(BaseModel):
    all_card_fight_details: list

@app.post("/predict-fight", dependencies=[Depends(verify_api_key)])
async def predict(data: PredictRequest):
    fighter_b = data.fighter_b.strip().title()
    fighter_r = data.fighter_r.strip().title()
    return do_predict_fight(fighter_b, fighter_r)

@app.post("/predict-card", dependencies=[Depends(verify_api_key)])
async def predict_card(data: PredictCardRequest):
    main_predictions = []
    prelim_predictions = []
    for fight_details in data.all_card_fight_details:
        fighter_b = fight_details["fighter_b"]
        fighter_r = fight_details["fighter_r"]
        fight_title = f"{fighter_b} vs. {fighter_r}"
        prediction = do_predict_fight(fighter_b, fighter_r)
        pred_obj = {
            "fight_id": fight_details["fight_id"],
            "fight_title": fight_title,
            "prediction": prediction,
            "ordering": fight_details["ordering"]
        }
        if fight_details["card_section"] == "main":
            main_predictions.append(pred_obj)
        else:
            prelim_predictions.append(pred_obj)

    return {
        "main": main_predictions,
        "prelims": prelim_predictions
    }

def do_predict_fight(fighter_b, fighter_r):
    if fighter_b not in fighter_name_to_id or fighter_r not in fighter_name_to_id:
        return "Unable to predict"

    return predict_fight([fighter_b, fighter_r])

def create_prediction_input(fighter_names):
    df_input = pd.DataFrame()
    r_fighter_id = fighter_name_to_id[fighter_names[0]]
    b_fighter_id = fighter_name_to_id[fighter_names[1]]

    r_fighter_dict = fighters_accum_stats[r_fighter_id]
    for stat in r_fighter_dict:
        df_input.at[0, "r_" + stat] = r_fighter_dict[stat]
    b_fighter_dict = fighters_accum_stats[b_fighter_id]
    for stat in b_fighter_dict:
        df_input.at[0, "b_" + stat] = b_fighter_dict[stat]

    return df_input

def order_pred_input_df(df):
    return df[['r_height', 'r_weight_lbs', 'r_reach_inches', 'b_height',
       'b_weight_lbs', 'b_reach_inches', 'r_sum_kd', 'r_sum_sig_landed',
       'r_sum_sig_atmp', 'r_sum_td_success', 'r_sum_td_atmp', 'r_sum_sub_att',
       'r_sum_rev', 'r_sum_sig_str_landed_head', 'r_sum_sig_str_atmp_head',
       'r_sum_sig_str_landed_body', 'r_sum_sig_str_atmp_body',
       'r_sum_sig_str_landed_leg', 'r_sum_sig_str_atmp_leg',
       'r_sum_sig_str_landed_distance', 'r_sum_sig_str_atmp_distance',
       'r_sum_sig_str_landed_clinch', 'r_sum_sig_str_atmp_clinch',
       'r_sum_sig_str_landed_ground', 'r_sum_sig_str_atmp_ground',
       'r_sum_ctrl_seconds', 'r_sum_rounds_fought', 'b_sum_kd',
       'b_sum_sig_landed', 'b_sum_sig_atmp', 'b_sum_td_success',
       'b_sum_td_atmp', 'b_sum_sub_att', 'b_sum_rev',
       'b_sum_sig_str_landed_head', 'b_sum_sig_str_atmp_head',
       'b_sum_sig_str_landed_body', 'b_sum_sig_str_atmp_body',
       'b_sum_sig_str_landed_leg', 'b_sum_sig_str_atmp_leg',
       'b_sum_sig_str_landed_distance', 'b_sum_sig_str_atmp_distance',
       'b_sum_sig_str_landed_clinch', 'b_sum_sig_str_atmp_clinch',
       'b_sum_sig_str_landed_ground', 'b_sum_sig_str_atmp_ground',
       'b_sum_ctrl_seconds', 'b_sum_rounds_fought']]

# first fighter name is blue fighter, second is red fighter
def predict_fight(fighter_names):
    df_input = create_prediction_input(fighter_names)
    df_input = order_pred_input_df(df_input)
    fight_pred = model.predict(df_input)
    if (fight_pred[0] == 0):
        return fighter_names[0]
    else:
        return fighter_names[1]

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
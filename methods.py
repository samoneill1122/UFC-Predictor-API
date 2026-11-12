import pandas as pd

# return a dataframe of all of the fighter stats up until a specific fight specified by the 
# date of said fight
def get_fighter_stats_df_raw(df, fighter_id, fight_date):
    return df[((df['r_fighter_id'] == fighter_id) | (df['b_fighter_id'] == fighter_id)) & (df['event_date'] < fight_date)]

# returns a data frame of the accumulated fighter stats and removes the opponent's stats
def get_accumulated_fighter_stats_df(df, fighter_id, sum_cols): 
    fighter_accum_stats_df = pd.DataFrame()
    for idx, row in df.reset_index(drop=True).iterrows():
        col_prefix = ""
        if fighter_id == df["b_fighter_id"].iloc[0]:
            col_prefix = "b_total"
        else:
            col_prefix = "r_total"

        for col in sum_cols: 
            if idx >= 1:
                fighter_accum_stats_df.at[0, col] = fighter_accum_stats_df.at[0, col] + df[col.replace("sum", col_prefix)].iloc[idx]
            else:
                fighter_accum_stats_df.at[0, col] = df[col.replace("sum", col_prefix)].iloc[idx]

        if idx >= 1:
            fighter_accum_stats_df.at[0, "sum_rounds_fought"] = fighter_accum_stats_df.at[0, "sum_rounds_fought"] + df["rounds_fought"].iloc[idx]
        else:
            fighter_accum_stats_df.at[0, "sum_rounds_fought"] = df["rounds_fought"].iloc[idx]

    return fighter_accum_stats_df

def convert_to_cms(val):
    if pd.isna(val):
        return val
    feet = int(val[0])
    inches = int(val[3:].replace("\"", ""))
    total_inches = (feet * 12) + inches

    return total_inches * 2.54

# returns a dataframe that replaces the stats of the fighter for each specific fight with their accumulative stats
# up until that fight
def build_accum_stats_df(df):
    accum_stats_dict = {}
    result_df = pd.DataFrame()

    sum_cols = ["sum_kd", "sum_sig_landed", "sum_sig_atmp", "sum_td_success", "sum_td_atmp", "sum_sub_att", "sum_rev", "sum_sig_str_landed_head", "sum_sig_str_atmp_head", "sum_sig_str_landed_body", "sum_sig_str_atmp_body", "sum_sig_str_landed_leg", "sum_sig_str_atmp_leg", "sum_sig_str_landed_distance", "sum_sig_str_atmp_distance", "sum_sig_str_landed_clinch", "sum_sig_str_atmp_clinch", "sum_sig_str_landed_ground", "sum_sig_str_atmp_ground", "sum_ctrl_seconds"]
    static_cols = ["height", "weight_lbs", "reach_inches"]

    for i, row in df.reset_index(drop=True).iterrows():
        r_fighter_id = row["r_fighter_id"]
        b_fighter_id = row["b_fighter_id"]

        for col in static_cols:
            result_df.at[i, "r_" + col] = row["r_" + col]
        for col in static_cols:
            result_df.at[i, "b_" + col] = row["b_" + col]

        calc_fighter_acum_stats(r_fighter_id, accum_stats_dict, result_df, sum_cols, static_cols, row, i, "r_")
        calc_fighter_acum_stats(b_fighter_id, accum_stats_dict, result_df, sum_cols, static_cols, row, i, "b_")
        set_target_col(row, r_fighter_id, b_fighter_id, result_df, i)

    return [result_df, accum_stats_dict]

def calc_fighter_acum_stats(fighter_id, accum_stats_dict, df, sum_cols, static_cols, row, curr_idx, fighter_prefix):
    if fighter_id in accum_stats_dict:
        for col in sum_cols:
            df.at[curr_idx, fighter_prefix + col] = accum_stats_dict[fighter_id][col]
            accum_stats_dict[fighter_id][col] = accum_stats_dict[fighter_id][col] + row[col.replace("sum", fighter_prefix + "total")]

        df.at[curr_idx, fighter_prefix + "sum_rounds_fought"] = accum_stats_dict[fighter_id]["sum_rounds_fought"]
        accum_stats_dict[fighter_id]["sum_rounds_fought"] = accum_stats_dict[fighter_id]["sum_rounds_fought"] + row["rounds_fought"]
    else:
        accum_stats_dict[fighter_id] = {}
        for col in sum_cols:
            accum_stats_dict[fighter_id][col] = row[col.replace("sum", fighter_prefix + "total")]
            df.at[curr_idx, fighter_prefix + col] = 0.0
        for col in static_cols:
            accum_stats_dict[fighter_id][col] = row[fighter_prefix + col]

        accum_stats_dict[fighter_id]["sum_rounds_fought"] = row["rounds_fought"]
        df.at[curr_idx, fighter_prefix + "sum_rounds_fought"] = 0.0

def set_target_col(row, r_fighter_id, b_fighter_id, df, curr_idx):
    if row["winner_id"] == r_fighter_id:
        df.at[curr_idx, "winner"] = "r"
    elif row["winner_id"] == b_fighter_id:
        df.at[curr_idx, "winner"] = "b"
    else:
        df.at[curr_idx, "winner"] = "n"

def create_prediction_input(fighter_names, fighter_name_to_id, accum_stats_dict):
    df_input = pd.DataFrame()
    r_fighter_id = fighter_name_to_id[fighter_names[0]]
    b_fighter_id = fighter_name_to_id[fighter_names[1]]

    r_fighter_dict = accum_stats_dict[r_fighter_id]
    for stat in r_fighter_dict:
        df_input.at[0, "r_" + stat] = r_fighter_dict[stat]
    b_fighter_dict = accum_stats_dict[b_fighter_id]
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
def predict_fight(fighter_names, fighter_name_to_id, accum_stats_dict, model):
    df_input = create_prediction_input(fighter_names, fighter_name_to_id, accum_stats_dict)
    df_input = order_pred_input_df(df_input)
    fight_pred = model.predict(df_input)
    return model.predict_proba(df_input)
    # if (fight_pred[0] == 0):
    #     return fighter_names[0]
    # else:
    #     return fighter_names[1]

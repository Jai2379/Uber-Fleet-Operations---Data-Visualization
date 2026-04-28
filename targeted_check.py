import sqlite3
import pandas as pd
from predict_cancellation import check_ride_risk 

def analyze_ride(target_id):
    # connect to the operations database
    conn = sqlite3.connect('uber_operations.db')

    # query the specific ride record
    query = "SELECT rowid, * FROM rides WHERE rowid = ?"
    df_row = pd.read_sql(query, conn, params=(target_id,))

    # validate record existence
    if df_row.empty:
        print(f"error: ride record {target_id} not found.")
        conn.close()
        return

    # extract features and strip the identifier
    ride_data = df_row.iloc[0].drop('rowid').to_dict() 

    # process through the inference engine
    print(f"--- analyzing operations record id: {target_id} ---")
    print(check_ride_risk(ride_data))
    
    conn.close()

# execution execution
if __name__ == "__main__":
    analyze_ride(1)
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

dummy_data    = False                   
feature_file  = "wallet_features.csv"   
output_file   = "wallet_scores.csv"     


feature_cols = ["tx_count", "avg_amount", "total_amount", "avg_fee", "avg_time_gap_seconds"]
Contamination = 0.05  # it is percentage that how much is suspicious
Random_State = 42

#---------------------Now we make fake wallet features for testing, 
def make_dummy_features(n_normal=200, n_suspicious=10, seed=42):
    rng = np.random.default_rng(seed)
    
    normal = pd.DataFrame({
        "wallet":       [f"wallet_{i:04d}" for i in range(n_normal)],
        "tx_count":     rng.integers(1,30,n_normal),
        "avg_amount":   rng.uniform(0.05,2.0,n_normal).round(4),
        "avg_fee":      rng.uniform(0.0001,0.001,n_normal).round(6),
        "avg_time_gap": rng.uniform(300,86400,n_normal).round(1) #second        
    })
    normal["total_amount"] = (normal["tx_count"]*normal["avg_amount"]).round(4)
    
    
    #suspicious wallets
    susp = pd.DataFrame({
        "wallet":       [f"wallet_S{i:03d}" for i in range(n_suspicious)],
        "tx_count":     rng.integers(300,800,n_suspicious),
        "avg_amount":   rng.uniform(0.0001,0.002,n_suspicious).round(6),
        "avg_fee":      rng.uniform(0.00001,0.00005,n_suspicious).round(7),
        "avg_time_gap": rng.uniform(1,30,n_suspicious).round(1)                      #bahut fast        
    })
    
    susp["total_amount"] = (susp["tx_count"]*susp["avg_amount"]).round(4)            #concatination of normal and suspicious wallets
    
    df = pd.concat([normal,susp], ignore_index = True)
    
    df = df.sample(frac=1, random_state= seed).reset_index(drop=True)                #mixing of suspicious and normal wallets 
    return df

#--------------- Checking data is dummy of featured.

if dummy_data:
    print("We are using dummy data for testing purpose.")
    df = make_dummy_features()
else: 
    print(f"Real feature loading...........: {feature_file}")
    df = pd.read_csv(feature_file)

print(f"          Total wallets: {len(df)}")
#------------safety check (Checking all the columns are in feature file or not.)
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise SystemExit(
        f"Error: These columns are not in feature file: {missing}\n"
        f"       Actual columns of file: {list(df.columns)}\n"
    )


#------------Code for model Training 
X = df[feature_cols].values

model = IsolationForest(
        n_estimators=100,
        contamination=Contamination,
        random_state=Random_State,
    )
model.fit(X)
joblib.dump(model, 'isolation_forest_model.pkl')
 
#----------------Scores
raw = -model.decision_function(X)

low, high = raw.min(),raw.max()

if high>low:
    suspicion = (raw -low) / (high- low) * 100
else:
    suspicion = np.zeros_like(raw)

df["suspicion_score"] = suspicion.round(2)

#predict(): -1 = anomaly, 1 = normal ->change in yes/no column
df["is_anomaly"] = np.where(model.predict(X) == -1,"Yes","No")

#------------------------------------------------------------------------------Sorting (like we are giving ranks, which one suspicious that is locate at the top)

df = df.sort_values("suspicion_score",ascending=False).reset_index(drop=True)
df.to_csv(output_file,index=False)

print(f"\n>>Finally Completed!!  here is : '{output_file}'.")
n_flagged = int((df["is_anomaly"] == "Yes").sum())
print("Top 10 most suspicious wallets: ")
cols_to_show = ["wallet","tx_count","avg_amount","suspicion_score","is_anomaly"]

print(df[cols_to_show].head(10).to_string(index=False))

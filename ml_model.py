import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def get_ml_signal(csv_path="RELIANCE_BO_ml_data.csv"):
        
    try:
        df = pd.read_csv(csv_path, index_col='Date')
    except FileNotFoundError:
        return {"ml_signal": "ERROR", "ml_confidence": 0.0}

    features = ['Returns', 'HL_Pct', 'MA50', 'MA200', 'RSI', 'High_Volume']
    X = df[features]
    y = df['Target']

    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    
    X_live = X.iloc[-1:] 

    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    
    probabilities = model.predict_proba(X_live)[0]
    
    prob_down = probabilities[0] * 100  
    prob_up = probabilities[1] * 100


    if prob_up >= 60.0:
        signal = "BUY"
        confidence = prob_up
    elif prob_down >= 60.0:
        signal = "SELL"
        confidence = prob_down
    else:
        signal = "HOLD"
        confidence = max(prob_up, prob_down) 

    print(f"[ML Engine] Signal Generated: {signal} ({confidence:.2f}% confidence)")
    
    return {
        "ml_signal": signal,
        "ml_confidence": round(confidence, 2)
    }

if __name__ == "__main__":
    result = get_ml_signal()
    print(result)
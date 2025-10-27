import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data_path):
    try:
        df = pd.read_csv(data_path, index_col=False)
    except Exception as e:
        raise FileNotFoundError(f"Error in file path {e}")
    
    x = df.drop('target', axis=1)
    y = df['target']

    scaler = StandardScaler()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


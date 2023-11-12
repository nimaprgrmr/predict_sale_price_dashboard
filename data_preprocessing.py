import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def make_year(column):
    year = column.split('-')[0]
    return int(year)


def make_month(column):
    month = column.split('-')[1]
    return int(month)


def make_day(column):
    day = column.split('-')[2][0:2]
    return int(day)


def make_date(column):
    date = column.split(" ")[0]
    return date


def convert_prd(column):
    if column == 401:
        return 1397
    if column == 901:
        return 1398
    if column == 1001:
        return 1399
    if column == 1101:
        return 1400
    if column == 1201:
        return 1401
    elif column == 1301:
        return 1402


def make_prd(column):
  id_prd = column.split('-')[3]
  return int(id_prd)


def read_data(path='./datasets/data_sale_bamland.csv'):
    df = pd.read_csv(path, header=None)
    columns = ['date', 'id_prd_to_plc', 'id_br', 'amount', 'price']
    df.columns = columns

    df['total_price'] = df['price'] * df['amount']
    df['year'] = df['date'].apply(make_year)
    df['month'] = df['date'].apply(make_month)
    df['day'] = df['date'].apply(make_day)
    df['date'] = df['date'].apply(make_date)
    df['id_prd_to_plc'] = df['id_prd_to_plc'].apply(convert_prd)

    # Create date column for splitting
    df['date'] = df['date'] + '-' + df['id_prd_to_plc'].astype(str)

    # Group by day
    new_data = df.groupby(df['date'], as_index=False).sum(numeric_only=True)
    new_data['id_br'] = 51338  # id bamland branch

    # Create again columns after make new df for each day sales
    new_data['id_br'] = new_data['id_br'].astype(int)
    new_data = new_data.drop(['year', 'month', 'day'], axis=1)
    new_data['year'] = new_data['date'].apply(make_year)
    new_data['month'] = new_data['date'].apply(make_month)
    new_data['day'] = new_data['date'].apply(make_day)
    new_data['id_prd_to_plc'] = new_data['date'].apply(make_prd)

    return new_data


def train_model(new_data):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler

    X = new_data[['id_prd_to_plc', 'year', 'month', 'day']]
    y = new_data['total_price']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1e-6, random_state=42)

    np.random.seed(42)
    modelRFR = RandomForestRegressor(2000, criterion='absolute_error')
    modelRFR.fit(X_train, y_train)

    return modelRFR, scaler


def save_model(model, scaler, path_model, path_scaler):
    import pickle

    pickle.dump(model, open(path_model, "wb"))
    pickle.dump(scaler, open(path_scaler, "wb"))

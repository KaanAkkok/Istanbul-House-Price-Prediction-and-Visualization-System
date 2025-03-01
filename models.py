import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.model_selection import KFold

def evaluate_model(model, X, y, X_scaled, needs_scaling):

    if needs_scaling:
        X_eval = X_scaled.to_numpy()
    else:
        X_eval = X.to_numpy()
    y_array = y.to_numpy()

    cv_scores = cross_val_score(model, X_eval, y_array, cv=5, scoring='r2')

    X_train, X_test, y_train, y_test = train_test_split(X_eval, y_array, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    prediction_std = None
    if hasattr(model, 'estimators_'):
        if isinstance(model, RandomForestRegressor):
            predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
            prediction_std = np.std(predictions, axis=0).mean()
        elif isinstance(model, GradientBoostingRegressor):
            staged_preds = np.array(list(model.staged_predict(X_test)))
            prediction_std = np.std(staged_preds, axis=0).mean()

    return {
        'cv_score_mean': cv_scores.mean(),
        'cv_score_std': cv_scores.std(),
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'prediction_std': prediction_std
    }

def train_models():

    df = pd.read_csv('house_prices.csv', sep=';', encoding='utf-8')
    df['Fiyat'] = df['Fiyat'].str.replace(' TL', '').str.replace('.', '').str.replace(',', '.').astype(float)

    X = df[['m² (Brüt)', 'Oda Sayısı']]
    X = pd.concat([
        X[['m² (Brüt)']],
        pd.get_dummies(df['Oda Sayısı'], prefix='oda'),
        pd.get_dummies(df['Bölge'], prefix='bolge')
    ], axis=1)
    y = df['Fiyat']

    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    X_array = X.to_numpy()

    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled['m² (Brüt)'] = scaler.fit_transform(X[['m² (Brüt)']])
    X_scaled_array = X_scaled.to_numpy()

    model_metrics = {}

    lr_model = LinearRegression()
    lr_model.fit(X_scaled_array, y)
    model_metrics['linear'] = evaluate_model(lr_model, X, y, X_scaled, True)

    rf_model = RandomForestRegressor(
        n_estimators=500, max_depth=10,
        min_samples_split=10, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_array, y)
    model_metrics['random_forest'] = evaluate_model(rf_model, X, y, X_scaled, False)

    svr_model = SVR(kernel='rbf', C=1000.0, epsilon=0.1, gamma='scale')
    svr_model.fit(X_scaled_array, y)
    model_metrics['svr'] = evaluate_model(svr_model, X, y, X_scaled, True)

    gb_model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1,
        max_depth=5, min_samples_split=5,
        min_samples_leaf=3, random_state=42
    )
    gb_model.fit(X_array, y)
    model_metrics['gradient_boosting'] = evaluate_model(gb_model, X, y, X_scaled, False)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.1,
        max_depth=5, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for train_idx, val_idx in kf.split(X_array):
        X_train_cv, X_val_cv = X_array[train_idx], X_array[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        xgb_model.fit(X_train_cv, y_train_cv)
        y_pred_cv = xgb_model.predict(X_val_cv)
        cv_scores.append(r2_score(y_val_cv, y_pred_cv))

    X_train, X_test, y_train, y_test = train_test_split(X_array, y, test_size=0.2, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    model_metrics['xgboost'] = {
        'cv_score_mean': np.mean(cv_scores),
        'cv_score_std': np.std(cv_scores),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'prediction_std': None
    }

    with open('house_price_models.pkl', 'wb') as f:
        pickle.dump({
            'linear': (lr_model, X_scaled.columns, scaler),
            'random_forest': (rf_model, X.columns, None),
            'svr': (svr_model, X_scaled.columns, scaler),
            'gradient_boosting': (gb_model, X.columns, None),
            'xgboost': (xgb_model, X.columns, None),
            'metrics': model_metrics
        }, f)

    return {
        'linear': (lr_model, X_scaled.columns),
        'random_forest': (rf_model, X.columns),
        'svr': (svr_model, X_scaled.columns),
        'gradient_boosting': (gb_model, X.columns),
        'xgboost': (xgb_model, X.columns)
    }

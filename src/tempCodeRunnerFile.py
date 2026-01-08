preds = model.predict(X_val.values)

rmse = np.sqrt(mean_squared_error(y_val, preds))
r2 = r2_score(y_val, preds)

print(f"R2   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")
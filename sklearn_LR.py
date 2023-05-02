import pandas as pd
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':

    # Load data
    df = pd.read_csv("datasets/df_final_combined_version3.csv")

    # Define variables
    X = df[["mask", "closed_border", "social_restrictions", "space_closures", "state_of_emergency", "testing_screening"]]
    y = df["New_deaths"]

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Print coefficients
    print("Intercept: ", model.intercept_)
    print("Coefficients: ", model.coef_)







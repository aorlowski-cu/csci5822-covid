import numpy as np
import pandas as pd

np.random.seed(123)

# Define the number of observations
n = 10000

# Define the intervention methods
mask = np.random.binomial(n=1, p=0.1, size=n)
closed_border = np.random.binomial(n=1, p=0.2, size=n)
social_restrictions = np.random.binomial(n=1, p=0.1, size=n)
space_closures = np.random.binomial(n=1, p=0.2, size=n)
state_of_emergency = np.random.binomial(n=1, p=0.3, size=n)
testing_screening = np.random.binomial(n=1, p=0.6, size=n)

# Define the prior distributions
mask_prior = np.random.binomial(n=1, p=0.8, size=n)
closed_border_prior = np.random.binomial(n=1, p=0, size=n)
social_restrictions_prior = np.random.binomial(n=1, p=0.6, size=n)
space_closures_prior = np.random.binomial(n=1, p=0.8, size=n)
state_of_emergency_prior = np.random.binomial(n=1, p=0, size=n)
testing_screening_prior = np.random.binomial(n=1, p=0.5, size=n)
new_cases_prior = np.random.normal(loc=0, scale=1, size=n)
new_deaths_prior = np.random.normal(loc=0, scale=1, size=n)

# Define the simulated dataset
df = pd.DataFrame({
    "mask": mask,
    "closed_border": closed_border,
    "social_restrictions": social_restrictions,
    "space_closures": space_closures,
    "state_of_emergency": state_of_emergency,
    "testing_screening": testing_screening,
    "mask_prior": mask_prior,
    "closed_border_prior": closed_border_prior,
    "social_restrictions_prior": social_restrictions_prior,
    "space_closures_prior": space_closures_prior,
    "state_of_emergency_prior": state_of_emergency_prior,
    "testing_screening_prior": testing_screening_prior,
    "New_cases": new_cases_prior,
    "New_deaths": new_deaths_prior,
})
df.to_csv("df_manufacture2.csv", index=False)

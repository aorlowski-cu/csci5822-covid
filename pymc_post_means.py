import pandas as pd
import pymc as pm

df = pd.read_csv("datasets/df_final_combined_version3.csv")

with pm.Model() as model:
    mask_prior = pm.Bernoulli("mask_prior", 0.5)
    closed_border_prior = pm.Bernoulli("closed_border_prior", 0.5)
    social_restrictions_prior = pm.Bernoulli("social_restrictions_prior", 0.5)
    space_closures_prior = pm.Bernoulli("space_closures_prior", 0.5)
    state_of_emergency_prior = pm.Bernoulli("state_of_emergency_prior", 0.5)
    testing_screening_prior = pm.Bernoulli("testing_screening_prior", 0.5)
    
    new_cases_prior = pm.Normal("new_cases_prior", mu=0, tau=1)
    new_deaths_prior = pm.Normal("new_deaths_prior", mu=0, tau=1)
    
    mask_likelihood = pm.Bernoulli("mask_likelihood", p=mask_prior, observed=df["mask"])
    closed_border_likelihood = pm.Bernoulli("closed_border_likelihood", p=closed_border_prior, observed=df["closed_border"])
    social_restrictions_likelihood = pm.Bernoulli("social_restrictions_likelihood", p=social_restrictions_prior, observed=df["social_restrictions"])
    space_closures_likelihood = pm.Bernoulli("space_closures_likelihood", p=space_closures_prior, observed=df["space_closures"])
    state_of_emergency_likelihood = pm.Bernoulli("state_of_emergency_likelihood", p=state_of_emergency_prior, observed=df["state_of_emergency"])
    testing_screening_likelihood = pm.Bernoulli("testing_screening_likelihood", p=testing_screening_prior, observed=df["testing_screening"])
    
    new_cases_likelihood = pm.Normal("new_cases_likelihood", mu=new_cases_prior, tau=1, observed=df["New_cases"])
    new_deaths_likelihood = pm.Normal("new_deaths_likelihood", mu=new_deaths_prior, tau=1, observed=df["New_deaths"])

    trace = pm.sample(1000)
    
mask_posterior = trace["mask_prior"].mean()

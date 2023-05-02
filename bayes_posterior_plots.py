import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


if __name__ == '__main__':
    np.random.seed(123)

    # Load the simulated dataset
    #df = pd.read_csv("datasets/df_manufacture2.csv")
    df = pd.read_csv("datasets/df_final_combined_version3.csv")

    

    # Define the PyMC3 model
    with pm.Model() as model:
        # Priors for the intervention effects
        beta_mask = pm.Normal("beta_mask", mu=0, sigma=1)
        beta_closed_border = pm.Normal("beta_closed_border", mu=0, sigma=1)
        beta_social_restrictions = pm.Normal("beta_social_restrictions", mu=0, sigma=1)
        beta_space_closures = pm.Normal("beta_space_closures", mu=0, sigma=1)
        beta_state_of_emergency = pm.Normal("beta_state_of_emergency", mu=0, sigma=1)
        beta_testing_screening = pm.Normal("beta_testing_screening", mu=0, sigma=1)
        
        # Linear model for new cases
        mu = (
            beta_mask * df["mask"] + 
            beta_closed_border * df["closed_border"] + 
            beta_social_restrictions * df["social_restrictions"] + 
            beta_space_closures * df["space_closures"] + 
            beta_state_of_emergency * df["state_of_emergency"] + 
            beta_testing_screening * df["testing_screening"]
        )
        
        # Likelihood for new cases
        sigma = pm.Exponential("sigma", 1)
        new_cases = pm.Normal("new_cases", mu=mu, sigma=sigma, observed=df["New_cases"])
      
            
      # Sampling
        trace = pm.sample(1000, tune=1000)
        print(pm.summary(trace))

        # plot posterior distributions and trace plots for only the `beta_mask` variable
        #az.plot_trace(trace, var_names=["beta_mask"])
        print(pm.summary(trace, var_names=["beta_mask"]))

        def plot_individual_traces(trace, var_names):
            for var_name in var_names:
                az.plot_trace(trace, var_names=[var_name], legend=True)
                plt.gca().set_ylabel(var_name)
                plt.gca().set_xlabel('Iteration')
                plt.title(f"Trace plot for {var_name}")
                plt.show()

        plot_individual_traces(trace, ['beta_mask', 'beta_closed_border', 'beta_social_restrictions','beta_space_closures','beta_state_of_emergency','beta_testing_screening'])



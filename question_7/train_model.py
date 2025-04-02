import pandas as pd
import numpy as np
from scipy.stats import beta

# Load segmented customer dataset
df = pd.read_csv("data/processed/Q7_banking_marketing_train_segmented.csv")

# --------------------------------------------
# Define Banking Campaign Variants
# --------------------------------------------
campaign_variants = [
    "New Account Bonus",
    "Spend-to-Earn Rewards",
    "Free Banking Services",
    "Loyalty Rewards Program"
]
timing_variants = ["Morning", "Afternoon", "Evening"]
channel_variants = ["Email", "SMS", "Push Notification"]

# Initialize parameters for Thompson Sampling (Success/Failure for each campaign)
segment_list = df["customer_segment"].unique()

# Bayesian Priors for Each Segment
historical_performance = df.groupby("customer_segment")["conversion_binary"].mean()
segment_campaign_params = {
    segment: {
        variant: {
            "alpha": max(1, historical_performance[segment] * 10),  
            "beta": max(1, (1 - historical_performance[segment]) * 10)
        } for variant in campaign_variants
    }
    for segment in segment_list
}

segment_timing_params = {
    segment: {variant: {"alpha": 1, "beta": 1} for variant in timing_variants}
    for segment in segment_list
}
segment_channel_params = {
    segment: {variant: {"alpha": 1, "beta": 1} for variant in channel_variants}
    for segment in segment_list
}

# --------------------------------------------
# Define Dynamic Fatigue Score Handling
# --------------------------------------------
low_fatigue_threshold = df["fatigue_score"].quantile(0.33)  
high_fatigue_threshold = df["fatigue_score"].quantile(0.66)  

# --------------------------------------------
# Function to Recommend the Best Campaign for Each Customer
# --------------------------------------------
def recommend_campaign(customer):
    """
    Selects the best campaign, timing, and channel based on:
    - Thompson Sampling with Bayesian Updating
    - Dynamic Fatigue Handling
    - Minimum Subscription Rate Floor
    """

    customer_segment = customer["customer_segment"]
    fatigue_score = customer["fatigue_score"]
    conversion_rate = max(customer["conversion_rate"], 0.01)  
    best_contact_time = customer["best_contact_time"]

    # Dynamically categorize fatigue levels
    if fatigue_score <= low_fatigue_threshold:
        fatigue_level = "low"
    elif fatigue_score >= high_fatigue_threshold:
        fatigue_level = "high"
    else:
        fatigue_level = "medium"

    # Adjust campaign selection using fatigue score dynamically
    adjusted_campaigns = campaign_variants.copy()
    if fatigue_level == "high":  
        adjusted_campaigns.remove("Spend-to-Earn Rewards")  
    elif fatigue_level == "low":
        adjusted_campaigns.append("New Account Bonus")  

    # Sample from Beta distributions for campaign selection
    best_campaign = max(
        adjusted_campaigns, key=lambda x: beta.rvs(
            max(segment_campaign_params[customer_segment][x]["alpha"] + conversion_rate * 10, 1),  
            max(segment_campaign_params[customer_segment][x]["beta"] + 5, 1)  # Small smoothing term
        )
    )

    # Select best timing based on past engagement
    if best_contact_time in timing_variants:
        best_timing = best_contact_time  
    else:
        best_timing = max(
            timing_variants, key=lambda x: beta.rvs(
                max(segment_timing_params[customer_segment][x]["alpha"], 1),
                max(segment_timing_params[customer_segment][x]["beta"], 1)
            )
        )

    # Sample from Beta distributions for channel selection
    best_channel = max(
        channel_variants, key=lambda x: beta.rvs(
            max(segment_channel_params[customer_segment][x]["alpha"], 1),
            max(segment_channel_params[customer_segment][x]["beta"], 1)
        )
    )

    return best_campaign, best_timing, best_channel

# --------------------------------------------
# Generate Recommendations for Each Customer
# --------------------------------------------
df[["recommended_campaign", "recommended_timing", "recommended_channel"]] = df.apply(recommend_campaign, axis=1, result_type="expand")

# Save recommendations
df.to_csv("data/processed/Q7_customer_campaign_recommendations_final.csv", index=False)
print("Final Optimized Banking Campaign Recommendations Generated & Saved!")


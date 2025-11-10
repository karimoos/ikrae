import pandas as pd
import numpy as np

# IKrae - Data Preparation and Graph Construction 
#Load your dataset
df = pd.read_csv("user_interactions.csv")

# columns: ['user_id', 'timestamp', 'device']
# If raw interaction logs are separate, first aggregate:
# df = logs.groupby('user_id').agg({'timestamp':'count', 'device': lambda x: x.mode()[0]}).reset_index()
# df.rename(columns={'timestamp': 'interaction_count'}, inplace=True)

# Step 1: Exclude inactive and automated traces
df = df[df['interaction_count'] >= 10]                         # remove users with <10 interactions
df = df[df['avg_response_time'] >= 1.0]                        # remove <1s automated traces

# Step 2: Compute percentiles to understand activity distribution
total_learners = len(df)
print(f"Initial learners: {total_learners}")

# Step 3: Categorize learners by activity level
def categorize_activity(n):
    if n > 1000:
        return 'high'
    elif n >= 200:
        return 'medium'
    else:
        return 'low'

df['activity_level'] = df['interaction_count'].apply(categorize_activity)

# Step 4: Stratified sampling
sample_size = 50000
strata = {'high': 0.4, 'medium': 0.4, 'low': 0.2}

sampled_df = pd.concat([
    df[df['activity_level'] == 'high'].sample(frac=strata['high'], random_state=42),
    df[df['activity_level'] == 'medium'].sample(frac=strata['medium'], random_state=42),
    df[df['activity_level'] == 'low'].sample(frac=strata['low'], random_state=42)
])

# If you need exact size ≈ 50k learners:
sampled_df = sampled_df.sample(n=min(sample_size, len(sampled_df)), random_state=42)

# Step 5: Report descriptive stats
stats = sampled_df['activity_level'].value_counts(normalize=True) * 100
interaction_sum = sampled_df['interaction_count'].sum()

print("Sample composition:")
print(stats.round(2))
print(f"Total sampled learners: {len(sampled_df)}")
print(f"Total interactions ≈ {interaction_sum/1e6:.2f} M")

# Step 6: Device diversity estimation (optional)
device_stats = sampled_df['device'].value_counts(normalize=True) * 100
print("Device distribution:")
print(device_stats.round(2))

# Step 7: Save sampled dataset
sampled_df.to_csv("sampled_learners.csv", index=False)

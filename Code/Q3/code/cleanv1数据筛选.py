import pandas as pd

# Load the CSV file
df = pd.read_csv("/mnt/data/cleaned_v1.csv")

# Specify the columns to be selected
selected_columns = [
    "match_id",
    "player1",
    "player2",
    "elapsed_time",
    "p1_1st_serve_success_ratio",
    "p1_1st_serve_win_ratio",
    "p1_2nd_serve_win_ratio",
    "p2_1st_serve_success_ratio",
    "p2_1st_serve_win_ratio",
    "p2_2nd_serve_win_ratio",
    "p1_unf_err_count2r",
    "p2_unf_err_count2r",
]

# Filter the DataFrame to include only the selected columns
filtered_df = df[selected_columns]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv("/mnt/data/filtered_data.csv", index=False)

# Add time information to CA_train.csv and CA_test.csv
# Query time data from MySQL and merge with existing CSV files

import pandas as pd
import subprocess
import os

# Change to MORL_CA_Framework directory
os.chdir('E:/MORL_CA_Framework')

# Query time data from MySQL
print("Querying time data from MySQL...")
result = subprocess.run([
    'mysql', '-uroot', '-h127.0.0.1', '-P3306', '-p123456', 'wanhua2015',
    '-e', 'select date, time from realdata;'
], capture_output=True, text=True)

# Parse MySQL output
lines = result.stdout.strip().split('\n')
header = lines[0].split('\t')
data_lines = lines[1:]

# Create DataFrame from time data
time_data = []
for line in data_lines:
    parts = line.split('\t')
    if len(parts) >= 2:
        date_str = parts[0]  # e.g., "15/08/20"
        time_str = parts[1]  # e.g., "00:05:00"
        # Combine date and time
        datetime_str = f"{date_str} {time_str}"
        time_data.append(datetime_str)

time_df = pd.DataFrame({'datetime': time_data})
print(f"Loaded {len(time_df)} time records")

# Load original CSV files
print("Loading CA_train.csv and CA_test.csv...")
train_data = pd.read_csv('CA_train.csv')
test_data = pd.read_csv('CA_test.csv')

print(f"CA_train.csv: {len(train_data)} rows")
print(f"CA_test.csv: {len(test_data)} rows")

# ChlorAlkali data split: train first 14022, test last 3505
# wanhua.csv has 17527 data rows (17528 - 1 header)
# train + test should equal 17527
train_size = len(train_data)
test_size = len(test_data)

print(f"\nExpected total: {train_size} + {test_size} = {train_size + test_size}")

# Assign time to train (first train_size rows) and test (last test_size rows)
train_time = time_df['datetime'].iloc[:train_size].reset_index(drop=True)
test_time = time_df['datetime'].iloc[train_size:train_size + test_size].reset_index(drop=True)

# Add datetime column to dataframes
train_data.insert(0, 'datetime', train_time)
test_data.insert(0, 'datetime', test_time)

# Save updated files
print("\nSaving updated files...")
train_data.to_csv('CA_train.csv', index=False)
test_data.to_csv('CA_test.csv', index=False)

print(f"Updated CA_train.csv with {len(train_data)} rows (including datetime column)")
print(f"Updated CA_test.csv with {len(test_data)} rows (including datetime column)")

# Verify
print("\nVerification - first 3 rows of CA_train.csv:")
print(train_data.head(3)[['datetime']].to_string())
print("\nVerification - first 3 rows of CA_test.csv:")
print(test_data.head(3)[['datetime']].to_string())
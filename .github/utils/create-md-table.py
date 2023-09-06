import sys
import os
import pandas as pd
import json
from datetime import date

dataframes = {}

today = date.today()

GITHUB_URL = sys.argv[1]

############### Change csv files to dataframes ##########################
for subdir, dirs, files in os.walk('./performances'):
    idx = 0
    for file in files:
        dataframes[f'df_{idx}'] = pd.read_csv(f'./performances/{file}')
        idx += 1

############ Concatenate dataframes and create the MD table ##############
result_df = pd.concat(dataframes, axis=0).reset_index()
result_df.set_index('Framework', inplace=True)
result_df = result_df.drop(columns=['level_0', 'level_1'])
result_markdown_table = result_df.to_markdown()
with open('result_summary_table.md', 'w') as file:
    file.write(result_markdown_table)

###################### Convert dataframe to json ##########################
frameworks = []
workloads = []
scripts = []
precisions = []
benchmark_types = []
tolerances = []
thresholds = []
results = []
passes_fails = []
results_dates = []
urls = []

result_df = result_df.reset_index()
for index, row in result_df.iterrows():
    frameworks.append(row['Framework'])
    workloads.append(row['Workload'])
    scripts.append(row['Script'])
    precisions.append(row['Precision'])
    benchmark_types.append(row['Benchmark Type'])
    tolerances.append(row['Tolerance'])
    thresholds.append(row['Threshold'])
    results.append(row['Result'])
    passes_fails.append(row['Pass/Fail'])
    results_dates.append(today.strftime("%Y-%m-%d"))
    urls.append(GITHUB_URL)

json_content = {
    "framework": frameworks,
    "workload": workloads,
    "script": scripts,
    "_precision": precisions,
    "benchmark_type": benchmark_types,
    "tolerance": tolerances,
    "threshold": thresholds,
    "result": results,
    "pass_fail": passes_fails,
    "results_date": results_dates,
    "url": urls
}
result_json = json.dumps(json_content)
print(result_json)

with open("result_summary_table.json", "w") as outfile:
    outfile.write(result_json)
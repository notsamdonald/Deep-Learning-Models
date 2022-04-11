import json

import matplotlib.pyplot as plt
import pandas as pd

with open('code_dataset.jsonl', 'r') as json_file:
    json_list = list(json_file)

code_list = []

for json_str in json_list:
    result = json.loads(json_str)
    code_list.append(result)

code_df = pd.DataFrame(code_list)

total = code_df['target'].sum()
proportion = total / code_df.shape[0]

print("Insecure code counts: {}, Total code counts: {}, Proportion {}".format(total, code_df.shape[0], proportion))

plt.hist(code_df['func'].str.len(), bins=100)
plt.show()

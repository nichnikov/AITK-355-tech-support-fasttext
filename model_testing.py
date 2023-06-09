import os
import pandas as pd
import requests

queries_df = pd.read_csv(os.path.join("data", "queries_for_testing.csv"), sep="\t")
print(queries_df)

queries = list(queries_df["query"])

results = []
for query in queries:
    request_dict = {"pubid": 477, "text": query}
    res = requests.post("http://0.0.0.0:8080/api/search", json=request_dict)
    print(query, res.json())    
    if res.json() is not None:
        res_dict = res.json()
        res_dict["query"] = query
    else:
        res_dict = {"query": query, "templateId": 0, "templateText": "", "score": 0.0}
    print(res_dict)        
    results.append(res_dict)

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join("data", "test_results.csv"), sep="\t", index=False)
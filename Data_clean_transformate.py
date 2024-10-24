import pandas as pd
import numpy as np
# Sample data
data = {'Name': [' Alice ', 'Bob', ' Charlie ', 'David', 'Eve', ' Alice '],
   'Age': [25, 31, None, 28, 25, 25],
'City': ['New York', ' Paris ', 'London', 'London', 'Tokyo', 'New York']}
df = pd.DataFrame(data)
 
df["Name"]=df["Name"].str.strip()
df["City"]=df["City"].str.lower()
df.drop_duplicates(inplace=True)
df["City"]=df["City"].str.replace(' ',"_")
# df['Age']=df["Age"].fillna(0)
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["City"]=np.where(df["City"]=="new_york", "nyc", df["City"])
df['Age']=pd.to_numeric(df["Age"],errors="coerce")
df["Age_Group"]=pd.cut(df["Age"], bins=[0,10,20,30,60], labels=[ "child", "young Adult", "Adult","Old"])
print(df)

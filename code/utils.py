from pathlib import Path

import pandas as pd


#Get dataset function
#==============================================================================
def list_csvs(directory):
    """Return the HAI CSV files in `directory`, sorted by name.

    Accepts both plain (.csv) and gzip-compressed (.csv.gz) files, as distributed
    by the official HAI repository. If both versions of a file exist, one is used.
    """
    files = {}
    for p in sorted(Path(directory).glob("*.csv*")):
        if p.name.endswith(".csv") or p.name.endswith(".csv.gz"):
            files.setdefault(p.name.split(".csv")[0], p)
    return [files[k] for k in sorted(files)]

def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())

def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])

#Normalization
#==============================================================================
def normalize(df, tag_min, tag_max):
    ndf = df.copy()
    for c in df.columns:
        if tag_min[c] == tag_max[c]:
            ndf[c] = df[c] - tag_min[c]
        else:
            ndf[c] = (df[c] - tag_min[c]) / (tag_max[c] - tag_min[c])
    return ndf

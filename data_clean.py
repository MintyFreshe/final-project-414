import pandas as pd
import csv
import re

df = pd.read_csv("FAO_CAHD_WIDEF.csv")

#drop zimbabwe, argentina, finland etc 

#drop countries with null in 2021 column

#print countries with null in 2021 column
countries_with_null_2021 = df[df["2021"].isna()]["REF_AREA_LABEL"]

df = df[~df["REF_AREA_LABEL"].isin(countries_with_null_2021)]

columns_to_drop = ["2017", "2018", "2019", "2020", "2022", "2023", "2024", "OBS_CONF_LABEL", "OBS_STATUS_LABEL", "REF_AREA", "DATABASE_ID","DATABASE_ID_LABEL","OBS_STATUS","OBS_CONF"]

df2 = df.drop(columns=columns_to_drop)

df2 = df2[df2["UNIT_MEASURE"] != "XDC_PS_D"]

df2["2021"] = pd.to_numeric(df2["2021"], errors="coerce")

def clean_column_name(value):
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")

wide_df = df2.pivot_table(
    index=["REF_AREA_LABEL"],
    columns=["INDICATOR_LABEL", "UNIT_MEASURE"],
    values="2021",
    aggfunc="first",
).reset_index()

wide_df.columns = [
    "_".join(clean_column_name(str(part)) for part in col if str(part))
    if isinstance(col, tuple)
    else clean_column_name(str(col))
    for col in wide_df.columns
]

wide_df = wide_df.dropna()

print(wide_df.head())

wide_df.to_csv("FAO_CAHD_WIDEF_CLEAN.csv", index=False, quoting=csv.QUOTE_ALL)

import os
import pandas as pd
import matplotlib

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# files and main column I am using
input_file = "FAO_CAHD_WIDEF_CLEAN.csv"
feature_column = "percentage_of_the_population_unable_to_afford_a_healthy_diet_percent_pt"
feature_label = "Percent of Population Unable to Afford a Healthy Diet"

clustered_output_file = "clustered_affordability_percent.csv"
summary_output_file = "cluster_summary_affordability_percent.csv"
countries_output_file = "countries_by_cluster_affordability_percent.csv"
countries_sample_output_file = "countries_by_cluster_affordability_percent_sample.csv"
elbow_chart_file = "elbow_method_affordability_percent.png"
average_chart_file = "cluster_average_affordability_percent.png"
boxplot_file = "cluster_affordability_percent_distribution.png"
sample_scatter_file = "countries_by_cluster_affordability_percent_sample_scatter.png"


print("Loading data...")
data = pd.read_csv(input_file)

print("\nColumns:")
print(data.columns.tolist())

print("\nFirst 5 rows:")
print(data.head())


# find the country column, since the cleaned file could use either style
possible_country_columns = ["REF_AREA_LABEL", "ref_area_label", "Country", "country"]
country_column = None

for col in possible_country_columns:
    if col in data.columns:
        country_column = col
        break

if country_column is None:
    raise ValueError("Could not find the country column.")

print("\nCountry column:", country_column)


# make sure the column for clustering is numeric
if feature_column not in data.columns:
    raise ValueError("Could not find the feature column.")

data[feature_column] = pd.to_numeric(data[feature_column], errors="coerce")

before_drop = len(data)
data = data.dropna(subset=[feature_column]).copy()
after_drop = len(data)

print("Dropped", before_drop - after_drop, "rows with missing values.")
print("Rows left:", after_drop)


# elbow method
feature_data = data[[feature_column]]
max_k = min(10, len(data) - 1)
k_values = list(range(2, max_k + 1))
inertia_values = []

print("\nTrying different k values...")

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(feature_data)
    inertia_values.append(model.inertia_)
    print("k =", k, "inertia =", round(model.inertia_, 4))

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia_values, marker="o")
plt.title("Elbow Method for Healthy Diet Affordability")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.savefig(elbow_chart_file, dpi=300)
plt.close()

print("\nSaved elbow chart:", elbow_chart_file)


# I picked 4 because it gives low, medium, high, and very high groups
chosen_k = 4
print("\nUsing k =", chosen_k)


print("\nRunning KMeans...")
final_model = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
data["cluster"] = final_model.fit_predict(feature_data)


# make cluster 0 the lowest average group
cluster_order = data.groupby("cluster")[feature_column].mean().sort_values().index.tolist()
cluster_label_map = {}

for new_label, old_label in enumerate(cluster_order):
    cluster_label_map[old_label] = new_label

data["cluster"] = data["cluster"].map(cluster_label_map)


print("\nMaking cluster summary...")
cluster_summary = (
    data.groupby("cluster")[feature_column]
    .agg(
        average_percent_unable_to_afford="mean",
        minimum_percent_unable_to_afford="min",
        maximum_percent_unable_to_afford="max",
        number_of_countries="count",
    )
    .reset_index()
)

print(cluster_summary)
cluster_summary.to_csv(summary_output_file, index=False)
print("Saved summary:", summary_output_file)


print("\nMaking countries by cluster table...")
countries_by_cluster = data[[country_column, "cluster", feature_column]].sort_values(
    by=["cluster", feature_column, country_column]
)

print(countries_by_cluster)
countries_by_cluster.to_csv(countries_output_file, index=False)
print("Saved countries file:", countries_output_file)


# smaller table so the scatter plot is easier to read
countries_by_cluster_sample = (
    countries_by_cluster.groupby("cluster", group_keys=False)
    .head(10)
    .reset_index(drop=True)
)

usa_rows = countries_by_cluster[countries_by_cluster[country_column] == "United States"]

if not usa_rows.empty:
    usa_row = usa_rows.iloc[[0]]
    sample_has_usa = (countries_by_cluster_sample[country_column] == "United States").any()

    if sample_has_usa == False:
        countries_by_cluster_sample = pd.concat(
            [countries_by_cluster_sample, usa_row],
            ignore_index=True,
        )
        countries_by_cluster_sample = countries_by_cluster_sample.sort_values(
            by=["cluster", feature_column, country_column]
        ).reset_index(drop=True)

print("\nSample table:")
print(countries_by_cluster_sample)

countries_by_cluster_sample.to_csv(countries_sample_output_file, index=False)
print("Saved sample file:", countries_sample_output_file)


print("\nMaking sample scatter plot...")
plt.figure(figsize=(14, 6))

for cluster_num in sorted(countries_by_cluster_sample["cluster"].unique()):
    cluster_sample = countries_by_cluster_sample[
        countries_by_cluster_sample["cluster"] == cluster_num
    ]
    plt.scatter(
        cluster_sample[country_column],
        cluster_sample[feature_column],
        label="Cluster " + str(cluster_num),
        s=70,
    )

plt.title("Sample Countries by Affordability Cluster")
plt.xlabel("Country")
plt.ylabel(feature_label)
plt.xticks(rotation=75, ha="right")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(sample_scatter_file, dpi=300)
plt.close()

print("Saved scatter plot:", sample_scatter_file)


print("\nMaking average cluster chart...")
plt.figure(figsize=(8, 5))
plt.bar(
    cluster_summary["cluster"].astype(str),
    cluster_summary["average_percent_unable_to_afford"],
)
plt.title("Average Population Unable to Afford a Healthy Diet by Cluster")
plt.xlabel("Cluster")
plt.ylabel(feature_label)
plt.tight_layout()
plt.savefig(average_chart_file, dpi=300)
plt.close()

print("Saved average chart:", average_chart_file)


print("\nMaking boxplot...")
plt.figure(figsize=(8, 5))
data.boxplot(column=feature_column, by="cluster")
plt.title("Population Unable to Afford a Healthy Diet by Cluster")
plt.suptitle("")
plt.xlabel("Cluster")
plt.ylabel(feature_label)
plt.tight_layout()
plt.savefig(boxplot_file, dpi=300)
plt.close()

print("Saved boxplot:", boxplot_file)


data.to_csv(clustered_output_file, index=False)
print("\nSaved final data:", clustered_output_file)

print("\nDone.")

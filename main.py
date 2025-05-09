# Data Mining Assignment 2
# Problem 1:K-Means Algorithm using Euclidean Distance
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler


def load_excel_file():
    while True:
        try:
            # file_path = input("Please enter the file path of the data you want to cluster: ")
            # data = pd.read_csv(file_path)
            data = pd.read_csv(
                "C:\\Users\\DR.Hisham\\Desktop\\Year 4\\Second and Final Semester!!\\Data Mining\\Assignments\\Assignment 2\\Assignment(2)\\Assignment(2)\\SS2025_Clustering_SuperMarketCustomers.csv")
            print(data.head())
            break
        except Exception as e:
            print(f"Error reading file: {e}. Please try again.")
    return data


def get_input(data):
    while True:
        try:
            k = int(input("Enter the number of clusters (k): "))
            if k > 0:
                break
            else:
                print("k must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer value for k.")
    while True:
        try:
            percentage_of_data = float(input("Enter the amount of data to be processed (e.g., 0.7 for 70%): "))
            if 0 < percentage_of_data <= 1:
                break
            else:
                print("Error: Enter a value between 0 and 1 (e.g., 0.7 for 70%).")
        except ValueError:
            print("Invalid input. Please enter a number like 0.7.")
    while True:
        try:
            min_threshold = float(input("Enter your minimum threshold for outlier detection: "))
            max_threshold = float(input("Enter your maximum threshold for outlier detection: "))
            if min_threshold < max_threshold:
                break
            else:
                print("Error: Minimum Threshold has to be less that the maximum")
        except ValueError:
            print("Invalid input. Please enter a number like 0.7.")
    amount_of_data = int(len(data) * percentage_of_data)
    trimmed_data = data.iloc[:amount_of_data]
    return trimmed_data, k, min_threshold, max_threshold


def preprocess_data(trimmed_data):
    # Drop Customer ID column
    trimmed_data = trimmed_data.drop(columns=["CustomerID"])
    # Drop Gender Column
    trimmed_data = trimmed_data.drop(columns=["Gender"])
    # No nulls
    # print(trimmed_data.isnull().sum())
    # Normalization of the data using min-max scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(trimmed_data)
    scaled_df = pd.DataFrame(scaled_data, columns=trimmed_data.columns)
    return scaled_df


def initial_centroid(trimmed_data, k):
    return trimmed_data.sample(n=k).values.tolist()


def calculate_euclidean_distance(centroid, value):
    sum_of_diff = 0
    for c, v in zip(centroid, value):
        diff = c - v
        sum_of_diff += diff ** 2
    euclidean_distance = math.sqrt(sum_of_diff)
    return euclidean_distance


def assign_values_to_cluster(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    # Calculate distance to each centroid and store the minimum distance
    for i, row in data.iterrows():
        distances = []
        for centroid in centroids:
            dis = calculate_euclidean_distance(centroid, row)
            distances.append(dis)
        closest_centroid = distances.index(min(distances))
        clusters[closest_centroid].append(row.values)
    return clusters


def update_centroid(clusters):
    updated_centroids = []
    for cluster in clusters:
        # Check that the cluster contains a value
        if cluster:
            cluster_df = pd.DataFrame(cluster)
            new_centroid = cluster_df.mean().values.tolist()
            updated_centroids.append(new_centroid)
        else:
            updated_centroids.append(None)

    return updated_centroids


def has_converged(previous_centroid, current_centroid, threshold=0.0001):
    for pre, cur in zip(previous_centroid, current_centroid):
        if calculate_euclidean_distance(pre, cur) > threshold:
            return False
    return True


def detect_outliers(clusters, centroids, min_threshold, max_threshold):
    cleaned_clusters = []
    outliers = []

    for i, cluster in enumerate(clusters):
        cleaned_cluster = []
        cluster_outliers = []

        for point in cluster:
            distance = calculate_euclidean_distance(centroids[i], point)
            if min_threshold <= distance <= max_threshold:
                cleaned_cluster.append(point)
            else:
                cluster_outliers.append(point)

        cleaned_clusters.append(cleaned_cluster)
        outliers.append(cluster_outliers)

    return cleaned_clusters, outliers


def k_means_algorithm(data, k, min_threshold, max_threshold, max_iterations=10):
    centroids = initial_centroid(data, k)

    for iteration in range(max_iterations):
        # print(f"\n--- Iteration {iteration + 1} ---")
        clusters = assign_values_to_cluster(data, centroids)

        clusters, outliers = detect_outliers(clusters, centroids, min_threshold, max_threshold)

        new_centroids = update_centroid(clusters)

        # Skip empty centroids
        if None in new_centroids:
            # print("Empty cluster:")
            centroids = initial_centroid(data, k)
            continue

        if has_converged(centroids, new_centroids):
            # print("No difference in clusters.Convergence reached")
            break
        centroids = new_centroids

    return clusters, outliers, centroids

def main():
    data = load_excel_file()
    f, k, min_threshold, max_threshold = get_input(data)
    preprocessed = preprocess_data(f)

    final_clusters, final_outliers, final_centroids = k_means_algorithm(
        preprocessed, k, min_threshold, max_threshold
    )

    # Display the size and content of each cluster
    for idx, cluster in enumerate(final_clusters):
        print(f"Cluster {idx + 1} size: {len(cluster)}")
        print(f"Contents of Cluster {idx + 1}:")
        cluster_df = pd.DataFrame(cluster, columns=preprocessed.columns)
        print(cluster_df.to_string(index=False))

# main()

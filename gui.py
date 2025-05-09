import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import main as dm

class KMeansApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Clustering GUI")
        self.root.configure(bg='#ADD8E6')  # Light blue background
        self.root.geometry("600x600")  # Set a proper window size

        self.file_path = ""
        self.data = None

        # Center Frame for Inputs and Output
        self.center_frame = tk.Frame(root, bg='#ADD8E6')
        self.center_frame.pack(expand=True, pady=20)

        # File selection
        tk.Label(self.center_frame, text="Select CSV File:", bg='#ADD8E6').pack(pady=10)
        self.browse_button = tk.Button(self.center_frame, text="Browse", command=self.load_file)
        self.browse_button.pack()

        # Status Label for Success/Error Messages
        self.status_label = tk.Label(self.center_frame, text="", bg='#ADD8E6', fg='green')
        self.status_label.pack(pady=5)

        # Inputs
        tk.Label(self.center_frame, text="Number of Clusters (k):", bg='#ADD8E6').pack(pady=5)
        self.k_entry = tk.Entry(self.center_frame)
        self.k_entry.pack()

        tk.Label(self.center_frame, text="Percentage of Data (0-1):", bg='#ADD8E6').pack(pady=5)
        self.percent_entry = tk.Entry(self.center_frame)
        self.percent_entry.pack()

        tk.Label(self.center_frame, text="Min Threshold:", bg='#ADD8E6').pack(pady=5)
        self.min_thresh_entry = tk.Entry(self.center_frame)
        self.min_thresh_entry.pack()

        tk.Label(self.center_frame, text="Max Threshold:", bg='#ADD8E6').pack(pady=5)
        self.max_thresh_entry = tk.Entry(self.center_frame)
        self.max_thresh_entry.pack()

        # Submit Button
        self.submit_button = tk.Button(self.center_frame, text="Submit", command=self.run_kmeans)
        self.submit_button.pack(pady=10)

        # Output
        self.output_text = tk.Text(self.center_frame, height=20, width=80)
        self.output_text.pack(pady=10)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_path = file_path
            try:
                self.data = pd.read_csv(self.file_path)
                # Update the persistent status label's text
                self.status_label.config(text="CSV file uploaded successfully!", fg='green')
            except Exception as e:
                self.status_label.config(text=f"Error: Could not read file - {e}", fg='red')

    def run_kmeans(self):
        if self.data is None:
            messagebox.showerror("Error", "No file loaded.")
            return

        try:
            k = int(self.k_entry.get())
            percentage = float(self.percent_entry.get())
            min_thresh = float(self.min_thresh_entry.get())
            max_thresh = float(self.max_thresh_entry.get())

            trimmed_data = self.data.iloc[:int(len(self.data) * percentage)]
            preprocessed = dm.preprocess_data(trimmed_data)
            final_clusters, final_outliers, final_centroids = dm.k_means_algorithm(preprocessed, k, min_thresh, max_thresh)

            self.output_text.delete("1.0", tk.END)
            for idx, (cluster, outlier_group) in enumerate(zip(final_clusters,final_outliers)):
                self.output_text.insert(tk.END, f"Cluster {idx + 1} size: {len(cluster)}\n")
                cluster_df = pd.DataFrame(cluster, columns=preprocessed.columns)
                self.output_text.insert(tk.END, cluster_df.to_string(index=False) + "\n\n")

                self.output_text.insert(tk.END, f"Outliers in Cluster {idx + 1}:\n")
                if outlier_group:
                    outliers_df = pd.DataFrame(outlier_group, columns=preprocessed.columns)
                    self.output_text.insert(tk.END, outliers_df.to_string(index=False) + "\n\n")
                else:
                    self.output_text.insert(tk.END, "No outliers in this cluster\n\n")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = KMeansApp(root)
    root.mainloop()
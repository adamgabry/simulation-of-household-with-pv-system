import pandas as pd

class CSVMerger:
    def __init__(self, csv_files, output_csv_path):
        """
        Initialize the CSVMerger instance with a list of csv file paths
        and the path for the output merged csv file.
        """
        self.csv_files = csv_files
        self.output_csv_path = output_csv_path

    def merge_and_sort_csv(self):
        """
        Merges multiple CSV files into one, sorts it by the first column, 
        and saves the result into a new CSV file.
        """
        # Initialize an empty DataFrame
        combined_df = pd.DataFrame()

        for file in self.csv_files:
            # Read each file
            temp_df = pd.read_csv(file, header=None, low_memory=False)
            # Append it to the combined DataFrame
            combined_df = pd.concat([combined_df, temp_df])

        # Drop rows that are entirely NA (if any)
        combined_df.dropna(how='all', inplace=True)

        # Assuming the first column is the ID to sort by, and it's numeric
        combined_df[0] = pd.to_numeric(combined_df[0], errors='coerce')
        combined_df.sort_values(by=0, inplace=True)

        # Reset the index of the combined dataframe
        combined_df.reset_index(drop=True, inplace=True)

        # Save the merged and sorted dataframe to a new CSV file
        combined_df.to_csv(self.output_csv_path, index=False, header=False)

        print(f"Merged CSV saved to {self.output_csv_path}")

"""
@Author: Adam Gabrys
@Date: 2023-10-24

@Description: This script contains a class, LargeDatabaseExtractor, which is used to extract data from a SQLite database. 
The class takes a database file as input during initialization. It has a method, extract_data, which takes a list of entity IDs, 
a SQL query, an output file name, and an optional batch size as parameters. The method modifies the SQL query to include the 
correct number of placeholders for the entity IDs, executes the query, and writes the results to a CSV file in batches. 
This approach is memory-efficient and can handle large amounts of data.
"""
import sqlite3
import csv

class LargeDatabaseExtractor:
    def __init__(self, db_file):
        """
        Initialize the LargeDatabaseExtractor object.

        Args:
            db_file (str): The path to the database file.

        Returns:
            None
        """
        self.db_file = db_file

    def extract_data(self, entity_ids, query, output_file, batch_size=1000):
        """
        Extracts data from the database based on the provided entity IDs and query,
        and writes the data to a CSV file.

        Parameters:
        - entity_ids (tuple or int): The entity IDs to filter the data.
        - query (str): The SQL query to execute.
        - output_file (str): The path to the output CSV file.
        - batch_size (int): The number of rows to fetch at a time from the database.

        Returns:
        None
        """
        # Ensure entity_ids is a tuple, even for a single value
        if not isinstance(entity_ids, tuple):
            entity_ids = (entity_ids,)

        # Connect to the database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Modifying the query to include all placeholders for entity_ids
        placeholders = ','.join(['?'] * len(entity_ids))  # Creating a string of placeholders
        modified_query = query.replace("IN ?", f"IN ({placeholders})")  # Replace 'IN ?' with 'IN (?, ?, ...)'

        # Try to prepare and execute the query with entity_ids as parameters
        try:
            cursor.execute(modified_query, entity_ids)
        except sqlite3.DatabaseError as e:
            print(f"Error executing query: {e}, trying to execute the query separately for every sensor.")
            raise e

        # Open the CSV file for writing
        with open(output_file, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            while True:
                try:
                    # Fetch a batch of data
                    data_to_write = cursor.fetchmany(batch_size)
                    if not data_to_write:
                        break  # Exit the loop if no more data is available
                    
                    # Write the batch of data to the CSV file
                    csv_writer.writerows(data_to_write)
                except sqlite3.DatabaseError as e:
                    print(f"Error fetching data: {e}. Some or all data were mallformed. The non-malformed data will be still written to the CSV file.")
                    continue  # Skip this batch and try the next batch

        # Close the database connection
        conn.close()

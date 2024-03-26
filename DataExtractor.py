import os
import pandas as pd

dataset_path = "archive/"
desired_modality = "01"
desired_emotions = ["03", "04"]
desired_intensity = "01"

happy_data = pd.DataFrame()
sad_data = pd.DataFrame()

csv_filenames = os.listdir(dataset_path)

for filename in csv_filenames:
    # Store the filename in parts array by splitting the filename by "-"
    parts = filename.split("-")
    modality_code = parts[0]
    emotion_code = parts[2]
    intensity_code = parts[3]

    # Check if the file matches the desired modality and emotion codes
    if modality_code == str(desired_modality) and emotion_code in desired_emotions and intensity_code == str(desired_intensity):
        # Construct the full file path
        file_path = f"{dataset_path}/{filename}"
        # Read the CSV file from the specified path
        csv_data = pd.read_csv(file_path)

        # Append data to the appropriate DataFrame based on the emotion code
        if emotion_code == "03":
            happy_data = pd.concat([happy_data, csv_data], ignore_index=True)
        elif emotion_code == "04":
            sad_data = pd.concat([sad_data, csv_data], ignore_index=True)


# Print the number of happy and sad data rows
print("The total number of happy data rows:", len(happy_data))
print("The total number of sad data rows:", len(sad_data))

# The data folder path
data_folder_path = "data/"

# Save the filtered data to new CSV files
happy_data.to_csv(f"{data_folder_path}happy.csv", index=False)
sad_data.to_csv(f"{data_folder_path}sad.csv", index=False)

# print that the data extraction is done
print("Data extraction is done!")

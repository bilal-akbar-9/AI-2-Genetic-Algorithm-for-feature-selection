import glob
import os
import pandas as pd

dataset_path = "archive/"
desired_modality = "01"
desired_emotions = ["03", "04"]

happy_data = []
sad_data = []

csv_filenames = os.listdir(dataset_path)

for filename in csv_filenames:
    parts = filename.split("-")
    modality_code = parts[0]
    emotion_code = parts[2]

    if modality_code == desired_modality and emotion_code in desired_emotions:
        file_path = os.path.join(dataset_path, filename)
        data = pd.read_csv(file_path)

        if emotion_code == "03":
            happy_data.append(data)
        elif emotion_code == "04":
            sad_data.append(data)

print("Number of happy data entries:", len(happy_data))
print("Number of sad data entries:", len(sad_data))

import os
import requests

# Create a directory to save the files
os.makedirs("Retrosheet_Game_Logs", exist_ok=True)

# Define the years you want to download
start_year = 1903
end_year = 2023

for year in range(start_year, end_year + 1):
    url = f"https://www.retrosheet.org/gamelogs/gl{year}.zip"
    response = requests.get(url)

    if response.status_code == 200:
        with open(f"Retrosheet_Game_Logs/gl{year}.zip", "wb") as file:
            file.write(response.content)
        print(f"Downloaded gl{year}.zip")
    else:
        print(f"Failed to download gl{year}.zip")

print("Download complete!")


import os
import requests


def download_river_level_data():
    url = "https://check-for-flooding.service.gov.uk/station/8288"
    download_url = "https://check-for-flooding.service.gov.uk/station-csv/8288"
    save_directory = "/Users/FreddieLewin/PycharmProjects/durham_river_level_predictor/data_directory"
    response = requests.get(url)
    if response.status_code == 200:
        download_response = requests.get(download_url)
        if download_response.status_code == 200:
            file_count = len([f for f in os.listdir(save_directory) if f.endswith('.csv')])
            file_name = f"river_level_data_{file_count + 1}.csv"
            file_path = os.path.join(save_directory, file_name)
            with open(file_path, "wb") as csv_file:
                csv_file.write(download_response.content)
            print(f"Data downloaded successfully. Saved to: {file_path}")
            return file_path
        else:
            print(f"Failed to download data. Status code: {download_response.status_code}")
    else:

        print(f"Failed to access the website. Status code: {response.status_code}")
    return -1




import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from scipy.special import gamma
from scipy.stats import poisson, ks_2samp, beta
from datetime import datetime, timedelta
from import_csv_data.gov_uk_data_csv_file import download_river_level_data


def calculate_15_minute_intervals(target_time_str):
    current_time = datetime.now().time()
    target_time = datetime.strptime(target_time_str, '%H:%M').time()
    if target_time <= current_time:
        print(f'Input time after current time!')
    time_difference = datetime.combine(datetime.today(), target_time) - datetime.combine(datetime.today(), current_time)
    intervals = round(time_difference.total_seconds() / (15 * 60))
    return intervals


def predicted_river_level_linear(start_time):
    number_of_intervals = calculate_15_minute_intervals(target_time_str=start_time)
    name_of_file = download_river_level_data().split('/')[-1]
    df_river_levels_curr = pd.read_csv(f'/Users/FreddieLewin/PycharmProjects/durham_river_level_predictor/data_directory/{name_of_file}')
    last_six_points = df_river_levels_curr['Height (m)'][-6:]
    last_six_points = last_six_points.reset_index(drop=True)

    plt.plot(last_six_points, label='Actual River Levels')

    X = np.arange(len(last_six_points))
    A = np.vstack([X, np.ones_like(X)]).T
    gradient, y_intercept = np.linalg.lstsq(A, last_six_points, rcond=None)[0]
    def func(x):
        return gradient * x + y_intercept
    x_vals = list(range(1, 5 + number_of_intervals))
    y_vals = list(map(func, x_vals))
    plt.plot(x_vals, y_vals, label='Linear Best Fit')
    plt.axhline(y=0.675, color='green', linestyle='--', label='Novice Limit')
    plt.axhline(y=0.6, color='purple', linestyle='--', label='Senior Limit')
    plt.show()

    y_val_at_session_start = y_vals[-1]
    y_val_curr = df_river_levels_curr['Height (m)'].tolist()[-1]
    if y_val_at_session_start < 0.6:
        print(f'All Sessions should be ON: Actual River Levels: {y_val_curr}, Predicted River Level: {y_val_at_session_start}')
    elif 0.6 < y_val_at_session_start < 0.675:
        print(f'Novices may not boat, Seniors can! Current River Level: {y_val_curr}, Predicted River Level at Session Start: {y_val_at_session_start}')
    else:
        print(f'No one can boat! Actual River Levels: {y_val_curr}, Predicted River Level: {y_val_at_session_start}')

# now the next step is to iterate over lambda values and fit a poisson pdf for each one. find the R squared for each ine and find the best
# maybe beta distubrion and use meta office rain predictor

def predicted_river_level_beta(start_time):
    number_of_intervals = calculate_15_minute_intervals(target_time_str=start_time)
    name_of_file = download_river_level_data().split('/')[-1]
    time.sleep(1)
    df_river_levels_curr = pd.read_csv(f'/Users/FreddieLewin/PycharmProjects/durham_river_level_predictor/data_directory/{name_of_file}')
    first_height = df_river_levels_curr['Height (m)'].iloc[0]
    extra_rows = pd.DataFrame({'Height (m)': [first_height] * 500})
    df_with_extra_rows = pd.concat([extra_rows, df_river_levels_curr]).reset_index(drop=True)
    plt.plot(df_with_extra_rows['Height (m)'])
    best_B_value = float('inf')
    best_A_value = float('inf')
    p_value_max = 0
    A_values = np.linspace(0, 10, 100)
    for A in A_values:
        B_values = np.linspace(0, 10, 100)

        for B in B_values:
            beta_pdf = beta.pdf(np.linspace(0, 1, len(df_with_extra_rows)), a=A, b=B)  # Calculate the Beta PDF
            beta_pdf_scaled = np.interp(beta_pdf, (beta_pdf.min(), beta_pdf.max()),
                                        (df_with_extra_rows['Height (m)'].min(), df_with_extra_rows['Height (m)'].max()))
            ks_statistic, p_value = ks_2samp(beta_pdf_scaled[494:], df_with_extra_rows['Height (m)'][494:])
            if p_value_max < p_value:
                best_B_value = B
                best_A_value = A
                plt.plot(df_with_extra_rows['Height (m)'])
                plt.plot(beta_pdf_scaled, color='blue', alpha=1, label=f'Beta PDF B: {B} A: {A}')
                plt.legend()
                plt.show()
                p_value_max = max(p_value_max, p_value)
    print(f'B: {best_B_value}, A: {best_A_value} P-Value Max {p_value_max}')
    # if p_value_max < 0.00005:
    #     predicted_river_level_linear(start_time=start_time)
    #     return 0

    river_level_curr = df_river_levels_curr['Height (m)'].tolist()[-1]

    beta_pdf = beta.pdf(np.linspace(0, 1, len(df_with_extra_rows) + number_of_intervals), a=best_A_value, b=best_B_value)
    beta_pdf_scaled = np.interp(beta_pdf, (beta_pdf.min(), beta_pdf.max()),
                                (df_with_extra_rows['Height (m)'].min(), df_with_extra_rows['Height (m)'].max()))
    predicted_river_level = beta_pdf_scaled[-1]

    if predicted_river_level < 0.6:
        print(
            f'All Sessions should be ON: Actual River Levels: {river_level_curr}, Predicted River Level: {predicted_river_level}')
    elif 0.6 < predicted_river_level < 0.675:
        print(
            f'Novices may not boat, Seniors can! Current River Level: {river_level_curr}, Predicted River Level at Session Start: {predicted_river_level}')
    else:
        print(
            f'No one can boat! Actual River Levels: {river_level_curr}, Predicted River Level: {predicted_river_level}')
predicted_river_level_beta('21:40')

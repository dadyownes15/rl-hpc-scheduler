import numpy as np
from typing import List
import pandas as pd
import datetime
import os
import math

class CarbonIntensityGrid:
    def __init__(self, path: str = None, date_column: str = 'Datetime (UTC)' , carbon_intensity_column: str = 'Carbon intensity gCO₂eq/kWh (direct) (normalized)', forecast_window_size: int = 24, forecast_type: str = 'daily'):
        if path is None:
            this_file_path = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(this_file_path)))
            path = os.path.join(project_root, "data/InputFiles/DK-DK2_2021-2023_hourly_combined_normalized.csv")

        self.carbon_intensity_path = path
        self.date_column = date_column
        self.carbon_intensity_column = carbon_intensity_column
        self.forecast_window_size =int (forecast_window_size)
        self.df = pd.read_csv(self.carbon_intensity_path)
        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], utc=True)
        
        self.forecast_type = forecast_type
        if self.forecast_type not in ['daily', 'continuous']:
            raise ValueError("forecast_type must be 'daily' or 'continuous'")
        
        # for daily forecast caching
        self.last_forecast_date = None
        self.daily_intensity_forecast = None

        if (self.carbon_intensity_column or self.date_column or self.carbon_intensity_column or self.carbon_intensity_path) == None:
            raise ValueError("Bad input")
             
    def forecast(self, current_date_time, itensities_only = False):
        """
        Return a 24 hour forecast given a timestamp.
        The behavior of this method depends on the `forecast_type` set during initialization.

        If `forecast_type` is 'daily':
        It returns a forecast for the current day (00:00 to 23:00). The carbon intensity values
        are fetched once per day and cached. The first column represents the time remaining
        in each hourly slot, clipped between 0 and 3600 seconds.

        If `forecast_type` is 'continuous':
        It returns a rolling 24-hour forecast starting from the next full hour. The first
        column represents the wait time in seconds from the current time to the beginning
        of each forecast slot.
        """
        
        current_dt = pd.to_datetime(current_date_time)
        if current_dt.tzinfo is None:
            current_date_time_utc = current_dt.tz_localize('UTC')
        else:
            current_date_time_utc = current_dt.tz_convert('UTC')

        if self.forecast_type == 'daily':
            grid_start_time = current_date_time_utc.normalize()

            if self.last_forecast_date != grid_start_time.date():
                self.last_forecast_date = grid_start_time.date()
                grid_end_time = grid_start_time + datetime.timedelta(days=1)
                
                intensity_df = self.df[(self.df[self.date_column] >= grid_start_time) & (self.df[self.date_column] < grid_end_time)]
                assert len(intensity_df) == 24, f"Expected 24 hourly data points for {grid_start_time.date()}, but got {len(intensity_df)}"

                self.daily_intensity_forecast = intensity_df[self.carbon_intensity_column].values

            time_remaning_column = np.zeros(shape=(self.forecast_window_size))

            for i in range(self.forecast_window_size):
                slot_end_time = grid_start_time + datetime.timedelta(hours=i + 1)
                time_remaining_seconds = (slot_end_time - current_date_time_utc).total_seconds()
                time_remaning_column[i] = np.clip(time_remaining_seconds, 0, 3600)
            
            if itensities_only == True:
                return np.array(self.daily_intensity_forecast)
            else:
                return np.hstack([
                    time_remaning_column.reshape(-1, 1),
                    self.daily_intensity_forecast.reshape(-1, 1)
                ])
        
        elif self.forecast_type == 'continuous':
            forecast_start_time = current_date_time_utc.ceil('h')
            forecast_end_time = forecast_start_time + datetime.timedelta(hours=self.forecast_window_size)

            intensity_df = self.df[(self.df[self.date_column] >= forecast_start_time) & (self.df[self.date_column] < forecast_end_time)]
            
            if len(intensity_df) != self.forecast_window_size:
                raise ValueError(f"Could not retrieve {self.forecast_window_size} hours of data for continuous forecast starting from {forecast_start_time}.")

            intensity_values = intensity_df[self.carbon_intensity_column].values
            
            wait_times_column = np.zeros(shape=(self.forecast_window_size))
            for i in range(self.forecast_window_size):
                slot_start_time = forecast_start_time + datetime.timedelta(hours=i)
                wait_time_seconds = (slot_start_time - current_date_time_utc).total_seconds()
                wait_times_column[i] = np.maximum(0, wait_time_seconds)
            
            return np.hstack([
                wait_times_column.reshape(-1, 1),
                intensity_values.reshape(-1, 1)
            ]) 
    
    def carbon_reward(self, job_start_time: datetime.datetime, reserved_time: int, carbon_consideration_index: int, numProcess: int):
        """Compute the carbon-cost reward for running *numProcess* processors
        from *start_time* for *reserved_time* seconds.

        The function currently supports only `forecast_type == 'daily'` and
        assumes that only the first-day, 24-hour carbon forecast is known. For
        multi-day reservations that forecast is simply repeated, mimicking the
        real use-case where future-day intensities are unknown.
        """

        if self.forecast_type != "daily":
            raise ValueError("Carbon reward is only implemented for daily forecasting so far")

        # ------------- validation ------------- #
        assert isinstance(job_start_time, datetime.datetime), "start_time must be datetime"
        assert isinstance(reserved_time, int) and reserved_time > 0, "reserved_time must be positive int seconds"
        assert isinstance(carbon_consideration_index, (int, float)) and carbon_consideration_index >= 0, "carbon_consideration_index must be positive number"
        assert isinstance(numProcess, int) and numProcess > 0, "numProcess must be positive int"

        end_time = job_start_time + datetime.timedelta(seconds=reserved_time)

        # ----- build per-hour usage vector (index 0-23 repeated per day) ----- #
        total_days = (end_time.date() - job_start_time.date()).days + 1
        usage_secs_per_hour = np.zeros(total_days * 24, dtype=int)

        hour_cursor = job_start_time
        while hour_cursor < end_time:
            next_hour_boundary = (hour_cursor + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            if next_hour_boundary <= hour_cursor:  # safety to guarantee progress
                next_hour_boundary = hour_cursor + datetime.timedelta(hours=1)

            interval_end = min(next_hour_boundary, end_time)

            day_offset = (hour_cursor.date() - job_start_time.date()).days
            hour_index = hour_cursor.hour + 24 * day_offset
            usage_secs_per_hour[hour_index] += int((interval_end - hour_cursor).total_seconds())

            hour_cursor = next_hour_boundary

        # ----- carbon intensity vector matching the same indexing ----- #
        first_day_intensity = self.forecast(job_start_time, itensities_only=True).astype(float).flatten()  # (24,)
        carbon_values = np.tile(first_day_intensity, total_days)  # length = total_days*24


        # ----- final reward ----- #
        carbon_cost = float(np.dot(usage_secs_per_hour, carbon_values))  # seconds × intensity
        print(carbon_cost)
        print(carbon_consideration_index)
        print(numProcess)
        current_reward = 1/(carbon_cost * numProcess) * carbon_consideration_index
        print(current_reward) 
        return current_reward
# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

import logging

import numpy as np
import pandas as pd

from forest.jasmine.data2mobmat import great_circle_dist
from forest.jasmine.mobmat2traj import locate_home
from forest.poplar.legacy.common_funcs import datetime2stamp, stamp2datetime

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger()


def gps_quality(data):
    """
    ------------------------------------------------------------------------------------------------------

    This function checks the quality of the GPS data.

    Parameters:
    ...........
    data: pd.DataFrame
        A dataframe containing the GPS data.

    Raises:
    ...........
    ValueError
        If the data does not have enough observations per hour.

    ------------------------------------------------------------------------------------------------------
    """

    end_time = data.timestamp.iloc[-1]
    start_time = data.timestamp.iloc[0]

    # calculate number of hours
    num_hours = int((end_time - start_time) / 3600 / 1000) + 1

    # check if there are at least 10 observations per hour
    quality_check = 0
    for hour in range(num_hours):
        hour_data = data[
            (data.timestamp >= start_time + hour * 3600 * 1000)
            & (data.timestamp < start_time + (hour + 1) * 3600 * 1000)
        ]
        if hour_data.shape[0] > 60:
            quality_check += 1

    if quality_check < 0.05 * num_hours:
        raise ValueError("Data does not have enough observations per hour.")


def trajectory_statistics(traj, df_list, timezone):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the statistics of the GPS data.

    Parameters:
    ...........
    traj: np.array
        A numpy array containing the GPS data.
    df_list: list
        A list containing the dataframes to be filled with the statistics.
    timezone: str
        The timezone of the GPS data.

    Returns:
    ...........
    df_list: list
        A list containing the dataframes filled with the statistics.

    ------------------------------------------------------------------------------------------------------
    """

    hourly, daily, summary = df_list

    hourly = gps_stats(traj, hourly, "hourly", timezone)

    daily = gps_stats(traj, daily, "daily", timezone)

    summary = summary_stats(daily, summary)

    return [hourly, daily, summary]


def gps_stats(traj, df, frequency, timezone):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the statistics of the GPS data.

    Parameters:
    ...........
    traj: np.array
        A numpy array containing the GPS data.
    df: pd.DataFrame
        A dataframe to be filled with the statistics.
    frequency: str
        The frequency of the statistics. Can be "hourly" or "daily".
    timezone: str
        The timezone of the GPS data.

    Returns:
    ...........
    df: pd.DataFrame
        A dataframe filled with the statistics.

    ------------------------------------------------------------------------------------------------------
    """

    obs_traj = traj[traj[:, 7] == 1, :]
    home_coords = locate_home(obs_traj, timezone)

    start_time_list = stamp2datetime(traj[0, 3], timezone)
    end_time_list = stamp2datetime(traj[-1, 6], timezone)
    if frequency == "hourly":
        start_time_list[4:6] = [0, 0]
        end_time_list[4:6] = [0, 0]
        offset = 3600
    else:
        start_time_list[3:6] = [0, 0, 0]
        end_time_list[3:6] = [0, 0, 0]
        offset = 3600 * 24
    start_stamp = datetime2stamp(start_time_list, timezone)
    end_stamp = datetime2stamp(end_time_list, timezone) + offset

    window = 3600
    if frequency == "daily":
        window *= 24

    no_windows = (end_stamp - start_stamp) // window

    for i in range(no_windows):
        start_time = start_stamp + i * window
        end_time = start_stamp + (i + 1) * window

        current_time_list = stamp2datetime(start_time, timezone)
        year, month, day, hour = current_time_list[:4]
        if frequency == "daily":
            date_str = f"{year:04}-{month:02}-{day:02}"
        else:
            date_str = f"{year:04}-{month:02}-{day:02} {hour:02}:00:00"

        index_rows = (traj[:, 3] < end_time) * (traj[:, 6] > start_time)

        if sum(index_rows) == 0:
            res = [pd.NA for _ in df.columns[1:]]
            res = [date_str] + res
            new_row = pd.Series(res, index=df.columns)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            continue

        current_traj = traj_smooth_ends(traj, start_time, end_time, index_rows)

        # observed time
        obs_dur = sum(
            (current_traj[:, 6] - current_traj[:, 3])[current_traj[:, 7] == 1]
        ) / 3600

        if frequency == "daily":
            day_index, night_index = day_night_split(current_traj, timezone)

            # observed time day
            obs_dur_day = sum(
                (current_traj[day_index, 6] - current_traj[day_index, 3])[
                    current_traj[day_index, 7] == 1
                ]
            ) / 3600
            # observed time night
            obs_dur_night = sum(
                (current_traj[night_index, 6] - current_traj[night_index, 3])[
                    current_traj[night_index, 7] == 1
                ]
            ) / 3600

        # distance travelled
        mov_vec = np.round(
            great_circle_dist(
                current_traj[:, 4],
                current_traj[:, 5],
                current_traj[:, 1],
                current_traj[:, 2],
            ),
            0,
        )
        dist_traveled = sum(mov_vec) / 1000

        # pause time + movement time
        pause_time = sum(
            (current_traj[:, 6] - current_traj[:, 3])[current_traj[:, 0] == 2]
        ) / 3600
        move_time = sum(
            (current_traj[:, 6] - current_traj[:, 3])[current_traj[:, 0] == 1]
        ) / 3600

        # home time + max dist from home
        d_home_1 = great_circle_dist(*home_coords, current_traj[:, 1], current_traj[:, 2])
        d_home_2 = great_circle_dist(*home_coords, current_traj[:, 4], current_traj[:, 5])
        d_home = (d_home_1 + d_home_2) / 2

        time_at_home = sum((current_traj[:, 6] - current_traj[:, 3])[d_home <= 50]) / 3600

        home_distances = np.concatenate((d_home_1, d_home_2))
        max_dist_home = max(home_distances) / 1000
        mean_dist_home = np.mean(home_distances) / 1000

        if frequency == "hourly":
            res = [
                date_str,
                obs_dur,
                move_time,
                pause_time,
                dist_traveled,
                time_at_home,
                max_dist_home,
                mean_dist_home,
            ]
        else:
            res = [
                date_str,
                obs_dur,
                obs_dur_day,
                obs_dur_night,
                move_time,
                pause_time,
                dist_traveled,
                time_at_home,
                max_dist_home,
                mean_dist_home,
            ]

        df = pd.concat([df, pd.DataFrame([res], columns=df.columns)], ignore_index=True)

    return df


def traj_smooth_ends(traj, start_time, end_time, index_rows):
    """
    ------------------------------------------------------------------------------------------------------

    This function smooths the ends of the GPS sub-trajectories, so that they start and end at the
     specified times.

    Parameters:
    ...........
    traj: np.array
        A numpy array containing the GPS data.
    start_time: float
        The start time of the sub-trajectory.
    end_time: float
        The end time of the sub-trajectory.
    index_rows: np.array
        A boolean array indicating the rows of the sub-trajectory.

    Returns:
    ...........
    current_traj: np.array
        A numpy array containing the smoothed sub-trajectory.

    ------------------------------------------------------------------------------------------------------
    """

    current_traj = traj[index_rows, :]

    if sum(index_rows) == 1:
        p0 = (start_time - current_traj[0, 3]) / (
            current_traj[0, 6] - current_traj[0, 3]
        )
        p1 = (end_time - current_traj[0, 3]) / (current_traj[0, 6] - current_traj[0, 3])
        x0, y0 = current_traj[0, [1, 2]]
        x1, y1 = current_traj[0, [4, 5]]
        current_traj[0, 1] = (1 - p0) * x0 + p0 * x1
        current_traj[0, 2] = (1 - p0) * y0 + p0 * y1
        current_traj[0, 3] = start_time
        current_traj[0, 4] = (1 - p1) * x0 + p1 * x1
        current_traj[0, 5] = (1 - p1) * y0 + p1 * y1
        current_traj[0, 6] = end_time

        return current_traj

    p0 = (current_traj[0, 6] - start_time) / (current_traj[0, 6] - current_traj[0, 3])
    p1 = (end_time - current_traj[-1, 3]) / (current_traj[-1, 6] - current_traj[-1, 3])
    current_traj[0, 1] = (1 - p0) * current_traj[0, 4] + p0 * current_traj[0, 1]
    current_traj[0, 2] = (1 - p0) * current_traj[0, 5] + p0 * current_traj[0, 2]
    current_traj[0, 3] = start_time
    current_traj[-1, 4] = (1 - p1) * current_traj[-1, 1] + p1 * current_traj[-1, 4]
    current_traj[-1, 5] = (1 - p1) * current_traj[-1, 2] + p1 * current_traj[-1, 5]
    current_traj[-1, 6] = end_time

    return current_traj


def day_night_split(current_traj, timezone):
    """
    ------------------------------------------------------------------------------------------------------

    This function splits the GPS sub-trajectory into day and night. Where day is defined as the time
     between 8am and 8pm and night is defined as the time between 8pm and 8am.

    Parameters:
    ...........
    current_traj: np.array
        A numpy array containing the GPS data.
    timezone: str
        The timezone of the GPS data.

    Returns:
    ...........
    day_index: np.array
        A boolean array indicating the rows of the sub-trajectory corresponding to day time.
    night_index: np.array
        A boolean array indicating the rows of the sub-trajectory corresponding to night time.

    ------------------------------------------------------------------------------------------------------
    """

    hours = []
    for i in range(current_traj.shape[0]):
        time_list = stamp2datetime(
            (current_traj[i, 3] + current_traj[i, 6]) / 2,
            timezone,
        )
        hours.append(time_list[3])

    hours_array = np.array(hours)
    day_index = (hours_array >= 8) * (hours_array <= 19)
    night_index = np.logical_not(day_index)

    return day_index, night_index


def summary_stats(daily, summary):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the summary statistics of the GPS data.

    Parameters:
    ...........
    daily: pd.DataFrame
        A dataframe containing the daily statistics.
    summary: pd.DataFrame
        A dataframe to be filled with the summary statistics.

    Returns:
    ...........
    summary: pd.DataFrame
        A dataframe filled with the summary statistics.

    ------------------------------------------------------------------------------------------------------
    """

    # number of days
    no_days = daily.shape[0]

    # total observed time
    total_observed_time = sum(daily.observed_time)

    # mean movement time
    mean_move_time = np.mean(daily.move_time)
    # sd movement time
    sd_move_time = np.std(daily.move_time)

    # mean pause time
    mean_pause_time = np.mean(daily.pause_time)
    # sd pause time
    sd_pause_time = np.std(daily.pause_time)

    # mean distance travelled
    mean_dist_travelled = np.mean(daily.dist_travelled)
    # sd distance travelled
    sd_dist_travelled = np.std(daily.dist_travelled)

    # mean home time
    mean_home_time = np.mean(daily.home_time)
    # sd home time
    sd_home_time = np.std(daily.home_time)

    # mean home max dist
    mean_home_max_dist = np.mean(daily.home_max_dist)
    # sd home max dist
    sd_home_max_dist = np.std(daily.home_max_dist)

    # mean home mean dist
    mean_home_mean_dist = np.mean(daily.home_mean_dist)
    # sd home mean dist
    sd_home_mean_dist = np.std(daily.home_mean_dist)

    summary.loc[0] = [
        no_days,
        total_observed_time,
        mean_move_time,
        sd_move_time,
        mean_pause_time,
        sd_pause_time,
        mean_dist_travelled,
        sd_dist_travelled,
        mean_home_time,
        sd_home_time,
        mean_home_max_dist,
        sd_home_max_dist,
        mean_home_mean_dist,
        sd_home_mean_dist,
    ]

    return summary

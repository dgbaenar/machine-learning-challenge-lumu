import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import pandas as pd

import seaborn as sns

from model.fraud_model import FraudModel

style.available
style.use('ggplot')


def calculate_score(input_file='./data/backtesting.csv',
                    output_file='./data/backtesting_result.csv'):
    # Read dataframe and score users
    df = pd.read_csv(input_file)
    proba, X = FraudModel.get_batch_model_response(df)
    # Create final columns
    df = pd.DataFrame({"ID_USER": df["ID_USER"],
                       "FRAUD_SCORE": proba,
                       "VARIABLES": X["variables"],
                       "THRESHOLD": X["threshold"],
                       "FECHA": df["FECHA"]
                       })
    df["IS_FRAUD"] = df["FRAUD_SCORE"] > df["THRESHOLD"]
    # Save dataframe
    df.to_csv(output_file, index=False)

    return df


def backtesting_plots(df):
    sns.ecdfplot(data=df, x="FRAUD_SCORE")
    plt.title('Backtesting: cumulative distribution')
    plt.savefig('./tests/img/cumulative_distribution.png')


def backtesting_thresholds(df):
    threshold_50 = len(df[df['FRAUD_SCORE'] > 0.5])
    threshold_60 = len(df[df['FRAUD_SCORE'] > 0.6])
    threshold_70 = len(df[df['FRAUD_SCORE'] > 0.7])
    threshold_80 = len(df[df['FRAUD_SCORE'] > 0.8])
    threshold_90 = len(df[df['FRAUD_SCORE'] > 0.9])
    total_users = len(df)
    df = pd.DataFrame({
        'THRESHOLDS': [0.5, 0.6, 0.7, 0.8, 0.9],
        'TOTAL_FRAUDSTERS': [threshold_50,
                             threshold_60,
                             threshold_70,
                             threshold_80,
                             threshold_90],
        'PERCENTAGE_FRAUDSTERS':
        [round((threshold_50/total_users)*100, 2),
         round((threshold_60/total_users)*100, 2),
         round((threshold_70/total_users)*100, 2),
         round((threshold_80/total_users)*100, 2),
         round((threshold_90/total_users)*100, 2)],
    })
    df['TOTAL_USERS'] = total_users
    df.to_csv('./data/backtesting_metrics.csv')


def get_percentiles():
    # Read and format backtesting dataset
    interest_cols = ["ID_USER", "FRAUD_SCORE", "FECHA"]
    data = pd.read_csv("./data/backtesting_result.csv", usecols=interest_cols)

    data = data.sort_values(by=["FECHA"], ascending=False)
    data['FRAUD_SCORE'] = data['FRAUD_SCORE'].round(3)

    # Calculate percentiles
    percentiles = list(np.percentile(
        data["FRAUD_SCORE"], np.arange(10, 110, 10)))
    intervals = [(round(percentiles[i], 3), round(percentiles[i+1], 3))
                 for i, _ in enumerate(percentiles) if i < 9]
    return intervals, data


def get_intervals(value, intervals):
    for interval in intervals:
        if value >= interval[0] and value < interval[1]:
            return interval
        if value >= 0 and value < interval[0]:
            return (0, interval[0])


def gropby_intervals(data, intervals):
    data['PROB_INTERVAL'] = data['FRAUD_SCORE'].apply(
        lambda x: get_intervals(x, intervals))
    # Group dataframe
    df_grouped = data.groupby(['FECHA',
                               'PROB_INTERVAL'])['PROB_INTERVAL'].count().to_frame()

    df_grouped.columns = ['COUNT_PROB_INTERVAL']
    df_grouped = df_grouped.reset_index(
        level=['FECHA', 'PROB_INTERVAL'])
    # Pivot table
    table = pd.pivot_table(df_grouped, values='COUNT_PROB_INTERVAL',
                           index=['FECHA'], columns=['PROB_INTERVAL'])
    table_prop = (table.div(table.sum(axis=1), axis=0)*100)

    return table, table_prop


def plot_scoring_intervals(table, table_prop):
    # plot a Stacked Bar Chart using matplotlib
    table_prop.plot(kind='bar',
                    stacked=True,
                    colormap='tab10',
                    title='Distribution of probability intervals by day',
                    figsize=(20, 12))

    for n, x in enumerate([*table.index.values]):
        for (proportion, count, y_loc) in zip(table_prop.loc[x],
                                              table.loc[x],
                                              table_prop.loc[x].cumsum()):

            plt.text(x=n - 0.17,
                     y=(y_loc - proportion) + (proportion / 2),
                     s=f'{count}\n({np.round(proportion, 1)}%)',
                     color="white",
                     fontsize=12,
                     fontweight="bold")

    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.xlabel("Week of the year")
    plt.xticks(rotation=0)
    plt.savefig("./metrics/img/scoring_interval.png")
    plt.close()

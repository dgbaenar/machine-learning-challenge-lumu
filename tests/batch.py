import model.backtesting as mb


def main():
    data = mb.calculate_score()
    mb.backtesting_plots(data)
    mb.backtesting_thresholds(data)

    intervals, data = mb.get_percentiles()
    table, table_prop = mb.gropby_intervals(data, intervals)
    mb.plot_scoring_intervals(table, table_prop)


if __name__ == '__main__':
    main()
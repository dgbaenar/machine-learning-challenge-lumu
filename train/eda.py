import pandas as pd

import sweetviz as sv


df = pd.read_csv("./data/raw_processed.csv")
my_report = sv.analyze(source=df,
                       target_feat="FRAUDE",
                       pairwise_analysis="on")
my_report.show_html(filepath="./templates/eda.html")

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
  sns.set_theme(style="ticks")
  palette = sns.color_palette("rocket_r")
  df = pd.read_excel('./../data/SWaT_A1_A2_Dec2015/drive-download-20220826T152933Z-001/Physical/SWaT_v2.xlsx', header=1)
  sub_system_P1 = ["FIT101", "LIT101", "MV101", "P101", "P102"]
  sub_system_P2 = ["AIT201", "AIT202", "AIT203", "FIT201", "MV201", "P201", "P202", "P203", "P204", "P205", "P206"]
  sub_system_P3 = ["DPIT301", "FIT301", "LIT301", "MV301",	"MV302", "MV303", "MV304", "P301", "P302"]
  sub_system_P4 = ["AIT401", "AIT402",	"FIT401", "LIT401", "P401", "P402", "P403", "P404", "UV401"]
  sub_system_P5 = ["AIT501", "AIT502", "AIT503", "AIT504",	"FIT501", "FIT502",	"FIT503", "FIT504",	"P501",	"P502",	"PIT501", "PIT502",	"PIT503"]
  sub_system_P6 = ["FIT601","P601", "P602", "P603"]

  actutaors_P1 = ["MV101", "P101", "P102"]
  actutaors_P2 = ["MV201", "P201", "P202", "P203", "P204", "P205", "P206"]
  actutaors_P3 = ["MV301", "MV302", "MV303", "MV304", "P301", "P302"]
  actutaors_P4 = ["LIT401", "P401", "P402", "P403", "P404", "UV401"]
  actutaors_P5 = ["P501",	"P502"]
  actutaors_P6 = ["P601", "P602"]

  # Following columns can be ignored because they only have single value:
  ignore_P1 = ["P102"]
  ignore_P2 = ["P201", "P202", "P204", "P206"]
  ignore_P3 = []
  ignore_P4 = ["P401", "P403", "P404"]
  ignore_P5 = ["P502"]
  ignore_P6 = ["P601", "P603"]

  final_actuators_P1 = list(set(actutaors_P1).difference(ignore_P1)) if ignore_P1 is not None else actutaors_P1
  final_actuators_P2 = list(set(actutaors_P2).difference(ignore_P2)) if ignore_P2 is not None else actutaors_P2
  final_actuators_P3 = list(set(actutaors_P3).difference(ignore_P3)) if ignore_P3 is not None else actutaors_P3
  final_actuators_P4 = list(set(actutaors_P4).difference(ignore_P4)) if ignore_P4 is not None else actutaors_P4
  final_actuators_P5 = list(set(actutaors_P5).difference(ignore_P5)) if ignore_P5 is not None else actutaors_P5
  final_actuators_P6 = list(set(actutaors_P6).difference(ignore_P6)) if ignore_P6 is not None else actutaors_P6

  num_cols_P1 = list(set(sub_system_P1).difference(set(final_actuators_P1)).difference(set(ignore_P1)))
  num_cols_P2 = list(set(sub_system_P2).difference(set(final_actuators_P2)).difference(set(ignore_P2)))
  num_cols_P3 = list(set(sub_system_P3).difference(set(final_actuators_P3)).difference(set(ignore_P3)))
  num_cols_P4 = list(set(sub_system_P4).difference(set(final_actuators_P4)).difference(set(ignore_P4)))
  num_cols_P5 = list(set(sub_system_P5).difference(set(final_actuators_P5)).difference(set(ignore_P5)))
  num_cols_P6 = list(set(sub_system_P6).difference(set(final_actuators_P6)).difference(set(ignore_P6)))

  sub_systems = [sub_system_P1, sub_system_P2, sub_system_P3, sub_system_P4, sub_system_P5, sub_system_P6]
  num_cols =  [num_cols_P1, num_cols_P2, num_cols_P3, num_cols_P4, num_cols_P5, num_cols_P6]
  cat_cols =  [final_actuators_P1, final_actuators_P2, final_actuators_P3, final_actuators_P4, final_actuators_P5, final_actuators_P6]
  ignore_cols = [ignore_P1, ignore_P2, ignore_P3, ignore_P4, ignore_P5, ignore_P6]

  print(df.columns)
  sns.lineplot(x = " Timestamp", y = "FIT101", data = df)
  plt.show()


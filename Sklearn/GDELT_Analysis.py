import csv

with open('20200202.export.CSV', 'r', encoding="ISO-8859-1") as csvfile:
     need  =[30, 31, 32, 33, 34, 35, 40, 41, 47, 48]
     f1 = open('clean_20200202.csv','w', newline='')
     writer = csv.writer(f1)
     writer.writerow(["事件类型", "潜在影响", "次数", "数据源", "文章数", "语气", "纬度1", "经度1", "纬度2", "经度2"])
     rows = csvfile.read().split("\n")
     for index, row in enumerate(rows):
         cols = row.split("\t")
         needCols = [cols[i - 1] if i - 1 < len(cols) and cols[i - 1] != "" else None for i in need]
         writer.writerow(needCols)
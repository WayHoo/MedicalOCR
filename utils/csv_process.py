import csv
import os


def write_test_sheet(file_name, file_path="D:\\TestSheetCSV\\"):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_name = os.path.join(file_path, file_name+".csv")
    print("csv file absolute path is %s" % file_name)
    with open(file_name, 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        header = ["name", "score"]
        content = [["Wang", "100"], ["Li", "80"]]
        writer.writerow(header)
        writer.writerows(content)


if __name__ == "__main__":
    write_test_sheet("test")

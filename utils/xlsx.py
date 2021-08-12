# coding=utf-8
import os
import openpyxl

__all__ = ["write_excel_xlsx"]


def write_excel_xlsx(path, xlsx_name, sheet_name, data):
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = os.path.join(path, xlsx_name + ".xlsx")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = sheet_name
    for i in range(0, len(data)):
        for j in range(0, len(data[i])):
            sheet.cell(row=i+1, column=j+1, value=str(data[i][j]))
    workbook.save(file_name)
    print("write test sheet to %s successful..." % file_name)

import pandas as pd
from tkinter import filedialog
import os

class DataSetGenerator:
    def __init__(self, path=None):
        self.excel_path = path if path is not None else filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls .csv")])
        self.data_frame = None
        self.mice_names = []
        # self.dataset = dataset if dataset is not None else pd.DataFrame(columns=['Mouse','day0','day1','day2','day3','day4','day5','day6','day7','day8','day9','day10'])
        # self.exp_name = None

    def initialize_data_frame(self):
        try:
            df = pd.read_excel(self.excel_path, engine='openpyxl', index_col=False)
            self.data_frame = df.dropna(axis='columns')
        except FileNotFoundError as file_not_found_msg:
            print(file_not_found_msg, "\nPlease try to select the file manually\n")
            self.excel_path = filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls .csv")])
            self.initialize_data_frame()

    def get_mice_name_list(self):
        try:
            for name in self.data_frame.loc[:, "Mice"]:
                if name not in self.mice_names:
                    self.mice_names.append(name)
        except KeyError as key_error_msg:
            print(key_error_msg)
            exit(-1)




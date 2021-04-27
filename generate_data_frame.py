import pandas as pd
from tkinter import filedialog
from mouse import Mouse
import os

class DataSetGenerator:
    def __init__(self, path=None):
        self.excel_path = path if path is not None else filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls .csv")])
        self.new_data_to_enter = None
        self.dataset = pd.DataFrame(columns=['Mouse','contraction_day0','contraction_day1','contraction_day2','contraction_day3','contraction_day4','contraction_day5','contraction_day6','contraction_day7','contraction_day8','contraction_day9','contraction_day10', \
                                             'scab_day0','scab_day1','scab_day2','scab_day3','scab_day4','scab_day5','scab_day6','scab_day7','scab_day8','scab_day9','scab_day10', \
                                             'wound_close_day0','wound_close_day1','wound_close_day2','wound_close_day3','wound_close4','wound_close_day5','wound_close_day6','wound_close_day7','wound_close_day8','wound_close_day9','wound_close_day10', \
                                             'size_in_pixels0', 'size_in_pixels_day1', 'size_in_pixels_day2', 'size_in_pixels_day3', 'size_in_pixels_day4', 'size_in_pixels_day5', 'size_in_pixels_day6', 'size_in_pixels_day7', 'size_in_pixels_day8', 'size_in_pixels_day9', 'size_in_pixels_day10', \
                                             'size_in_cm0', 'size_in_cm_day1', 'size_in_cm_day2', 'size_in_cm_day3', 'size_in_cm_day4', 'size_in_cm_day5', 'size_in_cm_day6', 'size_in_cm_day7', 'size_in_cm_day8', 'size_in_cm_day9', 'size_in_cm_day10', \
                                             'pictures_day0', 'pictures_day1', 'pictures_day2', 'pictures_day3', 'pictures_day4', 'pictures_day5', 'pictures_day6', 'pictures_day7', 'pictures_day8', 'pictures_day9', 'pictures_day10'])
        self.mice_names = []
        # self.exp_name = None

## FIXME tomer fix the error checking to print as you like
    def get_new_data_to_enter(self):
        try:
            # extract sheets from excel
            contraction = pd.read_excel(self.excel_path, engine='openpyxl', index_col=False, sheet_name='Contraction')
            scab = pd.read_excel(self.excel_path, engine='openpyxl', index_col=False, sheet_name='Scab')
            wound_close = pd.read_excel(self.excel_path, engine='openpyxl', index_col=False, sheet_name='Wound close')
            dataframes_from_excel = [contraction,scab,wound_close]

            # check if all sheets have the same columns
            if all([set(dataframes_from_excel[0].columns) == set(df.columns) for df in dataframes_from_excel]):
                print('All sheets have the same columns')
            else:
                print('Some sheets have different columns')

            # concatenate data sheets into one dataframe
            stack_df = pd.concat(dataframes_from_excel)
            stack_df = stack_df.reset_index(drop=True)
            stack_df.dropna(axis='columns')
            stack_df.dropna(axis = 'index')

            # check if all columns have valid names
            for col in stack_df.columns:
                if col != 'group' and col != 'Mice':
                    col = col.split()
                    if col[0] != 'Day' or not  0 <= int(col[1]) <= 10:
                        print("incorect columnes name in excel sheets:", col, "is not a valid column")

            self.new_data_to_enter = stack_df

        except FileNotFoundError as file_not_found_msg:
            print(file_not_found_msg, "\nPlease try to select the file manually\n")
            self.excel_path = filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls .csv")])
            self.initialize_data_frame()

    def get_mice_name_list(self):
        try:
            for name in self.new_data_to_enter.loc[:, "Mice"]:
                if name not in self.mice_names:
                    self.mice_names.append(name)
        except KeyError as key_error_msg:
            print(key_error_msg)
            exit(-1)


    def enter_new_mouse(self,mouse_name, exp_name):

        #prepere mouse values
        full_name = str(exp_name) + "_" + str(mouse_name)
        filtered_df = self.new_data_to_enter[self.new_data_to_enter['Mice'].str.contains(mouse_name)]
        contraction = filtered_df[filtered_df['group'].str.contains('contraction')].drop(['group','Mice'], axis='columns')
        scab = filtered_df[filtered_df['group'].str.contains('Scab')].drop(['group','Mice'], axis='columns')

        wound_close = filtered_df[filtered_df['group'].str.contains('Wound close')].drop(['group','Mice'], axis='columns')

        self.dataset = self.dataset.append({'Mouse': full_name}, ignore_index=True)
        mouse_index = self.dataset[self.dataset['Mouse'] == full_name].index.to_numpy()[0]

        #enter mouse values to dataset
        for day in contraction.columns:
            day_num = day.split()[1]
            if contraction.empty is not True:
                self.dataset.at[mouse_index,'contraction_day'+day_num] = contraction.iloc[0][day]
            if scab.empty is not True:
                self.dataset.at[mouse_index,'scab_day'+day_num] = scab.iloc[0][day]
            if wound_close.empty is not True:
                self.dataset.at[mouse_index,'wound_close_day'+day_num] = wound_close.iloc[0][day]







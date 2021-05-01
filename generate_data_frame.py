import pandas as pd
from tkinter import filedialog

sheet_names = ['Contraction', 'Scab', 'Wound close', 'Size by pixels', 'Absolute size']
columns_names = ['contraction', 'scab', 'wound_close', 'size_in_pixels', 'size_in_cm', 'pictures']
csv_path = "wound_measurement_dataset.csv"


class DataSetGenerator:
    def __init__(self, path=None):
        self.excel_path = path if path is not None else filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls .csv")])
        self.new_data_to_enter = None
        self.mice_names = []

        self.dataset = None
        self.create_new_dataset()

    def create_new_dataset(self):
        columns = ['Mouse']
        for column in columns_names:
            for i in range(11):
                columns.append(str(column) + '_day' + str(i))
        self.dataset = pd.DataFrame(columns=columns)
        # pd.DataFrame(columns=['Mouse', 'contraction_day0', 'contraction_day1', 'contraction_day2', 'contraction_day3',
        #                       'contraction_day4', 'contraction_day5', 'contraction_day6', 'contraction_day7',
        #                       'contraction_day8', 'contraction_day9', 'contraction_day10', \
        #                       'scab_day0', 'scab_day1', 'scab_day2', 'scab_day3', 'scab_day4', 'scab_day5', 'scab_day6',
        #                       'scab_day7', 'scab_day8', 'scab_day9', 'scab_day10', \
        #                       'wound_close_day0', 'wound_close_day1', 'wound_close_day2', 'wound_close_day3',
        #                       'wound_close_day4', 'wound_close_day5', 'wound_close_day6', 'wound_close_day7',
        #                       'wound_close_day8', 'wound_close_day9', 'wound_close_day10', \
        #                       'size_in_pixels_day0', 'size_in_pixels_day1', 'size_in_pixels_day2',
        #                       'size_in_pixels_day3', 'size_in_pixels_day4', 'size_in_pixels_day5',
        #                       'size_in_pixels_day6', 'size_in_pixels_day7', 'size_in_pixels_day8',
        #                       'size_in_pixels_day9', 'size_in_pixels_day10', \
        #                       'size_in_cm_day0', 'size_in_cm_day1', 'size_in_cm_day2', 'size_in_cm_day3',
        #                       'size_in_cm_day4', 'size_in_cm_day5', 'size_in_cm_day6', 'size_in_cm_day7',
        #                       'size_in_cm_day8', 'size_in_cm_day9', 'size_in_cm_day10', \
        #                       'pictures_day0', 'pictures_day1', 'pictures_day2', 'pictures_day3', 'pictures_day4',
        #                       'pictures_day5', 'pictures_day6', 'pictures_day7', 'pictures_day8', 'pictures_day9',
        #                       'pictures_day10'])

    def get_new_data_to_enter(self):
        try:
            # extract sheets from excel
            dataframes_from_excel = []
            for sheet in sheet_names:
                dataframes_from_excel.append(
                    pd.read_excel(self.excel_path, engine='openpyxl', index_col=False, sheet_name=sheet))
            # TOMER: I put this in a nice loop :)
            # contraction = pd.read_excel(self.excel_path, engine='openpyxl', index_col=False, sheet_name='Contraction')
            # scab = pd.read_excel(self.excel_path, engine='openpyxl', index_col=False, sheet_name='Scab')
            # wound_close = pd.read_excel(self.excel_path, engine='openpyxl', index_col=False, sheet_name='Wound close')
            # size_in_pixels = pd.read_excel(self.excel_path, engine='openpyxl', index_col=False, sheet_name='Size by pixels')
            # size_in_cm = pd.read_excel(self.excel_path, engine='openpyxl', index_col=False, sheet_name='Absolute size')
            #
            # dataframes_from_excel = [contraction, scab, wound_close, size_in_pixels, size_in_cm]

            # check if all sheets have the same columns
            if not all([set(dataframes_from_excel[0].columns) == set(df.columns) for df in dataframes_from_excel]):
                print('Some sheets have different columns')

            # concatenate data sheets into one dataframe
            stack_df = pd.concat(dataframes_from_excel)
            stack_df = stack_df.reset_index(drop=True)
            stack_df.dropna(axis=['columns', 'index'])
            # stack_df.dropna(axis='index')

            # check if all columns have valid names
            for col in stack_df.columns:
                if col != 'group' and col != 'Mice':
                    col = col.split()
                    if col[0] != 'Day' or not 0 <= int(col[1]) <= 10:
                        print("incorrect columns name in excel sheets:", col, "is not a valid column")

            self.new_data_to_enter = stack_df

        except FileNotFoundError as file_not_found_msg:
            print(file_not_found_msg, "\nPlease try to select the file manually\n")
            self.excel_path = filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls .csv")])
            self.get_new_data_to_enter()

    def get_mice_name_list(self):
        try:
            for name in self.new_data_to_enter.loc[:, "Mice"]:
                if name not in self.mice_names:
                    self.mice_names.append(name)
        except KeyError as key_error_msg:
            print(key_error_msg)
            exit(-1)

    def add_mice(self):
        for mouse in self.mice_names:
            self.enter_new_mouse(mouse_name=mouse, exp_name='exp1')

    def enter_new_mouse(self, mouse_name, exp_name):

        # prepare mouse values
        full_name = str(exp_name) + "_" + str(mouse_name)
        filtered_df = self.new_data_to_enter[self.new_data_to_enter['Mice'].str.contains(mouse_name)]
        contraction = filtered_df[filtered_df['group'].str.contains('contraction')].drop(['group', 'Mice'], axis='columns')
        scab = filtered_df[filtered_df['group'].str.contains('Scab')].drop(['group', 'Mice'], axis='columns')
        wound_close = filtered_df[filtered_df['group'].str.contains('Wound close')].drop(['group', 'Mice'], axis='columns')
        size_in_pixels = filtered_df[filtered_df['group'].str.contains('Size by pixels')].drop(['group', 'Mice'], axis='columns')
        size_in_cm = filtered_df[filtered_df['group'].str.contains('Absolute size')].drop(['group', 'Mice'], axis='columns')
        self.dataset = self.dataset.append({'Mouse': full_name}, ignore_index=True)
        mouse_index = self.dataset[self.dataset['Mouse'] == full_name].index.to_numpy()[0]

        # enter mouse values to dataset
        for day in contraction.columns:
            day_num = day.split()[1]
            if contraction.empty is not True:
                self.dataset.at[mouse_index, 'contraction_day'+day_num] = contraction.iloc[0][day]
            if scab.empty is not True:
                self.dataset.at[mouse_index, 'scab_day'+day_num] = scab.iloc[0][day]
            if wound_close.empty is not True:
                self.dataset.at[mouse_index, 'wound_close_day'+day_num] = wound_close.iloc[0][day]
            if size_in_pixels.empty is not True:
                self.dataset.at[mouse_index, 'size_in_pixels_day'+day_num] = size_in_pixels.iloc[0][day]
            if size_in_cm.empty is not True:
                self.dataset.at[mouse_index, 'size_in_cm_day'+day_num] = size_in_cm.iloc[0][day]


def prepare_dataset():
    data_generator = DataSetGenerator()
    data_generator.get_new_data_to_enter()
    data_generator.get_mice_name_list()
    data_generator.add_mice()
    # print(data_generator.dataset.to_string())
    # data_generator.dataset.to_csv(csv_path)

import pandas as pd
from tkinter import filedialog
import ntpath
import os
from PIL import Image

sheet_names = ['Contraction', 'Scab', 'Wound close', 'Size by pixels', 'Absolute size']
columns_names = ['contraction', 'scab', 'wound_close', 'size_in_pixels', 'size_in_cm', 'pictures']


class Picture:
    def __init__(self):
        self.mouse_name = None
        self.day = None
        self.contraction = None
        self.scab = None
        self.size_in_pixels = None
        self.size_in_cm = None
        self.wound_close = None
        self.pictures = []


class DataSet:
    def __init__(self, path=None):
        self.excel_path = path if path is not None else filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx .xls .csv")])
        self.new_data_to_enter = None
        self.mice_names = []
        self.data = None
        self.create_new_dataset()

    def update_dataset(self):
        pass
    def create_new_dataset(self):
        columns = ['Mouse']
        for column in columns_names:
            for i in range(11):
                columns.append(str(column) + '_day' + str(i))
        self.data = pd.DataFrame(columns=columns)

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
            stack_df.dropna(axis='columns')
            stack_df.dropna(axis='index')

            # check if all columns have valid names
            for col in stack_df.columns:
                if col != 'group' and col != 'Mice':
                    col = col.split()
                    if col[0] != 'Day' or not 0 <= int(col[1]) <= 10:
                        print("incorrect columns name in excel sheets:", col, "is not a valid column")


            # extract pictures from directory tree
            mice_list = stack_df['Mice'].unique()
            mice_pictures = pd.DataFrame(columns=stack_df.columns)
            dir_root = ntpath.dirname(self.excel_path)
            mice_dir_list = next(os.walk(dir_root))[1]

            for mouse in mice_dir_list:
                mouse_name = ''.join(mouse.split(sep="-"))
                if mouse_name not in mice_list:
                    print("mouse name: ", mouse," is not in excel file") # check mouse in directory tree match mice in excel
                    break
                mice_pictures = mice_pictures.append({'group':'pictures','Mice': mouse_name}, ignore_index=True)
                mouse_index = mice_pictures[mice_pictures['Mice'] == mouse_name].index.to_numpy()[0]

                for (root, dirs, files) in os.walk(dir_root+'/'+mouse):
                    day_num = ntpath.split(root)[1].split()
                    if day_num[0].lower() != 'day': continue # go over only directories of days
                    image_list = []
                    for image in files:
                        if not image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')): continue
                        image_list.append(Image.open(root +'/'+image))
                    mice_pictures.at[mouse_index, 'Day ' + day_num[1]] = image_list

            stack_df = stack_df.append(mice_pictures)
            stack_df = stack_df.reset_index(drop=True)
            stack_df.dropna(axis='columns')
            stack_df.dropna(axis='index')

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
        pictures = filtered_df[filtered_df['group'].str.contains('pictures')].drop(['group', 'Mice'], axis='columns')
        self.data = self.data.append({'Mouse': full_name}, ignore_index=True)
        mouse_index = self.data[self.data['Mouse'] == full_name].index.to_numpy()[0]

        # enter mouse values to dataset
        for day in contraction.columns:
            day_num = day.split()[1]
            if contraction.empty is not True:
                self.data.at[mouse_index, 'contraction_day' + day_num] = contraction.iloc[0][day]
            if scab.empty is not True:
                self.data.at[mouse_index, 'scab_day' + day_num] = scab.iloc[0][day]
            if wound_close.empty is not True:
                self.data.at[mouse_index, 'wound_close_day' + day_num] = wound_close.iloc[0][day]
            if size_in_pixels.empty is not True:
                self.data.at[mouse_index, 'size_in_pixels_day' + day_num] = size_in_pixels.iloc[0][day]
            if size_in_cm.empty is not True:
                self.data.at[mouse_index, 'size_in_cm_day' + day_num] = size_in_cm.iloc[0][day]
            if pictures.empty is not True:
                self.data.at[mouse_index, 'pictures_day' + day_num] = pictures.iloc[0][day]

    def get_pic_with_tag(self, mouse_name, day):
        pic = Picture()
        day = str(day)
        mouse_index = self.data[self.data['Mouse'] == mouse_name].index.to_numpy()[0]
        pic.mouse_name = mouse_name
        pic.day = day
        pic.scab = self.data.at[mouse_index, 'scab_day' + day]
        pic.contraction = self.data.at[mouse_index, 'contraction_day' + day]
        pic.size_in_pixels = self.data.at[mouse_index, 'size_in_pixels_day' + day]
        pic.size_in_cm = self.data.at[mouse_index, 'size_in_cm_day' + day]
        pic.wound_close = self.data.at[mouse_index, 'wound_close_day' + day]
        pic.pictures = self.data.at[mouse_index, 'pictures_day' + day]
        return pic


def prepare_dataset():

    data_generator = DataSet(path="/Users/regevazran/Desktop/technion/semester i/project c/data/mouse batches/AWHA-1/AWHA-1.xlsx")  # FIXME Tomer i changed the path and the name of the exel file from example_exp to AWHA-1
    data_generator.get_new_data_to_enter()
    data_generator.get_mice_name_list()
    data_generator.add_mice()
    return data_generator
    # print(data_generator.dataset.to_string())                 # FIXME delete
    # data_generator.dataset.at[0,'pictures_day0'][0].show()    # show picture example form the data set FIXME delete
    # data_generator.dataset.to_csv(csv_path)

import tkinter.messagebox
from tkinter.messagebox import askyesno
from PIL import Image
from easygui import *
import pandas as pd
import ntpath
import cv2
import os

sheet_names = ['Contraction', 'Scab', 'Wound close', 'Size by pixels', 'Absolute size']
columns_names = ['contraction', 'scab', 'wound_close', 'size_in_pixels', 'algo_size_in_pixels', 'min_bounding_radius_in_pixels', 'size_in_cm', 'pictures']  # Check columns names


class Picture:
    def __init__(self):
        self.mouse_name = None
        self.day = None
        self.contraction = None
        self.scab = None
        self.size_in_pixels = None
        self.algo_size_in_pixels = None
        self.min_bounding_radius_in_pixels = None
        self.size_in_cm = None
        self.wound_close = None
        self.pictures = []
        self.dataset_name = "MouseDataSet.csv"


def directories_hierarchy_error():
    cv2.imshow("Hierarchy", cv2.imread("DirectoriesHierarchy.jpg"))
    cv2.waitKey(0)
    exit(-1)


def error_in_mouse_input(error):
    error_msg = {'dataset': "Mouse name or day given is not in existing dataset!",
                 'name_exp': 'Mouse name given is not in the right experiment name format!\nAWHA<X>_P<X>',
                 'name_p': 'Mouse name given is not in the right P - name format!\nAWHA<X>_P<X>',
                 'format': 'Mouse name given is not in the right format!\nAWHA<X>_P<X>',
                 'day': 'Day given was either below the 0 or above 10!'}[error]
    print(error_msg)
    exit(-1)


class DataSet:
    def __init__(self, args):
        self.args = args
        self.excel_path = args.add_excel
        self.dataset_path = args.dataset
        self.dataset = None
        self.new_data_to_enter = None
        self.mice_names = []
        self.exp_name = ""

    def get_dataset(self):
        if self.dataset_path:
            if os.path.isfile(self.dataset_path):
                if self.dataset_path.split(".")[-1] == 'csv':
                    self.dataset = pd.read_csv(self.dataset_path)
                    print(f"Loaded existing dataset! {self.dataset_path.split('/')[-1]}")
                    return
                else:
                    print(f"The dataset given is not a csv file!\nPath is: {self.dataset_path}")
            else:
                print(f"The dataset path given does not exist!\nPath is: {self.dataset_path}")
        else:
            print(f"No dataset path was given by the user.")

        if self.excel_path == "":
            print(f"Neither Excel nor Dataset was given by the user!")
            exit(-1)

        self.create_new_dataframe()

    def create_new_dataframe(self):
        print(f"Creating new empty dataset.")
        columns = ['Mouse']
        for column in columns_names:
            for i in range(11):
                columns.append(str(column) + '_day' + str(i))
        self.dataset = pd.DataFrame(columns=columns)

    def update_dataset(self):
        if not self.excel_path:
            return
        if os.path.isfile(self.excel_path):
            if self.excel_path.split(".")[-1] == 'xlsx' or self.excel_path.split(".")[-1] == 'xls':
                self.get_new_data_to_enter()
            else:
                print(f"The excel path given is not an excel file!\nPath is: {self.excel_path}")
                return
        else:
            print(f"The excel path given does not exist!\nPath is: {self.excel_path}")
            return
        self.get_mice_name_list()
        self.add_mice_to_dataset()
        self.save_dataset_to_file()

    def save_dataset_to_file(self):
        self.dataset.to_csv("MouseDataSet.csv", index=False)

    def get_new_data_to_enter(self):
        self.get_exp_name()
        stack_df = self.get_raw_data_from_excel()
        stack_df = self.get_pictures_data(stack_df)
        self.new_data_to_enter = stack_df

    def get_exp_name(self):
        split_path = self.excel_path.split("/")
        exp_name = ""
        for path in split_path:
            if "AWHA" in path:
                exp_name = path
                break
        if not exp_name:
            print("Cant get experiment name out of excel path!")
            exit(-1)
        exp_name = exp_name.replace("-", "")
        print(f"Found experiment name as: {exp_name}")
        self.exp_name = exp_name

    def get_raw_data_from_excel(self):
        # extract sheets from excel
        dataframes_from_excel = []
        for sheet in sheet_names:
            dataframes_from_excel.append(
                pd.read_excel(self.excel_path, engine='openpyxl', index_col=False, sheet_name=sheet))
        # check if all sheets have the same columns
        if not all([set(dataframes_from_excel[0].columns) == set(df.columns) for df in dataframes_from_excel]):
            print(f'Some sheets have different columns. \n{dataframes_from_excel[0].columns}')
            exit(-1)

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
                    exit(-1)
        return stack_df

    def get_pictures_data(self, stack_df):
        # extract pictures from directory tree
        mice_list = stack_df['Mice'].unique()
        mice_pictures = pd.DataFrame(columns=stack_df.columns)
        dir_root = ntpath.dirname(self.excel_path)
        mice_dir_list = next(os.walk(dir_root))[1]

        for mouse in mice_dir_list:
            mouse_name = ''.join(mouse.split(sep="-"))
            if mouse_name not in mice_list:
                tkinter.messagebox.showerror("Cant find mouse images", f"Mouse {mouse_name} is not in the same folder as the excel given!")
                directories_hierarchy_error()
            mice_pictures = mice_pictures.append({'group': 'pictures', 'Mice': mouse_name}, ignore_index=True)
            mouse_index = mice_pictures[mice_pictures['Mice'] == mouse_name].index.to_numpy()[0]

            for (root, dirs, files) in os.walk(dir_root + '/' + mouse):
                day_num = ntpath.split(root)[1].split()
                # Iterate only in 'days' directories
                if day_num[0].lower() != 'day':
                    continue
                image_list = []
                for image in files:
                    # Take only pictures
                    if not image.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                        continue
                    image_list.append(Image.open(root + '/' + image))
                mice_pictures.at[mouse_index, 'Day ' + day_num[1]] = image_list

        # Add pictures dataframe to the whole dataframe
        stack_df = stack_df.append(mice_pictures)
        stack_df = stack_df.reset_index(drop=True)
        stack_df.dropna(axis='columns')
        stack_df.dropna(axis='index')
        return stack_df

    def get_mice_name_list(self):
        for name in self.new_data_to_enter.loc[:, "Mice"]:
            if name not in self.mice_names:
                self.mice_names.append(name)

    def add_mice_to_dataset(self):
        for mouse in self.mice_names:
            self.enter_new_mouse(mouse_name=mouse, exp_name=self.exp_name)

    def enter_new_mouse(self, mouse_name, exp_name):
        # prepare mouse values
        filtered_df = self.new_data_to_enter[self.new_data_to_enter['Mice'].str.contains(mouse_name)]
        # TODO: add this sequence into for loop
        contraction = filtered_df[filtered_df['group'].str.contains('contraction')].drop(['group', 'Mice'], axis='columns')
        scab = filtered_df[filtered_df['group'].str.contains('Scab')].drop(['group', 'Mice'], axis='columns')
        wound_close = filtered_df[filtered_df['group'].str.contains('Wound close')].drop(['group', 'Mice'], axis='columns')
        size_in_pixels = filtered_df[filtered_df['group'].str.contains('Size by pixels')].drop(['group', 'Mice'], axis='columns')
        size_in_cm = filtered_df[filtered_df['group'].str.contains('Absolute size')].drop(['group', 'Mice'], axis='columns')
        pictures = filtered_df[filtered_df['group'].str.contains('pictures')].drop(['group', 'Mice'], axis='columns')

        mouse_name_to_add = str(exp_name) + "_" + str(mouse_name)
        for i in range(len(self.dataset['Mouse'])):
            if mouse_name_to_add == self.dataset['Mouse'][i]:
                if not askyesno("Mouse already exist in dataset!", f"Mouse {mouse_name_to_add} already exist in dataset!\nDo you want to overwrite?"):
                    return
                else:
                    self.remove_old_mouse(mouse_name_to_add)
                    break

        self.dataset = self.dataset.append({'Mouse': mouse_name_to_add}, ignore_index=True)
        mouse_index = self.dataset[self.dataset['Mouse'] == mouse_name_to_add].index.to_numpy()[0]

        # enter mouse values to dataset
        for day in contraction.columns:  # TODO: check what are those weird ifs, try get into loop by columns
            day_num = day.split()[1]

            if contraction.empty is not True:
                self.dataset.at[mouse_index, 'contraction_day' + day_num] = contraction.iloc[0][day]
            if scab.empty is not True:
                self.dataset.at[mouse_index, 'scab_day' + day_num] = scab.iloc[0][day]
            if wound_close.empty is not True:
                self.dataset.at[mouse_index, 'wound_close_day' + day_num] = wound_close.iloc[0][day]
            if size_in_pixels.empty is not True:
                self.dataset.at[mouse_index, 'size_in_pixels_day' + day_num] = size_in_pixels.iloc[0][day]
            if size_in_cm.empty is not True:
                self.dataset.at[mouse_index, 'size_in_cm_day' + day_num] = size_in_cm.iloc[0][day]
            if pictures.empty is not True:
                self.dataset.at[mouse_index, 'pictures_day' + day_num] = pictures.iloc[0][day]

    def remove_old_mouse(self, mouse_name):
        index = self.dataset.index
        condition = self.dataset["Mouse"] == mouse_name
        mice_indices = index[condition]
        mice_indices_list = mice_indices.tolist()
        self.dataset = self.dataset.drop(mice_indices_list, axis=0)

    def get_last_day(self, mouse_name, cur_day):
        if cur_day is None:
            return None
        if mouse_name not in self.mice_names:
            print("Cant get last day mouse status - mouse name was not found in data set")
            return None
        if cur_day < 1:
            print("Cant get last day mouse status - no preliminary data found")
            return None

        last_day = cur_day - 1
        while last_day >= 0:
            mouse_index = self.dataset[self.dataset['Mouse'] == mouse_name].index.to_numpy()[0]
            data = self.dataset.at[mouse_index, 'algo_size_in_pixels_day' + str(last_day)]
            if data is not None:
                break
            last_day = last_day - 1

        return last_day

    def get_pic_with_tag(self, mouse_name, day):
        pic = Picture()
        day = str(day)
        if mouse_name not in self.mice_names:
            print("get_pic_with_tag: mouse name was not found in data set")
            return None
        mouse_index = self.dataset[self.dataset['Mouse'] == mouse_name].index.to_numpy()[0]
        pic.mouse_name = mouse_name
        pic.day = day
        pic.scab = self.dataset.at[mouse_index, 'scab_day' + day]
        pic.contraction = self.dataset.at[mouse_index, 'contraction_day' + day]
        pic.size_in_pixels = self.dataset.at[mouse_index, 'size_in_pixels_day' + day]
        pic.algo_size_in_pixels = self.dataset.at[mouse_index, 'algo_size_in_pixels_day' + day]
        pic.min_bounding_radius_in_pixels = self.dataset.at[mouse_index, 'min_bounding_radius_in_pixels_day' + day]
        pic.size_in_cm = self.dataset.at[mouse_index, 'size_in_cm_day' + day]
        pic.wound_close = self.dataset.at[mouse_index, 'wound_close_day' + day]
        pic.pictures = self.dataset.at[mouse_index, 'pictures_day' + day]
        return pic

    def update_mouse_in_dataset(self, mouse_name, day, data_value, type="radius"):  # data_type options: ("radius", "area") # FIXME need to test
        # extract current value from data set
        if type == "radius":
            data_type = str('min_bounding_radius_in_pixels_day') + str(day)
        elif type == "area":
            data_type = str('algo_size_in_pixels_day') + str(day)
        else:
            print("regev_update_dataset_merge_to_mine: invalid data type")
            return

        if mouse_name in self.mice_names:
            mouse_index = self.dataset[self.dataset['Mouse'] == mouse_name].index.to_numpy()[0]
        else:
            print("regev_update_dataset_merge_to_mine: mouse name was not found in data set")
            return None

        cur_dataset_value = self.dataset.at[
            mouse_index, data_type]  # dataset value is a list in a form of [avg value,value weight]

        # set new value
        size = 0 if cur_dataset_value is None else cur_dataset_value[0]
        weight = 1 if cur_dataset_value is None else cur_dataset_value[1] + 1

        self.dataset.at[mouse_index, data_type] = [(size + data_value) / weight, weight]
        return

    def check_day_number(self):
        if self.day < 0 or self.day > 10:
            error_in_mouse_input('day')

    def check_if_in_dataset(self):
        for i in range(len(self.dataset['Mouse'])):
            if self.mouse_name == self.dataset['Mouse'][i]:
                if not pd.isna(self.dataset[f'pictures_day{self.day}'][i]):
                    print("all is good")
                    return
                else:
                    break
        error_in_mouse_input("dataset")

    def check_format(self):
        split_name = self.mouse_name.split("_")
        if len(split_name) != 2:
            error_in_mouse_input("format")
        if "AWHA" not in split_name[0] or not split_name[0][-1].isdigit():
            error_in_mouse_input("name_exp")
        if "P" not in split_name[1] or not split_name[1][-1].isdigit():
            error_in_mouse_input("name_p")

    def get_day_from_user(self):
        day = enterbox("Get day to measure", "Get Day")
        self.day = int(day)

    def get_mouse_name_from_user(self):
        exp_name = enterbox("Get Experiment Name", "Get Mouse Name", "AWHA")
        mouse_name = enterbox("Get Mouse Name", "Get Mouse Name", "P")
        self.mouse_name = str(exp_name) + "_" + str(mouse_name)

    def check_input(self):
        print(f'Checking mouse name given {self.mouse_name} - day {self.day}')
        self.check_format()
        self.check_day_number()
        self.check_if_in_dataset()

    def get_mouse_data(self):
        # Check if there was a mouse name argument:
        if self.args.mouse:
            self.mouse_name = self.args.mouse
        else:
            self.get_mouse_name_from_user()

        if self.args.day >= 0:
            self.day = self.args.day
        else:
            self.get_day_from_user()

        self.check_input()

        return self.mouse_name, self.day


def prepare_dataset(args):
    # Create dataset object
    data_generator = DataSet(args)
    # Get existing dataset
    data_generator.get_dataset()
    # Check if excel was given to update it
    data_generator.update_dataset()

    return data_generator

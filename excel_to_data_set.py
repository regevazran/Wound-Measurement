from defines import *

class Mouse:
    def __init__(self):
        self.name = None
        self.contraction = None
        self.scab = None
        self.wound_close = None
        self.wound_size_by_pixel = None
        self.wound_size_by_cm = None
        self.pictures = None

    def add_name(self,mouse_name, exp_name):
        self.name = str(exp_name) + "_" + str(mouse_name)

    def add_day(self,day,contraction,scab,wound_close,wound_size_by_pixel,wound_size_by_cm,pictures):
        self.contraction[day] = contraction
        self.scab[day] = scab
        self.wound_close[day] = wound_close
        self.wound_size_by_pixel[day] = wound_size_by_pixel
        self.wound_size_by_cm[day] = wound_size_by_cm
        self.pictures[day] = pictures



class DataSetGenerator:
    def __init__(self, path, exp_name, dataset=None):
        self.excel_path = path
        self.dataset = dataset if dataset is not None else pd.DataFrame(columns=['Mouse','day0','day1','day2','day3','day4','day5','day6','day7','day8','day9','day10'])
        self.exp_name = exp_name


    def get_mice_name_list(self):
        pass

    def create_new_mouse(self,mouse_name):
        mouse = Mouse()
        mouse.add_name(mouse_name=mouse_name,exp_name=self.exp_name)

    def update_mice_features(self):
        pass
    def add_mice_to_dataset(self):
        pass
    def add_to_data_set(self,mouse):
        pass


import pandas as pd


class DataSetGenerator:
    def __init__(self, path, dataset=None):
        self.excel_path = path
        self.data_frame = None
        self.mice_names = []
        # self.dataset = dataset if dataset is not None else pd.DataFrame(columns=['Mouse','day0','day1','day2','day3','day4','day5','day6','day7','day8','day9','day10'])
        # self.exp_name = None

    def initialize_data_frame(self):
        df = pd.read_excel(self.excel_path, engine='openpyxl', index_col=False)
        self.data_frame = df.dropna(axis='columns')

    def get_mice_name_list(self):
        for name in self.data_frame.loc[:, "Mice"]:
            if name not in self.mice_names:
                self.mice_names.append(name)

    # def create_new_mouse(self,mouse_name):
    #     mouse = Mouse()
    #     mouse.add_name(mouse_name=mouse_name,exp_name=self.exp_name)

    def update_mice_features(self):
        pass
    def add_mice_to_dataset(self):
        pass
    def add_to_data_set(self,mouse):
        pass


from generate_data_frame import DataSetGenerator


def main():

    excel_path = "C:/Users/tomer/OneDrive/שולחן העבודה/try_for_mouse.xlsx"
    data_set = None
    data_generator = DataSetGenerator(path=excel_path, dataset=data_set)
    data_generator.initialize_data_frame()
    data_generator.get_mice_name_list()

    # data_generator.update_mice_features()
    # data_generator.add_mice_to_dataset()





if __name__ == '__main__':
    main()

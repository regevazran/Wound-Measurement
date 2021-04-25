from generate_data_frame import DataSetGenerator

# TODO: Regev, try to give datasetgenerator wrong path and see what happens.
# TODO: also, try to give it the right path. and try not give path at all. IT ALL WORKS! :)
def main():
    data_generator = DataSetGenerator()
    data_generator.initialize_data_frame()
    data_generator.get_mice_name_list()

    print(f"Hope to make you proud!!\nData frame is:\n{data_generator.data_frame}\nMice names are:\n{data_generator.mice_names}")
    # data_generator.update_mice_features()
    # data_generator.add_mice_to_dataset()

if __name__ == '__main__':
    main()

from generate_data_frame import DataSetGenerator

# TODO: Regev, try to give datasetgenerator wrong path and see what happens.
# TODO: also, try to give it the right path. and try not give path at all. IT ALL WORKS! :)
def main():
    data_generator = DataSetGenerator()
    data_generator.get_new_data_to_enter()
    data_generator.get_mice_name_list()

    for mouse in data_generator.mice_names:
        data_generator.enter_new_mouse(mouse_name=mouse,exp_name='exp1')

    print(data_generator.dataset.to_string())
    data_generator.dataset.to_csv("wound_measurement_dataset.csv")
if __name__ == '__main__':
    main()

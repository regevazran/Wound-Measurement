import image_processing
from generate_data_frame import prepare_dataset
from yolo.yolov5_executor import test_yolov5
import argparse

dataset_path = ""
excel_path = ""

# Regev, do not overwrite this path - add your own if you want to below it.
# excel_path = "C:/Users/tomer/OneDrive/שולחן העבודה/Technion/Project B/data/mouse batches/AWHA-1/AWHA-1.xlsx"
# dataset_path = "C:/Users/tomer/OneDrive/שולחן העבודה/Technion/Project B/data/wound_measurement_dataset.csv"


def parse_arguments():
    parser = argparse.ArgumentParser(prog="Wound Measurement", description="Starting Wound Measurement Main App!")
    parser.add_argument('--dataset', type=str, default="MouseDataSet.csv",
                        help='Data set location')
    parser.add_argument('-a', '--add-excel', type=str, default="",
                        help='Add new excel to existing data-set')
    parser.add_argument('-m', '--mouse', type=str, default="",
                        help='Start algorithm on specific mouse given.\nUsage: AWHA<X>_P<X>')
    parser.add_argument('-d', '--day', type=int, default=-1,
                        help='Start algorithm on specific day given.\nUsage: <positive int from 0 to 10>')
    parser.add_argument('-ty', '--test-yolo', action='store_true')
    return parser.parse_args()


def main():
    dataset = prepare_dataset(args)
    mouse_to_segment, day = dataset.get_mouse_data()
    image_pr = image_processing.image_process_algo_master(dataset)
    # TODO: move those mouse to segment and day values to get wound segmentation function correctly!
    image_pr.get_wound_segmentation(mouse_to_segment, day)


if __name__ == '__main__':
    args = parse_arguments()
    # If you want to add another excel to the dataset, please add as an argument or change the path above
    if excel_path != "":
        args.add_excel = excel_path
    if dataset_path != "":
        args.dataset = dataset_path
    if args.test_yolo:
        test_yolov5()
    else:
        main()


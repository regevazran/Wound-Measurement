import image_processing
from generate_data_frame import prepare_dataset
import argparse
data_set_path = "C:/Users/tomer/OneDrive/שולחן העבודה/Technion/Project B/data/wound_measurement_dataset.csv"


def parse_arguments():
    parser = argparse.ArgumentParser(prog="Wound Measurement", description="Starting Wound Measurement Main App!")
    parser.add_argument('-d', '--data-set', type=str, default=data_set_path,
                        help='Data set location')
    parser.add_argument('-a', '--add', type=str, default="",
                        help='Add new excel to existing data-set')
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset = prepare_dataset(args)
    # dataset.get_dataset_path()
    # image_process_algo = ImageProcessAlgoMaster(dataset)
    image_pr = image_processing.image_process_algo_master(dataset)
    image_pr.get_wound_segmentation()


if __name__ == '__main__':
    # main()
    from yolo.yolo_demo import test_yolo
    test_yolo()

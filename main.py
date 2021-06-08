from generate_data_frame import prepare_dataset
from image_processing import image_process_algo_master
from yolo.yolo_demo import *
csv_path = "wound_measurement_dataset.csv"


def main():
    dataset = prepare_dataset()
    image_process_algo = image_process_algo_master(dataset)
    image_process_algo.start()


if __name__ == '__main__':
    # main()
    yolo_demo()

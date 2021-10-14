from generate_data_frame import prepare_dataset
from image_processing import image_process_algo_master
csv_path = "wound_measurement_dataset.csv"


def main():
    dataset = prepare_dataset()
    image_process_algo = image_process_algo_master(dataset)

    image_process_algo.get_wound_segmentation()


if __name__ == '__main__':
    main()
    # yolo_demo()

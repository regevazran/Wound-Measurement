import generate_data_frame
from mice_algo_cv import mice_algo_cv_master


def main():

    generate_data_frame.prepare_dataset()

    mice_algo = mice_algo_cv_master()
    mice_algo.initialize_pictures_from_csv()
    mice_algo.do_something_with_pictures()
    mice_algo.post_process_pictures_data()


if __name__ == '__main__':
    main()

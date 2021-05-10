import generate_data_frame
from mice_algo_cv import mice_algo_cv_master


def main():
    data_generator = generate_data_frame.DataSetGenerator()
    generate_data_frame.prepare_dataset(data_generator)
    mice_names = data_generator.dataset['Mouse']
    print(mice_names[0])
    pics = generate_data_frame.DataSetGenerator.get_pic_with_tag(data_generator, mouse_name=mice_names[0],day= 0)
    pics.pictures[5].show()


    # mice_algo = mice_algo_cv_master()
    # mice_algo.initialize_pictures_from_csv()
    # mice_algo.do_something_with_pictures()
    # mice_algo.post_process_pictures_data()



if __name__ == '__main__':
    main()

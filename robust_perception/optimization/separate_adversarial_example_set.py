import matplotlib.pyplot as plt
import os
import pandas
import shutil

def main():
    col_names = ['process_num', 'iter_num', 'probability']

    package_directory = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(package_directory,
        '../data/experiment1/initial_optimization_run/results.csv')

    data = pandas.read_csv(csv_dir, names=col_names)
    probabilities = data.probability.tolist()[1:]
    process_num = data.process_num.tolist()[1:]

    process_num = [int(j) for j in process_num]
    probabilities = [float(i) for i in probabilities]

    counter = 0

    for num, prob in zip(process_num, probabilities):
        if prob < 0.5:
            file_name = '../data/experiment1/initial_optimization_run/{:05d}_3_color.png'.format(num)
            original_path = os.path.join(package_directory, file_name)

            new_file_name = '../data/experiment1/adversarial_example_set/{:05d}_3_color.png'.format(num)
            new_path = os.path.join(package_directory, new_file_name)
            shutil.copyfile(original_path, new_path)
            print('original_path: {}, new_path: {}'.format(original_path, new_path))
            counter += 1

    print(counter)

if __name__ == "__main__":
    main()
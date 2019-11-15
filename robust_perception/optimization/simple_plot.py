import matplotlib.pyplot as plt
import os
import pandas

def main():
    col_names = ['process_num', 'iter_num', 'probability']

    package_directory = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(package_directory,
        '../data/experiment1/initial_optimization_run/results.csv')

    data = pandas.read_csv(csv_dir, names=col_names)
    probabilities = data.probability.tolist()

    print(probabilities[1:len(probabilities)])
    plt.plot(probabilities[1:len(probabilities)])
    axes = plt.gca()
    axes.set_xlim([0, len(probabilities)])
    axes.set_ylim([0, 1])
    axes.grid()
    plt.show()

if __name__ == "__main__":
    main()
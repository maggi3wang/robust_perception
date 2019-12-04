import matplotlib.pyplot as plt
import os
import pandas
import numpy as np

# def main():
#     col_names = ['process_num', 'iter_num', 'probability']

#     package_directory = os.path.dirname(os.path.abspath(__file__))
#     csv_dir = os.path.join(package_directory,
#         '../data/experiment1/initial_optimization_run/results.csv')

#     data = pandas.read_csv(csv_dir, names=col_names)
#     probabilities = data.probability.tolist()

#     print(probabilities[1:len(probabilities)])
#     plt.plot(probabilities[1:len(probabilities)])
#     axes = plt.gca()
#     axes.set_xlim([0, len(probabilities)])
#     axes.set_ylim([0, 1])
#     axes.grid()
#     plt.show()

def plot_optimization_comparisons(method, csv_name, title):
    # col_names = ['process_num', 'iter_num', 'probability']
    col_names = ['process_num', 'iter_num', 'probability_1', 'probability_2', 'probability_3',
        'probability_4', 'probability_5', 'is_correct', 'time', 'blank']

    package_directory = os.path.dirname(os.path.abspath(__file__))
    opt_dir = os.path.join(package_directory,
        '../data/optimization_comparisons/{}'.format(method))

    csv_dir = os.path.join(opt_dir, csv_name)

    data = pandas.read_csv(csv_dir, names=col_names)
    probabilities = data.probability_3.tolist()
    probabilities = np.array(probabilities[1:len(probabilities)], dtype=float)

    print(probabilities[1:len(probabilities)])
    plt.plot(probabilities[1:len(probabilities)])
    axes = plt.gca()
    axes.set_xlim([0, len(probabilities)])
    axes.set_ylim([0, 1])
    axes.grid()

    axes.set_xlabel('Iteration')
    axes.set_ylabel('Probability')
    axes.set_title('{} Optimization Run (5 hours)'.format(title))

    plt.savefig(os.path.join(opt_dir, '{}_probability_plot.png'.format(method)))
    plt.show()

def main():
    plot_optimization_comparisons(method='slsqp', csv_name='results_for_plotting.csv', title='SLSQP')
    plot_optimization_comparisons(method='nelder_mead', csv_name='results.csv', title='Nelder-Mead')

if __name__ == "__main__":
    main()
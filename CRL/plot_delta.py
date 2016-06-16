from matplotlib import pyplot as plt

from Delta import Delta

if __name__ == '__main__':
    import numpy as np

    step = 1000
    l = np.arange(30, 30000 + 1, step)
    a = np.zeros((len(l), 2))
    for i in range(len(l)):
        e = l[i]
        d = Delta(energy=e, quiet=False, presise=True)
        a[i, 0] = d.delta
        a[i, 1] = d.analytical_delta

    fig = plt.figure()
    ax = fig.add_subplot(111)
    begin = 0
    ax.plot(l[begin:], (a[begin:, 0] - a[begin:, 1]) / a[begin:, 0] * 100, '-r.',
            label='Difference of Delta (Henke - analytical')
    # ax.plot(l, np.log(a[:, 0]), '-r.', label='Delta from Henke')
    # ax.plot(l, np.log(a[:, 1]), '-g.', label='Analytical Delta')
    ax.legend()
    ax.set_xlabel('Energy, [eV]')
    # ax.set_ylabel('Delta (log scale)')
    ax.set_ylabel('Relative Delta difference, %')
    ax.set_title('Comparison of analytical Delta with data from Henke\'s database')
    ax.grid()

    plt.show()

    print('')

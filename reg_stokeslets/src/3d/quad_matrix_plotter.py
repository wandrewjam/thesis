import numpy as np
import matplotlib.pyplot as plt


def main():
    data = np.load('stk_matrix_data.npz')
    cond_array, svd_array = data['cond_array'], data['svd_array']

    fig, ax = plt.subplots(1, 2, sharey='all')
    list8 = []
    list16 = []
    for row in svd_array:
        # i = row[1] * 2 + row[2]
        if row[0] == 8:
            # ax[0].plot([i] * len(row[3:1161]), row[3:1161], 'o')
            list8.append(row[3:1161])
        elif row[0] == 16:
            # ax[1].plot([i] * len(row[3:]), row[3:], 'o')
            list16.append(row[3:])
    ax[0].boxplot(list8)
    ax[1].boxplot(list16)
    ax[0].set_title('$N = 8$')
    ax[1].set_title('$N = 16$')
    ax[0].set_xlabel('Matrix #')
    ax[1].set_xlabel('Matrix #')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 2, sharey='all')
    list8 = []
    list16 = []
    for row in svd_array:
        # i = row[1] * 2 + row[2]
        if row[0] == 8:
            list8.append(row[3:1161])
        elif row[0] == 16:
            list16.append(row[3:])
    ax[0].boxplot(list8)
    ax[1].boxplot(list16)
    ax[0].set_yscale('log')
    ax[0].set_title('$N = 8$')
    ax[1].set_title('$N = 16$')
    ax[0].set_xlabel('Matrix #')
    ax[1].set_xlabel('Matrix #')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

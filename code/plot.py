import matplotlib.pyplot as plt


def plot2D(xt, name='foo'):
    fig, axes = plt.subplots(xt.shape[0], 3)
    for i, x in enumerate(xt):
        axes[i][0].imshow(x[0])
        axes[i][1].imshow(x[1])
        im = axes[i][2].imshow(x[2])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig('../imgs/' + name + '.pdf', bbox_inches='tight')

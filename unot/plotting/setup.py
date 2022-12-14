from distutils.core import setup
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def setup_plt():
    plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 11
    ## sns.set_condition(condition="talk", font_scale=0.9)
    sns.set_context(context="talk", font_scale=1.0)
    palette = ["#1A254B", "#114083", "#A7BED3", "#F2545B", "#A4243B"]
    palette_big = [
        # "#141d3d",  # 0e1a45
        "#1A254B",
        "#114083",
        "#386cb5",
        "#A7BED3",  # blue
        "#c3cdd6",
        "#cccccc",  # gray
        "#d1bebe",
        "#e08285",  # pink  f57176
        "#F2545B",  #  de6f74
        "#A4243B",
        "#8c0e25",  # "#7d0f23",
        # "#570111",
    ]

    cmap1 = LinearSegmentedColormap.from_list("BBR", palette_big)  # , N=40)
    plt.register_cmap(cmap=cmap1)

    cmap2 = LinearSegmentedColormap.from_list(
        "BBR_r_sharp", palette_big[::-1], N=len(palette_big)
    )
    plt.register_cmap(cmap=cmap2)

    cmap3 = LinearSegmentedColormap.from_list("BBR_r", palette_big[::-1])
    plt.register_cmap(cmap=cmap3)

    plt.rcParams["image.cmap"] = "BBR_r"


if __name__ == "__main__":
    setup_plt()
    n = 500
    x = np.random.rand(n, 2) * 2
    # y = np.random.randn(n, 2)
    # c = np.random.rand(n) * 2

    plt.figure(figsize=(13, 6))

    plt.subplot(1, 2, 1)

    plt.scatter(x[:, 0], x[:, 1], c=x[:, 0])
    # plt.scatter(y[:, 0], y[:, 1], c=c)
    plt.colorbar()
    plt.set_cmap("BBR_r_sharp")
    plt.clim(0, 2)

    plt.subplot(1, 2, 2)
    plt.scatter(x[:, 0], x[:, 1], c=x[:, 0])
    # plt.scatter(y[:, 0], y[:, 1], c=c)
    plt.colorbar()
    plt.set_cmap("BBR_r")
    plt.clim(0, 2)

    plt.savefig("./colorbar.pdf", format="pdf")
    # plt.show()

    cmaps = {}

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(category, cmap_list):
        # Create figure and adjust figure height to number of colormaps
        nrows = len(cmap_list)
        figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
        fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4 * 2, figh * 2))
        fig.subplots_adjust(
            top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99
        )
        axs[0].set_title(f"{category} colormaps", fontsize=14)

        for ax, name in zip(axs, cmap_list):
            ax.imshow(gradient, aspect="auto", cmap=mpl.colormaps[name])
            ax.text(
                -0.01,
                0.5,
                name,
                va="center",
                ha="right",
                fontsize=10,
                transform=ax.transAxes,
            )

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axs:
            ax.set_axis_off()

        # Save colormap list for later.
        cmaps[category] = cmap_list
        plt.show()

    plot_color_gradients("", ["BBR_r"])

import os
from typing import Dict, Optional, Tuple

import matplotlib
import matplotlib.ticker as tck
import numpy as np
import orqviz
from matplotlib import pyplot as plt
from orqviz.plot_utils import get_colorbar_from_ax

from .utils import generate_timestamp_str


def create_plot2(
    scan2D_result: orqviz.scans.Scan2DResult,
    fourier_result: orqviz.fourier.FourierResult,
    metrics_dict: Dict[str, float],
    label: str,
    fourier_res_x: int,
    fourier_res_y: int,
    directory_name: Optional[str] = None,
    timestamp_str: Optional[str] = None,
    unit: str = "tau",
    remove_constant: bool = False,
    include_all_metrics: bool = True,
    color_bar_font_size: Optional[int] = None,
    plot_title: str = "",
    fix_x_y_extents: bool = False,
    font_size: int = 15,
    figure_dpi: int = 500,
) -> None:
    """
    fourier_res: maximum frequency to keep on the plot (inclusive), calculated by
        summing up the coefficients of the operator
    """

    plt.rcParams["figure.dpi"] = figure_dpi
    plt.rcParams.update({"font.size": font_size})

    if timestamp_str is None:
        timestamp_str = generate_timestamp_str()

    directory_name = check_and_create_directory(directory_name, timestamp_str)

    if remove_constant:
        scan2D_result, fourier_result = remove_constant_term(
            fourier_result, scan2D_result
        )

    #fig, ax1 = plt.subplots(figsize=(6, 5))
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #fig, ax1 = plt.subplots(figsize=(7.5, 6.5))
    # das ist die richtige einstellung für tv
    #fig, ax1 = plt.subplots(figsize=(9, 8))
    # Create separate figures with specified sizes
    #fig1 = plt.figure(figsize=(10, 8))
    #fig2 = plt.figure(figsize=(10, 8))
        #fig, ax1 = plt.subplots(figsize=(10, 8))
    fig, ax1 = plt.subplots(figsize=(10, 8))
    #plot_scans(fig, ax1, scan2D_result, fix_x_y_extents, unit)
    plot_fourier(fig, ax1, fourier_result, fourier_res_x, fourier_res_y)

    #if color_bar_font_size is not None:
    adjust_color_bar(ax1, ax1, 30)

    title_of_the_plot = get_plot_title(include_all_metrics, metrics_dict, plot_title)

    fig.suptitle(title_of_the_plot)
    fig.tight_layout()
    plt.savefig(os.path.join(directory_name, f"{label}.png"))
    print(f"Saved plot to {os.path.join(directory_name, f'{label}.png')}")
    plt.clf()
    plt.close()


def remove_constant_term(
    fourier_result: orqviz.fourier.FourierResult,
    scan2D_result: orqviz.scans.Scan2DResult,
) -> Tuple[orqviz.scans.Scan2DResult, orqviz.fourier.FourierResult]:
    my_values = np.copy(fourier_result.values)
    constant_term = my_values[0][0]
    my_values[0][0] = 0
    fourier_result = orqviz.fourier.FourierResult(
        my_values, fourier_result.end_points_x, fourier_result.end_points_y
    )
    constant_term_sign = -1 if np.mean(scan2D_result.values) > 0 else 1
    scan2D_result = orqviz.scans.clone_Scan2DResult_with_different_values(
        scan2D_result,
        scan2D_result.values + constant_term_sign * np.abs(constant_term),
    )

    return scan2D_result, fourier_result


def get_x_y_extents(
    scan2D_result: orqviz.scans.Scan2DResult, fix_x_y_extents: bool = False
) -> Tuple[float, float]:
    
    min_x = scan2D_result.params_grid[0][0][0]
    max_x = scan2D_result.params_grid[0][-1][0]
    min_y = scan2D_result.params_grid[0][0][1]
    max_y = scan2D_result.params_grid[-1][0][1]

    if fix_x_y_extents:
        x_extent = np.pi
        y_extent = np.pi

    else:
        x_extent = max_x - min_x
        y_extent = max_y - min_y

    return x_extent, y_extent


def get_plot_title(
    include_all_metrics: bool,
    metrics_dict: Dict[str, float],
    plot_title: str = "",
) -> str:
    if include_all_metrics:
        roughness_label = " \n ".join(
            [f"Roughness index [{k}]: {'%.2f' % v}" for k, v in metrics_dict.items()]
        )
    else:
        tv = metrics_dict.get("tv", 0)
        fourier_density = metrics_dict.get("fourier density using norms", 0)
        roughness_label=""
        #roughness_label = f"\n Total Variation: {'%.2f' % tv} "
        #roughness_label = f"\n Total Variation: {'%.2f' % tv} \n Fourier density: {'%.2f' % fourier_density} "
    return plot_title + roughness_label


def check_and_create_directory(
    directory_name: Optional[str], timestamp_str: str
) -> str:
    # check if directory exists and create it if not
    if directory_name is None:
        if not os.path.exists(os.path.join(os.getcwd(), "results")):
            os.mkdir(os.path.join(os.getcwd(), "results"))
            print(f"Created directory {os.path.join(os.getcwd(), 'results')}")
    directory_name = os.path.join(os.getcwd(), f"results/plots_{timestamp_str}")
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
        print(f"Created directory {directory_name}")

    return directory_name

def plot_scans(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    scan2D_result: orqviz.scans.Scan2DResult,
    fix_x_y_extents: bool = False,
    unit: str = "tau",
) -> None:
    orqviz.scans.plot_2D_scan_result(scan2D_result, fig, ax)

    x_extent, y_extent = get_x_y_extents(scan2D_result, fix_x_y_extents)

    adjust_axis_units(ax, unit, x_extent, y_extent)

    ax.tick_params(axis="both", labelsize=15)
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=x_extent / 4))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=y_extent / 4))
    ax.set_xlabel("$\\gamma$")
    ax.set_ylabel("$\\beta$")

def plot_scans2(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    scan2D_result: orqviz.scans.Scan2DResult,
    fix_x_y_extents: bool = False,
    unit: str = "tau",
) -> None:
    plt.rcdefaults()  # Clear previous settings
    tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "text.latex.preamble":  "".join([r'\usepackage{amssymb}', r'\usepackage{amsmath}']),
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 30,
    "font.size": 30,
    # Make the legend/label fonts a little smaller
    "legend.fontsize":30,
    "xtick.labelsize":30,
    "ytick.labelsize":30
    }
    font = {'family': 'serif',
            'weight': 'normal',
            'size':30}

    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif') 
    
    plt.rcParams.update(tex_fonts)
    # Specify the font properties for the axis ticks
    for tick in ax.get_xticklabels():
        tick.set_fontname('serif')
        tick.set_fontweight('normal') 
        tick.set_fontsize(35)

    for tick in ax.get_yticklabels():
        print(tick)
        tick.set_fontname('serif')
        tick.set_fontweight('normal')
        tick.set_fontsize(35) 

    orqviz.scans.plot_2D_scan_result(scan2D_result, fig, ax)
    # Specify the font family and size
    # Apply the font settings using rc

    x_extent, y_extent = get_x_y_extents(scan2D_result, fix_x_y_extents)

    adjust_axis_units(ax, unit, x_extent, y_extent)

    ax.tick_params(axis="both", labelsize=30)
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=x_extent / 4))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=y_extent / 4))
    
    ax.set_xlabel("$\\gamma$",fontdict={'family': 'serif', 'size': 35})
    ax.set_ylabel("$\\beta$",fontdict={'family': 'serif', 'size': 35})
    ax.set_yticklabels(['t','$-0.5\pi$', '$-0.25\pi$','0', '$0.25\pi$', '$0.5\pi$'])
    ax.set_xticklabels(['t', '$-0.5\pi$', '$-0.25\pi$','0', '$0.25\pi$', '$0.5\pi$'])
    


def adjust_axis_units(
    ax: matplotlib.axes.Axes, unit: "str", x_extent: int, y_extent: int
) -> None:
    if unit == "tau":
        plot_for_tau(ax)

    elif unit == "pi":
        plot_for_pi(ax)
    else:
        raise ValueError("unit must be either 'tau' or 'pi'")

    ax.xaxis.set_major_locator(tck.MultipleLocator(base=x_extent / 4))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=y_extent / 4))


def plot_for_tau(ax: matplotlib.axes.Axes) -> None:
    tau = np.pi * 2
    ax.xaxis.set_major_formatter(
        tck.FuncFormatter(
            lambda val, pos: "{:.2f}$\\tau$".format(val / tau) if val != 0 else "0"
        )
    )
    ax.yaxis.set_major_formatter(
        tck.FuncFormatter(
            lambda val, pos: "{:.2f}$\\tau$".format(val / tau) if val != 0 else "0"
        )
    )


def plot_for_pi(ax: matplotlib.axes.Axes) -> None:
    ax.xaxis.set_major_formatter(
        tck.FuncFormatter(
            lambda val, pos: "{:.2f}$\\pi$".format(val / np.pi) if val != 0 else "0"
        )
    )
    ax.yaxis.set_major_formatter(
        tck.FuncFormatter(
            lambda val, pos: "{:.2f}$\\pi$".format(val / np.pi) if val != 0 else "0"
        )
    )



def plot_fourier(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    fourier_result: orqviz.fourier.FourierResult,
    fourier_res_x: int,
    fourier_res_y: int,
) -> None:
    orqviz.fourier.plot_2D_fourier_result(
        result=fourier_result,
        max_freq_x=fourier_res_x,
        max_freq_y=fourier_res_y,
        fig=fig,
        ax=ax,
    )

    x_axis_base = None
    if fourier_res_x <= 10:
        x_axis_base = 2
    elif fourier_res_x <= 20:
        x_axis_base = 4
    elif fourier_res_x <= 60:
        x_axis_base = 8
    elif fourier_res_x <= 100:
        x_axis_base = 16
    else:
        x_axis_base = 32

    ax.xaxis.set_major_locator(tck.MultipleLocator(base=x_axis_base))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=int(fourier_res_y / 2)))
    ax.tick_params(axis="both", labelsize=15)
    ax.set_xlabel("$f_{\\gamma}$")
    ax.set_ylabel("$f_{\\beta}$")

def plot_fourier2(
    fig: matplotlib.figure.Figure,
    ax: matplotlib.axes.Axes,
    fourier_result: orqviz.fourier.FourierResult,
    fourier_res_x: int,
    fourier_res_y: int,
) -> None:
    
    plt.rcdefaults()  # Clear previous settings
    tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "text.latex.preamble":  "".join([r'\usepackage{amssymb}', r'\usepackage{amsmath}']),
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize":30,
    "font.size":30,
    # Make the legend/label fonts a little smaller
    "legend.fontsize":30,
    "xtick.labelsize":30,
    "ytick.labelsize":30
    }
    font = {'family': 'serif',
            'weight': 'normal',
            'size':30}

    plt.rc('font', **font)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif') 
    
    plt.rcParams.update(tex_fonts)
    # Specify the font properties for the axis ticks
    for tick in ax.get_xticklabels():
        tick.set_fontname('serif')

    for tick in ax.get_yticklabels():
        tick.set_fontname('serif')
    orqviz.fourier.plot_2D_fourier_result(
        result=fourier_result,
        max_freq_x=fourier_res_x,
        max_freq_y=fourier_res_y,
        fig=fig,
        ax=ax,
    )
    #ax.set_yticklabels(['t', '0', '2', '4'])
    #ax.set_xticklabels(['t','0', '32'])
    for tick in ax.get_xticklabels():
        tick.set_fontname('serif')
        tick.set_fontweight('normal') 
        tick.set_fontsize(35)

    for tick in ax.get_yticklabels():
        print(tick)
        tick.set_fontname('serif')
        tick.set_fontweight('normal')
        tick.set_fontsize(35) 

    x_axis_base = None
    if fourier_res_x <= 10:
        x_axis_base = 2
    elif fourier_res_x <= 20:
        x_axis_base = 4
    elif fourier_res_x <= 60:
        x_axis_base = 8
    elif fourier_res_x <= 100:
        x_axis_base = 16
    else:
        x_axis_base = 32

    ax.xaxis.set_major_locator(tck.MultipleLocator(base=x_axis_base))
    #ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=int(fourier_res_y / 2)))
    ax.tick_params(axis="both", labelsize=30)
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_xticklabels(['0', '$8$', '$16$','$24$', '$32$'])
    ax.set_xlabel("$f_{\\gamma}$", fontdict={'family': 'serif', 'fontweight': 'normal', 'size': 35})
    ax.set_ylabel("$f_{\\beta}$", fontdict={'family': 'serif', 'fontweight': 'normal','size': 35})
    # Create separate figures with specified sizes
    #plt.figure(figsize=(9, 8))
   
    #ax.set_xlabel("$\\gamma$",fontdict={'family': 'serif', 'size':30})
    #ax.set_ylabel("$\\beta$",fontdict={'family': 'serif', 'size':30})
    #hfont = {'fontname': 'serif'}
    #plt.rcParams['text.usetex'] = True
    #plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{mathrsfs}'
    #ax.set_yticklabels(['t','$-0.5\pi$', '$-0.25\pi$','0', '$0.25\pi$', '$0.5\pi$'])
    
    #plt.xlabel(r'$\mathrm{f_\gamma}$', fontname='serif')
    #ax.set_xlabel("$\mathrm{f_\gamma}$",labelpad=2, fontsize=10,fontweight='normal', fontfamily="serif")


def adjust_color_bar2(
    ax1: matplotlib.axes.Axes, ax2: matplotlib.axes.Axes, color_bar_font_size: int = 35
) -> None:
    cbar1 = get_colorbar_from_ax(ax1)
    #cbar1.set_ticks([0, 0.09, 0.18, 0.27])

    cbar1.ax.tick_params(labelsize=color_bar_font_size)
    #cbar1.set_ticks([-2, 1, 4])
    #cbar1.set_ticklabels(['-5', '0', '5'])
    #cbar2 = get_colorbar_from_ax(ax2)
    #cbar2.ax.tick_params(labelsize=30)
    #cbar2.set_ticks([0, 0.25, 0.5])  # Set your desired tick positions here for cbar2



def adjust_color_bar(
    ax1: matplotlib.axes.Axes, ax2: matplotlib.axes.Axes, color_bar_font_size: int = 15
) -> None:
    cbar1 = get_colorbar_from_ax(ax1)
    cbar1.ax.tick_params(labelsize=color_bar_font_size)
    cbar2 = get_colorbar_from_ax(ax2)
    cbar2.ax.tick_params(labelsize=color_bar_font_size)

def create_plot(
    scan2D_result: orqviz.scans.Scan2DResult,
    fourier_result: orqviz.fourier.FourierResult,
    metrics_dict: Dict[str, float],
    label: str,
    fourier_res_x: int,
    fourier_res_y: int,
    directory_name: Optional[str] = None,
    timestamp_str: Optional[str] = None,
    unit: str = "tau",
    remove_constant: bool = False,
    include_all_metrics: bool = True,
    color_bar_font_size: Optional[int] = None,
    plot_title: str = "",
    fix_x_y_extents: bool = False,
    font_size: int = 15,
    figure_dpi: int = 500,
) -> None:
    """
    fourier_res: maximum frequency to keep on the plot (inclusive), calculated by
        summing up the coefficients of the operator
    """

    plt.rcParams["figure.dpi"] = figure_dpi
    plt.rcParams.update({"font.size": font_size})

    if timestamp_str is None:
        timestamp_str = generate_timestamp_str()

    directory_name = check_and_create_directory(directory_name, timestamp_str)

    if remove_constant:
        scan2D_result, fourier_result = remove_constant_term(
            fourier_result, scan2D_result
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    plot_scans(fig, ax1, scan2D_result, fix_x_y_extents, unit)

    plot_fourier(fig, ax2, fourier_result, fourier_res_x, fourier_res_y)

    if color_bar_font_size is not None:
        adjust_color_bar(ax1, ax2, color_bar_font_size)

    title_of_the_plot = get_plot_title(include_all_metrics, metrics_dict, plot_title)

    fig.suptitle(title_of_the_plot)
    fig.tight_layout()
    plt.savefig(os.path.join(directory_name, f"{label}.png"))
    print(f"Saved plot to {os.path.join(directory_name, f'{label}.png')}")
    plt.clf()
    plt.close()


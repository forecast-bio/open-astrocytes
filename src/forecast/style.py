"""TODO"""

#

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.style as mpls

from cycler import cycler

from typing import (
    Optional,
    Any,
)
from numpy.typing import (
    NDArray,
    ArrayLike,
)
from matplotlib.axes import (
    Axes,
)


##

HOUSE_COLORS = {
    'neutral_pink': '#F9F4F5',
    'blue': '#00A6DB',
    'yellow': '#FFCE02',
    'red': '#F95838',
    'pink': '#FFB6C6',
    'green': '#007C35',
    'grey': '#595959',
    # 'light_blue': ''
    # 'light_yellow': ''
    # 'light_red': ''
    # 'light_green': ''
}

CYCLE_ORDER = [
    'blue',
    'yellow',
    'green',
    'red',
]


##

class HouseStyle:

    def __init__( self,
            grids: bool = False,
        ):
        """
        TODO
        """

        self.house_params = {
            'font.sans-serif': [
                'Helvetica',
                'Helvetica Neue',
                'Arial',   
            ],
            'font.size': 14,
            #
            'figure.dpi': 150,
            'savefig.dpi': 400,
            #
            'axes.facecolor': HOUSE_COLORS['neutral_pink'],
            'axes.edgecolor': HOUSE_COLORS['grey'],
            'axes.grid': grids,
            'axes.grid.axis': 'y',
            'axes.spines.top': False,
            'axes.spines.bottom': False,
            'axes.spines.left': False,
            'axes.spines.right': False,
            #
            'axes.prop_cycle': cycler(
                color = [ HOUSE_COLORS[c] for c in CYCLE_ORDER ]
            ),
            #
            'axes.titlepad': 8,
        }

    def __enter__( self ) -> 'HouseStyle':

        # Cache previous settings
        # self._prev_params = {
        #     'axes.prop_cycle': mpl.rcParams['axes.prop_cycle'],
        # }
        self._prev_params = {
            k: mpl.rcParams[k]
            for k in self.house_params.keys()
        }

        # Set up house style
    #     mpl.rcParams['axes.prop_cycle'] = cycler(
    #         color = [ HOUSE_COLORS[c] for c in CYCLE_ORDER ]
    #     )
    #     mpl.rcParams['axes.facecolor'] = HOUSE_COLORS['neutral_pink']
    #     mpl.rcParams['axes.edgecolor'] = 'none'
    #    axisbelow=True, grid=True, prop_cycle=colors)
        for k, v in self.house_params.items():
            mpl.rcParams[k] = v
        
        return self

    def __exit__( self, exc_type, exc_val, exc_tb ):
        # Wrap up figure
        # plt.gcf().patch.set_linewidth( 1 )
        plt.gcf().patch.set_facecolor( HOUSE_COLORS['neutral_pink'] )

        plt.show()

        # Restore cached settings
        for k, v in self._prev_params.items():
            mpl.rcParams[k] = v

    #

    def show_micrograph( self,
            axes: Axes,
            image: ArrayLike,
            scale_x: float = 1.,
            scale_y: float = 1.,
            #
            *args, **kwargs
        ):
        image = np.array( image )
        axes.imshow( image, *args,
            cmap = 'bone',
            extent = (0, scale_x * image.shape[0], 0, scale_y * image.shape[1]),
            interpolation = 'none',
            **kwargs
        )

    def label( self,
            title: Optional[str] = None,
            subtitle: Optional[str] = None,
            xlabel: Optional[str] = None,
            ylabel: Optional[str] = None,
            data_xlim: Optional[tuple[Any, Any]] = None,
            xaxis_y: float = 0.,
            data_ylim: Optional[tuple[Any, Any]] = None,
            yaxis_x: float = 0.,
        ):

        if subtitle is not None:
            plt.suptitle( subtitle, ha = 'left',
                # pad = 24,
                x = 0.024,
                y = 0.98,
                # fontname = 'Times New Roman',
                # fontsize = 24,
                fontsize = 14,
                color = HOUSE_COLORS['grey'],
                # fontdict = {
                #     'fontsize': 24,
                #     'fontweight': 'bold',
                # }
            )

        if title is not None:
            plt.title( title, ha = 'left',
                x = -0.135,
                # y = 0.8,
                pad = 40,
                fontname = 'Times New Roman',
                fontsize = 24,
                # fontsize = 18,
                # color = HOUSE_COLORS['grey'],
                # transform = plt.gcf().transFigure,
            )

        if xlabel is not None:
            plt.xlabel( xlabel, loc = 'left',
                fontsize = 20,
                color = HOUSE_COLORS['grey'],
                labelpad = 10,
            )
            plt.gca().tick_params( axis = 'x',
                pad = 4,
            )
        if ylabel is not None:
            plt.ylabel( ylabel, loc = 'bottom',
                fontsize = 20,
                color = HOUSE_COLORS['grey'],
                labelpad = 4,
            )
            plt.gca().tick_params( axis = 'y',
                pad = 4,
            )
        
        if data_xlim is not None:
            plt.plot( data_xlim, [xaxis_y, xaxis_y], 'k-',
                linewidth = 1
            )
            tick_locator = ticker.AutoLocator()
            ticks = tick_locator.tick_values( data_xlim[0], data_xlim[1] )
            plt.gca().set_xticks( ticks )

        if data_ylim is not None:
            plt.plot( [yaxis_x, yaxis_x], data_ylim, 'k-',
                linewidth = 1
            )
            tick_locator = ticker.AutoLocator()
            ticks = tick_locator.tick_values( data_ylim[0], data_ylim[1] )
            plt.gca().set_yticks( ticks )


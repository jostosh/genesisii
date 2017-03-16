import tensorflow as tf
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

from plotly.grid_objs import Grid, Column
import time

from adam import AdamOptimizer
from eve import EveOptimizer
from rmseve import RMSEveOptimizer
from rmsprop import RMSPropOptimizer
from keras_eve import Eve
from keras.optimizers import Adadelta, Adagrad, SGD

from argparse import ArgumentParser


str_to_opt = {
    'Adam': AdamOptimizer,
    'Eve': EveOptimizer,
    'Eve Koushnik et. al': Eve,
    'RMSProp': RMSPropOptimizer,
    'RMSEve': RMSEveOptimizer,
    'AdaGrad': Adagrad,
    'SGD': SGD
}


def parse_params(params):
    return {p.split('=')[0]: float(p.split('=')[1]) for p in params.split('|')}

def execute(args):
    loss, x, x_mesh, y = define_loss()

    s = 5
    columns = []

    columns.append(Column(x_mesh, 'xmesh'))
    columns.append(Column(y,      'ymesh'))

    sess = tf.Session()

    if args.params:
        paths = {str_to_opt[s](lr=0.001, **parse_params(p)): {'x': [], 'y': [], 'name': s + ' ' + p.replace('_', ' ').title()}
                 for s, p in zip(args.optimizers, args.params)}
    else:
        paths = {str_to_opt[s](lr=0.001): {'x': [], 'y': [], 'name': s} for s in args.optimizers}

    #if args.plotd:
    #    feedback = {
    #        rmseve: {'x': [], 'y': [], 'name': 'd_RMSEve'}
    #    }

    maxlen = 0

    maxd = 0

    for opt in paths.keys():
        print(opt)
        updates = opt.get_updates(params=[x], loss=loss, constraints=[])
        sess.run(tf.global_variables_initializer())

        for i in range(2500):
            #if opt in feedback:
            #    _, xnum, ynum, d = sess.run([updates, x, loss, opt.d])
            #    feedback[opt]['y'].append(d)
            #    feedback[opt]['x'].append(d)

#                maxd = max(maxd, d)
 #           else:
            _, xnum, ynum = sess.run([updates, x, loss])
            paths[opt]['x'].append(xnum)
            paths[opt]['y'].append(ynum)


            if xnum < -1 or xnum > 1:
                maxlen = max(maxlen, i)
                break
    #print(maxd)
    #exit(0)

    layout = go.Layout(
        title='Optimizers' if not args.title else args.title,
        autosize=False,
        width=800,
        height=800,
        margin=dict(
            l=65,
            r=50,
            b=65,
            t=90
        ),
        font=dict(size=14, family='sans-serif'),
        paper_bgcolor='(0,0,0,0)',
        xaxis=go.XAxis(range=[0, 1], title='x'),
        yaxis=go.YAxis(range=[0, 1.1], title='loss'),
        updatemenus=[{
            'buttons': [
                {'args': [None, dict(frame=dict(duration=50, redraw=False),
                                     transition=dict(duration=10),
                                     fromcurrent=True,
                                     mode='immediate')],
                 'label': 'Play',
                 'method': 'animate'}
            ],
            'pad': {'r': 10, 't': 87},
            'showactive': True,
            'type': 'buttons'
        }],
        showlegend=True,
        # yaxis2=dict(
        #     title='d',
        #     titlefont=dict(
        #         color='rgb(148, 103, 189)'
        #     ),
        #     tickfont=dict(
        #         color='rgb(148, 103, 189)'
        #     ),
        #     overlaying='y',
        #     side='right',
        #     range=[0, maxd],
        #     showgrid=False
        # )
    )

    print("Paths computed")
    [columns.append(Column(d['x'][:1], 'x_{}'.format(d['name']))) for d in paths.values()]
    [columns.append(Column(d['y'][:1], 'y_{}'.format(d['name']))) for d in paths.values()]
    # [columns.append(Column(d['y'], 'd_{}'.format(d['name']))) for d in feedback.values()]


    len_by_opt = {}

    for opt, d in paths.items():
        l = len(d['x'])
        for i in range(l // s + 1):
            #if (i+1)*s <= len(d['x']):
            end = min((i+1)*s, l)
            columns.append(Column(d['x'][end-1:end], 'x_{}_{}'.format(d['name'], i)))
            columns.append(Column(d['y'][end-1:end], 'y_{}_{}'.format(d['name'], i)))

            len_by_opt[opt] = i

    last_ref = {
        opt: '' for opt in paths.keys()
    }

    print("Generated columns")
    grid = Grid(columns)
    py.grid_ops.upload(grid, 'optimizers' + str(time.time()), auto_open=False)

    frame_data = []

    function = [
        go.Scatter(
            xsrc=grid.get_column_reference('xmesh'),
            ysrc=grid.get_column_reference('ymesh'),
            hoverinfo='skip',
            mode='lines',
            showlegend=False
        )
    ] + [
        # go.Scatter(
        #     xsrc=grid.get_column_reference('xmesh'),
        #     ysrc=grid.get_column_reference('d_{}'.format(d['name'])),
        #     name='d RMSEve',
        #     line=dict(dash='dash'),
        #     yaxis='y2'
        # ) for d in feedback.values()
    ]

    data = [
        go.Scatter(
            xsrc=grid.get_column_reference('x_{}'.format(d['name'])),
            ysrc=grid.get_column_reference('y_{}'.format(d['name'])),
            name=d['name'],
            mode='markers',
            marker=go.Marker(size=18, symbol=opt_idx)
        ) for opt_idx, d in enumerate(paths.values())
    ]
    print("Generating frames")
    for i in range(maxlen // s):
        data_group = []
        print("Now at frame {}".format(i))
        for opt_idx, (opt, d) in enumerate(paths.items()):
            idx = min(len_by_opt[opt], i)
            # if (i + 1) * s < len(d['x']):
            data_group.append(
                go.Scatter(
                    xsrc=grid.get_column_reference('x_{}_{}'.format(d['name'], idx)),
                    ysrc=grid.get_column_reference('y_{}_{}'.format(d['name'], idx)),
                    name=d['name'],
                    mode='markers',
                    marker=go.Marker(size=18, symbol=opt_idx)
                )
            )
        frame_data.append({'data': function + data_group})


    #data += [go.Scatter3d(
    #    mode='lines', line=dict(width=8), **scatter_set
    #) for scatter_set in paths.values()]

    print("Generating plot")
    fig = go.Figure(data=function + data, layout=layout, frames=frame_data)
    py.create_animations(fig, filename='optimizers_animation_rms_1d' + str(time.time()))
    #py.plot(fig, filename='optimizers.html')


def define_loss():
    x = tf.Variable(0.0, dtype=tf.float32)
    cos_fac = 0.025
    cos_freq = 5
    loss = 1 - x + tf.cos(x * cos_freq * 2 * np.pi) * cos_fac
    x_mesh = np.linspace(0, 1, 100)
    y = 1 - x_mesh + np.cos(x_mesh * cos_freq * 2 * np.pi) * cos_fac
    return loss, x, x_mesh, y


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--optimizers", nargs='+',
                        default=['Adam', 'Eve', 'RMSProp', 'RMSEve', 'AdaGrad', 'SGD', 'Eve Koushnik et. al'])
    parser.add_argument("--params", nargs='+', default=[])
    parser.add_argument("--plotd", dest='plotd', action='store_true')
    parser.add_argument("--title", default=None)

    execute(parser.parse_args())


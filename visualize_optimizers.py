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

x = tf.Variable(-1.0, dtype=tf.float32)
y = tf.Variable(1e-10, dtype=tf.float32)

yfac = 0.01

loss = tf.square(x) - tf.square(y) * yfac

x_mesh, y_mesh = np.meshgrid(np.linspace(-1.0, 0.5, 15), np.linspace(-1.0, 5.0, 60))
z = x_mesh ** 2 - (y_mesh ** 2) * yfac
s = 150
step = 50
columns = []

columns.append(Column(x_mesh, 'xmesh'))
columns.append(Column(y_mesh, 'ymesh'))
columns.append(Column(z,      'zmesh'))


layout = go.Layout(
    title='Optimizers',
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
    scene=(dict(
    xaxis=dict(range=[-1,0, 0.5]),
    yaxis=dict(range=[-1, 5]),
    zaxis=dict(range=[np.min(z), 1.0])
    )),
    updatemenus= [{
       'buttons': [
           {'args': [None, dict(frame=dict(duration=300, redraw=False),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode='immediate')],
            'label': 'Play',
            'method': 'animate'}
       ],
        'pad': {'r': 10, 't': 87},
        'showactive': True,
        'type': 'buttons'
    }],
    showlegend=True
)

adam = AdamOptimizer()
eve = EveOptimizer()
rms = RMSPropOptimizer()
rmseve = RMSEveOptimizer()

sess = tf.Session()

paths = {
    adam: {'x': [], 'y': [], 'z': [], 'name': 'Adam'},
    eve:  {'x': [], 'y': [], 'z': [], 'name': 'Eve'},
    rms: {'x': [], 'y': [], 'z': [], 'name': 'RMSProp'},
    rmseve: {'x': [], 'y': [], 'z': [], 'name': 'RMSEve'},
}

maxlen = 0

for opt in paths.keys():
    updates = opt.get_updates([x, y], loss)
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        _, xnum, ynum, znum = sess.run([updates, x, y, loss])
        paths[opt]['x'].append(xnum)
        paths[opt]['y'].append(ynum)
        paths[opt]['z'].append(znum)
        if xnum < -1 or xnum > 1:
            print(i, xnum, ynum, znum)
            maxlen = max(maxlen, i)
            break
        if ynum < -1 or ynum > 5:
            print(i, xnum, ynum, znum)
            maxlen = max(i, maxlen)
            break

print("Paths computed")
[columns.append(Column(d['x'][:1], 'x_{}'.format(d['name']))) for d in paths.values()]
[columns.append(Column(d['y'][:1], 'y_{}'.format(d['name']))) for d in paths.values()]
[columns.append(Column(d['z'][:1], 'z_{}'.format(d['name']))) for d in paths.values()]

len_by_opt = {}

for opt, d in paths.items():
    l = len(d['x'])
    for i in range(l // s + 1):
        #if (i+1)*s <= len(d['x']):
        end = min((i+1)*s, l)
        columns.append(Column(d['x'][:end:step], 'x_{}_{}'.format(d['name'], i)))
        columns.append(Column(d['y'][:end:step], 'y_{}_{}'.format(d['name'], i)))
        columns.append(Column(d['z'][:end:step], 'z_{}_{}'.format(d['name'], i)))

        len_by_opt[opt] = i

        #else:
        #    columns.append(Column(d['x'][::step], 'x_{}_{}'.format(d['name'], i)))
        #    columns.append(Column(d['y'][::step], 'y_{}_{}'.format(d['name'], i)))
        #    columns.append(Column(d['z'][::step], 'z_{}_{}'.format(d['name'], i)))

        #    columns.append(Column([d['x'][-1]], 'x_{}_{}e'.format(d['name'], i)))
        #    columns.append(Column([d['y'][-1]], 'y_{}_{}e'.format(d['name'], i)))
        #    columns.append(Column([d['z'][-1]], 'z_{}_{}e'.format(d['name'], i)))

last_ref = {
    opt: '' for opt in paths.keys()
}

print("Generated columns")
grid = Grid(columns)
py.grid_ops.upload(grid, 'optimizers' + str(time.time()), auto_open=False)

frame_data = []

surface = [
    go.Surface(
        xsrc=grid.get_column_reference('xmesh'),
        ysrc=grid.get_column_reference('ymesh'),
        zsrc=grid.get_column_reference('zmesh'),
        opacity=0.9,
        contours=go.Contours(
            x=dict(show=True, width=0.05), y=dict(show=True, width=0.05)
        ),
        showscale=False,
        hoverinfo='skip'
    )
]

data = [
    go.Scatter3d(
        xsrc=grid.get_column_reference('x_{}'.format(d['name'])),
        ysrc=grid.get_column_reference('y_{}'.format(d['name'])),
        zsrc=grid.get_column_reference('z_{}'.format(d['name'])),
        name=d['name'],
        mode='lines',
        line=dict(width=8)
    ) for d in paths.values()
]

print("Generating frames")
for i in range(maxlen // s):
    data_group = []
    print("Now at frame {}".format(i))
    for opt, d in paths.items():
        idx = min(len_by_opt[opt], i)
        # if (i + 1) * s < len(d['x']):
        data_group.append(
            go.Scatter3d(
                xsrc=grid.get_column_reference('x_{}_{}'.format(d['name'], idx)),
                ysrc=grid.get_column_reference('y_{}_{}'.format(d['name'], idx)),
                zsrc=grid.get_column_reference('z_{}_{}'.format(d['name'], idx)),
                mode='lines',
                line=dict(width=8),
                name=d['name']
            )
        )
    frame_data.append({'data': data_group + surface})


#data += [go.Scatter3d(
#    mode='lines', line=dict(width=8), **scatter_set
#) for scatter_set in paths.values()]

print("Generating plot")
fig = go.Figure(data=data + surface, layout=layout, frames=frame_data)
py.create_animations(fig, filename='optimizers_animation' + str(time.time()))
#py.plot(fig, filename='optimizers.html')


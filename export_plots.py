from tensorflow.python.summary import event_accumulator
import argparse
import os
import pickle
import json
from tensorflow.python.summary.event_accumulator import IsTensorFlowEventsFile, EventAccumulator
from deeprl.common.hyper_parameters import HyperParameters
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import matplotlib as mpl
import tensorflow as tf
import pprint

import re

import plotly.plotly as py
import plotly.graph_objs as go

from plotly.grid_objs import Grid, Column

mpl.rc('text', usetex=True)
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': 14})
mpl.rc('xtick', labelsize=14)
mpl.rc('ytick', labelsize=14)
import colorlover as cl


colorscale = cl.scales['8']['qual']['Dark2']
colorscalen = []

for c in cl.to_numeric(colorscale):
    colorscalen.append((c[0]/255., c[1]/255, c[2]/255))
colorscalen.append((0., 0., 0.))
colorscalen.append((1., 0., 0.))


def event_arrays_to_np_arrays(event_array):
    value_by_step = {}
    np_arrays_x = []
    np_arrays_y = []
    for event in event_array:
        for scalar_event in event:
            if scalar_event.step not in value_by_step:
                value_by_step[scalar_event.step] = [scalar_event.value]
            else:
                value_by_step[scalar_event.step].append(scalar_event.value)
        np_arrays_x.append(np.asarray([se.step for se in event]))
        np_arrays_y.append(np.asarray([se.value for se in event]))

    continuous_arrays_x = []
    continuous_arrays_y = []

    min_x = -np.inf
    max_x = np.inf
    for x_arr, y_arr in zip(np_arrays_x, np_arrays_y):
        xc_arr = []
        yc_arr = []
        min_x = max(min_x, x_arr[0])
        max_x = min(max_x, x_arr[-1])

        for i in range(len(x_arr) - 1):
            xc_arr.append(np.linspace(x_arr[i], x_arr[i+1], x_arr[i+1] - x_arr[i] + 1, dtype=np.int64)[:-1])
            yc_arr.append(np.linspace(y_arr[i], y_arr[i+1], x_arr[i+1] - x_arr[i] + 1)[:-1])
        xc_arr.append(np.asarray([x_arr[-1]]))
        yc_arr.append(np.asarray([y_arr[-1]]))

        continuous_arrays_x.append(np.concatenate(xc_arr))
        continuous_arrays_y.append(np.concatenate(yc_arr))


    c_arrays_y = []
    for x_arr, y_arr in zip(continuous_arrays_x, continuous_arrays_y):
        idx1 = x_arr.tolist().index(min_x)
        idx2 = x_arr.tolist().index(max_x)

        #y_arr = np.asarray(y_arr[idx1:idx2])

        #print(x_arr[idx1:idx2], idx1, idx2, len(y_arr[idx1:idx2]))

        c_arrays_y.append(y_arr[idx1:idx2])
    #print(min_x, max_x)

    if args.interpolate:
        values = np.mean(np.stack(c_arrays_y), axis=0)
        errors = np.std(np.stack(c_arrays_y), axis=0)
        steps = np.arange(min_x, max_x)
        return steps, values, errors, np_arrays_x, np_arrays_y

    error_by_step = {}
    for step, val in value_by_step.items():
        error_by_step[step] = np.std(val)
        value_by_step[step] = np.mean(val)

    steps = np.asarray([k for k in value_by_step.keys()])
    values = np.asarray([v for v in value_by_step.values()])
    errors = np.asarray([v for v in error_by_step.values()])
    return steps, values, errors, np_arrays_x, np_arrays_y


def obtain_name(hp):

    return hp['optimizer'].capitalize()

    '''
    function_by_name = {
        'idx': lambda p: 'Fold ' + hp['idx'],
        'model': lambda p: {'default': 'Default CNN', 'spatial': 'SIWS CNN'}[hp[p]],
        'per_feature': lambda p: '/F' if (p in hp and hp[p] == True) else ''
    }

    if args.trace_by:
        return ' '.join(function_by_name[n](n) for n in args.trace_by)

    return hp['model'].upper().replace('_', '-') + \
        (' {}FP'.format('r' if hp['residual_prediction'] else '') if hp['frame_prediction'] else '') + \
        (' OT' if hp['optimality_tightening'] else '') + \
        (' relu' if 'activation' not in hp else hp['activation']) + \
        ' ({})'.format(hp['t_max'] if 't_max' in hp else '')
    '''


def export_plots():

    layout = go.Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgb(229,229,229)',
        xaxis=go.XAxis(
            gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False,
            title=args.xlabel
        ),
        yaxis=go.YAxis(
            gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False,
            title=args.ylabel
        ),
    )
    if args.xrange:
        layout.xaxis.range = args.xrange
    if args.yrange:
        layout.xaxis.range = args.yrange

    event_files_by_hp_by_group = {}
    file_count = 0
    for root, dir, files in os.walk(args.input_dir):

        if any([IsTensorFlowEventsFile(f) for f in files]) and root.split('/')[-1] == args.mode and \
                all('cross{}'.format(i) in os.listdir(root + '/../..') for i in range(5)):
            hyper_parameters = {
                'data': re.search('.*data=(.*?)\/.*', root).group(1),
                'optimizer': re.search('.*optimizer=(.*?)\/.*', root).group(1)
            }
            if args.data and hyper_parameters['data'] != args.data:
                continue

            print(root, dir, files)

            for param in args.ignore_params:
                if param in hyper_parameters:
                    del hyper_parameters[param]
            file_count += 1
            event_files = [os.path.join(root, f) for f in files if IsTensorFlowEventsFile(f)]

            hyper_parameters_str = json.dumps(hyper_parameters, sort_keys=True)

            if hyper_parameters[args.group_by] not in event_files_by_hp_by_group:
                event_files_by_hp_by_group[hyper_parameters[args.group_by]] = {hyper_parameters_str: event_files}
            elif hyper_parameters_str not in event_files_by_hp_by_group[hyper_parameters[args.group_by]]:
                event_files_by_hp_by_group[hyper_parameters[args.group_by]][hyper_parameters_str] = event_files
            else:
                event_files_by_hp_by_group[hyper_parameters[args.group_by]][hyper_parameters_str] += event_files

    for group, event_files_by_hp in event_files_by_hp_by_group.items():
        hp_idx = 0

        handles = []

        data_objs = []

        columns = []
        names = []
        max_x = np.inf
        max_x_idx = 0

        data_groups = []
        for hyper_parameters_str, event_files in sorted(event_files_by_hp.items()):
            hyper_parameters = json.loads(hyper_parameters_str)
            events_by_scalar = {}
            print("Currently looking at {} event files".format(len(event_files)))
            pprint.pprint(hyper_parameters)
            for event_file in event_files:
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                for scalar in ea.Tags()['scalars']:
                    if scalar != args.scalar:
                        continue
                    #if scalar != 'Evaluation/Score':
                    #    continue
                    items = ea.Scalars(scalar)
                    if scalar not in events_by_scalar:
                        events_by_scalar[scalar] = [items]
                    else:
                        events_by_scalar[scalar].append(items)

            for scalar, event_arrays in events_by_scalar.items():
                steps, values, errors, np_arrays_x, np_arrays_y = event_arrays_to_np_arrays(event_arrays)

                steps, values, errors = zip(*sorted(zip(steps, values, errors)))
                steps = np.asarray(steps)
                values = np.asarray(values)
                errors = np.asarray(errors)

                if steps[-1] < max_x:
                    max_x = steps[-1]
                    max_x_idx = len(steps)

                plt.fill_between(steps, values - errors, values + errors, facecolor=colorscalen[hp_idx], alpha=0.2)

                handles.append(plt.plot(steps, values, linewidth=3.0, color=colorscalen[hp_idx],
                                            label=obtain_name(hyper_parameters))[0])
                trace = go.Scatter(
                    x=np.concatenate([steps, steps[::-1]]),
                    y=np.concatenate([values + errors, (values - errors)[::-1]]),
                    fill='tozerox',
                    fillcolor=colorscale[hp_idx].replace('rgb', 'rgba').replace(')', ',0.2)'),  # 'rgba(0,100,80,0.2)',
                    line=go.Line(color='transparent'),
                    showlegend=False
                )
                line = go.Scatter(
                    x=steps,
                    y=values,
                    line=go.Line(color=colorscale[hp_idx], width=3),
                    mode='lines',
                    name=obtain_name(hyper_parameters)
                )
                names.append(obtain_name(hyper_parameters))

                if args.animation:
                    for i in range(0, len(steps), args.animation_step):
                        columns.append(Column(steps[:i:args.animation_step], 'x_{}_{}'.format(obtain_name(hyper_parameters), i)))
                        columns.append(Column(values[:i:args.animation_step], 'y_{}_{}'.format(obtain_name(hyper_parameters), i)))

                data_objs += [trace, line]

            hp_idx += 1

        position_by_group = {
            'oxflower': 'upper left',
            'cifar10': 'lower right',
            'cifar100': 'upper left',
            'mnist': 'lower right'
        }

        title = group.title() if not args.title else args.title.title()
        ylabel = args.scalar.title() if not args.ylabel else args.ylabel.title()
        fnm = '_'.join([group, args.scalar, args.image_suffix])

        if args.animation:
            import time
            grid = Grid(columns)
            py.grid_ops.upload(grid, fnm + str(time.time()), auto_open=False)
            data = go.Data([
                    go.Scatter(
                        xsrc=grid.get_column_reference('x_{}_0'.format(name)),
                        ysrc=grid.get_column_reference('y_{}_0'.format(name)),
                        line=go.Line(color=colorscale[hp_idx], width=3),
                        mode='lines',
                        name=name
                    ) for hp_idx, name in enumerate(names)
                ]
            )

            animation_layout = go.Layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgb(229,229,229)',
                title=title,
                showlegend=True,
                updatemenus= [{
                   'buttons': [
                       {'args': [None, dict(frame=dict(duration=50, redraw=True),
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
                xaxis=go.XAxis(
                    gridcolor='rgb(255,255,255)',
                    showgrid=True,
                    showline=False,
                    showticklabels=True,
                    tickcolor='rgb(127,127,127)',
                    ticks='outside',
                    zeroline=False,
                    title=args.xlabel,
                    #autorange=False,
                    range=[0, max_x]
                ),
                yaxis=go.YAxis(
                    gridcolor='rgb(255,255,255)',
                    showgrid=True,
                    showline=False,
                    showticklabels=True,
                    tickcolor='rgb(127,127,127)',
                    ticks='outside',
                    zeroline=False,
                    title=ylabel,
                    #autorange=False,
                    range=[0, 1]
                ),
            )
            #)
            frame_data = []
            for i in range(0, max_x_idx, args.animation_step):
                data_group = []
                for hp_idx, name in enumerate(names):
                    data_group.append(go.Scatter(
                        xsrc=grid.get_column_reference('x_{}_{}'.format(name, i)),
                        ysrc=grid.get_column_reference('y_{}_{}'.format(name, i)),
                        line=go.Line(color=colorscale[hp_idx], width=3),
                        mode='lines'
                    ))
                frame_data.append({'data': data_group})

            #frames = go.Frames(dict=frame_data)
            figure = go.Figure(data=data, layout=animation_layout, frames=frame_data)
            py.create_animations(figure, filename=fnm + 'animation' + str(time.time()))
        else:
            data = go.Data(data_objs)
            layout.title = title
            layout.yaxis.title = ylabel
            fig = go.Figure(data=data, layout=layout)
            py.plot(fig, filename=fnm + '.html')

            plt.xlabel(args.xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            if args.xrange:
                plt.xlim(args.xrange)
            if args.yrange:
                plt.ylim(args.yrange)
            plt.legend(handles=handles, loc=position_by_group[group], framealpha=0.)

            plt.savefig(os.path.join(args.output_dir, fnm + '.pdf'))
            plt.clf()

            write_tex(ylabel, args.xlabel, os.path.join(args.output_dir, fnm + '.pdf'), group)


def write_tex(ylabel, xlabel, path, group):
    str = "\\begin{{figure}}\n" \
          "\t \\centering\n"\
          "\t \\includegraphics[width=\\linewidth]{{{path}}}\n"\
          "\t \\caption{{Mean {ylabel} vs. {xlabel} on {group} in which the shaded areas indicate the standard deviations.}}\n"\
          "\t \\label{{fig:{group}:{ylabel}}}\n"\
          "\\end{{figure}}\n".format(path=path, ylabel=ylabel.lower(), xlabel=xlabel.lower(), group=group)

    with open(path.replace('.pdf', '.tex'), 'w') as f:
        f.write(str)


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=os.path.join(project_root, 'logs'))
    parser.add_argument("--output_dir", default=os.path.join(project_root, 'doc', 'im'))
    parser.add_argument("--scalar", default='Accuracy')
    parser.add_argument("--ignore_params", nargs='+', default=[])
    parser.add_argument("--data", default=None)
    parser.add_argument("--interpolate", dest='interpolate', action='store_true')
    parser.add_argument("--animation", dest='animation', action='store_true')
    parser.add_argument("--image_suffix", default="")
    parser.add_argument("--xlabel", default="Train step")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--xrange", nargs='+', default=[], type=int)
    parser.add_argument("--yrange", nargs='+', default=[], type=int)
    parser.add_argument("--trace_by", nargs='+', default=['optimizer'])
    parser.add_argument("--group_by", default='data')
    parser.add_argument("--mode", default='test')
    parser.add_argument("--animation_step", default=100, type=int)
    args = parser.parse_args()

    export_plots()
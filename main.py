import yaml
import os
import numpy as np
from scipy import signal
from sklearn.neighbors import NearestNeighbors
from matplotlib import animation
import matplotlib.pyplot as plt
import aux_fn
from frontend import myFig
import time

with open("param.yaml") as fp:
    params = yaml.load(fp, Loader=yaml.FullLoader)
if params['chebwin_attenuation'] > 0:
    window = signal.chebwin(params['window_size'], at=params['chebwin_attenuation'])
else:
    window = np.ones(params['window_size'])
#load source file
filename = "data/26.05.20/2020-02-14 172715.674766.wav"
feeder = aux_fn.FileFeeder(filename, params['window_size'], params['step'])
gen = iter(feeder)
#process geometry between mics
geom_filepath = os.path.join(os.path.dirname(filename),'arch.csv')
geom = np.genfromtxt(geom_filepath, delimiter=',', skip_header=True)
spherePts = aux_fn.gen_sphere(params['n_points'], params['radius'], True)

shiftsInTicsByDir = aux_fn.get_sphere_shifts(spherePts, geom, feeder.fs, params['sound_speed'])
shiftsInTicsByDir,idx = np.unique(shiftsInTicsByDir, return_index=True, axis=0)
spherePts = spherePts[idx, :]

polar_pts = aux_fn.cartesian_to_spherical(spherePts)

params['max_shift_array'] = np.zeros(len(params['xcorr_pairs']),dtype=np.int)
pairShiftsInTicsByDir = np.zeros((len(spherePts), len(params['xcorr_pairs'])))
for i, pair in enumerate(params['xcorr_pairs']):
    params['max_shift_array'][i] = np.ceil(np.linalg.norm(geom[pair[0], :]-geom[pair[1],:])*feeder.fs/params['sound_speed'])
    pairShiftsInTicsByDir[:, i] = shiftsInTicsByDir[:, pair[0]]-shiftsInTicsByDir[:, pair[1]]

knn = NearestNeighbors(n_neighbors=1)
knn.fit(pairShiftsInTicsByDir)

fig1 = plt.figure(FigureClass=myFig, figsize=(6,3))


def anim_func(frame=0):
    try:
        start = time.perf_counter()
        val = next(gen)
        #filter = aux_fn.gen_point_filter(val[:, 0], params['window_size'], window, params)
        #res = aux_fn.freq_filter(val, params['window_size'], feeder.fs, filter, window)
        res = aux_fn.freq_filter(val, params['window_size'], feeder.fs, [600, 1200], window)
        possible_shifts = aux_fn.get_xcorr_shifts(res, params)
        maxval, maxind = knn.kneighbors(possible_shifts)
        maxind = maxind[maxval < 10]
        res_ind = aux_fn.threaded_get_source(res, shiftsInTicsByDir, maxind)
        direction = polar_pts[res_ind,:]
        time_chunk = params['step']/feeder.fs
        time_process = (time.perf_counter() - start)
        time_boost = time_chunk/time_process
        if frame%10 == 0:
            print("Seconds passed: {:.1f}, speed {:.1f}x".format(feeder.pos/feeder.fs, time_boost))
        fig1.update(direction)
    except StopIteration:
        pass
    return fig1.phi_point, fig1.theta_point,


anim = animation.FuncAnimation(fig1,anim_func,interval=0, blit=True)
plt.show()

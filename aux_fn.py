import math
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.signal import argrelextrema
from threading import Thread
import queue

class FileFeeder:
    def __init__(self, filename, window_size, step):
        self.pos = 0
        self.window_size = window_size
        self.step = step
        self.fs, self.data = wavfile.read(filename)

    def __iter__(self):
        while self.pos+self.window_size < self.data.shape[0]:
            chunk = self.data[self.pos:self.pos+self.window_size, :]
            self.pos += self.step
            yield chunk


def gen_sphere(n_points, radius, flag_is_half=True):
    if flag_is_half:
        n_points = n_points * 2
    Area = 4 * math.pi / n_points
    Distance = math.sqrt(Area)
    M_theta = round(math.pi / Distance)
    d_theta = math.pi / M_theta
    d_phi = Area / d_theta
    pts = [[], [], []]
    for m in range(M_theta):
        Theta = math.pi * (m + 0.5) / M_theta
        if not (Theta > math.pi/2 and flag_is_half):
            M_phi = round(2 * math.pi * math.sin(Theta) / d_phi)
            for n in range(M_phi):
                Phi = 2 * math.pi * n / M_phi
                pts[0] += [radius * math.sin(Theta) * math.cos(Phi)]
                pts[1] += [radius * math.sin(Theta) * math.sin(Phi)]
                pts[2] += [radius * math.cos(Theta)]
    return np.array(pts).transpose()


def get_sphere_shifts(pts, geom, frame_rate, sound_speed):
    """
    :param pts: numpy array with 3 columns that represents points in cartesian coordinates
    :param geom: numpy array with 3 columns that represents mic coordinates
    :param frame_rate: frame rate
    :param sound_speed: speed of sound
    :return: shift array
    """
    dist2mic = np.zeros((pts.shape[0],geom.shape[0]))
    for chan in range(geom.shape[0]):
        dist2mic[:, chan] = np.sum(np.abs(pts-geom[chan, :])**2,axis=-1)**(1./2)
    minval = dist2mic.min(axis=1)
    dist2mic = dist2mic-np.tile(minval, (geom.shape[0], 1)).transpose()
    dist2mic = np.round(dist2mic*frame_rate/sound_speed).astype(np.int)
    return dist2mic


def cartesian_to_spherical(pts):
    """
    :param pts: numpy array with 3 columns that represents points in cartesian coordinates
    :return: points in spherical coordinates (r, phi, theta)
    """
    polar_pts = np.zeros((pts.shape[0], 3))
    polar_pts[:, 0] = np.sum(np.abs(pts)**2,axis=-1)**(1./2)
    polar_pts[:, 1] = -np.arctan2(pts[:, 1], pts[:, 0])+math.pi/2
    polar_pts[polar_pts[:, 1] < -math.pi, 1] = polar_pts[polar_pts[:, 1] < -math.pi, 1] + 2 * math.pi
    polar_pts[polar_pts[:, 1] > math.pi, 1] = polar_pts[polar_pts[:, 1] > math.pi, 1] - 2 * math.pi
    polar_pts[:, 2] = np.arcsin(pts[:, 2] / polar_pts[:,0])
    return polar_pts


def normalize(sig):
    t_mean = np.mean(sig, axis=0)
    t_std = np.std(sig, axis=0)
    if len(sig.shape) == 2:
        sig=(sig-t_mean[None,:])/t_std[None,:]
    else:
        sig = (sig - t_mean) / t_std
    return sig


def freq_filter(sig, L, fs, gate, window):
    sig = sig* window[:, None]
    Y = np.fft.fft(sig, L, axis=0)
    freqs = np.abs(np.fft.fftfreq(L, 1/fs))
    if len(gate) == 2:
        Y[freqs < gate[0], :] = 0
        Y[freqs > gate[1], :] = 0
    if isinstance(gate, bool) and len(gate) == L:
        Y[~gate, :] = 0
    result = np.fft.ifft(Y, L, axis=0)
    result = np.real(result)
    result = normalize(result)
    return result


def gen_point_filter(sig, L, window, params):
    sig=normalize(sig)
    sig = np.multiply(sig, window)
    Y = np.fft.fft(sig, L)
    Y = np.abs(Y)
    Y = Y[1:int(L/2)]
    qval = np.quantile(Y, params['percentile'])
    filter_initial = (Y >= qval)
    res = np.zeros(len(filter_initial))
    for i in range(int(len(filter_initial)/params['filter_harmonic_num'])):
        res[i] = np.sum(filter_initial[i:params['filter_harmonic_num']*(i+1):(i+1)])
    res = (res >= params['harmonic_to_pass'])
    res = np.concatenate(([0], res, [0], np.flip(res)))
    return res


def get_xcorr_shifts(sig, params):
    max_container=[]
    npairs=len(params['xcorr_pairs'])
    for i, pair in enumerate(params['xcorr_pairs']):
        max_shift = params['max_shift_array'][i]
        corr = signal.correlate(sig[:,pair[0]], sig[:, pair[1]])
        middle = int((len(corr) - 1) / 2)
        corr_proc = corr[middle - max_shift:middle + max_shift + 1]
        maxima, = argrelextrema(corr_proc, np.greater)
        SV = 0
        while len(maxima) > params['max_num_xcorr_maximums']:
            SV += 20
            corr_proc = np.convolve(corr, np.ones((SV,))/SV, mode='valid')
            corr_proc=corr_proc[middle-max_shift:middle+max_shift+1]
            maxima, = argrelextrema(corr_proc, np.greater)

        max_container.append(maxima-max_shift)

    combs = np.meshgrid(*tuple(max_container))
    combs = np.stack(combs, axis=npairs)
    combs = np.reshape(combs, (-1, npairs))
    return combs


def get_point_energy(sig, shift, idx, queue):
    dur = len(sig)-max(shift)
    dir_sig = np.zeros(dur)
    for i in range(sig.shape[1]):
        dir_sig += sig[shift[i]:shift[i]+dur, i]
    res = np.std(dir_sig)
    queue.put((res, idx))


def threaded_get_source(sig,shiftsInTicsByDir,maxind):
    que = queue.Queue()
    for idx in maxind:
        t = Thread(target=get_point_energy, args=(sig, shiftsInTicsByDir[idx, :], idx, que))
        t.start()
    maxval = 0
    maxind = -1
    while not que.empty():
        item = que.get()
        if maxval < item[0]:
            maxval = item[0]
            maxind = item[1]
    return maxind
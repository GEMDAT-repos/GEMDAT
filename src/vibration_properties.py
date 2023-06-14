import matplotlib.pyplot as plt
import numpy as np


def vibration_properties(sim_data, show_pics):
    ts = sim_data['time_step']
    fs = 1 / ts
    length = sim_data['nr_steps']
    if length % 2 != 0:
        length -= 1
    one_sided = np.zeros((sim_data['nr_diffusing'], length // 2 + 1))

    speed = np.zeros((sim_data['nr_diffusing'], sim_data['nr_steps']))
    freq_mean = np.zeros(sim_data['nr_diffusing'])
    atom = 0
    vib_count = 0
    amplitude = []
    for i in range(sim_data['nr_atoms']):
        if sim_data['diffusing_atom'][i]:
            atom += 1
            for time in range(1, sim_data['nr_steps']):
                speed[atom - 1, time] = sim_data['displacement'][i, time] \
                        - sim_data['displacement'][i, time - 1]
                if np.sign(speed[atom - 1, time]) == np.sign(speed[atom - 1,
                                                                   time - 1]):
                    amplitude[vib_count] += speed[atom - 1, time]
                else:
                    vib_count += 1
                    amplitude.append(speed[atom - 1, time])

            freq_mean[atom - 1] = np.mean(speed[atom - 1, :], axis=0)

            trans = np.fft.fft(speed[atom - 1, :])
            two_sided = np.abs(trans / length)
            one_sided[atom - 1, :] = two_sided[:length // 2 + 1]
            one_sided[atom - 1, 1:-1] = 2 * one_sided[atom - 1, 1:-1]

    _mean_vib, vibration_amp = np.mean(amplitude), np.std(amplitude)
    attempt_freq = np.mean(freq_mean)
    std_attempt_freq = np.std(freq_mean)

    if show_pics:
        plt.figure()
        h = plt.hist(amplitude, bins=100, density=True)
        plt.plot([-vibration_amp, vibration_amp], [8000, 8000],
                 'g',
                 linestyle=':',
                 linewidth=3)
        plt.xlim([-2, 2])
        h[2].set_linewidth(3)
        plt.title('Histogram of vibrational amplitudes with fitted Gaussian')
        plt.xlabel('Amplitude (Angstrom)')
        plt.ylabel('Occurrence (a.u.)')
        plt.gca().set_yticklabels([])
        plt.show()

        f = fs * np.arange(length // 2 + 1) / length
        plt.figure()
        sum_freqs = np.sum(one_sided, axis=0)
        smoothed = np.convolve(sum_freqs, np.ones(51), 'valid') / 51
        plt.plot(f, smoothed, linewidth=3)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Occurrence (a.u.)')
        plt.gca().set_yticklabels([])
        plt.plot([attempt_freq, attempt_freq], [0, 1], '-r', linewidth=3)
        plt.plot(
            [attempt_freq - std_attempt_freq, attempt_freq - std_attempt_freq],
            [0, 1],
            ':r',
            linewidth=3)
        plt.plot(
            [attempt_freq + std_attempt_freq, attempt_freq + std_attempt_freq],
            [0, 1],
            ':r',
            linewidth=3)
        plt.ylim([0, np.max(sum_freqs)])
        plt.xlim([0, 2.5E13])
        plt.show()

    return attempt_freq

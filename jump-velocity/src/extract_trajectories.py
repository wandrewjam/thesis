import numpy as np
import pandas as pd


def main():
    filenames = ['CCC-Whole-Blood-Trajectories.xlsx',
                 'HCC-Whole-Blood-Trajectories.xlsx',
                 'PRP HH HC CC.xlsx']

    data_frames = list()
    data_frames.append(pd.read_excel(
        filenames[0], sheet_name='Normalized Trajectories'
    ))

    data_frames.append(pd.read_excel(
        filenames[1], sheet_name='Normalized Trajectories'
    ))

    data_frames.append(pd.read_excel(filenames[2], sheet_name='CC'))
    data_frames.append(pd.read_excel(filenames[2], sheet_name='HC'))

    data_frames = data_frames[::-1]

    for frame in data_frames:
        cols = [c for c in frame.columns if c[:13] != 'Distance (um)'
                and c[:17] != 'Elapsed Time (ms)']
        frame.drop(labels=cols, axis='columns', inplace=True)
        frame.dropna(axis=0, how='all', inplace=True)

    processed_frames = list()
    for frame in data_frames:
        frame_list = [frame.iloc[:, 2*i:2*i+2].dropna()
                      for i in range(frame.shape[1] // 2)]
        frame_list = [trajectory.values[:, ::-1] * np.array([1. / 1000, 1])
                      for trajectory in frame_list]
        # frame_list = [trajectory[:-1] for trajectory in frame_list
        #               if (trajectory[-1, 0] < trajectory[-2, 0]
        #               or trajectory[-1, 1] < trajectory[-2, 1])]
        processed_frames.append(frame_list)
        # for frame in frame_list:
        #     assert np.all(frame[1:] >= frame[:-1])
    for frame_list in processed_frames:
        for j, frame in enumerate(frame_list):
            if frame[-1, 0] < frame[-2, 0] or frame[-1, 1] < frame[-2, 1]:
                frame_list[j] = frame[:-1]

    for frame_list in processed_frames:
        for frame in frame_list:
            assert np.all(frame[1:] >= frame[:-1])

    file_headers = ['hcp', 'ccp', 'hcw', 'ccw']
    for head, experiment in zip(file_headers, processed_frames):
        for j, trajectory in enumerate(experiment):
            np.savetxt(head + '-t{}.dat'.format(j), trajectory)


if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/vlado-data/'))

    main()

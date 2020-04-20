import numpy as np
import pandas as pd


def main():
    # filenames = ['CCC-Whole-Blood-Trajectories.xlsx',
    #              'HCC-Whole-Blood-Trajectories.xlsx',
    #              'PRP HH HC CC.xlsx']

    # filenames = ['Manuscript 1 FFF PRP Trajectories.xlsx', 'PRP HF.xlsx']

    # filenames = ['Sasha FFF Data.xlsx', 'Sasha HFF Data.xlsx']

    # filenames = ['PRP HV.xlsx']

    # filenames = ['Manuscript 1 VVV PRP Trajectories.xlsx']

    filenames = ['Emma_Results/HFF_Video_1_Analysis.xlsx',
                 'Emma_Results/FFF_Video_1_Analysis.xlsx']
    data_frames = list()
    # data_frames.append(pd.read_excel(
    #     filenames[0], sheet_name='Normalized Trajectories'
    # ))
    # data_frames.append(pd.read_excel(
    #     filenames[0], header=None
    # ))

    data_frames.append(pd.read_excel(
        filenames[0], sheet_name=None
    ))

    data_frames.append(pd.read_excel(
        filenames[1], sheet_name=None
    ))

    # data_frames.append(pd.read_excel(
    #     filenames[1], sheet_name='Normalized Trajectories'
    # ))
    # data_frames.append(pd.read_excel(
    #     filenames[1], sheet_name='Trajectories'
    # ))

    # data_frames.append(pd.read_excel(filenames[2], sheet_name='CC'))
    # data_frames.append(pd.read_excel(filenames[2], sheet_name='HC'))

    # data_frames = data_frames[::-1]
    #
    # for frame in data_frames:
    #     # cols = [c for c in frame.columns if c[:13] != 'Distance (um)'
    #     #         and c[:17] != 'Elapsed Time (ms)']
    #     cols = [c for c in frame.columns if frame[c][0] != 0]
    #     frame.drop(labels=cols, axis='columns', inplace=True)
    #     frame.dropna(axis=0, how='all', inplace=True)
    #
    # processed_frames = list()
    # for frame in data_frames:
    #     frame_list = [frame.iloc[:, 2*i:2*i+2].dropna()
    #                   for i in range(frame.shape[1] // 2)]
    #     frame_list = [trajectory.values[:, ::-1] * np.array([1. / 1000, 1])
    #                   for trajectory in frame_list]
    #     # frame_list = [trajectory[:-1] for trajectory in frame_list
    #     #               if (trajectory[-1, 0] < trajectory[-2, 0]
    #     #               or trajectory[-1, 1] < trajectory[-2, 1])]
    #     processed_frames.append(frame_list)
    #     # for frame in frame_list:
    #     #     assert np.all(frame[1:] >= frame[:-1])
    # for frame_list in processed_frames:
    #     for j, frame in enumerate(frame_list):
    #         if frame[-1, 0] < frame[-2, 0] or frame[-1, 1] < frame[-2, 1]:
    #             frame_list[j] = frame[:-1]
    #
    # for frame_list in processed_frames:
    #     for frame in frame_list:
    #         assert np.all(frame[1:] >= frame[:-1])

    processed_frames = list()
    for data_dict in data_frames:
        trajectory_list = list()
        for key, item in data_dict.items():
            if key[:8] == 'Platelet':
                trajectory = item.iloc[:, 20:22].dropna(how='all').values
                trajectory *= np.array([1./1000, 1])
                trajectory_list.append(trajectory)
        processed_frames.append(trajectory_list)

    # file_headers = ['hcp', 'ccp', 'hcw', 'ccw']
    # file_headers = ['hfp', 'ffp']
    file_headers = ['hfe', 'ffe']
    # file_headers = ['hvp']
    # file_headers = ['vvp']

    # Make sure not to overwrite existing files
    os.chdir(os.path.expanduser('~/thesis/vlado-data/'))
    file_list = os.listdir(os.getcwd())
    offset = [0, 0]
    for file in file_list:
        if file[:3] == 'hfe':
            if len(file) == 10:
                j = int(file[5])
            elif len(file) == 11:
                j = int(file[5:7])
            else:
                continue

            if j > offset[0]:
                offset[0] = j
        elif file[:3] == 'ffe':
            if len(file) == 10:
                j = int(file[5])
            elif len(file) == 11:
                j = int(file[5:7])
            else:
                continue

            if j > offset[1]:
                offset[1] = j

    for head, experiment in zip(file_headers, processed_frames):
        if head == 'hfe':
            i = 0
        elif head == 'ffe':
            i = 1

        for j, trajectory in enumerate(experiment):
            fname = head + '-t{}.dat'.format(j + offset[i] + 1)
            assert fname not in file_list
            np.savetxt(fname, trajectory)


if __name__ == '__main__':
    import os
    os.chdir(os.path.expanduser('~/thesis/vlado-data/Excel/'))

    main()

from binding_expt import save_info, parse_file
import os


def test_save_info():
    test_file = 'test'
    save_info(test_file, seed=1234, t_end=0.31415)
    pars = parse_file(test_file)

    txt_dir = os.path.expanduser('~/thesis/reg_stokeslets/par-files/')
    test_filename = txt_dir + test_file + '.txt'
    os.remove(test_filename)

    assert pars['seed'] == 1234 and pars['t_end'] == 0.31415

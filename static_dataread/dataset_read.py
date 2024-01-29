from static_dataread.svhn import load_svhn
from static_dataread.mnist import load_mnist
from static_dataread.usps import load_usps
from static_dataread.unaligned_data_loader import UnalignedDataLoader


def return_dataset(domain_name, usps, scale, all_use):
    if domain_name == 'svhn':
        train_image, train_label, test_image, test_label = load_svhn()
    if domain_name == 'mnist':
        train_image, train_label, test_image, test_label = load_mnist(scale=scale,
                                                                      usps=usps,
                                                                      all_use=all_use)
    if domain_name == 'usps':
        train_image, train_label, test_image, test_label = load_usps(all_use=all_use)

    return train_image, train_label, test_image, test_label, domain_name


def read_dataset(source_name, target_name, scale=False, all_use='no'):
    usps = False
    if source_name == "usps" or target_name == "usps":
        usps = True

    xs_train, ys_train, xs_test, ys_test, s_domain_name = return_dataset(source_name,
                                                                         usps=usps,
                                                                         scale=scale,
                                                                         all_use=all_use)
    xt_train, yt_train, xt_test, yt_test, t_domain_name = return_dataset(target_name,
                                                                         usps=usps,
                                                                         scale=scale,
                                                                         all_use=all_use)

    return xs_train, ys_train, xt_train, yt_train, \
           xs_test, ys_test, xt_test, yt_test,\
           s_domain_name, t_domain_name


def generate_dataset(xs, ys, xt, yt, s_domain_name, t_domain_name, batch_size, gpu):
    S = {}
    T = {}

    S['imags'] = xs
    S['label'] = ys
    T['imags'] = xt
    T['label'] = yt

    # synth时，scale为40
    # usps时，scale为28
    # 其他时候，scale为32
    scale = 40 if s_domain_name == 'synth' else 28 if t_domain_name == 'usps' or t_domain_name == 'usps' else 32

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size, gpu, scale=scale)
    dataset = train_loader.load_data()

    return dataset




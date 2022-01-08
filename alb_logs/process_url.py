from urllib.parse import urlparse
import pathlib
from collections import defaultdict, Counter
#import matplotlib.pyplot as plt
import pickle
import glob
import csv
from pathlib import Path, PurePosixPath
import re

cdir = '/home/sandeep/aws_athena/sessions/'
out_dir = '/home/sandeep/aws_athena/with_status/output/'

fieldnames=[
    'type',
    'client_ip',
    'domain_name',
    'user_agent',
    'time',
    'request_url',
    'request_verb',
    'elb_status_code',
    'target_status_code',
    'actions_executed',
    'target_processing_time']


def process_file(filename, status_dict):

    uuid_pat = re.compile('[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{10}')

    anomaly_dict = defaultdict(int)
    con_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    con_dict_file = Path(filename).with_stem('con_dict').with_suffix('.pkl')
    anomaly_dict_file = Path(filename).with_stem('anomaly_dict').with_suffix('.pkl')

    with open(filename, 'r') as f:
        # lines = f.readlines(
        csv_file = csv.DictReader(f, fieldnames=fieldnames)
        count = 0
        for row in csv_file:
            #print(row['request_url'], row['elb_status_code'])
            u = urlparse(row['request_url'])
            s = int(row['elb_status_code'])
            t = row['time']
            posixpath = PurePosixPath(u.path)
            assert posixpath.parts[0] == '/'
            if posixpath.parts[1] != 'sessions':
                anomaly_dict[u.path] += 1
                #print('anomaly ', posixpath.parts)
            else:
                try:
                    v = posixpath.parts[2]
                    controller = posixpath.parts[3]
                    rest = []
                    for p in posixpath.parts[4:]:
                        if uuid_pat.match(p):
                            #print('skipping ', p)
                            continue
                        rest.append(p)
                    #print('v=', posixpath.parts[2], ':controller=',posixpath.parts[3:],':rest=', u.params, u.query)
                    con_dict[controller][v]['_'.join(rest)] += 1
                except:
                    anomaly_dict[u.path] += 1
    # print('================\n')
    # for k, v in anomaly_dict.items():
    #     print(k, v)
    # print('================\n')
    pickle.dump(anomaly_dict, open(str(anomaly_dict_file), 'wb'))
    #pickle.dump(con_dict, open(str(con_dict_file), 'wb'))
    for k, v in con_dict.items():
        for k2, v2 in v.items():
            for k3, v3 in v2.items():
                con_dict_file = Path(filename).with_stem('con_dict' + '_' + k + '_' + k2 + '_' + k3).with_suffix('.pkl')
                pickle.dump(con_dict[k][k2][k3], open(str(con_dict_file), 'wb'))
                print(k, k2, k3, v3)

def process_file_old(filename, status_dict):

    base_name = pathlib.Path(filename).name
    print('start ', filename, base_name)
    with open(filename, 'r') as f:
        # lines = f.readlines(
        csv_file = csv.DictReader(f, fieldnames=fieldnames)
        query_count = 0
        params_count = 0
        call_dict = Counter()
        skip_dict = defaultdict(set)
        for row in csv_file:
            line = row[3]
            status = int(row[4])
            #line = line[1:-2]  # remove first and last dobule quote
            u = urlparse(line)
            upath = pathlib.Path(u.path)
            new_parts = []
            for idx, part in enumerate(upath.parts):
                if len(part) == 36 or len(part) == 192:  ## uuid
                    # print(part, ' is uuid')
                    # upath.parts[idx] = 'quiz'
                    new_parts.append('dummy')
                else:
                    new_parts.append(upath.parts[idx])
            new_tuple = tuple(new_parts)
            new_path = pathlib.Path(*new_tuple[2:])  # remove user-management
            # if status < 200 or status >= 300:
            #     skip_dict[str(new_path)].add(status)
            #     continue
            status_dict[str(new_path)][status] += 1
            call_dict[str(new_path)] += 1
            if u.query:
                query_count = query_count + 1
            if u.params:
                params_count = params_count + 1
        #print('status = ', status_dict)
        #skip_file = out_dir + 'skip' + base_name + '.pkl'
        # pickle_file = out_dir + base_name + '.pkl'
        # status_file = out_dir + 'status' + base_name + '.pkl'
        # with open(pickle_file, 'wb') as file_handle:
        #     pickle.dump(call_dict, file_handle, pickle.HIGHEST_PROTOCOL)
        # # with open(skip_file, 'wb') as file_handle:
        # #     pickle.dump(skip_dict, file_handle, pickle.HIGHEST_PROTOCOL)
        # with open(status_file, 'wb') as file_handle:
        #     pickle.dump(status_dict, file_handle, pickle.HIGHEST_PROTOCOL)

def display_skipdict():
    filelist = glob.glob(out_dir + 'skip*.csv.pkl')
    new_dict =  defaultdict(set)
    for dfile in filelist:
        with open(dfile, 'rb') as file_handle:
            d = pickle.load(file_handle)
            new_dict.update(d)
    print('dict has ', len(new_dict))
    for k, v in new_dict.items():
        print(k, v)


def display_dict():

    filelist = glob.glob(out_dir + '*.csv.pkl')
    new_dict = Counter()
    for dfile in filelist:
        with open(dfile, 'rb') as file_handle:
            d = pickle.load(file_handle)
            new_dict.update(d)

    has_zoom = 0
    for k in new_dict.keys():
        if 'zoom' in k:
            has_zoom = has_zoom + 1
            print(k, new_dict[k], ' zoom')
    print(has_zoom, ' has zoom')

    print('dict has ', len(new_dict))
    pic_dict = Counter()
    max_val = 0
    select_above = 1000
    for k,v in new_dict.items():
        if v > select_above:
            pic_dict[k[:40]] = v//select_above
            if (v//select_above == 1):
                print(k, '=', v)
            max_val = max(v, max_val)
    print('newdict has ', len(pic_dict))

    fig, ax = plt.subplots()
    plt.style.use('tableau-colorblind10')
    plt.rcParams.update({'figure.autolayout': True})

    ax.barh(list(pic_dict.keys()), list(pic_dict.values()), 3, align='center')

    title = 'number of hits in multiples of ' + str(select_above)
    ax.set_title(title)
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.tight_layout()
    plt.savefig('total.jpg')
    print(plt.style.available)

    plt.show()

def dict_to_csv(status_dict):
    print('dict has ', len(status_dict))
    csv_out_file = out_dir + '_status.csv'
    with open(csv_out_file, 'w') as csv_out_file_handle:
        csv_writer = csv.writer(csv_out_file_handle)
        for k, v in status_dict.items():
            for k2, v2 in v.items():
                csv_writer.writerow([k, k2, v2])

basedir='/home/sandeep/aws_athena/sessions/'

def pickle_load():
    filelist = glob.glob(basedir + 'con_dict*.pkl')
    for f in filelist:
        d = pickle.load(open(f, 'rb'))
        print(f, d)

def main():

    status_dict = defaultdict(lambda: defaultdict(int))
    filelist = ['/home/sandeep/aws_athena/sessions/sessions.csv']
    #assert(len(filelist) == 26)
    for filename in filelist:
        process_file(filename, status_dict)

    #pickle_load()
    # dict_to_csv(status_dict)
    # display_skipdict()
    # display_dict()


if __name__ == '__main__':
    main()


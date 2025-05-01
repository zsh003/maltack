import os
import sys
import lief
import time
import struct

import numpy as np
import pickle

import threading
import subprocess
from multiprocessing import Manager, Pool

from raw_features import ByteHistogram, ByteEntropyHistogram, PEFeatureExtractor

from feature_engineering import Feature_engineering


from tqdm import tqdm

if len(sys.argv) > 1:
    datapath = sys.argv[1]
else:
    datapath = "../dataset/test"

black_path = []
black_list = []
for parent, dirnames, filenames in os.walk(datapath):
    if "/miner" in parent:
        for filename in filenames:
            fp = os.path.join(parent, filename)
            black_path.append(fp)
            black_list.append(filename)
white_path = []
white_list = []
for parent, dirnames, filenames in os.walk(datapath):
    if "/not_miner" in parent:
        for filename in filenames:
            fp = os.path.join(parent, filename)
            white_path.append(fp)
            white_list.append(filename)
            
print("Found {0} miner samples.".format(len(black_path)))
print("Found {0} not miner samples.".format(len(white_path)))

with open("../models/black_path.pkl", "wb") as f:
    pickle.dump(black_path, f)
with open("../models/black_list.pkl", "wb") as f:
    pickle.dump(black_list, f)
with open("../models/white_path.pkl", "wb") as f:
    pickle.dump(white_path, f)
with open("../models/white_list.pkl", "wb") as f:
    pickle.dump(white_list, f)

test_path = []
for parent, dirnames, filenames in os.walk(datapath):
    for filename in filenames:
        fp = os.path.join(parent, filename)
        test_path.append(fp)

hash_list = [os.path.split(sp)[-1] for sp in test_path]
test_fixed_path = [os.path.join("../tmp", sp) for sp in hash_list]

test_num = len(hash_list)
print("Found {0} samples in total.".format(test_num))

with open("../models/hash_list.pkl", "wb") as f:
    pickle.dump(hash_list, f)
    
emp = threading.Semaphore(value=12)

pe = PEFeatureExtractor()
fn = Feature_engineering()

# ---------------------文件头修复------------------------

def fix_header(fp, ha):
    with open(fp, 'rb') as f:
        data = f.read()
    e_lfnew = data[0x3C: 0x40]
    offset = int.from_bytes(e_lfnew, byteorder='little', signed=True)
    new_data = b"MZ" + data[2:offset] + b"PE\0\0" + data[offset+4:]

    new_path = "../tmp/{0}".format(ha)
    with open(new_path, 'wb') as f:
        f.write(new_data)
    emp.release()

# ---------------------直方图特征------------------------

def histogram_feature(sample_path):
    with open(sample_path, "rb") as f:
        data = f.read()
    file_size = len(data)
    Histogram = ByteHistogram().raw_features(data, None)
    Byte_Entropy = ByteEntropyHistogram().raw_features(data, None)

    Sum = 0
    for i in range(len(Byte_Entropy)):
        Sum += Byte_Entropy[i]

    Histogram = np.array(Histogram) / file_size
    Byte_Entropy = np.array(Byte_Entropy) / Sum

    feature = np.concatenate((Histogram, Byte_Entropy), axis=-1)
    feature = list(feature)
    path = sample_path.replace("tmp", "histogram") + ".txt"
    with open(path, 'w') as f:
        for i in feature:
            f.write("{}\n".format(str(i)))


# ---------------------PE静态特征------------------------

pe_raw_vectors = Manager().list([0] * test_num)

def get_pe_raw_vector(idx, fp, res_default):
    res = res_default
    try:
        with open(fp, 'rb') as f:
            raw_data = f.read()
        res = pe.feature_vector(raw_data)
    except Exception:
        pass
    pe_raw_vectors[idx] = res


# ---------------------特征工程------------------------

feature_engineering_features = Manager().list([0] * test_num)

def get_fn(idx, fp):
    with open(fp, 'rb') as f:
        data = f.read()
    res = fn.get_feature_engineering(data)
    feature_engineering_features[idx] = res

    
if __name__ == '__main__':
    print("Preprecess started.")

    """

    # 修复MZ和PE头
    os.system("rm -rf ../tmp")
    os.makedirs("../tmp")
    table = []
    with tqdm(total=test_num, ncols=80, desc="fix") as pbar:
        for fp, ha in zip(test_path, hash_list):
            emp.acquire()
            t = threading.Thread(target=fix_header, args=(fp, ha), daemon=True)
            t.start()
            table.append(t)
            pbar.update(1)
    for t in table:
        t.join()

    """
    
    start_time = time.time()

    
    """
    # 直方图特征
    os.system("rm -rf ../histogram")
    os.makedirs("../histogram")
    with Pool(12) as pool:
        for fp in test_fixed_path:
            pool.apply_async(func=histogram_feature, args=(fp, ))
        pool.close()
        pool.join()
    end_time = time.time()
    print("hostogram: {0:.2f}s".format(end_time - start_time))
    start_time = end_time

    """

    # PE静态特征
    os.system("rm -rf ../pe_raw")
    os.makedirs("../pe_raw")
    res_default = np.zeros(shape=(967,), dtype=np.float32)
    with Pool(12) as pool:
        for i, fp in enumerate(test_fixed_path):
            pool.apply_async(func=get_pe_raw_vector, args=(i, fp, res_default))
        pool.close()
        pool.join()
    with open("../pe_raw/pe_raw_vectors.pkl", "wb") as f:
        pickle.dump(list(pe_raw_vectors), f)
    end_time = time.time()
    print("pe raw: {0:.2f}s".format(end_time - start_time))
    start_time = end_time

    """
    # 特征工程
    os.system("rm -rf ../feature_engineering")
    os.makedirs("../feature_engineering")
    with Pool(12) as pool:
        for i, fp in enumerate(test_fixed_path):
            pool.apply_async(func=get_fn, args=(i, fp))
        pool.close()
        pool.join()
    end_time = time.time()
    print("feature engineering: {0:.2f}s".format(end_time - start_time))

    with open("../feature_engineering/feature_engineering_features.pkl", 'wb') as f:
        pickle.dump(list(feature_engineering_features), f)
    
    """

    print("Preprecess done.")

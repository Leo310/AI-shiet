# SPDX-License-Identifier: MIT

import json
import logging
import time

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

DATA_PATH = "./data/HDFS.log"
LINE_COUNT = 100000
TRAIN_LINE_COUNT = 100000
OUTPUT_LOG = "./data/drainlog.log"
OUTPUT_PARSED = "./data/my_hdfs_train"
DRAINSETTINGS = "./drain3settings.ini"

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logger = logging.getLogger(__name__)
logging.basicConfig(filename=OUTPUT_LOG, level=logging.INFO, format='%(message)s')

config = TemplateMinerConfig()
config.load(DRAINSETTINGS)
config.profiling_enabled = True
template_miner = TemplateMiner(config=config)

line_count = 0

# dictionary: { BlockID: [ logKeys ] }
logSeqPerBlock = {}

start_time = time.time()

# "pretrain" drain and some logging
batch_start_time = start_time
batch_size = 1
with open(DATA_PATH) as f:
    for _ in range(TRAIN_LINE_COUNT):
        line = f.readline()
        line = line.rstrip()
        line = line.partition(": ")[2]
        result = template_miner.add_log_message(line)

        line_count += 1
        if line_count % batch_size == 0:
            time_took = time.time() - batch_start_time
            rate = batch_size / time_took
            # logger.info(f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
            # f"{len(template_miner.drain.clusters)} clusters so far.")
            batch_start_time = time.time()
        # if result["change_type"] != "none":
            result_json = json.dumps(result)
            logger.info(f"Input ({line_count}): " + line)
            logger.info("Result: " + result_json)

# fill in logSeqPerBlock dictionary
with open(DATA_PATH) as f:
    for _ in range(LINE_COUNT):
        line = f.readline()
        line = line.rstrip()
        line = line.partition(": ")[2]
        result = template_miner.add_log_message(line)
        # params of logkey to extract block_id
        params = template_miner.extract_parameters(
            result["template_mined"], line, exact_matching=False)
        if(params is not None):
            for i in range(len(params)):
                if(params[i].mask_name == "BLK"):
                    # params[i].value contains blockid
                    if params[i].value not in logSeqPerBlock:
                        logSeqPerBlock[params[i].value] = [result["cluster_id"]]
                    else:
                        logSeqPerBlock[params[i].value].append(result["cluster_id"])
                    break


# write logSeqPerBlock to file
with open(OUTPUT_PARSED, "w+") as f2:
    for key in logSeqPerBlock:
        f2.write(str(" ".join(str(x) for x in logSeqPerBlock[key]) + '\n'))


# logging
time_took = time.time() - start_time
rate = line_count / time_took
logger.info(f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
            f"{len(template_miner.drain.clusters)} clusters")

sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
for cluster in sorted_clusters:
    logger.info(cluster)

print("Prefix Tree:")
template_miner.drain.print_tree()

template_miner.profiler.report(0)

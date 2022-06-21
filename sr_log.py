import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict
from dataclasses_json import dataclass_json, config
import time
from functools import partial
from time import struct_time

time_str = "%Y-%m-%dT%H:%M:%SZ"
time_decoder = partial(lambda x: time.strptime(x, time_str))
time_encoder = partial(lambda x: time.strftime(time_str, x))


@dataclass_json
@dataclass
class Log:
    time: struct_time = field(
        metadata=config(
            encoder=time_encoder,
            decoder=time_decoder,
        )
    )
    query: str
    pmids: List[str]
    num_ret: int
    num_rel: int
    num_rel_ret: int
    logical_hash: str
    mission_hash: str
    has_seed_studies: bool
    raw: str


log_reg = re.compile(r'time="(.*)" level=info msg=".*\]\[query=(.*)\]\[lang=.*\]\[pmids=\[([ 0-9]*)\]\]\[numrel=(.*)\]\[numret=(.*)\]\[numrelret=(.*)\]')


def parse_log_line(line: str):
    groups = log_reg.match(line).groups()
    assert len(groups) == 6
    pmids = groups[2].split()
    log_time = time_decoder(groups[0])
    log = Log(time=log_time,
              query=groups[1],
              pmids=pmids,
              num_rel=int(groups[3]),
              num_ret=int(groups[4]),
              num_rel_ret=int(groups[5]),
              logical_hash=hashlib.sha256(bytes(groups[2] + str(log_time.tm_year) + str(log_time.tm_yday), 'utf-8')).hexdigest(),
              mission_hash=hashlib.sha256(bytes(groups[2], 'utf-8')).hexdigest(),
              has_seed_studies=len(pmids) > 0,
              raw=re.sub(r'\[username=.*\](\[query=)', r'\1', line))
    if log.num_rel == 0:
        log.logical_hash = "INVALID"
        log.mission_hash = "INVALID"
    return log


def load_sr_logs(fname: str) -> List[Log]:
    with open(fname, "r") as f:
        for line in f:
            yield Log.from_json(line)


def load_sr_sessions(fname: str) -> Dict[str, List[Log]]:
    with open(fname, "r") as f:
        data = json.load(f)
    for k, v in data.items():
        data[k] = Log.schema().load(v, many=True)
    return data

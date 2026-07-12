# OmniEvaluator
# Copyright (c) 2026-present NAVER Cloud Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import List, Set

logger = logging.getLogger(__name__)


def get_numa_node_cpus() -> List[Set[int]]:
    # CPU set per NUMA node (read from sysfs), each intersected with the CPUs this
    # process is actually allowed to run on. LLM CPU inference is memory-bandwidth
    # bound, so pinning one worker per NUMA node keeps memory access socket-local.
    # Falls back to a single node holding all allowed CPUs when sysfs NUMA info is
    # absent (non-Linux / container without /sys/devices/system/node).
    allowed = os.sched_getaffinity(0)
    base = "/sys/devices/system/node"
    nodes: List[Set[int]] = []
    try:
        for name in sorted(os.listdir(base)):
            if not (name.startswith("node") and name[4:].isdigit()):
                continue
            with open(os.path.join(base, name, "cpulist")) as cpulist_file:
                cpulist = cpulist_file.read()            # e.g. '0-13,28-41'
            cpus: Set[int] = set()
            for part in filter(None, cpulist.strip().split(",")):
                start, _, end = part.partition("-")
                cpus.update(range(int(start), int(end or start) + 1))
            cpus &= allowed
            if cpus:
                nodes.append(cpus)
    except (OSError, ValueError) as ex:
        logger.debug("NUMA node discovery failed (%s); assuming a single node", ex)
    return nodes or [set(allowed)]


def get_available_cpu_count() -> int:
    # Usable CPU count honoring the cgroup CPU quota (v2 cpu.max, then v1 cfs_quota),
    # falling back to this process's CPU affinity size. Mirrors the intent of
    # ``resource.get_available_cpu_memory`` but for CPU count rather than memory.
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as cpu_max_file:
            quota, period = cpu_max_file.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as quota_file:
            quota = int(quota_file.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as period_file:
            period = int(period_file.read())
        if quota > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))

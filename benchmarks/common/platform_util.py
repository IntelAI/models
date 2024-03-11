#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os  # noqa: F401
import re
import platform as system_platform
import subprocess
import sys

NUMA_NODES_STR_ = "NUMA node(s)"
CPU_SOCKETS_STR_ = "Socket(s)"
CORES_PER_SOCKET_STR_ = "Core(s) per socket"
THREADS_PER_CORE_STR_ = "Thread(s) per core"
LOGICAL_CPUS_STR_ = "CPU(s):"
NUMA_NODE_CPU_RANGE_STR_ = "NUMA node{} CPU(s):"
ONLINE_CPUS_LIST = "On-line CPU(s) list:"


class CPUInfo():
    """CPU information class."""

    def __init__(self):
        """Initialize CPU information class."""
        self._binding_data = CPUInfo._sort_membind_info(self._get_core_membind_info())

    @staticmethod
    def _get_core_membind_info():
        """
        Return sorted information about cores and memory binding.
        E.g.
        CPU ID, Socket ID, Node ID, HT CPU ID,
        0  ,     0    ,    0   ,     0
        1  ,     0    ,    0   ,     1
        :return: list with cpu, sockets, ht core and memory binding information
        :rtype: List[List[str, Any]]
        """
        args = ["lscpu", "--parse=CPU,Core,Socket,Node"]
        process_lscpu = subprocess.check_output(args, universal_newlines=True).split("\n")

        # Get information about core, node, socket and cpu. On a machine with no NUMA nodes, the last column is empty
        # so regex also check for empty string on the last column
        bind_info = []
        for line in process_lscpu:
            pattern = r"^([\d]+,[\d]+,[\d]+,([\d]+|$))"
            regex_out = re.search(pattern, line)
            if regex_out:
                bind_info.append(regex_out.group(1).strip().split(","))

        return bind_info

    @staticmethod
    def _sort_membind_info(membind_bind_info):
        """
        Sore membind info data.
        :param membind_bind_info: raw membind info data
        :type membind_bind_info: List[List[str]]
        :return: sorted membind info
        :rtype: List[List[Dict[str, int]]]
        """
        membind_cpu_list = []
        nodes_count = int(max(element[2] for element in membind_bind_info)) + 1
        # Sort list by Node id
        for node_number in range(nodes_count):
            node_core_list = []
            core_info = {}
            for entry in membind_bind_info:
                cpu_id = int(entry[0])
                core_id = int(entry[1])
                node_id = int(entry[2])
                # On a machine where there is no NUMA nodes, entry[3] could be empty, so set socket_id = -1
                if entry[3] != "":
                    socket_id = int(entry[3])
                else:
                    socket_id = -1

                # Skip nodes other than current node number
                if node_number != node_id:
                    continue

                # Add core info
                if cpu_id == core_id:
                    core_info.update({
                        core_id: {
                            "cpu_id": cpu_id,
                            "node_id": node_id,
                            "socket_id": socket_id,
                        },
                    })
                else:
                    # Add information about Hyper Threading
                    core_info[core_id]["ht_cpu_id"] = cpu_id

            # Change dict of dicts to list of dicts
            for iterator in range(len(core_info)):
                curr_core_id = len(core_info) * node_number + iterator
                single_core_info = core_info.get(curr_core_id)
                if single_core_info:
                    node_core_list.append(single_core_info)

            membind_cpu_list.append(node_core_list)

        return membind_cpu_list

    @property
    def sockets(self):
        """
        Return count of sockets available on server.
        :return: available cores
        :rtype: int
        """
        available_sockets = len(self._binding_data)
        return int(available_sockets)

    @property
    def cores(self):
        """
        Return amount of cores available on server.
        :return: amount of cores
        :rtype: int
        """
        available_cores = self.cores_per_socket * self.sockets
        return int(available_cores)  # type: ignore

    @property
    def cores_per_socket(self):
        """
        Return amount of available cores per socket.
        :return: amount of cores
        :rtype: int
        """
        available_cores_per_socket = len(self._binding_data[0])
        return available_cores_per_socket

    @property
    def binding_information(self):
        """
        Return information about cores and memory binding.
        Format:
        [
            [ # socket 0
                { # Core 0
                    "cpu_id": 0,
                    "node_id": 0,
                    "socket_id": 0,
                    "ht_cpu_id": 56
                }
            ],
            [ # socket 1
                { # Core 0
                    "cpu_id": 28,
                    "node_id": 1,
                    "socket_id": 1,
                    "ht_cpu_id": 84
                }
            ]
        ]
        :return: dict with cpu, sockets, ht core and memory binding information
        :rtype: List[List[Dict[str, int]]]
        """
        return self._binding_data


class PlatformUtil:
    '''
    This module implements a platform utility that exposes functions that
    detects platform information.
    '''

    def __init__(self, args):
        self.args = args
        self.num_cpu_sockets = 0
        self.num_cores_per_socket = 0
        self.num_threads_per_core = 0
        self.num_logical_cpus = 0
        self.num_numa_nodes = 0

        # Core list generated by numactl -H in the case where --numa-cores-per-instance is
        # being used. It then gets pruned based on the cpuset_cpus, in case docker is
        # limiting the cores that the container has access to
        self.cpu_core_list = []

        # Dictionary generated from the cpuset.cpus file (in linux_init) for the case where
        # docker is limiting the number of cores that the container has access to
        self.cpuset_cpus = None

        os_type = system_platform.system()
        if "Windows" == os_type:
            self.windows_init()
        elif "Mac" == os_type or "Darwin" == os_type:
            self.mac_init()
        elif "Linux" == os_type:
            self.linux_init()
        else:
            raise ValueError("Unable to determine Operating system type.")

    def _get_list_from_string_ranges(self, str_ranges):
        """
        Converts a string of numbered ranges (comma separated numbers or ranges) to an
        integer list. Duplicates should be removed and the integer list should be
        ordered.
        For example an input of "3-6,10,0-5" should return [0, 1, 2, 3, 4, 5, 6, 10]
        """
        result_list = []

        for section in str_ranges.split(","):
            if "-" in section:
                # Section is a range, so get the start and end values
                start, end = section.split("-")
                section_list = range(int(start), int(end) + 1)
                result_list += section_list
            elif len(section):
                # This section is either empty or just a single number and not a range
                result_list.append(int(section))

        # Remove duplicates
        result_list = list(set(result_list))

        return result_list

    def _get_cpuset(self):
        """
        Try to get the cpuset.cpus info, since lscpu does not know if docker has limited
        the cpuset accessible to the container
        """
        cpuset = ""
        cpuset_cpus_file = "/sys/fs/cgroup/cpuset/cpuset.cpus"
        if os.path.exists(cpuset_cpus_file):
            with open(cpuset_cpus_file, "r") as f:
                cpuset = f.read()

            if hasattr(self.args, "verbose") and self.args.verbose:
                print("cpuset.cpus: {}".format(cpuset))
        return cpuset

    def get_cpu_cores(self):
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
        cpu_cores = {}
        bind_cpu_cores = []
        core_id = None
        numa_id = None
        for line in lines:
            if line.startswith('processor'):
                cpu_id = int(line.split(':')[-1].strip())
            elif line.startswith('physical id'):
                numa_id = int(line.split(':')[-1].strip())
            elif line.startswith('core id'):
                core_id = int(line.split(':')[-1].strip())
                if numa_id not in cpu_cores:
                    cpu_cores[numa_id] = {}
                if core_id not in cpu_cores[numa_id]:
                    cpu_cores[numa_id][core_id] = []
                cpu_cores[numa_id][core_id].append(cpu_id)

        for numa_id, cores in cpu_cores.items():
            numa_slice = []
            for core_id, cpu_ids in cores.items():
                numa_slice.append(str(cpu_ids[0]))
            bind_cpu_cores.append(numa_slice)
        return bind_cpu_cores

    def linux_init(self):
        lscpu_cmd = "lscpu"
        try:
            lscpu_output = subprocess.check_output([lscpu_cmd],
                                                   stderr=subprocess.STDOUT)
            # handle python2 vs 3 (bytes vs str type)
            if isinstance(lscpu_output, bytes):
                lscpu_output = lscpu_output.decode('utf-8')

            cpu_info = lscpu_output.split('\n')

        except Exception as e:
            print("Problem getting CPU info: {}".format(e))
            sys.exit(1)

        core_list_per_node = {}
        online_cpus_list = ""

        # parse it
        for line in cpu_info:
            #      NUMA_NODES_STR_       = "NUMA node(s)"
            if line.find(NUMA_NODES_STR_) == 0:
                self.num_numa_nodes = int(line.split(":")[1].strip())
            #      CPU_SOCKETS_STR_      = "Socket(s)"
            elif line.find(CPU_SOCKETS_STR_) == 0:
                self.num_cpu_sockets = int(line.split(":")[1].strip())
            #      CORES_PER_SOCKET_STR_ = "Core(s) per socket"
            elif line.find(CORES_PER_SOCKET_STR_) == 0:
                self.num_cores_per_socket = int(line.split(":")[1].strip())
            #      THREADS_PER_CORE_STR_ = "Thread(s) per core"
            elif line.find(THREADS_PER_CORE_STR_) == 0:
                self.num_threads_per_core = int(line.split(":")[1].strip())
            #      LOGICAL_CPUS_STR_     = "CPU(s)"
            elif line.find(LOGICAL_CPUS_STR_) == 0:
                self.num_logical_cpus = int(line.split(":")[1].strip())
            #      ONLINE_CPUS_LIST      = "On-line CPU(s) list"
            elif line.find(ONLINE_CPUS_LIST) == 0:
                online_cpus_list = line.split(":")[1].strip()
            else:
                # Get the ranges of cores per node from NUMA node* CPU(s)
                for node in range(0, self.num_numa_nodes):
                    if line.find(NUMA_NODE_CPU_RANGE_STR_.format(str(node))) == 0:
                        range_for_node = line.split(":")[1].strip()
                        range_list_for_node = self._get_list_from_string_ranges(range_for_node)
                        core_list_per_node[node] = range_list_for_node

        # Try to get the cpuset.cpus info, since lscpu does not know if the cpuset is limited
        cpuset = self._get_cpuset()

        if cpuset:
            num_cores_arg = -1
            if hasattr(self.args, "num_cores"):
                num_cores_arg = self.args.num_cores
            # If the cpuset is the same as the online_cpus_list, then we are using the whole
            # machine, so let's avoid unnecessary complexity and don't bother with the cpuset_cpu list.
            # The cpuset_cpus list will also get populated if the num_cores arg is being specified,
            # since this list will be used to create the numactl args in base_model_init.py
            if (online_cpus_list != "" and online_cpus_list != cpuset) or online_cpus_list == "" or num_cores_arg != -1:
                self.cpuset_cpus = self._get_list_from_string_ranges(cpuset)

        # Uses numactl get the core number for each numa node and adds the cores for each
        # node to the cpu_cores_list array. Only do this if the command is trying to use
        # numa_cores_per_instance we can't count on numactl being installed otherwise and
        # this list is only used for the numactl multi-instance runs.
        num_physical_cores = self.num_cpu_sockets * self.num_cores_per_socket
        if self.num_numa_nodes > 0:
            cores_per_node = int(num_physical_cores / self.num_numa_nodes)
        else:
            cores_per_node = self.num_cores_per_socket
        if hasattr(self.args, "numa_cores_per_instance"):
            if self.num_numa_nodes > 0 and self.args.numa_cores_per_instance is not None:
                try:
                    # Get the list of cores
                    self.cpu_core_list = self.get_cpu_cores()

                    # If we have the cpuset list, cross check that list with our core list and
                    # remove cores that are not part of the cpuset list
                    if self.cpuset_cpus is not None:
                        for socket, core_list in enumerate(self.cpu_core_list):
                            self.cpu_core_list[socket] = [x for x in core_list if int(x) in self.cpuset_cpus]

                    if hasattr(self.args, "verbose") and self.args.verbose:
                        print("Core list: {}".format(self.cpu_core_list), flush=True)

                except Exception as e:
                    print("Warning: An error occured when getting the list of cores:\n {}".
                          format(e))

        if self.cpuset_cpus is not None:
            # Reformat the cpuset_cpus list so that it's split up by node
            for node in core_list_per_node.keys():
                core_list_per_node[node] = [x for x in core_list_per_node[node] if x in self.cpuset_cpus]
            self.cpuset_cpus = core_list_per_node

            # Remove cores that aren't part of the cpu_core_list
            for socket in self.cpuset_cpus.keys():
                if len(self.cpuset_cpus[socket]) > cores_per_node:
                    del self.cpuset_cpus[socket][cores_per_node:]

            # Remove keys with empty lists (sockets where there are no cores enabled in the cpuset)
            self.cpuset_cpus = {k: v for k, v in self.cpuset_cpus.items() if v}

            # Update the number of sockets based on the cpuset
            if len(self.cpuset_cpus.keys()) > 0:
                self.num_cpu_sockets = len(self.cpuset_cpus.keys())

    def windows_init(self):
        NUM_SOCKETS_STR_ = "DeviceID"
        CORES_PER_SOCKET_STR_ = "NumberOfCores"
        THREAD_COUNT_STR_ = "ThreadCount"
        NUM_LOGICAL_CPUS_STR_ = "NumberOfLogicalProcessors"
        num_threads = 0
        wmic_cmd = "wmic cpu get DeviceID, NumberOfCores, \
            NumberOfLogicalProcessors, ThreadCount /format:list"
        try:
            wmic_output = subprocess.check_output(wmic_cmd, shell=True)

            # handle python2 vs 3 (bytes vs str type)
            if isinstance(wmic_output, bytes):
                wmic_output = wmic_output.decode('utf-8')

            cpu_info = wmic_output.split('\r\r\n')

        except Exception as e:
            print("Problem getting CPU info: {}".format(e))
            sys.exit(1)

        # parse it
        for line in cpu_info:
            # CORES_PER_SOCKET_STR_ = "NumberOfCores"
            if line.find(CORES_PER_SOCKET_STR_) == 0:
                self.num_cores_per_socket = int(line.split("=")[1].strip())
            # NUM_LOGICAL_CPUS_STR_ = "NumberOfLogicalProcessors"
            elif line.find(NUM_LOGICAL_CPUS_STR_) == 0:
                self.num_logical_cpus = int(line.split("=")[1].strip())
            # THREAD_COUNT_STR_ = "ThreadCount"
            elif line.find(THREAD_COUNT_STR_) == 0:
                num_threads = int(line.split("=")[1].strip())

        self.num_cpu_sockets = len(re.findall(
            r'\b%s\b' % re.escape(NUM_SOCKETS_STR_), wmic_output))

        if self.num_cpu_sockets > 0 and num_threads:
            self.num_threads_per_core =\
                int(num_threads / self.num_cpu_sockets)

    def mac_init(self):
        raise NotImplementedError("Mac Support not yet implemented")

    @property
    def cores_per_socket(self):
        """
        Return amount of available cores per socket.
        :return: amount of cores
        :rtype: int
        """
        return int(self.num_cores_per_socket)  # type: ignore

    @property
    def sockets(self):
        """
        Return count of sockets available on server.
        :return: available cores
        :rtype: int
        """
        return int(self.num_cpu_sockets)  # type: ignore

    @property
    def cores(self):
        """
        Return amount of cores available on server.
        :return: amount of cores
        :rtype: int
        """
        available_cores = self.num_cores_per_socket * self.num_cpu_sockets
        return int(available_cores)  # type: ignore

    @property
    def logical_cores(self):
        """
        Return amount of logical cores available on server.
        :return: amount of logical cores
        :rtype: int
        """
        return int(self.num_logical_cpus)  # type: ignore

    @property
    def numa_nodes(self):
        """
        Return amount of numa nodes available on server.
        :return: amount of numa nodes
        :rtype: int
        """
        return int(self.num_numa_nodes)  # type: ignore

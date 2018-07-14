import copy
import math
import multiprocessing
import socket
import sys
import tempfile
import unittest

from functools import wraps

import torch
from torch import nn
from torch.distributed import c10d
from torch.nn.parallel import distributed_c10d

from common import TestCase
from common_cuda import TEST_MULTIGPU


TIMEOUT_DEFAULT = 5
TIMEOUT_OVERRIDE = {}


def get_timeout(test_id):
    return TIMEOUT_OVERRIDE.get(test_id.split('.')[-1], TIMEOUT_DEFAULT)


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('localhost', 0))
    sockname = sock.getsockname()
    sock.close()
    return sockname[1]


if not c10d.is_available():
    print('c10d not available, skipping tests')
    sys.exit(0)


class StoreTestBase(object):
    def _create_store(self, i):
        raise RuntimeError("not implemented")

    def _test_set_get(self, fs):
        fs.set("key0", "value0")
        fs.set("key1", "value1")
        fs.set("key2", "value2")
        self.assertEqual(b"value0", fs.get("key0"))
        self.assertEqual(b"value1", fs.get("key1"))
        self.assertEqual(b"value2", fs.get("key2"))

    def test_set_get(self):
        self._test_set_get(self._create_store())


class FileStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile()

    def tearDown(self):
        self.file.close()

    def _create_store(self):
        return c10d.FileStore(self.file.name)


class TCPStoreTest(TestCase, StoreTestBase):
    def _create_store(self):
        addr = 'localhost'
        port = find_free_port()
        return c10d.TCPStore(addr, port, True)


class RendezvousTest(TestCase):
    def test_unknown_handler(self):
        with self.assertRaisesRegex(RuntimeError, "^No rendezvous handler"):
            c10d.rendezvous('invalid://')


class RendezvousFileTest(TestCase):
    def test_common_errors(self):
        with self.assertRaisesRegex(ValueError, 'path missing'):
            gen = c10d.rendezvous('file://?rank=0&size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'rank parameter missing'):
            gen = c10d.rendezvous('file:///tmp/foo?size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'size parameter missing'):
            gen = c10d.rendezvous('file:///tmp/foo?rank=0')
            next(gen)

    def test_nominal(self):
        with tempfile.NamedTemporaryFile() as file:
            url = 'file://%s?size=%d' % (file.name, 2)
            gen0 = c10d.rendezvous(url + "&rank=0")
            store0, rank0, size0 = next(gen0)
            self.assertEqual(0, rank0)
            self.assertEqual(2, size0)
            gen1 = c10d.rendezvous(url + "&rank=1")
            store1, rank1, size1 = next(gen1)
            self.assertEqual(1, rank1)
            self.assertEqual(2, size1)

            # Set value on both stores
            store0.set("key0", "value0")
            store1.set("key1", "value1")

            # Cross check with get
            self.assertEqual(b"value0", store1.get("key0"))
            self.assertEqual(b"value1", store0.get("key1"))


class RendezvousTCPTest(TestCase):
    def test_common_errors(self):
        with self.assertRaisesRegex(ValueError, 'port number missing'):
            gen = c10d.rendezvous('tcp://127.0.0.1?rank=0&size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'rank parameter missing'):
            gen = c10d.rendezvous('tcp://127.0.0.1:23456?size=1')
            next(gen)
        with self.assertRaisesRegex(ValueError, 'size parameter missing'):
            gen = c10d.rendezvous('tcp://127.0.0.1:23456?rank=0')
            next(gen)

    def test_nominal(self):
        addr = 'localhost'
        port = find_free_port()
        url = 'tcp://%s:%d?size=%d' % (addr, port, 2)
        gen0 = c10d.rendezvous(url + "&rank=0")
        store0, rank0, size0 = next(gen0)
        self.assertEqual(0, rank0)
        self.assertEqual(2, size0)
        gen1 = c10d.rendezvous(url + "&rank=1")
        store1, rank1, size1 = next(gen1)
        self.assertEqual(1, rank1)
        self.assertEqual(2, size1)

        # Set value on both stores
        store0.set("key0", "value0")
        store1.set("key1", "value1")

        # Cross check with get
        self.assertEqual(b"value0", store1.get("key0"))
        self.assertEqual(b"value1", store0.get("key1"))


class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1

    @staticmethod
    def join_or_run(fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn(self)
        return wrapper

    # The main process spawns N subprocesses that run the test.
    # This function patches overwrites every test function to either
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    @classmethod
    def setUpClass(cls):
        for attr in dir(cls):
            if attr.startswith('test'):
                fn = getattr(cls, attr)
                setattr(cls, attr, cls.join_or_run(fn))

    @property
    def size(self):
        raise NotImplementedError

    def setUp(self):
        self.rank = self.MAIN_PROCESS_RANK
        self.file = tempfile.NamedTemporaryFile()
        self.port = find_free_port()
        self.processes = [self._spawn_process(rank) for rank in range(int(self.size))]

    def tearDown(self):
        for p in self.processes:
            p.terminate()
        self.file.close()

    def _spawn_process(self, rank):
        name = 'process ' + str(rank)
        process = multiprocessing.Process(target=self._run, name=name, args=(rank,))
        process.start()
        return process

    def _run(self, rank):
        self.rank = rank

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retreiving a corresponding test and executing it.
        getattr(self, self.id().split(".")[2])()
        sys.exit(0)

    def _join_processes(self, fn):
        timeout = get_timeout(self.id())
        for p in self.processes:
            p.join(timeout)

    def gpus_for_rank(self):
        """Multigpu tests are designed to simulate the multi nodes with multi
        GPUs on each node. Nccl backend requires equal #GPUs in each process.
        On a single node, all visible GPUs are evenly
        divided to subsets, each process only uses a subset.
        """
        visible_devices = list(range(torch.cuda.device_count()))
        gpus_per_process = torch.cuda.device_count() // self.size
        gpus_for_rank = []
        for rank in range(self.size):
            gpus_for_rank.append(visible_devices[rank * gpus_per_process: (rank + 1) * gpus_per_process])
        return gpus_for_rank


class ProcessGroupGlooTest(MultiProcessTestCase):
    @property
    def size(self):
        return 4

    def opts(self):
        opts = c10d.ProcessGroupGloo.Options()
        opts.timeout = 1.0
        opts.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        return opts

    def test_broadcast_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.size, self.opts())

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # Every rank is root once, every tensor index is root once
        for i in range(self.size):
            for j in range(2):
                xs = [
                    torch.Tensor([self.rank * self.size + 0.0]),
                    torch.Tensor([self.rank * self.size + 1.0]),
                ]

                broadcast(xs, i, j)
                self.assertEqual(torch.Tensor([i * self.size + j]), xs[0])
                self.assertEqual(torch.Tensor([i * self.size + j]), xs[1])

        # Test overloaded convenience function
        x = torch.Tensor([self.rank + 1.0])
        work = pg.broadcast(x, root=0)
        work.wait()
        self.assertEqual(torch.Tensor([1.0]), x)

    def test_allreduce_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.size, self.opts())

        def allreduce(x, op):
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce([x], opts)
            work.wait()

        # Sum
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.SUM)
        self.assertEqual(torch.Tensor([float(self.size * (self.size + 1) / 2)]), x)

        # Product
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.PRODUCT)
        self.assertEqual(torch.Tensor([float(math.factorial(self.size))]), x)

        # Min
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.MIN)
        self.assertEqual(torch.Tensor([1.0]), x)

        # Max
        x = torch.Tensor([self.rank + 1.0])
        allreduce(x, c10d.ReduceOp.MAX)
        self.assertEqual(torch.Tensor([self.size]), x)

        # Test overloaded convenience function (defaults to using sum)
        x = torch.Tensor([self.rank + 1.0])
        work = pg.allreduce(x)
        work.wait()
        self.assertEqual(torch.Tensor([float(self.size * (self.size + 1) / 2)]), x)


class ProcessGroupNCCLTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        if not hasattr(c10d, "ProcessGroupNCCL"):
            raise unittest.SkipTest("C10D is not built with NCCL process group,"
                                    " skipping test")

        self.rank = self.MAIN_PROCESS_RANK
        self.size = 1
        self.file = tempfile.NamedTemporaryFile()
        self.num_gpus = torch.cuda.device_count()

    def tearDown(self):
        self.file.close()

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_broadcast_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.size)

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # for every root tensor
        for rt in range(self.num_gpus):
            tensors = []
            for i in range(self.num_gpus):
                tensors.append(torch.Tensor([i]).cuda(i))

            broadcast(tensors, self.rank, rt)

            for i in range(self.num_gpus):
                self.assertEqual(tensors[i], tensors[rt])

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_allreduce_ops(self):
        store = c10d.FileStore(self.file.name)
        pg = c10d.ProcessGroupNCCL(store, self.rank, self.size)

        def allreduce(tensors, op):
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            work = pg.allreduce(tensors, opts)
            work.wait()

        # Sum
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.SUM)

        for i in range(self.num_gpus):
            self.assertEqual(
                torch.Tensor([float(self.num_gpus * (self.num_gpus + 1) / 2)]),
                tensors[i])

        # Product
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.PRODUCT)

        for i in range(self.num_gpus):
            self.assertEqual(
                torch.Tensor([float(math.factorial(self.num_gpus))]),
                tensors[i])

        # Min
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.MIN)

        for i in range(self.num_gpus):
            self.assertEqual(torch.Tensor([1.0]), tensors[i])

        # Max
        tensors = []
        for i in range(self.num_gpus):
            tensors.append(torch.Tensor([i + 1]).cuda(i))

        allreduce(tensors, c10d.ReduceOp.MAX)

        for i in range(self.num_gpus):
            self.assertEqual(torch.Tensor([self.num_gpus]), tensors[i])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class DistributedDataParallelTest(MultiProcessTestCase):
    @property
    def size(self):
        return torch.cuda.device_count()

    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_gloo_backend(self):
        store = c10d.TCPStore('localhost', self.port, self.rank == 0)
        opts = c10d.ProcessGroupGloo.Options()
        opts.devices = [c10d.ProcessGroupGloo.create_tcp_device(interface="lo")]
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.size, opts)

        model = Net()
        gpus = self.gpus_for_rank()[self.rank]

        # single gpu training setup
        model = copy.deepcopy(model)
        print(model, gpus[0])
        model.cuda(device=gpus[0])
        print(model)

        # DDP training setup
        # ddp_model = distributed_c10d._DistributedDataParallelC10d(
        #     copy.deepcopy(model),
        #     process_group,
        #     device_ids=gpus)
        # ddp_model.cuda(gpus[0])
        #
        # local_batch_size = len(gpus)
        # global_batch_size = self.size * local_batch_size
        # criterion = nn.MSELoss()
        # input_cpu = torch.randn(global_batch_size, 4)
        # target_cpu = torch.randn(global_batch_size, 4)

        # input = input_cpu.cuda(gpus[0])
        # target = target_cpu.cuda(gpus[0])
        #
        # def step_model(model, input, target, criterion):
        #         model.train()
        #         output = model(input_var)
        #         loss = criterion(output, target)
        #         loss.backward()
        #
        # for _ in range(2):
        #     # step single node model
        #     step_model(model, input, target, criterion)
        #
        #     # step ddp model
        #     step_model(ddp_model,
        #                input[rank * local_batch_size: (rank + 1) * local_batch_size],
        #                target[rank * local_batch_size: (rank + 1) * local_batch_size],
        #                criterion)
        #
        #     # Update weights and run a second iteration to shake out errors
        #     for param in model.parameters():
        #         param.data += param.grad
        #         param.grad = None
        #     for param in ddp_model.parameters():
        #         param.data += param.grad
        #         param.grad = None
        #
        #     self.assertEqual(len(model.parameters()), len(ddp_model.parameters()))
        #     for i, j in zip(model.parameters(), ddp_model.parameters()):
        #         self.assertEqual(i, j)
        #
        #     # Shuffle the input so that DDP input is different
        #     input = input[torch.randperm(batch_size)]


if __name__ == '__main__':
    unittest.main()

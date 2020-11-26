import random
from qd_agents.buffers.priority_buffer import PriorityBuffer

class SuperPQ():
    """
    Class that manages multiple priority queues (buffers)
    """
    def __init__(self, count, capacity):
        self.pq_buffers = []
        for _ in range(count):
            self.pq_buffers.append(PriorityBuffer(capacity))

    @property
    def is_empty(self):
        """
        check if all queues are empty
        """
        check = []
        for pqb in self.pq_buffers:
            check.append(pqb.is_empty)
        return all(check)

    def random_select(self, ignore_empty=False):
        """
        return one queue at random
        """
        candidates = [pqb for pqb in self.pq_buffers if not pqb.is_empty]
        if len(candidates) == 0:
            assert ignore_empty, "All priority buffer are unexpectedly empty"
            return self.pq_buffers[0]

        return random.choice(candidates)

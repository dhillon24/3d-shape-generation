import unittest
import torch
from metrics import earth_mover_distance_cpu, earth_mover_distance_gpu
from metrics import chamfer_distance

class TestMetrics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(1, 994, 3)
        self.y = torch.randn(1, 948, 3)

    def test_earth_mover_distance(self):
        emd1 = earth_mover_distance_cpu(self.x, self.y)
        print("emd (CPU): ", emd1.item())
        self.assertGreaterEqual(emd1.item(), 0.0)
        self.assertLessEqual(emd1.item(), 200.0)
        emd2 = earth_mover_distance_gpu(self.x, self.y)
        print("emd (GPU): ", emd2.item())
        self.assertGreaterEqual(emd2.item(), 0.0)
        self.assertLessEqual(emd2.item(), 200.0)

    def test_chamfer_distance(self):
        cd = chamfer_distance(self.x, self.y)
        print("cd: ", cd.item())
        self.assertGreaterEqual(cd.item(), 0.0)
        self.assertLessEqual(cd.item(), 200.0)

if __name__ == '__main__':
    unittest.main()
import unittest

import torch

from agriworld.coupling import CouplingHead
from agriworld.experts import (
    NitrogenExpert,
    RadiationExpert,
    StomatalExpert,
    TemperatureExpert,
    WaterExpert,
)


class FactorResponseTests(unittest.TestCase):
    def test_temperature_has_cardinal_response(self):
        expert = TemperatureExpert()
        response = expert(torch.tensor([[10.0, 20.0, 28.0, 38.0, 45.0]]))
        self.assertLess(response[0, 0], response[0, 1])
        self.assertGreater(response[0, 2], response[0, 3])
        self.assertAlmostEqual(response[0, 2].item(), 1.0, places=4)

    def test_water_response_is_monotonic(self):
        expert = WaterExpert()
        low, high = expert(torch.tensor([[0.16, 0.32]])).unbind(dim=1)
        self.assertLessEqual(low.item(), high.item())

    def test_nitrogen_response_is_monotonic(self):
        expert = NitrogenExpert()
        bio = torch.tensor([[8.0], [8.0]])
        score, uptake = expert(torch.tensor([[20.0], [150.0]]), bio)
        self.assertLess(score[0].item(), score[1].item())
        self.assertLess(uptake[0].item(), uptake[1].item())

    def test_radiation_and_vpd_directions(self):
        radiation = RadiationExpert()
        low_par, _ = radiation(torch.tensor([[5.0]]), torch.tensor([[3.0]]))
        high_par, _ = radiation(torch.tensor([[15.0]]), torch.tensor([[3.0]]))
        self.assertLess(low_par.item(), high_par.item())

        stomatal = StomatalExpert()
        response = stomatal(torch.tensor([[0.5, 3.0]]))
        self.assertGreater(response[0, 0].item(), response[0, 1].item())

    def test_coupling_preserves_water_and_nitrogen_order(self):
        coupling = CouplingHead()
        common = torch.tensor([[0.8]])
        low_w = coupling(
            common, torch.tensor([[0.2]]), common,
            common, torch.tensor([[100.0]]), torch.tensor([[0.2]]),
        )[0]
        high_w = coupling(
            common, torch.tensor([[0.8]]), common,
            common, torch.tensor([[100.0]]), torch.tensor([[0.2]]),
        )[0]
        self.assertLess(low_w.item(), high_w.item())


if __name__ == "__main__":
    unittest.main()



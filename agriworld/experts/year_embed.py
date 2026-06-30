"""
YearEmbedding — 年份嵌入模块
===============================
将年份 (2019-2023) 映射为 4 维可学习向量，
注入 CouplingHead 让模型区分不同年份的气候背景。
"""

import torch
import torch.nn as nn

YEARS = [2019, 2020, 2021, 2022, 2023]


class YearEmbedding(nn.Module):
    def __init__(self, years: list = YEARS, dim: int = 4):
        super().__init__()
        self.year_to_idx = {y: i for i, y in enumerate(years)}
        self.embed = nn.Embedding(len(years), dim)
        nn.init.zeros_(self.embed.weight)

    def forward(self, year):
        """
        Args:
            year: int 或 [B] tensor of ints
        Returns:
            [B, dim] embedding
        """
        if isinstance(year, int):
            year = torch.tensor([year], dtype=torch.long)
        idx = torch.tensor([self.year_to_idx.get(int(y.item()), 0) for y in year],
                           dtype=torch.long, device=year.device)
        return self.embed(idx)

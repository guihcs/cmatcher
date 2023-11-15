import unittest
from cqa_search import GraphData
from torch_geometric.loader import DataLoader
import torch

class GraphDataTest(unittest.TestCase):

    def test_correct_batching(self):
        # GraphDataBatch(rsi=[1], rpi=[1], rni=[1], cqs=[1, 429], cqp=[1, 429], cqn=[1, 429], x_s=[37], x_sf=[33, 459],
        #                x_p=[28], x_pf=[24, 459], x_n=[9], x_nf=[6, 459], edge_index_s=[2, 112], edge_index_p=[2, 100],
        #                edge_index_n=[2, 20], edge_feat_s=[112], edge_feat_sf=[10, 50], edge_feat_p=[100],
        #                edge_feat_pf=[12, 50], edge_feat_n=[20], edge_feat_nf=[7, 50])



        # GraphDataBatch(rsi=[2], rpi=[2], rni=[2], cqs=[2, 429], cqp=[2, 429], cqn=[2, 429], x_s=[69], x_sf=[62, 459],
        #                x_p=[54], x_pf=[46, 459], x_n=[30], x_nf=[22, 459], edge_index_s=[2, 213], edge_index_p=[2, 193],
        #                edge_index_n=[2, 62], edge_feat_s=[213], edge_feat_sf=[20, 50], edge_feat_p=[193],
        #                edge_feat_pf=[23, 50], edge_feat_n=[62], edge_feat_nf=[17, 50])
        data = [
            GraphData(
                rsi=torch.LongTensor([0]),
                rpi=torch.LongTensor([0]),
                rni=torch.LongTensor([0]),
                cqs=torch.LongTensor(torch.arange(0, 429).unsqueeze(0)),
                cqp=torch.LongTensor(torch.arange(0, 429).unsqueeze(0)),
                cqn=torch.LongTensor(torch.arange(0, 429).unsqueeze(0)),
                x_s=torch.LongTensor(torch.arange(0, 37) % 33),
                x_sf=torch.LongTensor(torch.arange(0, 33 * 459).reshape(33, 459)),
                x_p=torch.LongTensor(torch.arange(0, 28)),
                x_pf=torch.LongTensor(torch.arange(0, 24 * 459).reshape(24, 459)),
                x_n=torch.LongTensor(torch.arange(0, 9)),
                x_nf=torch.LongTensor(torch.arange(0, 6 * 459).reshape(6, 459)),
                edge_index_s=torch.LongTensor(torch.arange(0, 2 * 112).reshape(2, 112)),
                edge_index_p=torch.LongTensor(torch.arange(0, 2 * 100).reshape(2, 100)),
                edge_index_n=torch.LongTensor(torch.arange(0, 2 * 20).reshape(2, 20)),
                edge_feat_s=torch.LongTensor(torch.arange(0, 112)),
                edge_feat_sf=torch.LongTensor(torch.arange(0, 10 * 50).reshape(10, 50)),
                edge_feat_p=torch.LongTensor(torch.arange(0, 100)),
                edge_feat_pf=torch.LongTensor(torch.arange(0, 12 * 50).reshape(12, 50)),
                edge_feat_n=torch.LongTensor(torch.arange(0, 20)),
                edge_feat_nf=torch.LongTensor(torch.arange(0, 7 * 50).reshape(7, 50)),
            ),

        # GraphDataBatch(rsi=[1], rpi=[1], rni=[1], cqs=[1, 429], cqp=[1, 429], cqn=[1, 429], x_s=[32],
        #                x_sf=[29, 459], x_p=[26], x_pf=[22, 459], x_n=[21], x_nf=[16, 459], edge_index_s=[2, 101],
        #                edge_index_p=[2, 93], edge_index_n=[2, 42], edge_feat_s=[101], edge_feat_sf=[10, 50],
        #                edge_feat_p=[93], edge_feat_pf=[11, 50], edge_feat_n=[42], edge_feat_nf=[10, 50])
            GraphData(
                rsi=torch.LongTensor([0]),
                rpi=torch.LongTensor([0]),
                rni=torch.LongTensor([0]),
                cqs=torch.LongTensor(torch.arange(0, 429).unsqueeze(0)),
                cqp=torch.LongTensor(torch.arange(0, 429).unsqueeze(0)),
                cqn=torch.LongTensor(torch.arange(0, 429).unsqueeze(0)),
                x_s=torch.LongTensor(torch.arange(0, 32) % 29),
                x_sf=torch.LongTensor(torch.arange(0, 29 * 459).reshape(29, 459)),
                x_p=torch.LongTensor(torch.arange(0, 26)),
                x_pf=torch.LongTensor(torch.arange(0, 22 * 459).reshape(22, 459)),
                x_n=torch.LongTensor(torch.arange(0, 21)),
                x_nf=torch.LongTensor(torch.arange(0, 16 * 459).reshape(16, 459)),
                edge_index_s=torch.LongTensor(torch.arange(0, 2 * 101).reshape(2, 101)),
                edge_index_p=torch.LongTensor(torch.arange(0, 2 * 93).reshape(2, 93)),
                edge_index_n=torch.LongTensor(torch.arange(0, 2 * 42).reshape(2, 42)),
                edge_feat_s=torch.LongTensor(torch.arange(0, 101)),
                edge_feat_sf=torch.LongTensor(torch.arange(0, 10 * 50).reshape(10, 50)),
                edge_feat_p=torch.LongTensor(torch.arange(0, 93)),
                edge_feat_pf=torch.LongTensor(torch.arange(0, 11 * 50).reshape(11, 50)),
                edge_feat_n=torch.LongTensor(torch.arange(0, 42)),
                edge_feat_nf=torch.LongTensor(torch.arange(0, 10 * 50).reshape(10, 50)),
            ),

        ]

        for batch in DataLoader(data, batch_size=2):
            self.assertEqual(str(batch), 'GraphDataBatch(rsi=[2], rpi=[2], rni=[2], cqs=[2, 429], cqp=[2, 429], cqn=[2, 429], x_s=[69], x_sf=[62, 459], x_p=[54], x_pf=[46, 459], x_n=[30], x_nf=[22, 459], edge_index_s=[2, 213], edge_index_p=[2, 193], edge_index_n=[2, 62], edge_feat_s=[213], edge_feat_sf=[20, 50], edge_feat_p=[193], edge_feat_pf=[23, 50], edge_feat_n=[62], edge_feat_nf=[17, 50])')
            self.assertTrue(torch.equal(batch.rsi, torch.LongTensor([0, 37])))
            self.assertTrue(torch.equal(batch.rpi, torch.LongTensor([0, 28])))
            self.assertTrue(torch.equal(batch.rni, torch.LongTensor([0, 9])))
            self.assertTrue(torch.equal(batch.cqs, torch.LongTensor(torch.arange(0, 429).unsqueeze(0).repeat(2, 1))))
            self.assertTrue(torch.equal(batch.cqp, torch.LongTensor(torch.arange(0, 429).unsqueeze(0).repeat(2, 1))))
            self.assertTrue(torch.equal(batch.cqn, torch.LongTensor(torch.arange(0, 429).unsqueeze(0).repeat(2, 1))))
            self.assertTrue(torch.equal(batch.x_s, torch.LongTensor(torch.cat([torch.arange(0, 37) % 33, 33 + torch.arange(0, 32) % 29]))))









if __name__ == '__main__':
    unittest.main()

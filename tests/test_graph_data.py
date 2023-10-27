import unittest
from cqa_search import GraphData
from torch_geometric.loader import DataLoader
import torch

class GraphDataTest(unittest.TestCase):

    def test_correct_batching(self):

        data = [
            GraphData(
                rsi=torch.LongTensor([0]),
                rpi=torch.LongTensor([0]),
                rni=torch.LongTensor([0]),
                cqs=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                cqp=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                cqn=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                x_s=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                x_sf=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                x_p=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                x_pf=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                x_n=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                x_nf=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                edge_index_s=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                edge_index_p=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                edge_index_n=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                edge_feat_s=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                edge_feat_sf=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                edge_feat_p=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                edge_feat_pf=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                edge_feat_n=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                edge_feat_nf=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
            ),
            GraphData(
                rsi=torch.LongTensor([0]),
                rpi=torch.LongTensor([0]),
                rni=torch.LongTensor([0]),

                cqs=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                cqp=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                cqn=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                x_s=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                x_sf=torch.LongTensor([[1, 2, 3],
                                       [4, 5, 6]]),
                x_p=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                x_pf=torch.LongTensor([[1, 2, 3],
                                       [4, 5, 6]]),
                x_n=torch.LongTensor([[1, 2, 3],
                                      [4, 5, 6]]),
                x_nf=torch.LongTensor([[1, 2, 3],
                                       [4, 5, 6]]),
                edge_index_s=torch.LongTensor([[1, 2, 3],
                                               [4, 5, 6]]),
                edge_index_p=torch.LongTensor([[1, 2, 3],
                                               [4, 5, 6]]),
                edge_index_n=torch.LongTensor([[1, 2, 3],
                                               [4, 5, 6]]),
                edge_feat_s=torch.LongTensor([[1, 2, 3],
                                              [4, 5, 6]]),
                edge_feat_sf=torch.LongTensor([[1, 2, 3],
                                               [4, 5, 6]]),
                edge_feat_p=torch.LongTensor([[1, 2, 3],
                                              [4, 5, 6]]),
                edge_feat_pf=torch.LongTensor([[1, 2, 3],
                                               [4, 5, 6]]),
                edge_feat_n=torch.LongTensor([[1, 2, 3],
                                              [4, 5, 6]]),
                edge_feat_nf=torch.LongTensor([[1, 2, 3],
                                               [4, 5, 6]]),
            )
        ]

        for batch in DataLoader(data, batch_size=2):
            self.assertTrue(torch.equal(batch.rsi, torch.LongTensor([0, 2])))
            self.assertTrue(torch.equal(batch.rpi, torch.LongTensor([0, 2])))
            self.assertTrue(torch.equal(batch.rni, torch.LongTensor([0, 2])))

            self.assertTrue(torch.equal(batch.cqs, torch.LongTensor([[1, 2, 3],
                                                                    [4, 5, 6],
                                                                    [1, 2, 3],
                                                                    [4, 5, 6]])))

            self.assertTrue(torch.equal(batch.cqp, torch.LongTensor([[1, 2, 3],
                                                                    [4, 5, 6],
                                                                    [1, 2, 3],
                                                                    [4, 5, 6]])))

            self.assertTrue(torch.equal(batch.cqn, torch.LongTensor([[1, 2, 3],
                                                                    [4, 5, 6],
                                                                    [1, 2, 3],
                                                                    [4, 5, 6]])))

            self.assertTrue(torch.equal(batch.x_s, torch.LongTensor([[1, 2, 3],
                                                                    [4, 5, 6],
                                                                    [3, 4, 5],
                                                                    [6, 7, 8]])))

            self.assertTrue(torch.equal(batch.x_sf, torch.LongTensor([[1, 2, 3],
                                                                    [4, 5, 6],
                                                                    [1, 2, 3],
                                                                    [4, 5, 6]])))

            self.assertTrue(torch.equal(batch.x_p, torch.LongTensor([[1, 2, 3],
                                                                    [4, 5, 6],
                                                                    [3, 4, 5],
                                                                    [6, 7, 8]])))

            self.assertTrue(torch.equal(batch.x_pf, torch.LongTensor([[1, 2, 3],
                                                                    [4, 5, 6],
                                                                    [1, 2, 3],
                                                                    [4, 5, 6]])))

            self.assertTrue(torch.equal(batch.x_n, torch.LongTensor([[1, 2, 3],
                                                                    [4, 5, 6],
                                                                    [3, 4, 5],
                                                                    [6, 7, 8]])))

            self.assertTrue(torch.equal(batch.x_nf, torch.LongTensor([[1, 2, 3],
                                                                    [4, 5, 6],
                                                                    [1, 2, 3],
                                                                    [4, 5, 6]])))

            self.assertTrue(torch.equal(batch.edge_index_s, torch.LongTensor([[1, 2, 3, 3, 4, 5],
                                                                              [4, 5, 6, 6, 7, 8]])))

            self.assertTrue(torch.equal(batch.edge_feat_sf, torch.LongTensor([[1, 2, 3],
                                                                            [4, 5, 6],
                                                                            [1, 2, 3],
                                                                            [4, 5, 6]])))

            self.assertTrue(torch.equal(batch.edge_index_p, torch.LongTensor([[1, 2, 3, 3, 4, 5],
                                                                              [4, 5, 6, 6, 7, 8]])))

            self.assertTrue(torch.equal(batch.edge_feat_pf, torch.LongTensor([[1, 2, 3],
                                                                              [4, 5, 6],
                                                                              [1, 2, 3],
                                                                              [4, 5, 6]])))

            self.assertTrue(torch.equal(batch.edge_index_n, torch.LongTensor([[1, 2, 3, 3, 4, 5],
                                                                              [4, 5, 6, 6, 7, 8]])))

            self.assertTrue(torch.equal(batch.edge_feat_nf, torch.LongTensor([[1, 2, 3],
                                                                              [4, 5, 6],
                                                                              [1, 2, 3],
                                                                              [4, 5, 6]])))





if __name__ == '__main__':
    unittest.main()

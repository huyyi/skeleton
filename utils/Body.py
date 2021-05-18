import numpy as np
import matplotlib.pyplot as plt

inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]


class NtuBody(object):
    def __init__(self) -> None:
        super().__init__()
        self.joints = 25
        self.adjacent_matrix = np.zeros((self.joints, self.joints))
        for idx in [(x - 1, y - 1) for (x, y) in inward_ori_index]:
            self.adjacent_matrix[idx] = 1
        self.adjacent_matrix_with_I = self.adjacent_matrix + np.identity(self.joints)

    @staticmethod
    def get_symmetric_adj(A: np.array) -> np.array:
        """上三角矩阵生成对称阵

        Returns:
            np.array: [description]
        """
        return A + (A.T - np.diag(A.diagonal()))

    @staticmethod
    def normalize_adjacency_matrix(A: np.array) -> np.array:
        node_degrees = A.sum(-1)
        degs_inv_sqrt = np.power(node_degrees, -0.5)
        norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
        return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

    def plt_body(self):
        normalized_A = self.normalize_adjacency_matrix(self.adjacent_matrix)
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(self.adjacent_matrix_with_I, cmap='gray')
        ax[1].imshow(self.adjacent_matrix, cmap='gray')
        ax[2].imshow(normalized_A, cmap='gray')
        plt.show()


if __name__ == "__main__":
    ntu = NtuBody()
    ntu.plt_body()

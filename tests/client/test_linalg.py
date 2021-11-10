import numpy as np
import tinychain as tc
import unittest

from testutils import ClientTest

ENDPOINT = "/transact/hypothetical"


class LinearAlgebraTests(ClientTest):
    def testNorm(self):
        shape = [2, 3, 4]
        matrices = np.arange(24).reshape(shape)

        cxt = tc.Context()
        cxt.matrices = tc.tensor.Dense.load(shape, tc.I32, matrices.flatten().tolist())
        cxt.result = tc.linalg.norm(tensor=cxt.matrices)

        expected = [np.linalg.norm(matrix) for matrix in matrices]

        actual = self.host.post(ENDPOINT, cxt)
        actual = actual[tc.uri(tc.tensor.Dense)][1]

        self.assertEqual(actual, expected)

    def testQR(self):
        matrix = np.arange(6).reshape([2, 3])

        cxt = tc.Context()
        cxt.matrices = tc.tensor.Dense.load([1, 2, 3], tc.I32, matrix.flatten().tolist())
        cxt.qr = tc.linalg.qr
        cxt.result = cxt.qr(matrices=cxt.matrices)

        expected = [np.linalg.qr(matrix)]

        actual = self.host.post(ENDPOINT, cxt)
        print(expected)
        print(actual)

if __name__ == "__main__":
    unittest.main()

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
        shape = [3, 4]
        matrix = np.arange(12).reshape(shape)

        cxt = tc.Context()
        cxt.matrix = tc.tensor.Dense.load(shape, tc.F32, matrix.flatten().tolist())
        cxt.qr = tc.linalg.qr
        cxt.result = cxt.qr(matrix=cxt.matrix)

        expected = np.linalg.qr(matrix)

        import json
        print(json.dumps(tc.to_json(cxt), indent=4))

        actual = self.host.post(ENDPOINT, cxt)
        print(expected[0].shape, expected[1].shape)
        print(actual)

if __name__ == "__main__":
    unittest.main()

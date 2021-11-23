import numpy as np
import numba
import meshio
from scipy.sparse import coo_matrix, csc_matrix


# extracts mesh, saving, elements, border_elements, coordinates, number of elements, number of dof per element,
# number of DOFs, and the element type (QUAD4, QUAD8)
def extract_mesh(mesh_data):
    coord = mesh_data.points.T[:2]
    connectivity = mesh_data.cells[-1].data
    connectivity_surface = mesh_data.cells[-2].data

    elem_phys = mesh_data.cell_data['gmsh:physical']
    elem_phys_dict = mesh_data.field_data

    _type = mesh_data.cells[-1].type
    elem_type = -999
    if _type == 'quad':
        elem_type = 3
    elif _type == 'quad8':
        elem_type = 16
    elif _type == 'triangle':
        elem_type = 2
    elif _type == 'triangle6':
        elem_type = 9
    else:
        print('Not supported!')

    return connectivity, connectivity_surface, coord, elem_phys, elem_phys_dict, \
           connectivity.shape[0], 2 * connectivity.shape[1], 2 * coord.shape[1], elem_type


# quadrature points, from 1-3 QP can be currently chosen
def select_qp(elem_type, n_qp):
    if elem_type == 3 or elem_type == 16:
        if n_qp == 1:
            QPs = np.array([[0., 0]])
            wQP = np.array([2. * 2.])
        elif n_qp == 2:
            QPs = np.array([[-np.sqrt(1. / 3), -np.sqrt(1. / 3)],
                            [np.sqrt(1. / 3), -np.sqrt(1. / 3)],
                            [-np.sqrt(1. / 3), np.sqrt(1. / 3)],
                            [np.sqrt(1. / 3), np.sqrt(1. / 3)]])
            wQP = np.array([1, 1, 1, 1.])
        elif n_qp == 3:
            QPs = np.array([[-np.sqrt(3. / 5), -np.sqrt(3. / 5)],
                            [0, -np.sqrt(3. / 5)],
                            [np.sqrt(3. / 5), -np.sqrt(3. / 5)],
                            [-np.sqrt(3. / 5), 0],
                            [0, 0],
                            [np.sqrt(3. / 5), 0],
                            [-np.sqrt(3. / 5), np.sqrt(3. / 5)],
                            [0, np.sqrt(3. / 5)],
                            [np.sqrt(3. / 5), np.sqrt(3. / 5)]])
            wQP = np.array([5. / 9 * 5. / 9,
                            8. / 9 * 5. / 9,
                            5. / 9 * 5. / 9,
                            5. / 9 * 8. / 9,
                            8. / 9 * 8. / 9,
                            5. / 9 * 8. / 9,
                            5. / 9 * 5. / 9,
                            8. / 9 * 5. / 9,
                            5. / 9 * 5. / 9])
    elif elem_type == 2 or elem_type == 9:
        if n_qp == 1:
            QPs = np.array([[1. / 3, 1. / 3]])
            wQP = np.array([0.5])
        elif n_qp == 2:
            QPs = np.array([[1. / 6, 1. / 6],
                            [2. / 3, 1. / 6],
                            [1. / 6, 2. / 3]])
            wQP = np.array([1. / 6, 1. / 6, 1. / 6])
        elif n_qp == 3:
            QPs = np.array([[1. / 3, 1. / 3],
                            [3. / 5, 1. / 5],
                            [1. / 5, 3. / 5],
                            [1. / 5, 1. / 5]])
            wQP = np.array([-9. / 32, 25. / 96, 25. / 96, 25. / 96])
        elif n_qp == 4:
            QPs = np.array([[0., 0.],
                            [0.5, 0.],
                            [1., 0.],
                            [0.5, 0.5],
                            [0., 1.],
                            [0., 0.5],
                            [1. / 3, 1. / 3]])
            wQP = np.array([1. / 40, 1. / 15, 1. / 40, 1. / 15, 1. / 40, 1. / 15, 9. / 40])
    return QPs, wQP


# shape function for QUAD4 and QUAD8
def select_shape_functions(elem_type):
    if elem_type == 2:
        @numba.njit
        def N(xi, eta):
            return np.array([1. - xi - eta,
                             xi,
                             eta])

        @numba.njit
        def dNdxi(xi, eta):
            return np.array([[-1., 1., 0.],
                             [-1., 0., 1.]])

    elif elem_type == 9:
        @numba.njit
        def N(xi, eta):
            return np.array([(1 - xi - eta) * (1 - 2 * xi - 2 * eta),
                             xi * (2 * xi - 1),
                             eta * (2 * eta - 1),
                             4 * xi * (1 - xi - eta),
                             4 * xi * eta,
                             4 * eta * (1 - xi - eta)])

        @numba.njit
        def dNdxi(xi, eta):
            return np.array([[4 * xi + 4 * eta - 3, 4 * xi - 1, 0, -4 * (2 * xi + eta - 1), 4 * eta, -4 * eta],
                             [4 * xi + 4 * eta - 3, 0, 4 * eta - 1, -4 * xi, 4 * xi, -4 * (xi + 2 * eta - 1)]])

    elif elem_type == 3:
        @numba.njit
        def N(xi, eta):
            return np.array([0.25 * (1 - xi) * (1 - eta),
                             0.25 * (1 + xi) * (1 - eta),
                             0.25 * (1 + xi) * (1 + eta),
                             0.25 * (1 - xi) * (1 + eta)])

        @numba.njit
        def dNdxi(xi, eta):
            return np.array([[-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
                             [-0.25 * (1 - xi), -0.25 * (1 + xi), 0.25 * (1 + xi), 0.25 * (1 - xi)]])

    elif elem_type == 16:
        @numba.njit
        def N(xi, eta):
            return np.array([-0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
                             -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
                             -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
                             -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
                             0.5 * (1 - xi * xi) * (1 - eta),
                             0.5 * (1 + xi) * (1 - eta * eta),
                             0.5 * (1 - xi * xi) * (1 + eta),
                             0.5 * (1 - xi) * (1 - eta * eta)])

        @numba.njit
        def dNdxi(xi, eta):
            return np.array([[xi*(0.5-0.5*eta)-0.25*eta*eta+0.25*eta, xi*(0.25-0.5*eta)-0.25*xi*xi+0.5*eta],
                             [xi*(0.5-0.5*eta)+0.25*eta*eta-0.25*eta, -0.25*xi*xi+xi*(0.5*eta-0.25)+0.5*eta],
                             [xi*(0.5*eta+0.5)+0.25*eta*eta+0.25*eta, 0.25*xi*xi+xi*(0.5*eta+0.25)+0.5*eta],
                             [xi*(0.5*eta+0.5)-0.25*eta*eta-0.25*eta, 0.25*xi*xi+xi*(-0.5*eta-0.25)+0.5*eta],
                             [xi*(eta-1), 0.5*xi*xi-0.5],
                             [0.5-0.5*eta*eta, -(xi+1)*eta],
                             [-xi*(eta+1), 0.5-0.5*xi*xi],
                             [0.5*eta*eta-0.5, (xi-1)*eta]]).T

    @numba.njit
    def N_block(xi, eta, n_d=2):
        Ni = N(xi, eta)
        return np.kron(Ni, np.eye(n_d))

    @numba.njit
    def dNdX(X, Y, xi, eta):
        dNdxi_ = dNdxi(xi, eta)
        coord_ = np.stack((X, Y))
        tmp = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                for k in range(X.shape[0]):
                    tmp[i, j] += coord_[i, k] * dNdxi_[j, k]
        return dNdxi_.T.dot(np.linalg.inv(tmp))

    @numba.njit
    def B_block(X, Y, xi, eta):
        dNdX_ = dNdX(X, Y, xi, eta)
        return np.kron(dNdX_.T, np.eye(2))

    @numba.njit
    def B_block_symmetric(X, Y, xi, eta):
        dNdX_ = dNdX(X, Y, xi, eta)
        B_mat = np.zeros((3, 2*dNdX_.shape[0]))
        B_mat[0, ::2] = dNdX_[:, 0]
        B_mat[1, 1::2] = dNdX_[:, 1]
        B_mat[2, 1::2] = dNdX_[:, 0]
        B_mat[2, ::2] = dNdX_[:, 1]
        return B_mat

    return N, dNdxi, N_block, B_block, B_block_symmetric


class Mesh:
    """
    This class extracts GMSH files and supports linear/quadratic triangular and quad meshes. It provides several utility functions
    """
    def __init__(self, mesh_file, n_qp=2):
        self.mesh_data = meshio.read(mesh_file)
        self.connectivity, self.connectivity_surface, self.coord, self.elem_phys, self.elem_phys_name, \
        self.n_el, self.dof_per_el, self.n_dof, self.elem_type = extract_mesh(self.mesh_data)
        self.el_centers = np.zeros((self.connectivity.shape[0], 2))
        for i, el in enumerate(self.connectivity):
            self.el_centers[i] = self.coord[:, el].mean(1)
        self.N, self.dNdxi, self.N_block, self.B_block, self.B_block_symmetric = select_shape_functions(self.elem_type)

        self.QPs, self.wQP = select_qp(self.elem_type, n_qp)
        self.n_qp = self.QPs.shape[0]

    # calculate physical weights for each element and qp
    def calculate_physical_weights(self):
        weights = np.zeros((self.n_el, self.n_qp))
        for i, el in enumerate(self.connectivity):
            X, Y = self.coord[:, el]
            for j, qp in enumerate(self.QPs):
                dNdxi_ = self.dNdxi(qp[0], qp[1])
                weights[i, j] = self.wQP[j] * abs(np.linalg.det(np.einsum('ik, jk -> ij', np.stack((X, Y)), dNdxi_)))
        return weights

    # calculate the B matrix for each element and QP
    def calculate_B_all(self):
        B_all = np.zeros((self.n_el, self.n_qp, 4, self.dof_per_el))
        for i, el in enumerate(self.connectivity):
            X, Y = self.coord[:, el]
            for j, qp in enumerate(self.QPs):
                B_all[i, j] = self.B_block(X, Y, qp[0], qp[1])
        return B_all

    # calculate the B matrix for each element and QP
    def calculate_B_symmetric_all(self):
        B_all = np.zeros((self.n_el, self.n_qp, 3, self.dof_per_el))
        for i, el in enumerate(self.connectivity):
            X, Y = self.coord[:, el]
            for j, qp in enumerate(self.QPs):
                B_all[i, j] = self.B_block_symmetric(X, Y, qp[0], qp[1])
        return B_all

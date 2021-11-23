from mesh import Mesh
import numpy as np
from tqdm import tqdm
from scipy.sparse import csc_matrix, coo_matrix, linalg as sla


def transformMesh(mesh, dofs_list, disps_list, periodic, nu=0.3):
    """Compute displacement field for a [0,1]*[0,1] domain by solving a linear elastic problem.
    Displacements are prescribed on the interface and boundaries. disps_list can contain for each parameter sample
    list of displacements.

    Keyword arguments:
    mesh -- reference mesh, mesh object defined in mesh.py
    dofs_list -- list of nodes that should be deformed
    disps_list -- list of list of displacements at nodes specified in dofs_list
    periodic -- either fixed boundaries or periodic boundaries
    nu -- Poisson ratio, changes the compressibility
    """
    # get mesh information
    connectivity = mesh.connectivity  # connectivity list of elements
    coord = mesh.coord  # coordinates of element nodes
    N_el = mesh.n_el  # number of elements
    N_qp = mesh.n_qp  # number of QP
    N_dof = mesh.n_dof  # number of DOFs

    ##### LOAD SHAPE FUNCTIONS AND CALCULATE DERIVATIVES #####
    weights = mesh.calculate_physical_weights()  # weights of quadrature point of every element
    B_symmetric_all = mesh.calculate_B_symmetric_all()  # needed to solve FE problem
    B_all = mesh.calculate_B_all()  # needed to obtain deformation F and Jacobian det(F)

    ##### ASSEMBLE GLOBAL STIFFNESS MATRIX #####
    E = 1  # Young's modulus

    # Material matrix D
    D = np.array([[1 - nu, nu, 0],
                  [nu, 1 - nu, 0],
                  [0, 0, (1 - 2 * nu) / 2]])
    D = D * E / (1 + nu) / (1 - 2 * nu)

    # assemble global stiffness matrix
    rows = []
    cols = []
    data = []

    # loop over all elements and quadrature points
    for i, el in enumerate(connectivity):
        Le = np.stack((el * 2, el * 2 + 1)).T.reshape(-1)  # for assembly
        Be = B_symmetric_all[i]  # pick the B matrix of current element
        we = weights[i]  # pick weights of current element

        # sum up the contribution for each QP
        ke = np.zeros((len(Le), len(Le)))
        for j in range(N_qp):
            ke += np.dot(Be[j].T, np.dot(D, Be[j])) * we[j]

        # assemble the element contribution to global DOF
        for k in range(len(Le)):
            for l in range(len(Le)):
                rows.append(Le[k])
                cols.append(Le[l])
                data.append(ke[k, l])

    ##### BOUNDARY CONDITIONS AS CONSTRAINT MATRIX #####

    # locate nodes on each boundary edge
    nodes_left = np.where(coord[0, :] <= 0 + 1e-6)[0]
    nodes_right = np.where(coord[0, :] >= 1 - 1e-6)[0]
    nodes_top = np.where(coord[1, :] >= 1 - 1e-6)[0]
    nodes_bot = np.where(coord[1, :] <= 0 + 1e-6)[0]

    ##### PBC #####
    # corner nodes/dofs
    node_p1 = np.intersect1d(np.where(coord[0, :] <= 0 + 1e-6)[0], np.where(coord[1, :] <= 0 + 1e-6)[0])
    node_p2 = np.intersect1d(np.where(coord[0, :] >= 1 - 1e-6)[0], np.where(coord[1, :] <= 0 + 1e-6)[0])
    node_p3 = np.intersect1d(np.where(coord[0, :] >= 1 - 1e-6)[0], np.where(coord[1, :] >= 1 - 1e-6)[0])
    node_p4 = np.intersect1d(np.where(coord[0, :] <= 0 + 1e-6)[0], np.where(coord[1, :] >= 1 - 1e-6)[0])

    dofs_p1 = np.concatenate((node_p1 * 2, node_p1 * 2 + 1))
    dofs_p2 = np.concatenate((node_p2 * 2, node_p2 * 2 + 1))
    dofs_p3 = np.concatenate((node_p3 * 2, node_p3 * 2 + 1))
    dofs_p4 = np.concatenate((node_p4 * 2, node_p4 * 2 + 1))

    # remove corner nodes/dofs from edges
    nodes_left = np.setdiff1d(np.setdiff1d(np.setdiff1d(np.setdiff1d(nodes_left, node_p1), node_p2), node_p3),
                              node_p4)
    nodes_bot = np.setdiff1d(np.setdiff1d(np.setdiff1d(np.setdiff1d(nodes_bot, node_p1), node_p2), node_p3),
                             node_p4)
    nodes_right = np.setdiff1d(np.setdiff1d(np.setdiff1d(np.setdiff1d(nodes_right, node_p1), node_p2), node_p3),
                               node_p4)
    nodes_top = np.setdiff1d(np.setdiff1d(np.setdiff1d(np.setdiff1d(nodes_top, node_p1), node_p2), node_p3),
                             node_p4)

    # find which node is opposite of which node
    depMatrix_right = np.zeros((nodes_right.shape[0], 2), dtype=int)
    for i in range(nodes_right.shape[0]):
        for j in range(nodes_left.shape[0]):
            if np.isclose(coord[1, nodes_right[i]], coord[1, nodes_left[j]]):
                depMatrix_right[i, :] = [nodes_right[i], nodes_left[j]]
                break
    depMatrix_top = np.zeros((nodes_top.shape[0], 2), dtype=int)
    for i in range(nodes_top.shape[0]):
        for j in range(nodes_bot.shape[0]):
            if np.isclose(coord[0, nodes_top[i]], coord[0, nodes_bot[j]]):
                depMatrix_top[i, :] = [nodes_top[i], nodes_bot[j]]
                break

    # construct the dofs from nodes
    dep_dofs = np.concatenate((2 * depMatrix_right + 1, 2 * depMatrix_top))
    fixed_dofs = np.concatenate((2 * depMatrix_right, 2 * depMatrix_top + 1)).T.reshape(-1)

    if periodic:
        # constrain bottom right with bottom left, top left with bottom left, top right and bottom right
        # and concatenate
        corners = np.concatenate((np.stack((dofs_p2, dofs_p1)).T,
                                  np.stack((dofs_p4, dofs_p1)).T,
                                  np.stack((dofs_p3, dofs_p2)).T))
        dep_dofs = np.concatenate((dep_dofs, corners))

        # construct constraint matrix C
        Crows = []
        Ccols = []
        Cdata = []

        for i, dof in enumerate(dep_dofs):
            Crows.append(i)
            Ccols.append(dof[0])
            Cdata.append(1)
            Crows.append(i)
            Ccols.append(dof[1])
            Cdata.append(-1)
        Crows.append(dep_dofs.shape[0])
        Ccols.append(dofs_p1[0])
        Cdata.append(1)
        Crows.append(dep_dofs.shape[0] + 1)
        Ccols.append(dofs_p1[1])
        Cdata.append(1)

        # fix x-component of left and right edge and y-component of top and bottom edge
        for i, dof in enumerate(fixed_dofs):
            Crows.append(dep_dofs.shape[0] + 2 + i)
            Ccols.append(dof)
            Cdata.append(1)

        # constraint matrix for applying displacement on interface
        for i, dof in enumerate(dofs_list):
            Crows.append(i + dep_dofs.shape[0] + 2 + fixed_dofs.shape[0])
            Ccols.append(dof)
            Cdata.append(1)

        # concatenate both constraint matrices
        rows, cols, data = np.array(rows), np.array(cols), np.array(data)
        Crows, Ccols, Cdata = np.array(Crows), np.array(Ccols), np.array(Cdata)
        rows = np.concatenate((rows, Crows + N_dof, np.array(Ccols)))
        cols = np.concatenate((cols, Ccols, Crows + N_dof))
        data = np.concatenate((data, Cdata, Cdata))

        A = csc_matrix(coo_matrix((data, (rows, cols))))

        ##### SOLVER ###########################
        # compute sparse LU decomposition
        lu = sla.splu(csc_matrix(A))

        # set up right hand side
        F = np.zeros(A.shape[0])

        # loop over all prescribed displacements in disps_list
        us = []
        detFs = []
        Fs = []
        for _, disp in tqdm(enumerate(disps_list)):
            F[-len(dofs_list):] = disp

            # solve the linear system of equation
            u = lu.solve(F)[:N_dof]
            us.append(u)

            # loop over all elements and quadrature points and compute F/detF at quadrature points
            deformationF = np.zeros((N_el, N_qp, 9))
            detF = np.zeros((N_el, N_qp))
            for i, el in enumerate(connectivity):
                Le = np.stack((el * 2, el * 2 + 1)).T.reshape(-1)  # to assemble matrix
                ue = u[Le]
                Be = B_all[i]
                for j in range(N_qp):
                    dudX = np.dot(Be[j], ue).reshape(2, 2).T
                    deformationF[i, j, [0, 1, 3, 4]] = dudX.reshape(-1) + [1, 0, 0, 1]
                    deformationF[i, j, 8] = 1
                    detF[i, j] = np.linalg.det(dudX + np.eye(2))
            detFs.append(detF)
            Fs.append(deformationF)

    else:
        ix = np.setdiff1d(range(N_dof), np.concatenate((dofs_list, fixed_dofs, dofs_p1, dofs_p2, dofs_p3, dofs_p4)))
        Kff = csc_matrix(coo_matrix((data, (rows, cols))))[np.ix_(ix, ix)]
        Kfi = csc_matrix(coo_matrix((data, (rows, cols))))[np.ix_(ix, dofs_list)]

        # loop over all prescribed displacements in disps_list
        us = []
        detFs = []
        Fs = []
        for _, disp in tqdm(enumerate(disps_list)):
            u = np.zeros(N_dof)
            u[ix] = sla.bicgstab(Kff, -Kfi @ disp)[0]
            u[dofs_list] = disp

            us.append(u)

            # loop over all elements and quadrature points and compute F/detF at quadrature points
            deformationF = np.zeros((N_el, N_qp, 9))
            detF = np.zeros((N_el, N_qp))
            for i, el in enumerate(connectivity):
                Le = np.stack((el * 2, el * 2 + 1)).T.reshape(-1)  # to assemble matrix
                ue = u[Le]
                Be = B_all[i]
                for j in range(N_qp):
                    dudX = np.dot(Be[j], ue).reshape(2, 2).T
                    deformationF[i, j, [0, 1, 3, 4]] = dudX.reshape(-1) + [1, 0, 0, 1]
                    deformationF[i, j, 8] = 1
                    detF[i, j] = np.linalg.det(dudX + np.eye(2))
            detFs.append(detF)
            Fs.append(deformationF)
    return us, Fs, detFs

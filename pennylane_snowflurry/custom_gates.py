from pennylane.operation import Operation
from functools import lru_cache
import numpy as np
import pennylane as qml
from copy import copy


class TDagger(Operation):
    r"""ajoint(T)(pi/2)(wires)
    The single-qubit ajoint of T operation
    
    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None
    
    @staticmethod
    @lru_cache()
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            ndarray: matrix
        """
        return np.array([[1, 0], [0, 0.70710678-0.70710678j]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        Returns:
            array: eigenvalues
        """
        return np.array([1, 0.70710678-0.70710678j])

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        """
        return [qml.adjoint(qml.T(qml.T)(wires))]

    def pow(self, z):
        z = z % 8
        pow_map = {
            0: lambda op: [],
            1: lambda op: [copy(op)],
            2: lambda op: [qml.adjoint(qml.S)(wires=op.wires)],
            4: lambda op: [qml.Z(wires=op.wires)],
            6: lambda op: [qml.S(wires=op.wires)],
            7: lambda op: [qml.T(wires=op.wires)]
        }
        return pow_map.get(z, lambda op: [qml.PhaseShift(np.pi * z / 4, wires=op.wires)])(
            self
        )

    def adjoint(self):
        return qml.T(self.wires)

    def single_qubit_rot_angles(self):
        return [np.pi / 4, 0, 0]



class X90(Operation):
    r"""RX(pi/2)(wires)
    The single-qubit rotation of 90 degrees around the X axis
    
    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None
    
    @staticmethod
    @lru_cache()
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            ndarray: matrix
        """
        return qml.RX.compute_matrix(np.pi/2)

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        Returns:
            array: eigenvalues
        """
        return np.linalg.eigvals(X90.compute_matrix())

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        """
        return [qml.RX(np.pi/2, wires)]

    def pow(self, z):
        z = z % 8
        angle = z * np.pi / 2
        return [qml.RX(angle, self.wires)]
            
    def adjoint(self):
        return XM90(self.wires)

    def single_qubit_rot_angles(self):
        return [np.pi / 2, np.pi/2, -np.pi/2]


class XM90(Operation):
    r"""RX(-pi/2)(wires)
    The single-qubit rotation of -90 degrees around the X axis
    
    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None
    @staticmethod
    @lru_cache()
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            ndarray: matrix
        """
        return qml.RX.compute_matrix(-np.pi/2)

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        Returns:
            array: eigenvalues
        """
        return np.linalg.eigvals(XM90.compute_matrix())

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        """
        return [qml.RX(-np.pi/2, wires)]

    def pow(self, z):
        z = z % 8
        angle = -z * np.pi / 2
        return [qml.RX(angle, self.wires)]
         
    def adjoint(self):
        return X90(self.wires)

    def single_qubit_rot_angles(self):
        return [np.pi / 2, -np.pi/2, -np.pi/2]


class Y90(Operation):
    r"""RY(pi/2)(wires)
    The single-qubit rotation of 90 degrees around the Y axis
    
    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None
    
    @staticmethod
    @lru_cache()
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            ndarray: matrix
        """
        return qml.RY.compute_matrix(np.pi/2)

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        Returns:
            array: eigenvalues
        """
        return np.linalg.eigvals(Y90.compute_matrix())

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        """
        return [qml.RY(np.pi/2, wires)]

    def pow(self, z):
        z = z % 8
        angle = z * np.pi / 2
        return [qml.RY(angle, self.wires)]
            
    def adjoint(self):
        return YM90(self.wires)

    def single_qubit_rot_angles(self):
        return [0, np.pi/2, 0]


class YM90(Operation):
    r"""RY(-pi/2)(wires)
    The single-qubit rotation of -90 degrees around the Y axis
    
    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None
    @staticmethod
    @lru_cache()
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            ndarray: matrix
        """
        return qml.RY.compute_matrix(-np.pi/2)

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        Returns:
            array: eigenvalues
        """
        return np.linalg.eigvals(YM90.compute_matrix())

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        """
        return [qml.RY(-np.pi/2, wires)]

    def pow(self, z):
        z = z % 8
        angle = -z * np.pi / 2
        return [qml.RY(angle, self.wires)]
       
    def adjoint(self):
        return Y90(self.wires)     

    def single_qubit_rot_angles(self):
        return [0, -np.pi/2, 0]


class Z90(Operation):
    r"""RZ(pi/2)(wires)
    The single-qubit rotation of 90 degrees around the Z axis
    
    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None
    
    @staticmethod
    @lru_cache()
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            ndarray: matrix
        """
        return qml.RZ.compute_matrix(np.pi/2)

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        Returns:
            array: eigenvalues
        """
        return np.linalg.eigvals(Z90.compute_matrix())

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        """
        return [qml.RZ(np.pi/2, wires)]

    def pow(self, z):
        z = z % 8
        angle = z * np.pi / 2
        return [qml.RZ(angle, self.wires)]
            
    def adjoint(self):
        return ZM90(self.wires)

    def single_qubit_rot_angles(self):
        return [np.pi/2, 0, 0]


class ZM90(Operation):
    r"""RZ(-pi/2)(wires)
    The single-qubit rotation of -90 degrees around the Z axis
    
    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    batch_size = None
    
    @staticmethod
    @lru_cache()
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        Returns:
            ndarray: matrix
        """
        return qml.RZ.compute_matrix(-np.pi/2)

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        Returns:
            array: eigenvalues
        """
        return np.linalg.eigvals(ZM90.compute_matrix())

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        """
        return [qml.RZ(-np.pi/2, wires)]

    def pow(self, z):
        z = z % 8
        angle = -z * np.pi / 2
        return [qml.RZ(angle, self.wires)]
            
    def adjoint(self):
        return Z90(self.wires)

    def single_qubit_rot_angles(self):
        return [-np.pi/2, 0, 0]

class XYmodel2sites(object):
    
    dtype = np.complex128  # double-precision floating point
    d = 2  # single-site basis size

    Sz1 = np.array([[0.5, 0], [0, -0.5]], dtype)  # single-site S^z
    Sx1 = np.array([[0, 0.5], [0.5, 0]], dtype) 
    Sy1 = np.array([[0, -1j*0.5], [1j*0.5, 0]], dtype) 
    Sp1 = np.array([[0, 1], [0, 0]], dtype)  # single-site S^+
    Id = np.array([[1, 0], [0, 1]], dtype)  # single-site S^+

    sso = {"Id":Id, "Sx": Sx1, "Sy": Sy1, "Sz": Sz1, "Sp": Sp1, "Sm": Sp1.transpose()}  # single-site operators

    def __init__(self, J=1., gamma=0.,hz=0., hx=0., boundary_condition=open_bc):
        """
        `hz` can be either a number (for a constant magnetic field) or a
        callable (which is called with the site index and returns the
        magnetic field on that site).  The same goes for `hx`.
        """


        self.J = J
        self.gamma=gamma
        self.boundary_condition = boundary_condition
        if isinstance(hz, Callable):
            self.hz = hz
        else:
            self.hz = lambda site_index: hz
        if isinstance(hx, Callable):
            self.hx = hx
        else:
            self.hx = lambda site_index: hx

        if hx == 0:
            # S^z sectors corresponding to the single site basis elements
            self.single_site_sectors = np.array([0.5, -0.5])
        else:
            # S^z is not conserved
            self.single_site_sectors = np.array([0, 0])

    def H1(self, site_index):
        half_hz = .5 * self.hz(site_index)
        half_hx = .5 * self.hx(site_index)
        return np.array([[half_hz, half_hx], [half_hx, -half_hz]], self.dtype)

    def H2(self, Sx1, Sy1, Sx2, Sy2):  # two-site part of H
        """Given the operators S^z and S^+ on two sites in different Hilbert spaces
        (e.g. two blocks), returns a Kronecker product representing the
        corresponding two-site term in the Hamiltonian that joins the two sites.
        """
        return (
            (self.J ) * ((1+self.gamma)*kron(Sx1, Sx2)+(1-self.gamma)*kron(Sy1, Sy2)) )


    def initial_block(self, site_index):
        if self.boundary_condition == open_bc:
            # conn refers to the connection operator, that is, the operator on the
            # site that was most recently added to the block.  We need to be able
            # to represent S^z and S^+ on that site in the current basis in order
            # to grow the chain.
            operator_dict = {
                "H": self.H1(site_index),
                "conn_Sx": self.Sx1,
                "conn_Sy": self.Sy1,
            }
        else:
            # Since the PBC block needs to be able to grow in both directions,
            # we must be able to represent the relevant operators on both the
            # left and right sites of the chain.
            operator_dict = {
                "H": self.H1(site_index),
                "l_Sx": self.Sx1,
                "r_Sx": self.Sx1,
                "l_Sy": self.Sy1,
                "r_Sy": self.Sy1,
            }
        return Block(length=1, basis_size=self.d, operator_dict=operator_dict,
                     basis_sector_array=self.single_site_sectors)

    def enlarge_block(self, block, direction, site):
  
        mblock = block.basis_size
        o = block.operator_dict

        s= site.operator_dict

        # Create the new operators for the enlarged block.  Our basis becomes a
        # Kronecker product of the Block basis and the single-site basis.  NOTE:
        # `kron` uses the tensor product convention making blocks of the second
        # array scaled by the first.  As such, we adopt this convention for
        # Kronecker products throughout the code.
        if self.boundary_condition == open_bc:
            enlarged_operator_dict = {
                "H": kron(o["H"], identity(self.d)) +
                     kron(identity(mblock), s["H"]) +
                     self.H2(o["conn_Sx"],o["conn_Sy"], s["conn_Sx"], s["conn_Sy"]),
                "conn_Sx": kron(identity(mblock), s["conn_Sx"]),
                "conn_Sy": kron(identity(mblock), s["conn_Sy"])
            }
        else:
            assert direction in ("l", "r")
            if direction == "l":
                enlarged_operator_dict = {
                    "H": kron(o["H"], identity(self.d)) +
                         kron(identity(mblock), self.H1(bare_site_index)) +
                         self.H2(o["l_Sx"], o["l_Sy"], self.Sx1, self.Sy1),
                    "l_Sx": kron(identity(mblock), self.Sx1),
                    "l_Sy": kron(identity(mblock), self.Sy1),
                    "r_Sx": kron(o["r_Sx"], identity(self.d)),
                    "r_Sy": kron(o["r_Sy"], identity(self.d)),
                }
            else:
                enlarged_operator_dict = {
                    "H": kron(o["H"], identity(self.d)) +
                         kron(identity(mblock), self.H1(bare_site_index)) +
                         self.H2(o["r_Sx"], o["r_Sy"], self.Sx1, self.Sy1),
                    "l_Sx": kron(o["l_Sx"], identity(self.d)),
                    "l_Sy": kron(o["l_Sy"], identity(self.d)),
                    "r_Sx": kron(identity(mblock), self.Sx1),
                    "r_Sy": kron(identity(mblock), self.Sy1),
                }

        # This array keeps track of which sector each element of the new basis is
        # in.  `np.add.outer()` creates a matrix that adds each element of the
        # first vector with each element of the second, which when flattened
        # contains the sector of each basis element in the above Kronecker product.
        enlarged_basis_sector_array = np.add.outer(block.basis_sector_array, self.single_site_sectors).flatten()

        return EnlargedBlock(length=(block.length + 1),
                             basis_size=(block.basis_size * self.d),
                             operator_dict=enlarged_operator_dict,
                             basis_sector_array=enlarged_basis_sector_array)

    def construct_superblock_hamiltonian(self, sys_enl, env_enl):
        sys_enl_op = sys_enl.operator_dict
        env_enl_op = env_enl.operator_dict
        if self.boundary_condition == open_bc:
            # L**R
            H_int = self.H2(sys_enl_op["conn_Sx"], sys_enl_op["conn_Sy"], env_enl_op["conn_Sx"], env_enl_op["conn_Sy"])
        else:
            assert self.boundary_condition == periodic_bc
            # L*R*
            H_int = (self.H2(sys_enl_op["r_Sx"], sys_enl_op["r_Sy"], env_enl_op["l_Sx"], env_enl_op["l_Sy"]) +
                     self.H2(sys_enl_op["l_Sx"], sys_enl_op["l_Sy"], env_enl_op["r_Sx"], env_enl_op["r_Sy"]))
        return (kron(sys_enl_op["H"], identity(env_enl.basis_size)) +
                kron(identity(sys_enl.basis_size), env_enl_op["H"]) +
                H_int)
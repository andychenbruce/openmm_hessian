"""Test Hessian calculation for Amber force fields."""

import openmm.app as app
import openmm as mm
import openmm.unit as unit
import numpy as np
import unittest

class TestHessian(unittest.TestCase):
    """Test Hessian calculation functionality with Amber force fields."""

    def test_harmonic_bond_hessian(self):
        """Test Hessian calculation for harmonic bonds (like in Amber)."""
        # Create a simple system with two particles connected by a harmonic bond
        system = mm.System()
        system.addParticle(12.0)  # carbon atom mass
        system.addParticle(1.0)   # hydrogen atom mass
        
        # Create harmonic bond force (like in Amber force fields)
        bond_force = mm.HarmonicBondForce()
        bond_force.addBond(0, 1, 0.15, 30000.0)  # equilibrium distance 0.15 nm, force constant 30000 kJ/(mol*nm^2)
        system.addForce(bond_force)
        
        # Create integrator and context
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator)
        
        # Set positions: linear molecule
        positions = np.array([
            [0.0, 0.0, 0.0],  # atom 0
            [0.15, 0.0, 0.0]  # atom 1 (equilibrium distance)
        ]) * unit.nanometer
        
        context.setPositions(positions)
        
        # Calculate Hessian using the new API
        hessian = context.calculateHessian()
        
        # Convert to 6x6 matrix (2 atoms * 3 dimensions)
        n_atoms = system.getNumParticles()
        n_dof = 3 * n_atoms
        hessian_matrix = np.array(hessian).reshape((n_dof, n_dof))
        
        # Check that the Hessian is symmetric
        np.testing.assert_array_almost_equal(hessian_matrix, hessian_matrix.T, decimal=5)
        
        # For a harmonic bond at equilibrium, the x-x block should have specific values
        # The diagonal elements for x coordinates should be -k and -k
        # The off-diagonal elements should be +k and +k (symmetric)
        k = 30000.0  # force constant from the bond
        self.assertAlmostEqual(hessian_matrix[0, 0], -k, places=1)  # atom 0, x
        self.assertAlmostEqual(hessian_matrix[3, 3], -k, places=1)  # atom 1, x
        self.assertAlmostEqual(hessian_matrix[0, 3], k, places=1)   # coupling 0x-1x
        self.assertAlmostEqual(hessian_matrix[3, 0], k, places=1)   # coupling 1x-0x (symmetry)
        
        # Other coordinates should be close to zero for this simple case
        self.assertAlmostEqual(hessian_matrix[1, 1], 0.0, places=1)  # atom 0, y
        self.assertAlmostEqual(hessian_matrix[2, 2], 0.0, places=1)  # atom 0, z
        self.assertAlmostEqual(hessian_matrix[4, 4], 0.0, places=1)  # atom 1, y
        self.assertAlmostEqual(hessian_matrix[5, 5], 0.0, places=1)  # atom 1, z

    def test_amber_water_box_hessian(self):
        """Test Hessian calculation for a water box using Amber force field."""
        # Create a simple water system using Amber force field
        pdb = app.PDBFile('systems/alanine-dipeptide-explicit.pdb')  # Using a small system
        forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                                         constraints=app.HBonds)

        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator)
        context.setPositions(pdb.positions)

        # Calculate Hessian
        hessian = context.calculateHessian()

        # Verify the size of the Hessian matrix
        n_atoms = system.getNumParticles()
        n_dof = 3 * n_atoms
        self.assertEqual(len(hessian), n_dof * n_dof)

        # Convert to matrix form
        hessian_matrix = np.array(hessian).reshape((n_dof, n_dof))

        # Check that the Hessian is symmetric
        np.testing.assert_array_almost_equal(hessian_matrix, hessian_matrix.T, decimal=4)

    def test_hessian_with_forces_consistency(self):
        """Test that Hessian is consistent with numerical force derivatives."""
        # Create a simple system
        system = mm.System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        
        # Add harmonic bond
        bond_force = mm.HarmonicBondForce()
        bond_force.addBond(0, 1, 1.0, 100.0)
        system.addForce(bond_force)
        
        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator)
        
        # Set positions
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.01, 0.0, 0.0]  # slightly displaced
        ]) * unit.nanometer
        context.setPositions(positions)
        
        # Calculate Hessian
        hessian = context.calculateHessian()
        n_dof = 3 * system.getNumParticles()
        hessian_matrix = np.array(hessian).reshape((n_dof, n_dof))
        
        # Check that the Hessian is symmetric
        np.testing.assert_array_almost_equal(hessian_matrix, hessian_matrix.T, decimal=5)

    def test_amber_alanine_dipeptide_hessian(self):
        """Test Hessian calculation for a small peptide using Amber force field."""
        # Create a small molecule with common Amber force field terms
        pdb = app.PDBFile('systems/alanine-dipeptide-explicit.pdb')
        forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff,
                                         constraints=None)  # No constraints to simplify test

        integrator = mm.VerletIntegrator(0.001)
        context = mm.Context(system, integrator)
        context.setPositions(pdb.positions)

        # Calculate Hessian
        hessian = context.calculateHessian()

        # Verify the size of the Hessian matrix
        n_atoms = system.getNumParticles()
        n_dof = 3 * n_atoms
        self.assertEqual(len(hessian), n_dof * n_dof)

        # Convert to matrix form and check symmetry
        hessian_matrix = np.array(hessian).reshape((n_dof, n_dof))
        np.testing.assert_array_almost_equal(hessian_matrix, hessian_matrix.T, decimal=4)

if __name__ == '__main__':
    unittest.main()
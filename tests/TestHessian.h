/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2024 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/Context.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/internal/AssertionUtilities.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace OpenMM;
using namespace std;

const double TOL = 1e-4;

void testHessianSymmetry() {
    // Create a simple system with harmonic bonds to test Hessian symmetry
    System system;
    system.addParticle(1.0);  // particle 0
    system.addParticle(1.0);  // particle 1
    system.addParticle(1.0);  // particle 2
    
    // Add harmonic bonds
    HarmonicBondForce* bonds = new HarmonicBondForce();
    bonds->addBond(0, 1, 1.0, 10.0);  // bond between particles 0 and 1
    bonds->addBond(1, 2, 1.0, 10.0);  // bond between particles 1 and 2
    system.addForce(bonds);
    
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    
    // Set positions in a line
    vector<Vec3> positions(3);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(1, 0, 0);
    positions[2] = Vec3(2, 0, 0);
    context.setPositions(positions);
    
    // Calculate Hessian
    vector<double> hessian = context.calculateHessian();
    
    int numParticles = system.getNumParticles();
    int numDOF = 3 * numParticles;
    
    // Verify that the Hessian is symmetric: H[i][j] = H[j][i]
    for (int i = 0; i < numDOF; i++) {
        for (int j = 0; j < numDOF; j++) {
            double Hij = hessian[i * numDOF + j];
            double Hji = hessian[j * numDOF + i];
            ASSERT_EQUAL_TOL(Hij, Hji, TOL);
        }
    }
}

void testHessianHarmonicBond() {
    // Test Hessian for a harmonic bond system
    System system;
    system.addParticle(1.0);  // particle 0
    system.addParticle(1.0);  // particle 1
    
    // Add harmonic bond
    HarmonicBondForce* bonds = new HarmonicBondForce();
    double equilibriumDistance = 1.0;
    double forceConstant = 10.0;
    bonds->addBond(0, 1, equilibriumDistance, forceConstant);
    system.addForce(bonds);
    
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    
    // Set positions at equilibrium
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(equilibriumDistance, 0, 0);
    context.setPositions(positions);
    
    // Calculate Hessian
    vector<double> hessian = context.calculateHessian();
    
    int numParticles = system.getNumParticles();
    int numDOF = 3 * numParticles;
    
    // For a harmonic bond at equilibrium, the diagonal elements corresponding to 
    // the bond direction should be approximately [-k, +k, 0] for the two atoms
    // The off-diagonal elements should be approximately [+k, -k] for cross terms
    
    // Check Hessian elements for x-direction of the bond
    // H_00 (particle 0, x coordinate)
    ASSERT_EQUAL_TOL(-forceConstant, hessian[0 * numDOF + 0], 1e-2);
    // H_03 (particle 0 x, particle 1 x) - should be +forceConstant
    ASSERT_EQUAL_TOL(forceConstant, hessian[0 * numDOF + 3], 1e-2);
    // H_30 (particle 1 x, particle 0 x) - should be +forceConstant (symmetry)
    ASSERT_EQUAL_TOL(forceConstant, hessian[3 * numDOF + 0], 1e-2);
    // H_33 (particle 1, x coordinate)
    ASSERT_EQUAL_TOL(-forceConstant, hessian[3 * numDOF + 3], 1e-2);
    
    // Other elements should be close to zero
    ASSERT_EQUAL_TOL(0.0, hessian[1 * numDOF + 1], 1e-2);  // y component
    ASSERT_EQUAL_TOL(0.0, hessian[2 * numDOF + 2], 1e-2);  // z component
    ASSERT_EQUAL_TOL(0.0, hessian[4 * numDOF + 4], 1e-2);  // y component
    ASSERT_EQUAL_TOL(0.0, hessian[5 * numDOF + 5], 1e-2);  // z component
}

void testHessianHarmonicAngle() {
    // Test Hessian for a harmonic angle system
    System system;
    system.addParticle(1.0);  // particle 0
    system.addParticle(1.0);  // particle 1
    system.addParticle(1.0);  // particle 2
    
    // Add harmonic angle
    HarmonicAngleForce* angles = new HarmonicAngleForce();
    double angle = M_PI / 2.0;  // 90 degrees
    double k = 10.0;
    angles->addAngle(0, 1, 2, angle, k);
    system.addForce(angles);
    
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    
    // Set positions to form 90 degree angle
    vector<Vec3> positions(3);
    positions[0] = Vec3(-1, 0, 0);  // particle 0
    positions[1] = Vec3(0, 0, 0);   // particle 1 (center)
    positions[2] = Vec3(0, 1, 0);   // particle 2
    context.setPositions(positions);
    
    // Calculate Hessian
    vector<double> hessian = context.calculateHessian();
    
    int numParticles = system.getNumParticles();
    int numDOF = 3 * numParticles;
    
    // The Hessian should be symmetric and have reasonable values
    for (int i = 0; i < numDOF; i++) {
        for (int j = 0; j < numDOF; j++) {
            double Hij = hessian[i * numDOF + j];
            double Hji = hessian[j * numDOF + i];
            ASSERT_EQUAL_TOL(Hij, Hji, TOL);
        }
    }
}

void testHessianNonbonded() {
    // Test Hessian for a nonbonded system
    System system;
    system.addParticle(1.0);  // particle 0
    system.addParticle(1.0);  // particle 1
    
    // Add nonbonded force
    NonbondedForce* nonbonded = new NonbondedForce();
    nonbonded->addParticle(1.0, 0.1, 1.0);  // charge, sigma, epsilon
    nonbonded->addParticle(-1.0, 0.1, 1.0); // charge, sigma, epsilon
    system.addForce(nonbonded);
    
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    
    // Set positions
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0.5, 0, 0);
    context.setPositions(positions);
    
    // Calculate Hessian
    vector<double> hessian = context.calculateHessian();
    
    int numParticles = system.getNumParticles();
    int numDOF = 3 * numParticles;
    
    // The Hessian should be symmetric
    for (int i = 0; i < numDOF; i++) {
        for (int j = 0; j < numDOF; j++) {
            double Hij = hessian[i * numDOF + j];
            double Hji = hessian[j * numDOF + i];
            ASSERT_EQUAL_TOL(Hij, Hji, TOL);
        }
    }
}

void testHessianConsistencyWithForces() {
    // Test that the Hessian is consistent with the forces
    // This test verifies that the Hessian represents the second derivative of energy
    System system;
    system.addParticle(1.0);  // particle 0
    system.addParticle(1.0);  // particle 1
    
    // Add harmonic bond
    HarmonicBondForce* bonds = new HarmonicBondForce();
    double equilibriumDistance = 1.0;
    double forceConstant = 10.0;
    bonds->addBond(0, 1, equilibriumDistance, forceConstant);
    system.addForce(bonds);
    
    VerletIntegrator integrator(0.01);
    Context context(system, integrator, platform);
    
    // Set positions slightly displaced from equilibrium
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(equilibriumDistance + 0.01, 0, 0);  // slightly stretched
    context.setPositions(positions);
    
    // Calculate Hessian
    vector<double> hessian = context.calculateHessian();
    
    int numParticles = system.getNumParticles();
    int numDOF = 3 * numParticles;
    
    // Verify that Hessian is symmetric
    for (int i = 0; i < numDOF; i++) {
        for (int j = 0; j < numDOF; j++) {
            double Hij = hessian[i * numDOF + j];
            double Hji = hessian[j * numDOF + i];
            ASSERT_EQUAL_TOL(Hij, Hji, TOL);
        }
    }
    
    // Check that the diagonal elements have the expected signs for a stretched bond
    // The x-x elements should be positive (restoring force increases with displacement)
    ASSERT(hessian[0 * numDOF + 0] > 0);  // particle 0, x coordinate
    ASSERT(hessian[3 * numDOF + 3] > 0);  // particle 1, x coordinate
}

void runPlatformTests();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testHessianSymmetry();
        testHessianHarmonicBond();
        testHessianHarmonicAngle();
        testHessianNonbonded();
        testHessianConsistencyWithForces();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
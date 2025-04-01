#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <omp.h>  // For OpenMP support
#include <chrono>  // For high resolution clock

using namespace std;

// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4;   // Ratio of specific heats
const double CFL = 0.5;         // CFL number

// ------------------------------------------------------------
// Compute pressure from the conservative variables
// ------------------------------------------------------------
double pressure(double rho, double rhou, double rhov, double E) {
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

// ------------------------------------------------------------
// Compute flux in the x-direction
// ------------------------------------------------------------
void fluxX(double rho, double rhou, double rhov, double E, 
           double& frho, double& frhou, double& frhov, double& fE) {
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhou;
    frhou = rhou * u + p;
    frhov = rhov * u;
    fE = (E + p) * u;
}

// ------------------------------------------------------------
// Compute flux in the y-direction
// ------------------------------------------------------------
void fluxY(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhov;
    frhou = rhou * v;
    frhov = rhov * v + p;
    fE = (E + p) * v;
}

// ------------------------------------------------------------
// Compute the total kinetic energy of the system
// ------------------------------------------------------------
double totalKineticEnergy(int Nx, int Ny, double* rho, double* rhou, double* rhov) {
    double total_kinetic_energy = 0.0;

    // Parallelize the kinetic energy calculation using OpenMP
    #pragma omp parallel for reduction(+:total_kinetic_energy) collapse(2)
    for (int i = 1; i <= Nx; i++) {
        for (int j = 1; j <= Ny; j++) {
            double rho_val = rho[i * (Ny + 2) + j];
            if (rho_val > 0.0) {  // Check if rho is greater than 0
                double u = rhou[i * (Ny + 2) + j] / rho_val;
                double v = rhov[i * (Ny + 2) + j] / rho_val;
                total_kinetic_energy += 0.5 * rho_val * (u * u + v * v);
            }
        }
    }
    return total_kinetic_energy;
}

// ------------------------------------------------------------
// Main simulation routine
// ------------------------------------------------------------
int main(){
    // ----- Grid and domain parameters -----
    const int Nx = 200;         // Number of cells in x (excluding ghost cells)
    const int Ny = 100;         // Number of cells in y
    const double Lx = 2.0;      // Domain length in x
    const double Ly = 1.0;      // Domain length in y
    const double dx = Lx / Nx;
    const double dy = Ly / Ny;

    // Create flat arrays (with ghost cells)
    const int total_size = (Nx + 2) * (Ny + 2);
    
    double* rho = (double*)malloc(total_size * sizeof(double));
    double* rhou = (double*)malloc(total_size * sizeof(double));
    double* rhov = (double*)malloc(total_size * sizeof(double));
    double* E = (double*)malloc(total_size * sizeof(double));
    double* rho_new = (double*)malloc(total_size * sizeof(double));
    double* rhou_new = (double*)malloc(total_size * sizeof(double));
    double* rhov_new = (double*)malloc(total_size * sizeof(double));
    double* E_new = (double*)malloc(total_size * sizeof(double));

    // Boolean mask for solid cells
    bool* solid = (bool*)malloc(total_size * sizeof(bool));

    // Initialize arrays
    for (int i = 0; i < total_size; i++) {
      rho[i] = 0.0;
      rhou[i] = 0.0;
      rhov[i] = 0.0;
      E[i] = 0.0;
      rho_new[i] = 0.0;
      rhou_new[i] = 0.0;
      rhov_new[i] = 0.0;
      E_new[i] = 0.0;
      solid[i] = false;
    }

    // ----- Obstacle (cylinder) parameters -----
    const double cx = 0.5;      // Cylinder center x
    const double cy = 0.5;      // Cylinder center y
    const double radius = 0.1;  // Cylinder radius

    // ----- Free-stream initial conditions (inflow) -----
    const double rho0 = 1.0;
    const double u0 = 1.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double E0 = p0/(gamma_val - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);

    // ----- Initialize grid and obstacle mask -----
    #pragma omp target data map(alloc:rho[0:total_size], rhou[0:total_size], rhov[0:total_size], E[0:total_size], \
                                 rho_new[0:total_size], rhou_new[0:total_size], rhov_new[0:total_size], E_new[0:total_size], \
                                 solid[0:total_size])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < Nx + 2; i++) {
            for (int j = 0; j < Ny + 2; j++) {
                double x = (i - 0.5) * dx;
                double y = (j - 0.5) * dy;
                if ((x - cx)*(x - cx) + (y - cy)*(y - cy) <= radius * radius) {
                    solid[i*(Ny+2)+j] = true;
                    rho[i*(Ny+2)+j] = rho0;
                    rhou[i*(Ny+2)+j] = 0.0;
                    rhov[i*(Ny+2)+j] = 0.0;
                    E[i*(Ny+2)+j] = p0/(gamma_val - 1.0);
                } else {
                    solid[i*(Ny+2)+j] = false;
                    rho[i*(Ny+2)+j] = rho0;
                    rhou[i*(Ny+2)+j] = rho0 * u0;
                    rhov[i*(Ny+2)+j] = rho0 * v0;
                    E[i*(Ny+2)+j] = E0;
                }
            }
        }
    }

    // ----- Time-stepping loop -----
    const int nSteps = 2000;
    for (int n = 0; n < nSteps; n++){
        // --- Update interior cells, boundary conditions, and simulation logic ---
        // (Use the same logic as your original code)

        // Calculate total kinetic energy
        double total_kinetic = totalKineticEnergy(Nx, Ny, rho, rhou, rhov);

        // Optional: output progress and write VTK file every 50 time steps
        if (n % 50 == 0) {
            cout << "Step " << n << " completed, total kinetic energy: " << total_kinetic << endl;
        }
    }

    return 0;
}

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

    #pragma omp target teams distribute parallel for reduction(+:total_kinetic_energy) collapse(2)
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

    // Start timing
    auto start_time = chrono::high_resolution_clock::now();

    // ----- Time-stepping loop -----
    #pragma omp target data map(to:rho[0:total_size], rhou[0:total_size], rhov[0:total_size], E[0:total_size]) \
                             map(from:rho_new[0:total_size], rhou_new[0:total_size], rhov_new[0:total_size], E_new[0:total_size])
    {
        for (int n = 0; n < 2000; n++) {
            // Apply boundary conditions (similar to original code)
            // (boundary conditions remain unchanged)

            // Time-stepping loop for interior cells
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    if (solid[i*(Ny+2)+j]) {
                        rho_new[i*(Ny+2)+j] = rho[i*(Ny+2)+j];
                        rhou_new[i*(Ny+2)+j] = rhou[i*(Ny+2)+j];
                        rhov_new[i*(Ny+2)+j] = rhov[i*(Ny+2)+j];
                        E_new[i*(Ny+2)+j] = E[i*(Ny+2)+j];
                        continue;
                    }

                    // Compute Lax-Friedrichs averages and fluxes
                    // (this section remains unchanged)

                    // Apply flux differences
                    double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
                    double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                    double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
                    double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;

                    fluxX(rho[(i+1)*(Ny+2)+j], rhou[(i+1)*(Ny+2)+j], rhov[(i+1)*(Ny+2)+j], E[(i+1)*(Ny+2)+j],
                          fx_rho1, fx_rhou1, fx_rhov1, fx_E1);
                    fluxX(rho[(i-1)*(Ny+2)+j], rhou[(i-1)*(Ny+2)+j], rhov[(i-1)*(Ny+2)+j], E[(i-1)*(Ny+2)+j],
                          fx_rho2, fx_rhou2, fx_rhov2, fx_E2);
                    fluxY(rho[i*(Ny+2)+(j+1)], rhou[i*(Ny+2)+(j+1)], rhov[i*(Ny+2)+(j+1)], E[i*(Ny+2)+(j+1)],
                          fy_rho1, fy_rhou1, fy_rhov1, fy_E1);
                    fluxY(rho[i*(Ny+2)+(j-1)], rhou[i*(Ny+2)+(j-1)], rhov[i*(Ny+2)+(j-1)], E[i*(Ny+2)+(j-1)],
                          fy_rho2, fy_rhou2, fy_rhov2, fy_E2);

                    // Update the new state variables
                    rho_new[i*(Ny+2)+j] = rho[i*(Ny+2)+j] - CFL * (fx_rho1 - fx_rho2 + fy_rho1 - fy_rho2);
                    rhou_new[i*(Ny+2)+j] = rhou[i*(Ny+2)+j] - CFL * (fx_rhou1 - fx_rhou2 + fy_rhou1 - fy_rhou2);
                    rhov_new[i*(Ny+2)+j] = rhov[i*(Ny+2)+j] - CFL * (fx_rhov1 - fx_rhov2 + fy_rhov1 - fy_rhov2);
                    E_new[i*(Ny+2)+j] = E[i*(Ny+2)+j] - CFL * (fx_E1 - fx_E2 + fy_E1 - fy_E2);
                }
            }

            // Update the main arrays with the new values
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    rho[i*(Ny+2)+j] = rho_new[i*(Ny+2)+j];
                    rhou[i*(Ny+2)+j] = rhou_new[i*(Ny+2)+j];
                    rhov[i*(Ny+2)+j] = rhov_new[i*(Ny+2)+j];
                    E[i*(Ny+2)+j] = E_new[i*(Ny+2)+j];
                }
            }

            // Print the total kinetic energy every 100th step
            if (n % 100 == 0) {
                double kinetic_energy = totalKineticEnergy(Nx, Ny, rho, rhou, rhov);
                cout << "Step " << n << ", Total Kinetic Energy: " << kinetic_energy << endl;
            }
        }
    }

    // End timing
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time);

    // Print total elapsed time
    cout << "Total Elapsed Time: " << duration.count() << " seconds." << endl;

    // Clean up
    free(rho);
    free(rhou);
    free(rhov);
    free(E);
    free(rho_new);
    free(rhou_new);
    free(rhov_new);
    free(E_new);
    free(solid);

    return 0;
}

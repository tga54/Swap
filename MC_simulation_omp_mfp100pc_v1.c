#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h> // Include the OpenMP library
#define PI 3.14159265358979323846
#define NR 601
#define NZ 601
#define n_particles 100000
#define n_steps 1000000
#define dt 1000
const double sigma_C = 255e-27, c_pcyr = 0.307, ratio = 40.46285, grid_per_order = 100.0, r_disk = 10000.0;
double R[NR], Z[NZ];

void initialize_to_0(double *arr, int arr_size) {
    for (int i = 0; i < arr_size; i++){
        arr[i] = 0.0;
    }
}

void initialize_grids(void){
    for (int i = 0; i < NR; i++) {
        R[i] = pow(10.0, (double)i / grid_per_order);
    }
    for (int i = 0; i < NZ; i++) {
        Z[i] = pow(10.0, (double)i / grid_per_order);
    }
}

void initialize_pos(double *x, double *y, double *z, double r_disk){
    srand(time(NULL));
    for (int i = 0; i < n_particles; i++){
        double rand_R_squared = r_disk * r_disk * (double)rand() / (double)RAND_MAX; // random radius of rdr ~ dr^2 from 0 to r_disk^2
        double rand_theta = 2.0 * PI *(double)rand() / (double)RAND_MAX; // random angle d\theta from 0 to 2*pi
        x[i] = sqrt(rand_R_squared) * cos(rand_theta);
        y[i] = sqrt(rand_R_squared) * sin(rand_theta);
        z[i] = 0.0;
    }
}

void initialize_H(double *ndis_H2, double *ndis_HI, double *ndis_HII) {  
// nH = 0.0: no decay; nH > 0: flat hydrogen distribution; nH = -1.0:we take the distribution of H_2 and HI from doi:10.1093/mnras/staa1017, and HII from doi: 10.1111/j.1365-2966.2004.08349.x 
        for (int i = 0; i < NR; i++) {
            for (int j = 0; j < NZ; j++) {
                double r0 = sqrt(R[i]*R[i] + Z[j]*Z[j]);
                ndis_H2[i*NZ+j] = ratio * 2200. / 4. / 45. * exp(-12000./R[i] - R[i] / 1500.) / pow(cosh(Z[j] / 2. / 45.), 2);
                ndis_HI[i*NZ+j] = ratio * 53. / 4. / 85. * exp(-4000./R[i] - R[i] / 7000.) / pow(cosh(Z[j] / 2. / 85.), 2);
                ndis_HII[i*NZ+j] = 0.00015 * (1. + 3.7*log(1 + r0 / 20000.) / (r0/20000.) - 1.0277);
                // ndis_H[i*NZ+j] = ndis_H2[i*NZ+j];
            }
        }
}

double interp_2D(double *ndis_H, double r_temp, double z_temp){

    double r1, r2, z1, z2, n11, n12, n21, n22, area, log_r_temp, log_z_temp; 
    log_r_temp = log10(r_temp);
    log_z_temp = log10(z_temp);
    int ind_r = (int)(log_r_temp * grid_per_order);
    int ind_z = (int)(log_z_temp * grid_per_order);

    if (ind_r < 0 || ind_r >= NR - 1 || ind_z < 0 || ind_z >= NZ - 1) {
        return 0.0;
    }
    else{
        r1 = log10(R[ind_r]);
        r2 = log10(R[ind_r + 1]);
        z1 = log10(Z[ind_z]);
        z2 = log10(Z[ind_z + 1]);
        n11 = ndis_H[ind_r * NZ + ind_z];
        n12 = ndis_H[ind_r * NZ + ind_z + 1];
        n21 = ndis_H[(ind_r + 1) * NZ + ind_z];
        n22 = ndis_H[(ind_r + 1) * NZ + ind_z + 1];
        area  = (r2 - r1) * (z2 - z1);
        double f = 1.0 / area * ( n11 * (r2 - log_r_temp) * (z2 - log_z_temp) + n21 * (log_r_temp - r1) * (z2 - log_z_temp) + n12 * (r2 - log_r_temp) * (log_z_temp - z1) + n22 * (log_r_temp - r1) * (log_z_temp - z1));
        if (f < 0.0){ return 0.0; }
        else{ return f; }
    }
}

int main(void) {
    omp_set_num_threads(10);
    
    int record_size = 100000000; // int record_size = n_particles * n_steps / 10000;

    // Dynamically allocate memory for ndis and temp
    double l_scatter = c_pcyr * dt / 3.0;  // 100 pc
    printf("Scattering length is %f pc.", l_scatter);
    double *grammage_H2 = malloc(record_size * sizeof(double));
    double *grammage_HI = malloc(record_size * sizeof(double));
    double *grammage_HII = malloc(record_size * sizeof(double));

    double *time_record = malloc(record_size * sizeof(double));
    double *x = malloc(n_particles * sizeof(double));
    double *y = malloc(n_particles * sizeof(double));
    double *z = malloc(n_particles * sizeof(double));
    double *gram_temp_H2 = malloc(n_particles * sizeof(double));
    double *gram_temp_HI = malloc(n_particles * sizeof(double));
    double *gram_temp_HII = malloc(n_particles * sizeof(double));

    double *ndis_H2 = malloc(NR * NZ * sizeof(double));
    double *ndis_HI = malloc(NR * NZ * sizeof(double));
    double *ndis_HII = malloc(NR * NZ * sizeof(double));

    int ind1 = 0, ind2 = 0;

    srand(time(NULL));

    initialize_grids();
    initialize_pos(x, y, z, r_disk);
    initialize_to_0(gram_temp_H2, n_particles);
    initialize_to_0(gram_temp_HI, n_particles);
    initialize_to_0(gram_temp_HII, n_particles);

    initialize_to_0(grammage_H2, record_size);
    initialize_to_0(grammage_HI, record_size);
    initialize_to_0(grammage_HII, record_size);

    initialize_to_0(time_record, record_size);
    initialize_H(ndis_H2, ndis_HI, ndis_HII);

    unsigned int seeds[omp_get_max_threads()];
    for (int i = 0; i < omp_get_max_threads(); i++)
        seeds[i] = time(NULL) + i * 1337;  // or use better entropy

    for (int t = 0; t < n_steps; t++){
        #pragma omp parallel for
        for (int i = 0; i < n_particles; i++) {
            int tid = omp_get_thread_num();
            double cos_theta_v = 2.0 * (double)rand_r(&seeds[tid]) / RAND_MAX - 1.0;
            double sin_theta_v = sqrt(1.0 - cos_theta_v * cos_theta_v);
            double phi_v = 2.0 * PI * (double)rand_r(&seeds[tid]) / RAND_MAX;
            x[i] += l_scatter * sin_theta_v * cos(phi_v);
            y[i] += l_scatter * sin_theta_v * sin(phi_v);
            z[i] += l_scatter * cos_theta_v;
            double r_temp = sqrt(x[i]*x[i] + y[i]*y[i]);
            double z_temp = fabs(z[i]);

            double nH2 = interp_2D(ndis_H2, r_temp, z_temp);
            double nHI = interp_2D(ndis_HI, r_temp, z_temp);
            double nHII = interp_2D(ndis_HII, r_temp, z_temp);

            gram_temp_H2[i] += l_scatter * nH2; 
            gram_temp_HI[i] += l_scatter * nHI; 
            gram_temp_HII[i] += l_scatter * nHII; 

            // double dist2 = pow(r_temp - 8000.0, 2.0) + pow(z_temp, 2.0); 
            double dist = sqrt( pow(fabs(r_temp - 8000.0), 2.0) + pow(z_temp, 2.0));
            if (dist < l_scatter) {
                // printf("Found %d particle !!", ind_flag);
                int idx2;
                #pragma omp atomic capture
                idx2 = ind2++;
                if (idx2 < record_size) {
                    grammage_H2[idx2] = gram_temp_H2[i];
                    grammage_HI[idx2] = gram_temp_HI[i];
                    grammage_HII[idx2] = gram_temp_HII[i];
                    time_record[idx2] = (double)t;
                    // r_record[idx2] = dist;
                }
                else{
                    perror("Error ! Grammage array is too small !");
                }
            }
        }
        if (t * 10 % n_steps == 0){
            printf("Time step is: %d \n", t);
        }
    }

    char filename1[60], filename2[60], filename3[60], filename4[60];
    snprintf(filename1, sizeof(filename1), "MC_gram_H2_record_t1e9_disk10kpc_mfp100pc_v5.bin");
    snprintf(filename2, sizeof(filename2), "MC_gram_HI_record_t1e9_disk10kpc_mfp100pc_v5.bin");
    snprintf(filename3, sizeof(filename3), "MC_gram_HII_record_t1e9_disk10kpc_mfp100pc_v5.bin");
    snprintf(filename4, sizeof(filename4), "MC_time_record_t1e9_disk10kpc_mfp100pc_v5.bin");

    FILE *file1 = fopen(filename1, "wb");
    fwrite(grammage_H2, sizeof(double), ind2, file1);
    fclose(file1);

    FILE *file2 = fopen(filename2, "wb");
    fwrite(grammage_HI, sizeof(double), ind2, file2);
    fclose(file2);

    FILE *file3 = fopen(filename3, "wb");
    fwrite(grammage_HII, sizeof(double), ind2, file3);
    fclose(file3);

    FILE *file4 = fopen(filename4, "wb");
    fwrite(time_record, sizeof(double), ind2, file4);
    fclose(file4);

    free(grammage_H2);
    free(grammage_HI);
    free(grammage_HII);
    free(time_record);
    // free(r_record);
    free(x);
    free(y);
    free(z);
    free(gram_temp_H2);
    free(gram_temp_HI);
    free(gram_temp_HII);
    free(ndis_H2);
    free(ndis_HI);
    free(ndis_HII);
    return 0;
}

//Version history 
//#pragma comment(linker, "/STACK:40000000")
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h> // Include for boolean data type
#include "SFMT.h"
#include <time.h> // for running time checks
#include <stdlib.h>  // Include the stdlib.h header for random number generation
#include <omp.h>

#define NUM_BATCHES 64 //divide the simulations to be completed in batches
#define N_ENSEMBLE 100 //total number of trajectories per batch
#define N_STEPS 2500 // the max amount of steps of each trajectory

// Build with: gcc -O3 -msse2 -DHAVE_SSE2 -DSFMT_MEXP=607 -o mytest1 SFMT.c mytest1.c -lm
// Run with ./mytest1
// On windowes the error -1073741571 is stackoverflow. To fix this issue increase the buffer size with the build command
// gcc -O3 -msse2 -DHAVE_SSE2 -DSFMT_MEXP=607 -o "%e" -Xlinker --stack=10000000 SFMT.c "%f" 

//new build command with OpenMP for parallel computing
//gcc -fopenmp -O3 -msse2 -DHAVE_SSE2 -DSFMT_MEXP=607 -o "%e" -Xlinker --stack=10000000000 SFMT.c "%f"
//run
//"./%e" -s

//Build command on Linux
//gcc -fopenmp -O3 -msse2 -DHAVE_SSE2 -DSFMT_MEXP=607 -o QTM_obs_v2 SFMT.c QTM_obs_v2.c -lm
//run with ./QTM_obs_v2 -s

// Custom choice function
void generate_randmat(sfmt_t* sfmt, double randmat[N_ENSEMBLE][N_STEPS]) {
	// sfmt is already initialized and we only draw the random numbers
	// for example : double rand_value = sfmt_genrand_real1(rng); rng here is sfmt
	
	const int NUM = N_ENSEMBLE * N_STEPS;
    const int R_SIZE = 2 * NUM;
    int size;
    
    size = sfmt_get_min_array_size64(sfmt);
    if (size < R_SIZE) {
		size = R_SIZE;
    }
	
    uint64_t *array;

	#if defined(__APPLE__) || \
		(defined(__FreeBSD__) && __FreeBSD__ >= 3 && __FreeBSD__ <= 6)
		//printf("malloc used\n");
		array = malloc(sizeof(double) * size);
		if (array == NULL) {
		printf("can't allocate memory.\n");
		}
	#elif defined(_POSIX_C_SOURCE)
		//printf("posix_memalign used\n");
		if (posix_memalign((void **)&array, 16, sizeof(double) * size) != 0) {
		printf("can't allocate memory.\n");
		}
	//#elif defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
		//printf("memalign used\n");
		//array = memalign(16, sizeof(double) * size);
		//if (array == NULL) {
		//printf("can't allocate memory.\n");
		//return 1;
		//}
	#else /* in this case, gcc doesn't support SSE2 */
		//printf("malloc used\n");
		array = malloc(sizeof(double) * size);
		if (array == NULL) {
		printf("can't allocate memory.\n");
		}
	#endif
	
    sfmt_fill_array64(sfmt, array, size);
    
    // end of the rand part
    int rnd_ind = 0;    

    for (int i = 0; i < N_ENSEMBLE; i++) {
        for (int j = 0; j < N_STEPS; j++) {
            randmat[i][j] = sfmt_to_res53(array[rnd_ind++]);  // Generate a random number between 0 and 1
        }
    }
    free(array);
}

struct step{
	int x;
	int y;
};

struct step make_step(double rand_value, double F){	
	struct step step0;
	double norm = 1 / (2 + exp(F/2) + exp(-F/2));
	double p_right = norm * exp(F/2);
	double p_left = norm * exp(-F/2);
	double p_up = norm;
	double p_down = norm;
	if (rand_value < p_right) {
	  // Right
		step0.x = 1;
		step0.y = 0;
	} else if (rand_value < p_right + p_left) {
	  // left
		step0.x = -1;
		step0.y = 0;
	} else if (rand_value < p_right + p_left + p_up) {
	  // up
		step0.x = 0;
		step0.y = 1;
	} else {
	  // down
		step0.x = 0;
		step0.y = -1;
	}
	
	//printf("%d and %d \n", step0.x, step0.y);
	//printf("%f \n", rand_value);
	
	return step0;
}

void get_x(int x0, int y0, int N, double F, double* randvec, double* x_final, double* x2_final){
	//int N = N_STEPS;
	double *x_obs = (double *)malloc((N+1) * sizeof(double));
	double *y_obs = (double *)malloc((N+1) * sizeof(double));
    
    struct step current_step;
    x_obs[0] = (double)(x0);
    y_obs[0] = (double)(y0);
	
	//int obs_realizations=0;
	double xf = 0;
	double xf2 = 0;
	
	double x_bare = 0;
	
	double p_in = 0;
	
	// evaluate where the particle will be with no obstacles
	for (int j = 0; j<N; j++) {
		current_step = make_step(randvec[j], F);
		x_bare = x_bare + current_step.x;
	}
	
	// Running on all the possible positions of only a single obstacle in the effective area.
	for (int sx = x0 - 5; sx <= x0 + N; sx++) {
		for (int sy = x0 - 5; sy <= x0 + 5; sy++) {
			
			for (int ii = 0; ii<N+1; ii++) {
				x_obs[ii] = 0;
				y_obs[ii] = 0;
			}
			
			
			//this is new
			if (abs(sx) + abs(sy) <= N) {
				
				if (sx == 0 && sy == 0) {
					x_obs[N]=0;
					y_obs[N]=0;
				}
				else{
					for (int i = 0; i<N; i++) {
					
						current_step = make_step(randvec[i], F);						
						
						x_obs[i+1] = x_obs[i] + current_step.x;
						y_obs[i+1] = y_obs[i] + current_step.y;
						
						//printf("step:\n");
						//printf("%d %d\n", current_step.x, current_step.y);
						
						if (x_obs[i+1] == sx && y_obs[i+1]==sy){
							x_obs[i+1] = x_obs[i];
							y_obs[i+1] = y_obs[i];
							//printf("%f \n", S_max);
							//printf("%d %d\n", sx, sy);
						}
						//printf("finale pos:\n");
						//printf("%d %d\n", x_obs[1], y_obs[1]);
						
					}
				}
				
				p_in = p_in +1;
				xf = xf + (x_obs[N] - x0);
				xf2 = xf2 + pow(x_obs[N] - x0, 2);		
				//printf("%d \n", xf);
			}
			
			//printf("%d \n", xf);
		}
	}
	
	xf = xf + (1-p_in) * tanh(F/4) * N;
	xf2 = xf2 + (1-p_in) * (pow(tanh(F/4) * N, 2) + (1-pow(tanh(F/4), 2)) * N /2);
	
	xf = xf - x_bare;
	xf2 = xf2 - pow(x_bare, 2);
	//printf("%d \n", cnt);
	
    *x_final = xf;
	*x2_final = xf2;
    free(x_obs);
	free(y_obs);
	return;
}

void writeIntArrayToBinaryFile(int *array, int numElements, const char *fileName) {
    // Open the binary file in write mode ("wb")
    FILE *file = fopen(fileName, "wb");

    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the integer array to the binary file
    fwrite(array, sizeof(int), numElements, file);

    // Close the file
    fclose(file);
}

void writeDoubleArrayToBinaryFile(double *array, int numElements, const char *fileName) {
    // Open the binary file in write mode ("wb")
    FILE *file = fopen(fileName, "wb");

    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the integer array to the binary file
    fwrite(array, sizeof(double), numElements, file);

    // Close the file
    fclose(file);
}

int generate_random_seed() {
    srand(time(NULL));  // Seed the random number generator with the current time
    return rand();
}

void run_sim(int iteration_thread, int seed, double F){
	sfmt_t sfmt;
	sfmt_init_gen_rand(&sfmt, seed);
	// N_steps length of each trajectory	
    
    //// The rand part
    
    //////////////////////////
    //redefine steps_x and steps_y with malloc.
    //memory defined on heap, should be faster.
	// Dynamically allocate memory for steps_x and steps_y
	double (*randmat)[N_STEPS] = malloc(N_ENSEMBLE * sizeof(double[N_STEPS]));

    if (randmat == NULL) {
        perror("Memory allocation failed");
    }
    ///////////////////////
    //populate steps with random values
    generate_randmat(&sfmt, randmat);
	
	//////////////////////////////////////////
	//The trajectory for each tracer
	int traj_id;
	
	double *x_samples = (double *)malloc(N_ENSEMBLE * sizeof(double));
	double *x2_samples = (double *)malloc(N_ENSEMBLE * sizeof(double));
	
    double x_final, x2_final;    
    
    /////////////////////////////////////////////
	int i =  iteration_thread;
	//printf("Batch number %d \n", i);
	
	int N_array[] = {1, 2, 3, 4, 5, 6, 9, 10, 15, 16, 24, 25, 39, 40, 62, 63, 99, 100, 157, 158, 250, 251, 397, 398, 630, 631, 999, 1000, 1584, 1585, 2499, 2500};
	int N_array_size = 32;
	
	double x_avg[N_array_size];
	double x2_avg[N_array_size];
	
	
	for (int N_ind = 0; N_ind < N_array_size; N_ind++){
		
		x_avg[N_ind] = 0;
		x2_avg[N_ind] = 0;
		
		int N = N_array[N_ind];
		/////////////////////////////////
		for (traj_id =0; traj_id<N_ENSEMBLE; traj_id++){
			
			//Initialize the traj result to an unlikely value (we have a drift to the right)
			x_samples[traj_id] = -999.0;
			x2_samples[traj_id] = -999.0;
			
			//printf("%f \n", S_max);
			
			//set starting position
			int x0 = 0;
			int y0 = 0;
			
			get_x(x0, y0, N, F, randmat[traj_id], &x_final, &x2_final);
			//Note that the & operator is removed in this case because randmat[traj_id] already represents a pointer to an array of integers.
			
			x_samples[traj_id] = x_final;
			x2_samples[traj_id] = x2_final;
			//printf("final pos is (%d, %d) \n", x_samples[traj_id], x2_samples[traj_id]);
			
			x_avg[N_ind] = x_avg[N_ind] + x_samples[traj_id];
			x2_avg[N_ind] = x2_avg[N_ind] + x2_samples[traj_id];
		}
		////////////////////////////////////////////////////////////
		x_avg[N_ind] = x_avg[N_ind] / (double)(N_ENSEMBLE);
		x2_avg[N_ind] = x2_avg[N_ind] / (double)(N_ENSEMBLE);
		//test area
		printf("Velocity responce in n: %.8f\n", x_avg[N_ind] / N );
		printf("N_STEPS = %d\n", N);
		//end of test area
	}		
	//save all to files
	//we are saving x and x^2 already averaged over ensemble and obstacle realizations
	char filename_x[100], filename_x2[100];
	int numElements = N_array_size;
	
	// Call the function to write the array to a binary file
	sprintf(filename_x, "data_x_run_%d_F_%.3lf_N_%d.bin", i, F, N_STEPS);
	sprintf(filename_x2, "data_x2_run_%d_F_%.3lf_N_%d.bin", i, F, N_STEPS);
	
	writeDoubleArrayToBinaryFile(x_avg, numElements, filename_x);
	writeDoubleArrayToBinaryFile(x2_avg, numElements, filename_x2);
	//printf("finished batch \n");
    ////////////////////////////////////////////////////////////////////////////

    
	free(randmat);
	free(x_samples);
	free(x2_samples);
}

int main(int argc, char* argv[]) {
    // setting the constants
    // double F_array[] = {0.5, 1.5, 3.0};
    // int F_array_size = 3;
    
	double F_array[] = {6.0, 7.0, 8.0, 10.0};
	int F_array_size = 4;
	double F;
	
	int Iterations = NUM_BATCHES;
	
	int starting_seed = generate_random_seed();  // Generate a random seed
	int seed_array[F_array_size * Iterations];
	
	sfmt_t sfmt;
	sfmt_init_gen_rand(&sfmt, starting_seed);
	
	for (int j = 0; j < F_array_size * Iterations; j++) {
		seed_array[j] = sfmt_genrand_uint32(&sfmt);
	}
	
	for (int ind_F = 0; ind_F < F_array_size; ind_F++){
		F = F_array[ind_F];
		#pragma omp parallel for
		for (int i = 0; i < Iterations; i++) {
			//printf("seed is %d \n", seed_array[F_array_size * i + ind_F]);
			run_sim(i, seed_array[F_array_size * i + ind_F], F);
		}
		#pragma omp barrier // Add a barrier to synchronize threads

	}
    return 0;   
}


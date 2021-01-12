#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define sz 256

int process_offset = 0;
int num_of_supporting_processes = 0;
int dims[2];
const int count = 4;
MPI_Status status[count];
MPI_Request req[count];
int sz_block;
int *a_all, *b_all, *c_all;
int *recovered_processes;
int *destination;

int correct = 1;

typedef enum {
    ERROR_IN_BLOCKS_GENERATION = 1,
    ERROR_IN_FORWARDING = 2,
    ERROR_IN_SENDING_RESULTS = 3,
} ERRORS;

int IsSupportProcess(int rank) {
    return rank >= dims[0] * dims[1];
}

int GetSupportProcess() {
    return num_of_supporting_processes + dims[0] * dims[1] - process_offset;
}

FILE *GetFile(char *matrix_name, int num_proc, char *mode) {
    char str[10];
    sprintf(str, "%d", num_proc);
    char *name = strcat (matrix_name, strcat(str, ".txt"));
    FILE *f = fopen(name, mode);
    return f;
}

void CreateControlPoint(char *matrix_name, int num_proc, int *buf) {
    FILE *f = GetFile(matrix_name, num_proc, "w");

    for (int i = 0; i < sz_block; ++i) {
        for (int j = 0; j < sz_block; ++j) {
            fprintf(f, "%d ", buf[i * sz_block + j]);
        }
    }
    fprintf(f, "\n");
    fclose(f);
}

void ReadFromControlPoint(int *buf, char *matrix_name, int num_proc) {
    FILE *f = GetFile(matrix_name, num_proc, "r");
    char *str_buf = (char *) malloc(sz_block * sz_block * sizeof(char));
    while (fgets(str_buf, sz_block * sz_block, f));
    for (int i = 0; i < sz_block; ++i) {
        for (int j = 0; j < sz_block; ++j) {
            sscanf(str_buf, "%d", &buf[i * sz_block + j]);
        }
    }
}

void BlockGeneration(int rank, int *a, int *b, int *c) {
    int i_coord = rank / dims[0];
    int j_coord = rank % dims[0];
    for (int i = 0; i < sz_block; i++) {
        for (int j = 0; j < sz_block; j++) {
            a[i * sz_block + j] = rand() % 10000;
            a_all[(i + i_coord * sz_block) * sz + (j_coord * sz_block + j)] = a[i * sz_block + j];
            b[i * sz_block + j] = rand() % 10000;
            b_all[(i + i_coord * sz_block) * sz + (j_coord * sz_block + j)] = b[i * sz_block + j];
            c[i * sz_block + j] = 0;
            c_all[(i + i_coord * sz_block) * sz + (j_coord * sz_block + j)] = 0;
        }
    }
}

void GetGeneratedMatrix(int *a_tmp, int proc) {
    int num_incorrect_process;
    int has_error = MPI_Recv(a_tmp, sz_block * sz_block, MPI_INT, proc, 2020, MPI_COMM_WORLD, &status[0]);
    if (has_error) {
        num_incorrect_process = proc;
        destination = 0;
        MPI_Send(&num_incorrect_process, 1, MPI_INT, GetSupportProcess(),
                 ERROR_IN_BLOCKS_GENERATION, MPI_COMM_WORLD);
        return;
    }

    int curr_proc = proc;
    if (IsSupportProcess(proc)) {
        curr_proc = recovered_processes[proc];
    }
    int i_coord_tmp = curr_proc / dims[0];
    int j_coord_tmp = curr_proc % dims[0];
    for (int i = 0; i < sz_block; ++i) {
        for (int j = 0; j < sz_block; ++j) {
            a_all[(i + i_coord_tmp * sz_block) * sz + (j_coord_tmp * sz_block + j)] = a_tmp[i * sz_block + j];
        }
    }
}

void Forwarding(int *a, int *b, int *recv_tmp_a, int *recv_tmp_b, int rank) {
    int incorrect_process;
    int i_coord = rank / dims[0];
    int j_coord = rank % dims[0];
    CreateControlPoint("a", rank, a);
    MPI_Isend(a, sz_block * sz_block, MPI_INT, i_coord * dims[0] + (j_coord - i_coord + dims[0]) % dims[0], 2019,
              MPI_COMM_WORLD, &req[0]);
    CreateControlPoint("b", rank, b);

    MPI_Isend(b, sz_block * sz_block, MPI_INT, ((i_coord - j_coord + dims[0]) % dims[0]) * dims[0] + j_coord, 2019,
              MPI_COMM_WORLD, &req[1]);
    int source_a = i_coord * dims[0] + (j_coord + i_coord) % dims[0];
    int has_error_a = MPI_Irecv(recv_tmp_a, sz_block * sz_block, MPI_INT, source_a, 2019,
                                MPI_COMM_WORLD, &req[2]);
    if (has_error_a) {
        incorrect_process = source_a;
        *destination = rank;
        MPI_Send(&incorrect_process, 1, MPI_INT, GetSupportProcess(), ERROR_IN_FORWARDING, MPI_COMM_WORLD);
    }
    int source_b = ((i_coord + j_coord + dims[0]) % dims[0]) * dims[0] + j_coord;
    int has_error_b = MPI_Irecv(recv_tmp_b, sz_block * sz_block, MPI_INT, source_b, 2019,
                                MPI_COMM_WORLD, &req[3]);
    if (has_error_b) {
        incorrect_process = source_b;
        *destination = rank;
        MPI_Send(&incorrect_process, 1, MPI_INT, GetSupportProcess(), ERROR_IN_FORWARDING, MPI_COMM_WORLD);
    }
}

void GetComputedResults() {
    int incorrect_process;
    int *c_tmp = (int *) malloc(sz_block * sz_block * sizeof(int));
    for (int proc = 1; proc < dims[0] * dims[1] + num_of_supporting_processes; ++proc) {
        int has_error = MPI_Recv(c_tmp, sz_block * sz_block, MPI_INT, proc, 2021, MPI_COMM_WORLD, &status[0]);
        if (has_error) {
            incorrect_process = proc;
            MPI_Send(&incorrect_process, 1, MPI_INT, GetSupportProcess(), ERROR_IN_FORWARDING, MPI_COMM_WORLD);
        }
        int i_coord_tmp = proc / dims[0];
        int j_coord_tmp = proc % dims[0];
        for (int i = 0; i < sz_block; ++i) {
            for (int j = 0; j < sz_block; ++j) {
                c_all[(i + i_coord_tmp * sz_block) * sz + (j_coord_tmp * sz_block + j)] = c_tmp[i * sz_block + j];
            }
        }
    }
    free(c_tmp);
}

int TryToRecover(int incorrect_process, int curr_process, ERRORS err, int destination) {
    printf("Error type: %s\n", (const char *) err);
    printf("Error in process %d\n", incorrect_process);
    if (num_of_supporting_processes - process_offset < 0) {
        printf("Can't restart equations, don't have enough supporting processes\n");
        return 0;
    }
    recovered_processes[curr_process] = incorrect_process;
    switch (err) {
        case ERROR_IN_BLOCKS_GENERATION: {
            int *a = (int *) malloc(sz_block * sz_block * sizeof(int));
            int *b = (int *) malloc(sz_block * sz_block * sizeof(int));
            int *c = (int *) calloc(sz_block * sz_block, sizeof(int));
            BlockGeneration(incorrect_process, a, b, c);
            MPI_Send(a, sz_block * sz_block, MPI_INT, destination, 2020,
                     MPI_COMM_WORLD);
            process_offset += 1;
            break;
        }
        case ERROR_IN_FORWARDING: {
            int *a = (int *) malloc(sz_block * sz_block * sizeof(int));
            ReadFromControlPoint(a, "a", incorrect_process);
            MPI_Send(a, sz_block * sz_block, MPI_INT, destination, 2019,
                     MPI_COMM_WORLD);

            int *b = (int *) malloc(sz_block * sz_block * sizeof(int));
            ReadFromControlPoint(b, "b", incorrect_process);
            recovered_processes[curr_process] = incorrect_process;
            MPI_Send(b, sz_block * sz_block, MPI_INT, destination, 2019,
                     MPI_COMM_WORLD);

            int *c = (int *) malloc(sz_block * sz_block * sizeof(int));
            ReadFromControlPoint(c, "c", incorrect_process);
            recovered_processes[curr_process] = incorrect_process;
            MPI_Send(c, sz_block * sz_block, MPI_INT, destination, 2019,
                     MPI_COMM_WORLD);
            process_offset += 1;
            break;
        }
        case ERROR_IN_SENDING_RESULTS: {
            int *c = (int *) malloc(sz_block * sz_block * sizeof(int));
            ReadFromControlPoint(c, "c", incorrect_process);
            MPI_Send(c, sz_block * sz_block, MPI_INT, 0, 2021, MPI_COMM_WORLD);
            process_offset += 1;
            break;
        }
    }
    return 1;
}

int ErrorHandler(int *destination) {
    int program_is_correct = 1;
    for (int proc = dims[0] * dims[1]; proc < dims[0] * dims[1] + num_of_supporting_processes; ++proc) {
        int incorrect_proc;
        int rc1 = MPI_Irecv(&incorrect_proc, 1, MPI_INT, 0, ERROR_IN_BLOCKS_GENERATION,
                            MPI_COMM_WORLD, &req[0]);
        if (!rc1) {
            int result = TryToRecover(incorrect_proc, proc, ERROR_IN_BLOCKS_GENERATION, *destination);
            program_is_correct &= result;
        }

        int rc2 = MPI_Irecv(&incorrect_proc, 1, MPI_INT, 0, ERROR_IN_FORWARDING,
                            MPI_COMM_WORLD, &req[0]);
        if (!rc2) {
            int result = TryToRecover(incorrect_proc, proc, ERROR_IN_FORWARDING, *destination);
            program_is_correct &= result;
        }

        int rc3 = MPI_Irecv(&incorrect_proc, 1, MPI_INT, 0, ERROR_IN_SENDING_RESULTS,
                            MPI_COMM_WORLD, &req[0]);
        if (!rc3) {
            int result = TryToRecover(incorrect_proc, proc, ERROR_IN_SENDING_RESULTS, 0);
            program_is_correct &= result;
        }
    }
    return program_is_correct;
}

void swap(int *a, int *b) {
    int *tmp = *a;
    *a = *b;
    *b = *tmp;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int i, j, k;
    int *a, *b, *c;

    int ndims = 2;
    a_all = (int *) malloc(sz * sz * sizeof(int));
    b_all = (int *) malloc(sz * sz * sizeof(int));
    c_all = (int *) calloc(sz * sz, sizeof(int));

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // создаем массив, в котором указывается, какой дополнительный процесс выполняет функции одного из основных процессов
    recovered_processes = (int *) malloc(size * sizeof(int));
    for (int idx = 0; idx < size; ++idx) {
        recovered_processes[idx] = -1;
    }

    dims[0] = 0;
    dims[1] = 0;
    MPI_Dims_create(size, ndims, dims);

    if (dims[0] == dims[1]) {
        if (rank == 0) printf("No supporting processes\n");
    } else {
        int size_of_square = dims[1];
        num_of_supporting_processes = size - size_of_square * size_of_square;
        if (rank == 0) printf("%d supporting processes\n", num_of_supporting_processes);
        dims[0] = dims[1];
    }

    sz_block = sz / dims[0];
    a = (int *) malloc(sz_block * sz_block * sizeof(int));
    b = (int *) malloc(sz_block * sz_block * sizeof(int));
    c = (int *) calloc(sz_block * sz_block, sizeof(int));

    if (rank == 0) printf("MPI processors: %d\n", size);
    int i_coord = rank / dims[0];
    int j_coord = rank % dims[0];

    // обработка ошибок
    if (IsSupportProcess(rank)) {
        correct = ErrorHandler(destination);
    }

    // процессы генерируют блоки
    BlockGeneration(rank, a, b, c);

    // процессы 1...n - присылают сгенерированные матрицы 0 процессу
    // процесс 0 получает блоки, из них составляет полные матрицы a_all, b_all
    if (rank == 0) {
        for (int proc = 1; proc < size; ++proc) {
            int *a_tmp = (int *) malloc(sz_block * sz_block * sizeof(int));
            GetGeneratedMatrix(a_tmp, proc);
            free(a_tmp);
        }
    } else if (!IsSupportProcess(rank)) {
        MPI_Send(a, sz_block * sz_block, MPI_INT, 0, 2020,
                 MPI_COMM_WORLD);
    }
    if (rank == 0) {
        for (int proc = 1; proc < size; ++proc) {
            int *b_tmp = (int *) malloc(sz_block * sz_block * sizeof(int));
            GetGeneratedMatrix(b_tmp, proc);
            free(b_tmp);
        }
    } else if (!IsSupportProcess(rank)) {
        MPI_Send(b, sz_block * sz_block, MPI_INT, 0, 2020,
                 MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int *recv_tmp_a = (int *) malloc(sz_block * sz_block * sizeof(int));
    int *recv_tmp_b = (int *) malloc(sz_block * sz_block * sizeof(int));

    if (!IsSupportProcess(rank)) {
        // начальная пересылка
        Forwarding(a, b, recv_tmp_a, recv_tmp_b, rank);
        MPI_Waitall(count, &req[0], &status[0]);
        MPI_Barrier(MPI_COMM_WORLD);
        swap(a, recv_tmp_a);
        swap(b, recv_tmp_b);

        //умножение и циклическая пересылка
        for (int iteration = 0; iteration < dims[0]; ++iteration) {
            for (i = 0; i < sz_block; i++) {
                for (k = 0; k < sz_block; k++) {
                    for (j = 0; j < sz_block; j++) {
                        c[i * sz_block + j] += a[i * sz_block + k] * b[k * sz_block + j];
                    }
                }
            }
            CreateControlPoint("c", rank, c);
            Forwarding(a, b, recv_tmp_a, recv_tmp_b, rank);
            MPI_Waitall(count, &req[0], &status[0]);
            MPI_Barrier(MPI_COMM_WORLD);
            swap(a, recv_tmp_a);
            swap(b, recv_tmp_b);
        }
    }

    // 0 процесс принимает посчитанные блоки
    if (rank == 0) {
        for (i = 0; i < sz_block; ++i) {
            for (j = 0; j < sz_block; ++j) {
                c_all[(i + i_coord * sz_block) * sz + (j_coord * sz_block + j)] = c[i * sz_block + j];
            }
        }
        GetComputedResults();
    } else if (!IsSupportProcess(rank)) {
        MPI_Send(c, sz_block * sz_block, MPI_INT, 0, 2020, MPI_COMM_WORLD);
    }

    free(recv_tmp_a);
    free(recv_tmp_b);

    MPI_Barrier(MPI_COMM_WORLD);
    free(a);
    free(b);
    free(c);
    if (rank == 0) {
        if (correct) {
            printf("Results are correct\n");
        } else {
            printf("Results are not correct\n");
        }
    }
    MPI_Finalize();
    return 0;
}
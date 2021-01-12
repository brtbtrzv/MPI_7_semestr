#include <iostream>
#include <vector>
#include <mpi.h>
#include <optional>

const size_t NUM_PROCESSES = 8; // задана транспьютерная матрица NUM_PROCESSES x NUM_PROCESSES
const size_t LENGTH = 100;
const int DIMS = 2;

std::vector<std::vector<size_t> > from_left(NUM_PROCESSES);
std::vector<std::vector<size_t> > from_top(NUM_PROCESSES);

void FillMatrix() {
    for (auto &row : from_left) {
        row.resize(NUM_PROCESSES);
    }
    for (auto &row : from_top) {
        row.resize(NUM_PROCESSES);
    }

    from_left[0][0] = LENGTH;
    for (int i = 1; i < NUM_PROCESSES; ++i) {
        from_left[0][i] = from_left[0][i - 1] / 2;
    }
    for (int i = 1; i < NUM_PROCESSES; ++i) {
        for (int j = 0; j < NUM_PROCESSES; ++j) {
            if (j != 0) {
                if (i == NUM_PROCESSES - 1) {
                    from_left[i][j] = (from_left[i][j - 1] + from_top[i][j - 1]);
                } else {
                    from_left[i][j] = (from_left[i][j - 1] + from_top[i][j - 1]) / 2;
                }
            }
            if (j == NUM_PROCESSES - 1) {
                from_top[i][j] = from_top[i - 1][j] + from_left[i - 1][j];
            } else {
                from_top[i][j] = from_left[i - 1][j] + from_top[i - 1][j] - from_left[i - 1][j + 1];
            }
        }
    }
}

bool AllowDivision(size_t rank) {
    return ((rank / NUM_PROCESSES) != (NUM_PROCESSES - 1)) &&
           ((rank % NUM_PROCESSES) != (NUM_PROCESSES - 1));
}

size_t RightCell(size_t rank) {
    return rank + 1;
}

size_t LeftCell(size_t rank) {
    return rank - 1;
}

size_t BottomCell(size_t rank) {
    return rank + NUM_PROCESSES;
}

size_t UpperCell(size_t rank) {
    return rank - NUM_PROCESSES;
}

std::pair<char *, char *> Division(char *message, size_t size) {
    size_t middle = size / 2;

    return std::pair<char *, char *>(message, message + middle);
}

size_t GetLength(int rank) {
    auto i = rank / NUM_PROCESSES;
    auto j = rank % NUM_PROCESSES;
    return from_left[i][j] + from_top[i][j];
}

int main(int argc, char **argv) {
    FillMatrix();

    MPI_Init(&argc, &argv);

    int processes_num;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &processes_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dims[DIMS] = {0, 0};
    MPI_Dims_create(processes_num, DIMS, dims);


    if (rank == 0) {
        char *message = new char[LENGTH];
        for (int i = 0; i < LENGTH; ++i) {
            message[i] = '0' + i % 10;
        }

        auto divided_message = Division(message, GetLength(rank));
        auto right_cell = RightCell(rank);
        auto bottom_cell = BottomCell(rank);
        MPI_Bsend(&(divided_message.first), from_left[right_cell / NUM_PROCESSES][right_cell % NUM_PROCESSES],
                  MPI_CHAR, RightCell(rank), 3, MPI_COMM_WORLD);
        MPI_Bsend(&(divided_message.second), from_top[bottom_cell / NUM_PROCESSES][bottom_cell % NUM_PROCESSES],
                  MPI_CHAR, BottomCell(rank), 3, MPI_COMM_WORLD);

        std::cout << "\nSent from 0\n";
    } else {
        char *message = new char[GetLength(rank)];
        auto count_top = from_top[rank / NUM_PROCESSES][rank % NUM_PROCESSES];
        auto count_left = from_left[rank / NUM_PROCESSES][rank % NUM_PROCESSES];
        if (rank / NUM_PROCESSES != 0) {
            MPI_Request request;
            MPI_Irecv(&message[0], count_top, MPI_CHAR, UpperCell(rank), 3, MPI_COMM_WORLD, &request);
            MPI_Status status;
            if (MPI_Wait(&request, &status) == MPI_SUCCESS) {
                std::cout << rank << " received from top \n";
            }
        }
        if (rank % NUM_PROCESSES != 0) {
            MPI_Request request;
            MPI_Irecv(&message[count_top], count_left, MPI_CHAR, LeftCell(rank), 3, MPI_COMM_WORLD, &request);
            MPI_Status status;
            if (MPI_Wait(&request, &status) == MPI_SUCCESS) {
                std::cout << rank << " received from left \n";
            }
        }
        if (rank == NUM_PROCESSES * NUM_PROCESSES - 1) {
            std::cout << "Success!\n";
            for (char* c = message; c < message + LENGTH; ++c) {
                std::cout << *c;
            }
            std::cout << "\n";
        } else if (AllowDivision(rank)) {
            auto divided_message = Division(message, GetLength(rank));
            auto right_cell = RightCell(rank);
            auto bottom_cell = BottomCell(rank);
            MPI_Bsend(&(divided_message.first), from_left[right_cell / NUM_PROCESSES][right_cell % NUM_PROCESSES],
                      MPI_CHAR, RightCell(rank), 3, MPI_COMM_WORLD);
            MPI_Bsend(&(divided_message.second), from_top[bottom_cell / NUM_PROCESSES][bottom_cell % NUM_PROCESSES],
                      MPI_CHAR, BottomCell(rank), 3, MPI_COMM_WORLD);
        } else {
            auto count = from_left[rank / NUM_PROCESSES][rank % NUM_PROCESSES] +
                         from_top[rank / NUM_PROCESSES][rank % NUM_PROCESSES];
            if (rank / NUM_PROCESSES == NUM_PROCESSES - 1) {
                MPI_Bsend(&message, count, MPI_CHAR, RightCell(rank), 3, MPI_COMM_WORLD);
            } else if (rank % NUM_PROCESSES == NUM_PROCESSES - 1) {
                MPI_Bsend(&message, count, MPI_CHAR, BottomCell(rank), 3, MPI_COMM_WORLD);
            } else {
                throw std::exception();
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

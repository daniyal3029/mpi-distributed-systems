# MPI Distributed Systems Projects

This repository contains three comprehensive Parallel and Distributed Computing (PDC) projects implemented in C using the Message Passing Interface (MPI).

## Project Structure

| File | Project Title | Description |
| :--- | :--- | :--- |
| **[q1.c](q1.c)** | Hierarchical Tree Collectives | Implementation of K-ary tree-based Broadcast and Reduction with performance comparison against naive flat collectives. |
| **[q2.c](q2.c)** | Deadlock-Avoiding Exchange | Demonstration of a deadlock-free multi-phase message exchange system using rank-ordering and a READY-ACK handshake protocol. |
| **[q3.c](q3.c)** | Distributed Priority Queue | A concurrent task processing system using a Binary Min-Heap at the root, comparing blocking, non-blocking, and batching communication. |

## Getting Started (Linux/Ubuntu)

These projects are designed for **OpenMPI**. Ensure you have it installed:
```bash
sudo apt update && sudo apt install openmpi-bin libopenmpi-dev
```

### Compilation
Use `mpicc` with the math library (`-lm`) for Q1:
```bash
mpicc q1.c -o q1 -lm
mpicc q2.c -o q2
mpicc q3.c -o q3
```

### Execution
Run with 8 processes (or any preferred number):
```bash
# Question 1: mpirun -np <P> ./q1 <k> <root> <N>
mpirun -np 8 ./q1 2 0 1000

# Question 2
mpirun -np 8 ./q2

# Question 3: mpirun -np <P> ./q3 <tasks_per_worker>
mpirun -np 8 ./q3 10
```

## Running on Windows (MS-MPI)

If running on Windows, ensure **Microsoft MPI** is installed.

### Helper Scripts
I have provided PowerShell helper scripts for each question to automate compilation (using MSYS2 GCC) and execution:
- `run_q1.ps1`
- `run_q2.ps1`
- `run_q3.ps1`

## Technical Highlights

### 1. Hierarchical Tree (Q1)
- **Complexity**: Reduces communication from $O(P)$ to $O(\log_k P)$.
- **Tree Logic**: Dynamically calculates parents and children for an arbitrary root process.
- **Safety**: Uses a strict receive-then-forward sequence to avoid circular waits.

### 2. Deadlock Avoidance (Q2)
- **Rank Ordering**: Uses $i < j$ logic to determine who sends first, breaking the symmetry that causes deadlocks.
- **Handshake Protocol**: Implements a three-way sync (READY → ACK → DATA) to remove reliance on internal MPI buffering.

### 3. Distributed Min-Heap (Q3)
- **Data Structure**: Binary Min-Heap implementation with $O(\log N)$ insertion/extraction.
- **Batching**: Bundles multiple tasks into single packets to amortize network latency.
- **Asynchronous**: Demonstrates the use of `MPI_Isend`, `MPI_Irecv`, and `MPI_Test` for overlapping computation and communication.

## Sample Output Visualization
The projects include detailed logs and ASCII visualizations:
```text
Level 0: [P0 (root)]
Level 1: [P1] [P2]
Level 2: [P3] [P4] [P5] [P6]

[Process 1] received data from parent 0
[Process 1] forwarding to child 3
...
Tree Collectives: CORRECT
```

## Contributing
To contribute to this project, please fork the repository and submit a pull request with your changes. Ensure that your code is formatted according to standard C conventions and includes proper documentation. Additionally, please run the provided test cases to verify the correctness of your changes.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
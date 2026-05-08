/*
 * ============================================================
 *  Hierarchical Tree-Based Broadcast and Reduction on a Large Cluster
 * ============================================================
 *
 * COMPILATION:
 *   mpicc q1.c -o q1 -lm
 *
 * EXECUTION:
 *   mpirun -np 8 ./q1 2 0 1000
 *   Arguments: k=2 (fan-out), root=0, N=1000 (array size)
 *
 * DESCRIPTION:
 *   Implements naive (flat) and hierarchical (k-ary tree) versions of:
 *     - Broadcast
 *     - Reduction (sum)
 *   Measures wall-clock time for each, prints communication logs,
 *   validates correctness, and displays a performance table.
 *
 * EXPECTED OUTPUT (8 processes, k=2, root=0, N=1000):
 *   Tree structure, comm logs, final sum, CORRECT/INCORRECT, timing table.
 *
 * AUTHOR: Generated for PDC Assignment
 * ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* ── MPI message tags ─────────────────────────────────────── */
#define TAG_NAIVE_BCAST   10
#define TAG_TREE_BCAST    20
#define TAG_NAIVE_RED     30
#define TAG_TREE_RED      40

/* ── Tree descriptor ──────────────────────────────────────── */
typedef struct {
    int rank;           /* this process rank            */
    int size;           /* total number of processes    */
    int k;              /* fan-out                      */
    int root;           /* logical tree root            */
    int parent;         /* -1 if root                   */
    int  children[64];  /* child ranks (max 64)         */
    int  nchildren;     /* actual child count           */
} Tree;

/* ═══════════════════════════════════════════════════════════
 *  Helper: map logical rank → MPI rank
 *  We re-root the tree at `root` by treating `root` as
 *  logical rank 0, then mapping back.
 *  logical rank  l  →  MPI rank  (l + root) % size
 *  MPI rank      r  →  logical rank (r - root + size) % size
 * ═══════════════════════════════════════════════════════════ */
static inline int to_logical(int mpi_rank, int root, int size) {
    return (mpi_rank - root + size) % size;
}
static inline int to_mpi(int logical, int root, int size) {
    return (logical + root) % size;
}

/* ═══════════════════════════════════════════════════════════
 *  build_tree()
 *  Populates the Tree struct for the calling process.
 * ═══════════════════════════════════════════════════════════ */
void build_tree(Tree *t, int rank, int size, int k, int root) {
    t->rank     = rank;
    t->size     = size;
    t->k        = k;
    t->root     = root;
    t->nchildren = 0;

    int l = to_logical(rank, root, size); /* my logical rank */

    /* ── parent ──────────────────────────────────────────── */
    if (l == 0) {
        t->parent = -1;                  /* root has no parent */
    } else {
        int lp = (l - 1) / k;
        t->parent = to_mpi(lp, root, size);
    }

    /* ── children ────────────────────────────────────────── */
    for (int i = 1; i <= k; i++) {
        int lc = k * l + i;              /* child logical rank */
        if (lc < size) {
            t->children[t->nchildren++] = to_mpi(lc, root, size);
        }
    }
}

/* ═══════════════════════════════════════════════════════════
 *  print_tree_structure()
 *  Root collects and prints the tree topology.
 * ═══════════════════════════════════════════════════════════ */
void print_tree_structure(Tree *t) {
    int rank = t->rank, size = t->size, root = t->root;

    /* Each process sends its children list to root */
    int buf[66]; /* buf[0]=nchildren, buf[1..] = child ranks */
    buf[0] = t->nchildren;
    for (int i = 0; i < t->nchildren; i++) buf[i+1] = t->children[i];

    if (rank == root) {
        /* Print own line */
        printf("\n╔══════════════════════════════════════════╗\n");
        printf("║          K-ARY TREE STRUCTURE            ║\n");
        printf("╚══════════════════════════════════════════╝\n");
        printf("Process %d -> parent: (root) | children:", rank);
        for (int i = 0; i < t->nchildren; i++) printf(" %d", t->children[i]);
        if (t->nchildren == 0) printf(" (none)");
        printf("\n");

        /* Receive from all others */
        for (int src = 0; src < size; src++) {
            if (src == root) continue;
            int rbuf[66];
            MPI_Recv(rbuf, 66, MPI_INT, src, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int nc = rbuf[0];
            /* find parent of src */
            int lsrc = to_logical(src, root, size);
            int lpar = (lsrc == 0) ? -1 : (lsrc-1)/t->k;
            int par  = (lpar == -1) ? -1 : to_mpi(lpar, root, size);
            printf("Process %d -> parent: %d      | children:", src, par);
            for (int i = 0; i < nc; i++) printf(" %d", rbuf[i+1]);
            if (nc == 0) printf(" (none — leaf)");
            printf("\n");
        }
        printf("\n");
    } else {
        MPI_Send(buf, 66, MPI_INT, root, 99, MPI_COMM_WORLD);
    }
}

/* ═══════════════════════════════════════════════════════════
 *  print_ascii_tree()
 *  ASCII visualization of the k-ary tree.
 * ═══════════════════════════════════════════════════════════ */
void print_ascii_tree(int size, int k, int root) {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║           ASCII TREE DIAGRAM             ║\n");
    printf("╚══════════════════════════════════════════╝\n");

    /* BFS level-by-level */
    int *queue = (int *)malloc(size * sizeof(int));
    int *level = (int *)malloc(size * sizeof(int));
    int head = 0, tail = 0;

    queue[tail] = 0;        /* logical ranks */
    level[tail] = 0;
    tail++;

    int cur_level = 0;
    printf("Level 0: [P%d (root)]\n", root);

    while (head < tail) {
        int l = queue[head];
        int lv = level[head];
        head++;

        if (lv > cur_level) {
            cur_level = lv;
            printf("Level %d:", cur_level);
        }

        for (int i = 1; i <= k; i++) {
            int lc = k * l + i;
            if (lc < size) {
                int mpi_c = to_mpi(lc, root, size);
                printf(" [P%d]", mpi_c);
                queue[tail]   = lc;
                level[tail++] = lv + 1;
            }
        }
    }
    printf("\n\n");

    free(queue);
    free(level);
}

/* ═══════════════════════════════════════════════════════════
 *  PART 1 — naive_broadcast()
 *  Root sends array B to every other process individually.
 *  Communication complexity: O(P)
 * ═══════════════════════════════════════════════════════════ */
double naive_broadcast(int *B, int N, Tree *t) {
    int rank = t->rank, size = t->size, root = t->root;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (rank == root) {
        for (int dest = 0; dest < size; dest++) {
            if (dest == root) continue;
            /* MPI_Send(buf, count, datatype, dest, tag, comm) */
            MPI_Send(B, N, MPI_INT, dest, TAG_NAIVE_BCAST, MPI_COMM_WORLD);
            printf("  [Naive Bcast] Process %d → sent to process %d\n", rank, dest);
        }
    } else {
        /* MPI_Recv(buf, count, datatype, source, tag, comm, status) */
        MPI_Recv(B, N, MPI_INT, root, TAG_NAIVE_BCAST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("  [Naive Bcast] Process %d ← received from root %d\n", rank, root);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime() - t0;
}

/* ═══════════════════════════════════════════════════════════
 *  PART 2 — tree_broadcast()
 *  Root sends to its k children; each child forwards to its
 *  own children; continues until all processes have data.
 *  Communication complexity: O(log_k P)
 *
 *  Deadlock prevention:
 *    A process first receives from parent, THEN sends to children.
 *    This strict ordering guarantees no circular waits.
 * ═══════════════════════════════════════════════════════════ */
double tree_broadcast(int *B, int N, Tree *t) {
    int rank = t->rank, root = t->root;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Step 1: receive from parent (root skips this step) */
    if (rank != root) {
        MPI_Recv(B, N, MPI_INT, t->parent, TAG_TREE_BCAST,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("  [Tree  Bcast] Process %d ← received from parent %d\n",
               rank, t->parent);
    }

    /* Step 2: forward to children */
    for (int i = 0; i < t->nchildren; i++) {
        int child = t->children[i];
        MPI_Send(B, N, MPI_INT, child, TAG_TREE_BCAST, MPI_COMM_WORLD);
        printf("  [Tree  Bcast] Process %d → forwarding to child %d\n",
               rank, child);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime() - t0;
}

/* ═══════════════════════════════════════════════════════════
 *  PART 3 — naive_reduction()
 *  Every process sends its local array to root.
 *  Root accumulates (sums) all values.
 *  Communication complexity: O(P)
 *  Returns global sum to root; 0 to others.
 * ═══════════════════════════════════════════════════════════ */
double naive_reduction(int *A, int N, long long *global_sum, Tree *t) {
    int rank = t->rank, size = t->size, root = t->root;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    if (rank == root) {
        /* Accumulate own contribution */
        long long sum = 0;
        for (int i = 0; i < N; i++) sum += A[i];

        /* Receive from all other processes */
        int *tmp = (int *)malloc(N * sizeof(int));
        for (int src = 0; src < size; src++) {
            if (src == root) continue;
            MPI_Recv(tmp, N, MPI_INT, src, TAG_NAIVE_RED,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("  [Naive Red  ] Root %d ← received from process %d\n",
                   root, src);
            for (int i = 0; i < N; i++) sum += tmp[i];
        }
        free(tmp);
        *global_sum = sum;
    } else {
        MPI_Send(A, N, MPI_INT, root, TAG_NAIVE_RED, MPI_COMM_WORLD);
        printf("  [Naive Red  ] Process %d → sent to root %d\n", rank, root);
        *global_sum = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime() - t0;
}

/* ═══════════════════════════════════════════════════════════
 *  PART 4 — tree_reduction()
 *  Each process computes its local partial sum.
 *  Leaf nodes send up; intermediate nodes accumulate
 *  children's partial sums then add their own before sending up.
 *  Communication complexity: O(log_k P)
 *
 *  Deadlock prevention:
 *    A process first receives from ALL its children (blocking),
 *    accumulates, then sends ONE message upward to parent.
 *    Since only leaves start immediately and internal nodes
 *    wait, there is no circular dependency.
 * ═══════════════════════════════════════════════════════════ */
double tree_reduction(int *A, int N, long long *global_sum, Tree *t) {
    int rank = t->rank, root = t->root;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Step 1: compute local partial sum */
    long long partial = 0;
    for (int i = 0; i < N; i++) partial += A[i];

    /* Step 2: receive partial sums from all children and accumulate */
    for (int i = 0; i < t->nchildren; i++) {
        long long child_sum = 0;
        int child = t->children[i];
        MPI_Recv(&child_sum, 1, MPI_LONG_LONG, child, TAG_TREE_RED,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("  [Tree  Red  ] Process %d ← partial sum from child %d = %lld\n",
               rank, child, child_sum);
        partial += child_sum;
    }

    /* Step 3: send accumulated sum upward (unless we are root) */
    if (rank != root) {
        MPI_Send(&partial, 1, MPI_LONG_LONG, t->parent, TAG_TREE_RED,
                 MPI_COMM_WORLD);
        printf("  [Tree  Red  ] Process %d → sending partial sum %lld to parent %d\n",
               rank, partial, t->parent);
        *global_sum = 0;
    } else {
        *global_sum = partial;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime() - t0;
}

/* ═══════════════════════════════════════════════════════════
 *  validate_results()
 *  Root serially computes the expected sum and compares.
 * ═══════════════════════════════════════════════════════════ */
void validate_results(long long tree_sum, long long naive_sum,
                      long long serial_sum, int root, int rank) {
    if (rank != root) return;

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║           CORRECTNESS VALIDATION         ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("  Serial   sum = %lld\n", serial_sum);
    printf("  Naive    sum = %lld\n", naive_sum);
    printf("  Tree     sum = %lld\n", tree_sum);

    if (tree_sum == serial_sum && naive_sum == serial_sum) {
        printf("\n  ✔  Tree Collectives: CORRECT\n");
        printf("  ✔  Naive Collectives: CORRECT\n");
    } else {
        if (tree_sum != serial_sum)
            printf("\n  ✘  Tree Collectives: INCORRECT (expected %lld, got %lld)\n",
                   serial_sum, tree_sum);
        if (naive_sum != serial_sum)
            printf("  ✘  Naive Collectives: INCORRECT (expected %lld, got %lld)\n",
                   serial_sum, naive_sum);
    }
}

/* ═══════════════════════════════════════════════════════════
 *  print_performance_table()
 * ═══════════════════════════════════════════════════════════ */
void print_performance_table(int P, int k, double t_nb, double t_tb,
                              double t_nr, double t_tr) {
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                  PERFORMANCE COMPARISON                     ║\n");
    printf("╠══════╦═════╦══════════════╦══════════════╦══════════════╦═══════════════╣\n");
    printf("║  P   ║  k  ║  Naive_Bcast ║  Tree_Bcast  ║  Naive_Red   ║  Tree_Red     ║\n");
    printf("╠══════╬═════╬══════════════╬══════════════╬══════════════╬═══════════════╣\n");
    printf("║  %-3d ║  %-2d ║  %-10.6f  ║  %-10.6f  ║  %-10.6f  ║  %-11.6f  ║\n",
           P, k, t_nb, t_tb, t_nr, t_tr);
    printf("╚══════╩═════╩══════════════╩══════════════╩══════════════╩═══════════════╝\n");

    printf("\n  Speedup (Bcast): %.2fx  |  Speedup (Red): %.2fx\n",
           (t_tb > 0) ? t_nb / t_tb : 0.0,
           (t_tr > 0) ? t_nr / t_tr : 0.0);
}

/* ═══════════════════════════════════════════════════════════
 *  print_complexity_discussion()
 * ═══════════════════════════════════════════════════════════ */
void print_complexity_discussion(int P, int k, int root) {
    if (root != 0 && root != MPI_PROC_NULL) return; /* only called from root */
    double logkP = (k > 1) ? log((double)P) / log((double)k) : (double)P;

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║              COMPLEXITY & COMPARISON ANALYSIS                ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("  ▸ Naive Broadcast  : O(P)         = O(%d) messages from root\n", P);
    printf("  ▸ Tree  Broadcast  : O(log_%d(P))  = O(%.1f) levels\n", k, logkP);
    printf("  ▸ Naive Reduction  : O(P)         = O(%d) messages to root\n", P);
    printf("  ▸ Tree  Reduction  : O(log_%d(P))  = O(%.1f) levels\n", k, logkP);
    printf("\n  WHY TREE IS FASTER:\n");
    printf("  • Naive: Root is a bottleneck — it sends/receives P-1 msgs sequentially.\n");
    printf("  • Tree : Communication load is distributed. Each internal node handles\n");
    printf("    only k children. Messages travel in parallel across levels.\n");
    printf("  • As P grows, tree gives exponential savings (log vs linear).\n");
    printf("  • With k=%d and P=%d, tree uses ~%.0f levels vs %d sequential sends.\n\n",
           k, P, ceil(logkP), P-1);
}

/* ══════════════════════════════════════════════════════════
 *  MAIN
 * ══════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {

    /* ── MPI Initialization ───────────────────────────────── */
    /* MPI_Init: initializes MPI execution environment */
    MPI_Init(&argc, &argv);

    int rank, size;
    /* MPI_Comm_rank: returns this process's rank in the communicator */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    /* MPI_Comm_size: returns total number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ── Parse command-line arguments ───────────────────────
     *  Usage: mpirun -np P ./q1 k root N
     */
    if (argc < 4) {
        if (rank == 0)
            fprintf(stderr,
                    "Usage: mpirun -np <P> ./q1 <k> <root> <N>\n"
                    "  k    = tree fan-out  (e.g. 2)\n"
                    "  root = root process  (e.g. 0)\n"
                    "  N    = array size    (e.g. 1000)\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int k    = atoi(argv[1]);
    int root = atoi(argv[2]);
    int N    = atoi(argv[3]);

    /* Basic sanity checks */
    if (k < 1 || root < 0 || root >= size || N < 1) {
        if (rank == 0)
            fprintf(stderr, "Invalid arguments: k>=1, 0<=root<P, N>=1\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* ── Print header ────────────────────────────────────── */
    if (rank == root) {
        printf("\n╔══════════════════════════════════════════════════════════════╗\n");
        printf("║   Hierarchical Tree-Based Broadcast & Reduction              ║\n");
        printf("║   P=%d  k=%d  root=%d  N=%d                                 \n",
               size, k, root, N);
        printf("╚══════════════════════════════════════════════════════════════╝\n");
    }

    /* ── Build logical k-ary tree ────────────────────────── */
    Tree t;
    build_tree(&t, rank, size, k, root);

    /* ── Print tree structure (from root) ────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    print_tree_structure(&t);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == root) {
        print_ascii_tree(size, k, root);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ── Allocate arrays ─────────────────────────────────── */
    /* B: broadcast buffer (root initialises with 1..N) */
    int *B = (int *)malloc(N * sizeof(int));
    /* A: local data array (random values per process)   */
    int *A = (int *)malloc(N * sizeof(int));
    if (!B || !A) {
        fprintf(stderr, "Process %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Root fills broadcast buffer */
    if (rank == root) {
        for (int i = 0; i < N; i++) B[i] = i + 1;   /* 1, 2, 3, ... N */
    } else {
        memset(B, 0, N * sizeof(int));
    }

    /* Each process fills its local array with random values */
    srand(rank * 1234 + 5678);
    for (int i = 0; i < N; i++) A[i] = rand() % 100; /* 0..99 */

    /* ── Root computes serial sum for validation ─────────── */
    /* Root gathers all local arrays to compute expected total sum */
    long long serial_sum = 0;
    {
        /* Collect all arrays at root to compute ground truth */
        int *all = NULL;
        if (rank == root) {
            all = (int *)malloc((long long)size * N * sizeof(int));
            /* Copy own data */
            memcpy(all, A, N * sizeof(int));
        }
        /* Use MPI_Gather to collect all A arrays */
        /* MPI_Gather(sendbuf, sendcount, sendtype,
                      recvbuf, recvcount, recvtype, root, comm) */
        MPI_Gather(A, N, MPI_INT,
                   (rank == root) ? all : NULL,
                   N, MPI_INT, root, MPI_COMM_WORLD);

        if (rank == root) {
            for (int p = 0; p < size; p++)
                for (int i = 0; i < N; i++)
                    serial_sum += all[p * N + i];
            free(all);
        }
    }

    /* ─────────────────────────────────────────────────────
     *  PART 1: NAIVE BROADCAST
     * ───────────────────────────────────────────────────── */
    if (rank == root)
        printf("══════════════════ NAIVE BROADCAST ══════════════════\n");
    MPI_Barrier(MPI_COMM_WORLD);

    double t_naive_bcast = naive_broadcast(B, N, &t);

    /* ─────────────────────────────────────────────────────
     *  PART 2: TREE BROADCAST
     *  Reset buffer first, then re-seed root
     * ───────────────────────────────────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == root) {
        printf("\n══════════════════ TREE BROADCAST  ══════════════════\n");
        for (int i = 0; i < N; i++) B[i] = i + 1;
    } else {
        memset(B, 0, N * sizeof(int));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double t_tree_bcast = tree_broadcast(B, N, &t);

    /* ─────────────────────────────────────────────────────
     *  PART 3: NAIVE REDUCTION
     * ───────────────────────────────────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == root)
        printf("\n══════════════════ NAIVE REDUCTION  ══════════════════\n");
    MPI_Barrier(MPI_COMM_WORLD);

    long long naive_sum = 0;
    double t_naive_red = naive_reduction(A, N, &naive_sum, &t);

    /* ─────────────────────────────────────────────────────
     *  PART 4: TREE REDUCTION
     * ───────────────────────────────────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == root)
        printf("\n══════════════════ TREE REDUCTION   ══════════════════\n");
    MPI_Barrier(MPI_COMM_WORLD);

    long long tree_sum = 0;
    double t_tree_red = tree_reduction(A, N, &tree_sum, &t);

    /* ── Output results (from root only) ─────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == root) {
        validate_results(tree_sum, naive_sum, serial_sum, root, rank);
        print_complexity_discussion(size, k, rank);
        print_performance_table(size, k,
                                t_naive_bcast, t_tree_bcast,
                                t_naive_red,   t_tree_red);
        printf("\n  Final global sum: %lld\n\n", tree_sum);
    }

    /* ── Cleanup ──────────────────────────────────────────── */
    free(B);
    free(A);

    /* MPI_Finalize: cleans up MPI execution environment */
    MPI_Finalize();
    return EXIT_SUCCESS;
}

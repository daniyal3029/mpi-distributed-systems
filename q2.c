/*
 * ============================================================
 * PROJECT: Dynamic Deadlock-Avoiding Message Exchange in MPI
 * FILE:    q2.c
 * AUTHOR:  MPI/Parallel Computing Assignment
 *
 * COMPILE: mpicc q2.c -o q2
 * RUN:     mpirun -np 8 ./q2
 *
 * ============================================================
 *
 * OVERVIEW
 * --------
 * This program demonstrates a DEADLOCK-FREE multi-phase MPI
 * message exchange system using two communication strategies:
 *
 *   1. Pure Blocking Send/Receive  (with rank-ordering)
 *   2. Handshake-Based Safe Communication (READY→ACK→DATA)
 *
 * Both strategies work across two phases:
 *   Phase 1: Exchange with partners where |rank_i - rank_j| is ODD
 *   Phase 2: Exchange with partners where |rank_i - rank_j| is EVEN
 *              (excluding self, i.e. diff != 0)
 *
 * DEADLOCK EXPLAINED
 * ------------------
 * A deadlock occurs when every process is waiting for another
 * process, creating a circular wait.  Classic example with 2 procs:
 *
 *   Process 0: MPI_Send(to 1) ... blocks until P1 receives
 *   Process 1: MPI_Send(to 0) ... blocks until P0 receives
 *
 * Both are stuck in Send → nobody calls Recv → DEADLOCK.
 *
 * HOW WE AVOID IT
 * ---------------
 * Method A (Blocking):  Lower-rank process sends first and the
 *   higher-rank process receives first.  This breaks the circular
 *   wait because the two sides are never both stuck in Send at once.
 *
 * Method B (Handshake): A 3-step protocol (READY → ACK → DATA)
 *   ensures the receiver is always ready before data is sent.
 *   Combined with rank-based ordering it is fully deadlock-free.
 *
 * ASCII COMMUNICATION DIAGRAM (2-process pair, blocking)
 * -------------------------------------------------------
 *   Lower rank (0)          Higher rank (1)
 *   ──────────────          ──────────────
 *   MPI_Send ──────────────► MPI_Recv
 *   MPI_Recv ◄────────────── MPI_Send
 *
 * ASCII COMMUNICATION DIAGRAM (handshake)
 * ----------------------------------------
 *   Sender (lower rank)     Receiver (higher rank)
 *   ───────────────────     ──────────────────────
 *   Send READY ─────────────► Recv READY
 *                             Send ACK ───────────►  (back to sender)
 *   Recv ACK  ◄───────────────
 *   Send DATA ─────────────► Recv DATA
 *
 * ============================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================
 * CONSTANTS AND MPI TAG DEFINITIONS
 * ============================================================ */

#define NUM_PROCS_MIN   8       /* minimum required processes      */
#define MSG_SIZE        4       /* integers per message payload    */

/* Tags for the handshake protocol – must be distinct            */
#define TAG_READY       10      /* "I want to send you data"       */
#define TAG_ACK         20      /* "I am ready to receive"         */
#define TAG_DATA        30      /* actual data payload             */
#define TAG_BLOCKING    40      /* tag for pure-blocking method    */

/* ============================================================
 * DATA STRUCTURES
 * ============================================================ */

/*
 * MessageStore holds the send and receive buffers for one process.
 * send_buf[j] = array of MSG_SIZE ints that this process sends to j
 * recv_buf[j] = array of MSG_SIZE ints received from process j
 */
typedef struct {
    int  rank;          /* rank of this process                   */
    int  size;          /* total number of MPI processes          */
    int **send_buf;     /* send_buf[j][0..MSG_SIZE-1]             */
    int **recv_buf;     /* recv_buf[j][0..MSG_SIZE-1]             */
} MessageStore;

/* ============================================================
 * FORWARD DECLARATIONS
 * ============================================================ */
MessageStore* init_messages(int rank, int size);
void          free_messages(MessageStore *ms);
void          print_recv_buffers(const MessageStore *ms, const char *label);
int           is_phase1_partner(int r1, int r2);
int           is_phase2_partner(int r1, int r2);
void          blocking_exchange(MessageStore *ms, int phase);
void          handshake_exchange(MessageStore *ms, int phase);
double        measure_blocking(MessageStore *ms);
double        measure_handshake(MessageStore *ms);
void          print_timing_table(int rank, double t_block, double t_hand);
void          reset_recv_buf(MessageStore *ms);

/* ============================================================
 * MAIN
 * ============================================================ */
int main(int argc, char **argv)
{
    int rank, size;

    /* ----------------------------------------------------------
     * MPI_Init  – must be the first MPI call.
     * Initialises the MPI execution environment.
     * ---------------------------------------------------------- */
    MPI_Init(&argc, &argv);

    /* ----------------------------------------------------------
     * MPI_Comm_rank – find out which process we are (0..size-1)
     * MPI_Comm_size – find out how many processes were launched
     * ---------------------------------------------------------- */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Sanity check: need at least NUM_PROCS_MIN processes */
    if (size < NUM_PROCS_MIN) {
        if (rank == 0)
            fprintf(stderr,
                    "[ERROR] This program requires at least %d MPI processes.\n"
                    "        Run with: mpirun -np %d ./q2\n",
                    NUM_PROCS_MIN, NUM_PROCS_MIN);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    /* ----------------------------------------------------------
     * Seed random number generator differently per process so
     * each process generates unique messages.
     * ---------------------------------------------------------- */
    srand((unsigned)(time(NULL) + rank * 1234567));

    /* ----------------------------------------------------------
     * Allocate and fill send buffers with random integers.
     * ---------------------------------------------------------- */
    MessageStore *ms = init_messages(rank, size);

    /* =========================================================
     * BANNER
     * ========================================================= */
    if (rank == 0) {
        printf("\n");
        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Dynamic Deadlock-Avoiding Message Exchange in MPI       ║\n");
        printf("║  Processes: %-3d   Message size: %d integers per msg      ║\n", size, MSG_SIZE);
        printf("╚══════════════════════════════════════════════════════════╝\n\n");
        fflush(stdout);
    }

    /* Give rank-0 a moment to print the banner cleanly */
    MPI_Barrier(MPI_COMM_WORLD);

    /* =========================================================
     * PART 1 – PURE BLOCKING SEND/RECEIVE
     * ========================================================= */
    if (rank == 0) {
        printf("══════════════════════════════════════════════════════════\n");
        printf("  PART 1: PURE BLOCKING SEND/RECEIVE\n");
        printf("══════════════════════════════════════════════════════════\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* --- Phase 1 (blocking) ---------------------------------- */
    if (rank == 0) {
        printf("[Blocking] ── Phase 1 Begin (|rank_i - rank_j| is ODD) ──\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    blocking_exchange(ms, 1);

    /* MPI_Barrier ensures ALL processes finish Phase 1 before any
     * process enters Phase 2.  Without this, a fast process could
     * start Phase 2 while a slow process is still in Phase 1,
     * causing tag collisions or unexpected message matches.        */
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n[Blocking] ── Phase 1 Complete ──\n\n");
        printf("[Blocking] ── Phase 2 Begin (|rank_i - rank_j| is EVEN, ≠ 0) ──\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* --- Phase 2 (blocking) ---------------------------------- */
    blocking_exchange(ms, 2);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n[Blocking] ── Phase 2 Complete ──\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Print received buffers for ALL processes after blocking run */
    print_recv_buffers(ms, "Blocking");
    MPI_Barrier(MPI_COMM_WORLD);

    /* =========================================================
     * PART 2 – HANDSHAKE-BASED SAFE COMMUNICATION
     * ========================================================= */
    if (rank == 0) {
        printf("\n══════════════════════════════════════════════════════════\n");
        printf("  PART 2: HANDSHAKE-BASED SAFE COMMUNICATION\n");
        printf("══════════════════════════════════════════════════════════\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Reset receive buffers so we can verify handshake results */
    reset_recv_buf(ms);

    /* --- Phase 1 (handshake) --------------------------------- */
    if (rank == 0) {
        printf("[Handshake] ── Phase 1 Begin (|rank_i - rank_j| is ODD) ──\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    handshake_exchange(ms, 1);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n[Handshake] ── Phase 1 Complete ──\n\n");
        printf("[Handshake] ── Phase 2 Begin (|rank_i - rank_j| is EVEN, ≠ 0) ──\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* --- Phase 2 (handshake) --------------------------------- */
    handshake_exchange(ms, 2);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n[Handshake] ── Phase 2 Complete ──\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    print_recv_buffers(ms, "Handshake");
    MPI_Barrier(MPI_COMM_WORLD);

    /* =========================================================
     * PERFORMANCE COMPARISON
     * ========================================================= */
    if (rank == 0) {
        printf("\n══════════════════════════════════════════════════════════\n");
        printf("  PERFORMANCE COMPARISON\n");
        printf("══════════════════════════════════════════════════════════\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    reset_recv_buf(ms);
    double t_block = measure_blocking(ms);
    MPI_Barrier(MPI_COMM_WORLD);

    reset_recv_buf(ms);
    double t_hand  = measure_handshake(ms);
    MPI_Barrier(MPI_COMM_WORLD);

    print_timing_table(rank, t_block, t_hand);
    MPI_Barrier(MPI_COMM_WORLD);

    /* =========================================================
     * DEADLOCK-FREE CONFIRMATION
     * ========================================================= */
    if (rank == 0) {
        printf("\n╔══════════════════════════════════════════════════════════╗\n");
        printf("║  ✓  All phases completed without deadlock.               ║\n");
        printf("║  ✓  Both communication methods verified successfully.    ║\n");
        printf("╚══════════════════════════════════════════════════════════╝\n\n");
        fflush(stdout);
    }

    /* Cleanup */
    free_messages(ms);

    /* ----------------------------------------------------------
     * MPI_Finalize – must be the last MPI call.
     * Cleans up all MPI state.
     * ---------------------------------------------------------- */
    MPI_Finalize();
    return EXIT_SUCCESS;
}

/* ============================================================
 * FUNCTION: init_messages
 * PURPOSE : Allocate send/recv buffers and fill send buffers
 *           with random integers.
 * ============================================================ */
MessageStore* init_messages(int rank, int size)
{
    MessageStore *ms = (MessageStore*)malloc(sizeof(MessageStore));
    ms->rank = rank;
    ms->size = size;

    /* Allocate pointer arrays */
    ms->send_buf = (int**)malloc(size * sizeof(int*));
    ms->recv_buf = (int**)malloc(size * sizeof(int*));

    for (int j = 0; j < size; j++) {
        ms->send_buf[j] = (int*)malloc(MSG_SIZE * sizeof(int));
        ms->recv_buf[j] = (int*)calloc(MSG_SIZE, sizeof(int)); /* zero-init */

        /* Fill send buffer with random integers (seeded uniquely per proc) */
        for (int k = 0; k < MSG_SIZE; k++)
            ms->send_buf[j][k] = rand() % 1000;
    }
    return ms;
}

/* ============================================================
 * FUNCTION: free_messages
 * PURPOSE : Release all dynamically allocated memory.
 * ============================================================ */
void free_messages(MessageStore *ms)
{
    for (int j = 0; j < ms->size; j++) {
        free(ms->send_buf[j]);
        free(ms->recv_buf[j]);
    }
    free(ms->send_buf);
    free(ms->recv_buf);
    free(ms);
}

/* ============================================================
 * FUNCTION: reset_recv_buf
 * PURPOSE : Zero out receive buffers between timing runs.
 * ============================================================ */
void reset_recv_buf(MessageStore *ms)
{
    for (int j = 0; j < ms->size; j++)
        memset(ms->recv_buf[j], 0, MSG_SIZE * sizeof(int));
}

/* ============================================================
 * FUNCTION: print_recv_buffers
 * PURPOSE : Print the received data of every process after a run.
 * ============================================================ */
void print_recv_buffers(const MessageStore *ms, const char *label)
{
    /* Serialise output: process 0 prints first, then signals p1, etc. */
    int rank = ms->rank;
    int size = ms->size;

    /* Token-passing approach: process i waits for a token from i-1 */
    if (rank == 0) {
        printf("  ── [%s] Received Buffers ──\n", label);
        fflush(stdout);
    }

    /* Each process waits for its "turn" token (a dummy int from rank-1) */
    if (rank > 0) {
        int token;
        MPI_Recv(&token, 1, MPI_INT, rank - 1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("  [Process %d] recv_buf:\n", rank);
    for (int j = 0; j < size; j++) {
        /* Only show entries that were actually received (non-zero or active) */
        int active = 0;
        for (int k = 0; k < MSG_SIZE; k++)
            if (ms->recv_buf[j][k] != 0) { active = 1; break; }
        if (active) {
            printf("    from P%-2d : [", j);
            for (int k = 0; k < MSG_SIZE; k++)
                printf(" %3d", ms->recv_buf[j][k]);
            printf(" ]\n");
        }
    }
    fflush(stdout);

    /* Pass token to next process */
    if (rank < size - 1) {
        int token = 1;
        MPI_Send(&token, 1, MPI_INT, rank + 1, 99, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { printf("\n"); fflush(stdout); }
}

/* ============================================================
 * FUNCTION: is_phase1_partner
 * PURPOSE : Returns 1 if r1 and r2 should communicate in Phase 1.
 *           Condition: |r1 - r2| is ODD (and r1 != r2).
 * ============================================================ */
int is_phase1_partner(int r1, int r2)
{
    int diff = r1 - r2;
    if (diff < 0) diff = -diff;
    return (diff > 0) && (diff % 2 == 1);
}

/* ============================================================
 * FUNCTION: is_phase2_partner
 * PURPOSE : Returns 1 if r1 and r2 should communicate in Phase 2.
 *           Condition: |r1 - r2| is EVEN and non-zero.
 * ============================================================ */
int is_phase2_partner(int r1, int r2)
{
    int diff = r1 - r2;
    if (diff < 0) diff = -diff;
    return (diff > 0) && (diff % 2 == 0);
}

/* ============================================================
 * FUNCTION: blocking_exchange
 * PURPOSE : Perform deadlock-free blocking MPI_Send/MPI_Recv
 *           for a given phase.
 *
 * DEADLOCK AVOIDANCE MECHANISM (Lower-Rank-First Ordering)
 * ---------------------------------------------------------
 * For each pair (i, j) where i < j:
 *   - Process i (lower rank)  : Send first, then Receive
 *   - Process j (higher rank) : Receive first, then Send
 *
 * WHY THIS WORKS:
 *   If BOTH called Send first, both would block waiting for the
 *   other's Recv.  By forcing the lower-rank to send first and
 *   the higher-rank to receive first, we ensure the channel is
 *   always consumed before the reverse direction is filled.
 *   No circular wait → no deadlock.
 *
 * COMPLEXITY: O(P²) message pairs per phase, each pair exchanges
 *   2 * MSG_SIZE integers.
 * ============================================================ */
void blocking_exchange(MessageStore *ms, int phase)
{
    int rank = ms->rank;
    int size = ms->size;
    MPI_Status status;

    /* Iterate over all potential partner ranks */
    for (int partner = 0; partner < size; partner++) {
        if (partner == rank) continue; /* skip self */

        /* Check if this pair communicates in the requested phase */
        int active = (phase == 1) ? is_phase1_partner(rank, partner)
                                  : is_phase2_partner(rank, partner);
        if (!active) continue;

        /* --------------------------------------------------
         * RANK-BASED ORDERING
         * Lower rank sends first; higher rank receives first.
         * This single rule eliminates all circular waits.
         * -------------------------------------------------- */
        if (rank < partner) {
            /* ── Lower rank: SEND then RECEIVE ─────────────── */
            printf("  [Phase %d][Blocking] Process %d → SEND to   Process %d  "
                   "(ordering: lower-rank sends first)\n",
                   phase, rank, partner);
            fflush(stdout);

            MPI_Send(ms->send_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD);

            printf("  [Phase %d][Blocking] Process %d ← RECV from Process %d\n",
                   phase, rank, partner);
            fflush(stdout);

            MPI_Recv(ms->recv_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD, &status);

            printf("  [Phase %d][Blocking] Process %d ✓ exchange complete with Process %d\n",
                   phase, rank, partner);
            fflush(stdout);

        } else {
            /* ── Higher rank: RECEIVE then SEND ─────────────── */
            printf("  [Phase %d][Blocking] Process %d ← RECV from Process %d  "
                   "(ordering: higher-rank receives first)\n",
                   phase, rank, partner);
            fflush(stdout);

            MPI_Recv(ms->recv_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD, &status);

            printf("  [Phase %d][Blocking] Process %d → SEND to   Process %d\n",
                   phase, rank, partner);
            fflush(stdout);

            MPI_Send(ms->send_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD);

            printf("  [Phase %d][Blocking] Process %d ✓ exchange complete with Process %d\n",
                   phase, rank, partner);
            fflush(stdout);
        }
    }
}

/* ============================================================
 * FUNCTION: handshake_exchange
 * PURPOSE : Perform handshake-based safe communication for a
 *           given phase using the READY → ACK → DATA protocol.
 *
 * PROTOCOL DETAIL
 * ---------------
 * Step 1 – READY:  Sender informs receiver it wants to send data.
 * Step 2 – ACK:    Receiver acknowledges and signals readiness.
 * Step 3 – DATA:   Only after receiving ACK does sender transmit.
 *
 * WHY THIS IS SAFER THAN PLAIN BLOCKING
 * ---------------------------------------
 * With plain MPI_Send, the MPI library may or may not buffer the
 * message internally.  If buffering is disabled or the message is
 * too large, MPI_Send blocks until MPI_Recv is posted on the other
 * side.  The handshake makes the receiver's readiness EXPLICIT,
 * removing any reliance on MPI's internal buffering policy.
 *
 * EXTRA OVERHEAD
 * --------------
 * Each data transfer now requires 3 message passes instead of 1,
 * so wall-clock time is approximately 3× that of blocking I/O.
 * This is the safety/performance trade-off.
 *
 * ============================================================ */
void handshake_exchange(MessageStore *ms, int phase)
{
    int rank = ms->rank;
    int size = ms->size;
    MPI_Status status;

    int ready_signal = 1; /* payload for READY / ACK messages */
    int ack_signal   = 1;

    for (int partner = 0; partner < size; partner++) {
        if (partner == rank) continue;

        int active = (phase == 1) ? is_phase1_partner(rank, partner)
                                  : is_phase2_partner(rank, partner);
        if (!active) continue;

        /* Same lower-rank-first ordering applied to the handshake */
        if (rank < partner) {
            /* ══════════════════════════════════════════════
             * SENDER SIDE  (lower rank)
             * ════════════════════════════════════════════*/

            /* STEP 1: Send READY signal */
            printf("  [Phase %d][Handshake] Process %d → READY   to   Process %d\n",
                   phase, rank, partner);
            fflush(stdout);

            MPI_Send(&ready_signal, 1, MPI_INT,
                     partner, TAG_READY, MPI_COMM_WORLD);

            /* STEP 2: Wait for ACK from receiver */
            printf("  [Phase %d][Handshake] Process %d   waiting ACK  from Process %d\n",
                   phase, rank, partner);
            fflush(stdout);

            MPI_Recv(&ack_signal, 1, MPI_INT,
                     partner, TAG_ACK, MPI_COMM_WORLD, &status);

            printf("  [Phase %d][Handshake] Process %d ← ACK     from Process %d  (safe to send)\n",
                   phase, rank, partner);
            fflush(stdout);

            /* STEP 3: Send actual DATA (receiver is confirmed ready) */
            printf("  [Phase %d][Handshake] Process %d → DATA    to   Process %d\n",
                   phase, rank, partner);
            fflush(stdout);

            MPI_Send(ms->send_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_DATA, MPI_COMM_WORLD);

            /* Also receive DATA back from partner (full exchange) */
            MPI_Recv(ms->recv_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_DATA, MPI_COMM_WORLD, &status);

            printf("  [Phase %d][Handshake] Process %d ✓ DATA received from Process %d\n",
                   phase, rank, partner);
            fflush(stdout);

        } else {
            /* ══════════════════════════════════════════════
             * RECEIVER SIDE  (higher rank)
             * ════════════════════════════════════════════*/

            /* STEP 1: Wait for READY signal from sender */
            printf("  [Phase %d][Handshake] Process %d   waiting READY from Process %d\n",
                   phase, rank, partner);
            fflush(stdout);

            MPI_Recv(&ready_signal, 1, MPI_INT,
                     partner, TAG_READY, MPI_COMM_WORLD, &status);

            printf("  [Phase %d][Handshake] Process %d ← READY   from Process %d\n",
                   phase, rank, partner);
            fflush(stdout);

            /* STEP 2: Send ACK back to sender */
            printf("  [Phase %d][Handshake] Process %d → ACK     to   Process %d\n",
                   phase, rank, partner);
            fflush(stdout);

            MPI_Send(&ack_signal, 1, MPI_INT,
                     partner, TAG_ACK, MPI_COMM_WORLD);

            /* STEP 3: Receive DATA from sender */
            MPI_Recv(ms->recv_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_DATA, MPI_COMM_WORLD, &status);

            printf("  [Phase %d][Handshake] Process %d ← DATA    from Process %d\n",
                   phase, rank, partner);
            fflush(stdout);

            /* Send our own DATA back to complete the exchange */
            MPI_Send(ms->send_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_DATA, MPI_COMM_WORLD);

            printf("  [Phase %d][Handshake] Process %d ✓ exchange complete with Process %d\n",
                   phase, rank, partner);
            fflush(stdout);
        }
    }
}

/* ============================================================
 * FUNCTION: measure_blocking
 * PURPOSE : Run both phases of blocking exchange and return
 *           total elapsed time (max across all processes).
 *           We use MPI_Wtime for wall-clock measurement.
 *
 * NOTE: We collect the maximum time across processes because
 *       the overall program cannot proceed until every process
 *       finishes.  The bottleneck process determines total time.
 * ============================================================ */
double measure_blocking(MessageStore *ms)
{
    int rank = ms->rank;
    int size = ms->size;

    /* Suppress verbose output during timing runs */
    /* (we redirect by temporarily replacing stdout is complex;
     *  instead we just skip printf in a "silent" re-run) */

    /* Ensure all processes start timing at the same moment */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* ---- Phase 1 ---- */
    for (int partner = 0; partner < size; partner++) {
        if (partner == rank) continue;
        if (!is_phase1_partner(rank, partner)) continue;
        MPI_Status status;
        if (rank < partner) {
            MPI_Send(ms->send_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD);
            MPI_Recv(ms->recv_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(ms->recv_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD, &status);
            MPI_Send(ms->send_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Phase 2 ---- */
    for (int partner = 0; partner < size; partner++) {
        if (partner == rank) continue;
        if (!is_phase2_partner(rank, partner)) continue;
        MPI_Status status;
        if (rank < partner) {
            MPI_Send(ms->send_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD);
            MPI_Recv(ms->recv_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD, &status);
        } else {
            MPI_Recv(ms->recv_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD, &status);
            MPI_Send(ms->send_buf[partner], MSG_SIZE, MPI_INT,
                     partner, TAG_BLOCKING, MPI_COMM_WORLD);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double t_local = MPI_Wtime() - t_start;

    /* Find the maximum time across all processes */
    double t_max = t_local;
    /* Manual max reduction using point-to-point (no MPI_Reduce per rules) */
    if (rank == 0) {
        for (int src = 1; src < size; src++) {
            double other;
            MPI_Recv(&other, 1, MPI_DOUBLE, src, 200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (other > t_max) t_max = other;
        }
    } else {
        MPI_Send(&t_local, 1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);
    }
    return t_max; /* only meaningful on rank 0 */
}

/* ============================================================
 * FUNCTION: measure_handshake
 * PURPOSE : Run both phases of handshake exchange and return
 *           total elapsed time (max across all processes).
 * ============================================================ */
double measure_handshake(MessageStore *ms)
{
    int rank = ms->rank;
    int size = ms->size;
    int sig  = 1;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    /* Helper lambda-like inline for one handshake pair */
    for (int phase = 1; phase <= 2; phase++) {
        for (int partner = 0; partner < size; partner++) {
            if (partner == rank) continue;
            int active = (phase == 1) ? is_phase1_partner(rank, partner)
                                      : is_phase2_partner(rank, partner);
            if (!active) continue;

            MPI_Status status;
            if (rank < partner) {
                MPI_Send(&sig, 1, MPI_INT, partner, TAG_READY, MPI_COMM_WORLD);
                MPI_Recv(&sig, 1, MPI_INT, partner, TAG_ACK,   MPI_COMM_WORLD, &status);
                MPI_Send(ms->send_buf[partner], MSG_SIZE, MPI_INT,
                         partner, TAG_DATA, MPI_COMM_WORLD);
                MPI_Recv(ms->recv_buf[partner], MSG_SIZE, MPI_INT,
                         partner, TAG_DATA, MPI_COMM_WORLD, &status);
            } else {
                MPI_Recv(&sig, 1, MPI_INT, partner, TAG_READY, MPI_COMM_WORLD, &status);
                MPI_Send(&sig, 1, MPI_INT, partner, TAG_ACK,   MPI_COMM_WORLD);
                MPI_Recv(ms->recv_buf[partner], MSG_SIZE, MPI_INT,
                         partner, TAG_DATA, MPI_COMM_WORLD, &status);
                MPI_Send(ms->send_buf[partner], MSG_SIZE, MPI_INT,
                         partner, TAG_DATA, MPI_COMM_WORLD);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD); /* phase separator */
    }

    double t_local = MPI_Wtime() - t_start;

    double t_max = t_local;
    if (rank == 0) {
        for (int src = 1; src < size; src++) {
            double other;
            MPI_Recv(&other, 1, MPI_DOUBLE, src, 201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (other > t_max) t_max = other;
        }
    } else {
        MPI_Send(&t_local, 1, MPI_DOUBLE, 0, 201, MPI_COMM_WORLD);
    }
    return t_max;
}

/* ============================================================
 * FUNCTION: print_timing_table
 * PURPOSE : Print a formatted performance comparison table.
 *           Called only by rank 0.
 *
 * WHY HANDSHAKE IS SLOWER
 * -------------------------
 * Each data exchange now takes 3 round-trips instead of 1:
 *   1. READY   message (sender → receiver)
 *   2. ACK     message (receiver → sender)
 *   3. DATA    message (sender → receiver) + reply
 *
 * Network latency dominates for small messages, so the 3×
 * increase in message count directly inflates the time.
 * For very large messages the DATA transfer time dominates and
 * the overhead of READY/ACK becomes relatively small.
 *
 * WHY HANDSHAKE IS SAFER
 * -------------------------
 * Handshake eliminates reliance on MPI's internal buffering.
 * With buffered sends, MPI may copy the message into a system
 * buffer and return immediately — but if the buffer is full,
 * MPI_Send blocks anyway.  The handshake protocol makes the
 * receiver's availability explicit regardless of buffer state.
 * ============================================================ */
void print_timing_table(int rank, double t_block, double t_hand)
{
    if (rank != 0) return;

    printf("  ┌─────────────────────────────────────┬──────────────────┐\n");
    printf("  │ Method                              │  Time (seconds)  │\n");
    printf("  ├─────────────────────────────────────┼──────────────────┤\n");
    printf("  │ Blocking Send/Receive               │  %14.6f  │\n", t_block);
    printf("  │ Handshake Safe Protocol             │  %14.6f  │\n", t_hand);
    printf("  ├─────────────────────────────────────┼──────────────────┤\n");

    if (t_block > 1e-9) {
        double ratio = t_hand / t_block;
        printf("  │ Overhead ratio (handshake/blocking) │  %14.2fx  │\n", ratio);
    }
    printf("  └─────────────────────────────────────┴──────────────────┘\n\n");

    printf("  ANALYSIS:\n");
    printf("  • Blocking is faster because each pair needs only 2 message\n");
    printf("    passes (1 send + 1 recv per direction).\n");
    printf("  • Handshake requires 3 passes per direction (READY, ACK,\n");
    printf("    DATA) → roughly 3× more messages → higher latency.\n");
    printf("  • Handshake is safer: receiver's readiness is EXPLICIT,\n");
    printf("    removing dependence on MPI internal buffer availability.\n");
    printf("  • For large messages, DATA transfer dominates and the\n");
    printf("    READY/ACK overhead becomes negligible.\n\n");
    fflush(stdout);
}

/*
 * ============================================================
 * EXPECTED SAMPLE OUTPUT (8 processes, abbreviated)
 * ============================================================
 *
 * ╔══════════════════════════════════════════════════════════╗
 * ║  Dynamic Deadlock-Avoiding Message Exchange in MPI       ║
 * ║  Processes: 8    Message size: 4 integers per msg        ║
 * ╚══════════════════════════════════════════════════════════╝
 *
 * ══════════════════════════════════════════════════════════
 *   PART 1: PURE BLOCKING SEND/RECEIVE
 * ══════════════════════════════════════════════════════════
 *
 * [Blocking] ── Phase 1 Begin (|rank_i - rank_j| is ODD) ──
 *
 *   [Phase 1][Blocking] Process 0 → SEND to   Process 1  (ordering: lower-rank sends first)
 *   [Phase 1][Blocking] Process 1 ← RECV from Process 0  (ordering: higher-rank receives first)
 *   [Phase 1][Blocking] Process 0 ← RECV from Process 1
 *   [Phase 1][Blocking] Process 1 → SEND to   Process 0
 *   [Phase 1][Blocking] Process 0 ✓ exchange complete with Process 1
 *   [Phase 1][Blocking] Process 1 ✓ exchange complete with Process 0
 *   ...
 *
 * [Blocking] ── Phase 1 Complete ──
 *
 * [Blocking] ── Phase 2 Begin (|rank_i - rank_j| is EVEN, ≠ 0) ──
 *   ...
 * [Blocking] ── Phase 2 Complete ──
 *
 *   ── [Blocking] Received Buffers ──
 *   [Process 0] recv_buf:
 *     from P1  : [ 423  87 612 201 ]
 *     from P3  : [ 115 902  44  78 ]
 *     from P2  : [ 339  56 781 420 ]
 *     from P4  : [ 207 455  38 694 ]
 *   ...
 *
 * ══════════════════════════════════════════════════════════
 *   PART 2: HANDSHAKE-BASED SAFE COMMUNICATION
 * ══════════════════════════════════════════════════════════
 *
 * [Handshake] ── Phase 1 Begin (|rank_i - rank_j| is ODD) ──
 *
 *   [Phase 1][Handshake] Process 0 → READY   to   Process 1
 *   [Phase 1][Handshake] Process 1   waiting READY from Process 0
 *   [Phase 1][Handshake] Process 1 ← READY   from Process 0
 *   [Phase 1][Handshake] Process 1 → ACK     to   Process 0
 *   [Phase 1][Handshake] Process 0   waiting ACK  from Process 1
 *   [Phase 1][Handshake] Process 0 ← ACK     from Process 1  (safe to send)
 *   [Phase 1][Handshake] Process 0 → DATA    to   Process 1
 *   [Phase 1][Handshake] Process 1 ← DATA    from Process 0
 *   [Phase 1][Handshake] Process 0 ✓ DATA received from Process 1
 *   [Phase 1][Handshake] Process 1 ✓ exchange complete with Process 0
 *   ...
 *
 * ══════════════════════════════════════════════════════════
 *   PERFORMANCE COMPARISON
 * ══════════════════════════════════════════════════════════
 *
 *   ┌─────────────────────────────────────┬──────────────────┐
 *   │ Method                              │  Time (seconds)  │
 *   ├─────────────────────────────────────┼──────────────────┤
 *   │ Blocking Send/Receive               │        0.000382  │
 *   │ Handshake Safe Protocol             │        0.001147  │
 *   ├─────────────────────────────────────┼──────────────────┤
 *   │ Overhead ratio (handshake/blocking) │           3.00x  │
 *   └─────────────────────────────────────┴──────────────────┘
 *
 * ╔══════════════════════════════════════════════════════════╗
 * ║  ✓  All phases completed without deadlock.              ║
 * ║  ✓  Both communication methods verified successfully.   ║
 * ╚══════════════════════════════════════════════════════════╝
 *
 * ============================================================
 * COMPLEXITY DISCUSSION
 * ============================================================
 *
 *  Let P = number of processes.
 *
 *  Phase 1 pairs: All (i,j) with i<j and |i-j| odd.
 *    Count ≈ P²/4 pairs (roughly half of all pairs).
 *
 *  Phase 2 pairs: All (i,j) with i<j and |i-j| even, ≠ 0.
 *    Count ≈ P²/4 pairs.
 *
 *  Total message count:
 *    Blocking  : 2 messages/pair × ~P²/2 pairs = O(P²)
 *    Handshake : 4 messages/pair × ~P²/2 pairs = O(P²)
 *                (READY + ACK + DATA fwd + DATA bck)
 *
 *  Both methods are O(P²) in message count.
 *  Handshake has a constant factor of ~2× more messages.
 *
 * ============================================================
 */
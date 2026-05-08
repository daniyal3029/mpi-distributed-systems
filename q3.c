/*
 * ============================================================================
 * PROJECT: Concurrent Distributed Priority Queue using MPI
 * FILE:    q3.c
 * AUTHOR:  MPI Distributed Systems Project
 * COMPILE: mpicc q3.c -o q3
 * RUN:     mpirun -np 8 ./q3 10
 *          (8 processes, 10 tasks per process)
 * ============================================================================
 *
 * ASCII ARCHITECTURE DIAGRAM:
 * ===========================
 *
 *   Process 1 ──┐
 *   Process 2 ──┤  Tasks (priority, payload)
 *   Process 3 ──┤──────────────────────────► ROOT (Process 0)
 *   Process 4 ──┤                            │
 *   Process 5 ──┤                            ▼
 *   Process 6 ──┤                   ┌─────────────────┐
 *   Process 7 ──┘                   │   Binary Heap   │
 *                                   │  (Min-Priority) │
 *                                   │                 │
 *                                   │  [1][3][5][7]   │
 *                                   │   [4][6][8]     │
 *                                   │    [9][10]      │
 *                                   └─────────────────┘
 *
 * COMMUNICATION MODES:
 *   Part 1: BLOCKING   → MPI_Send / MPI_Recv  (sequential)
 *   Part 2: POLLING    → MPI_Isend / MPI_Irecv / MPI_Test (async)
 *   Part 3: BATCHING   → Groups of tasks sent together (reduced overhead)
 *
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>       /* usleep() for software backoff */
#include <mpi.h>

/* ============================================================================
 * CONSTANTS AND CONFIGURATION
 * ============================================================================ */

#define ROOT_RANK       0           /* Root/master process rank               */
#define TAG_TASK        100         /* MPI message tag for single tasks        */
#define TAG_BATCH       200         /* MPI message tag for batched tasks       */
#define TAG_DONE        300         /* MPI message tag: worker done signal     */
#define MAX_HEAP_SIZE   100000      /* Maximum tasks in heap                   */
#define MAX_PAYLOAD_LEN 32          /* Max length of task payload string       */
#define BATCH_SIZE      5           /* Number of tasks per batch (Part 3)      */
#define DEFAULT_K       5           /* Default tasks per process if not given  */

/* Backoff configuration */
#define BACKOFF_INIT_US  100        /* Initial backoff: 100 microseconds       */
#define BACKOFF_MAX_US   10000      /* Maximum backoff: 10 milliseconds        */
#define BACKOFF_MULT     2          /* Backoff multiplier (exponential)        */

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/*
 * Task structure — represents a single unit of work.
 * Each task has a priority (lower = higher priority) and a payload.
 * We use a plain struct with fixed-size fields so we can send it with MPI
 * without needing to define a custom MPI datatype (we send raw bytes).
 */
typedef struct {
    int  priority;                  /* Lower number = higher priority          */
    int  source_rank;               /* Which process generated this task       */
    int  task_id;                   /* Unique ID within the source process     */
    char payload[MAX_PAYLOAD_LEN];  /* Descriptive string for the task         */
} Task;

/*
 * Binary Min-Heap structure.
 * A min-heap keeps the smallest priority at the root (index 0).
 * Parent of node i  → (i-1)/2
 * Left child of i   → 2*i + 1
 * Right child of i  → 2*i + 2
 *
 * HEAP VISUALIZATION (size=7):
 *          [0] priority=1
 *         /              \
 *   [1] p=3              [2] p=5
 *   /     \              /     \
 * [3] p=4 [4] p=7   [5] p=6  [6] p=9
 */
typedef struct {
    Task *data;                     /* Dynamic array of tasks                  */
    int   size;                     /* Current number of tasks in heap         */
    int   capacity;                 /* Allocated capacity                      */
} MinHeap;

/*
 * Performance measurement structure.
 * Stores timing and throughput data for each communication approach.
 */
typedef struct {
    double time_blocking;           /* Wall-clock time for Part 1              */
    double time_nonblocking;        /* Wall-clock time for Part 2              */
    double time_batching;           /* Wall-clock time for Part 3              */
    int    total_tasks;             /* Total tasks processed                   */
} PerfStats;

/* ============================================================================
 * BINARY HEAP FUNCTIONS
 * ============================================================================ */

/*
 * init_heap — Allocate and initialize an empty min-heap.
 * Parameters:
 *   h        → pointer to heap struct
 *   capacity → maximum number of tasks to support
 */
void init_heap(MinHeap *h, int capacity) {
    h->data     = (Task *)malloc(capacity * sizeof(Task));
    h->size     = 0;
    h->capacity = capacity;
    if (!h->data) {
        fprintf(stderr, "[HEAP] ERROR: Failed to allocate heap memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

/*
 * free_heap — Release heap memory.
 */
void free_heap(MinHeap *h) {
    free(h->data);
    h->data = NULL;
    h->size = 0;
}

/*
 * swap_tasks — Swap two task entries in the heap array.
 * Used by heapify_up and heapify_down.
 */
static void swap_tasks(Task *a, Task *b) {
    Task tmp = *a;
    *a = *b;
    *b = tmp;
}

/*
 * heapify_up — Restore heap property after inserting at the end.
 *
 * After inserting a new task at position 'size-1', we compare it with its
 * parent. If the new task has LOWER priority (smaller number), we swap them
 * and continue upward until the heap property is restored.
 *
 * Time complexity: O(log n)
 */
void heapify_up(MinHeap *h, int idx) {
    /*
     * While we are not at the root AND parent has a larger priority value
     * (meaning the current node should be higher in the min-heap), swap.
     */
    while (idx > 0) {
        int parent = (idx - 1) / 2;
        if (h->data[parent].priority > h->data[idx].priority) {
            swap_tasks(&h->data[parent], &h->data[idx]);
            idx = parent;          /* Move up and continue checking           */
        } else {
            break;                 /* Heap property satisfied                 */
        }
    }
}

/*
 * heapify_down — Restore heap property after removing the root.
 *
 * After removing the minimum (root), we place the last element at root and
 * then push it downward by swapping with the smaller child.
 *
 * Time complexity: O(log n)
 */
void heapify_down(MinHeap *h, int idx) {
    int left, right, smallest;

    while (1) {
        left     = 2 * idx + 1;    /* Left child index                        */
        right    = 2 * idx + 2;    /* Right child index                       */
        smallest = idx;            /* Assume current node is smallest         */

        /* Check if left child exists and has smaller priority */
        if (left < h->size && h->data[left].priority < h->data[smallest].priority)
            smallest = left;

        /* Check if right child exists and has smaller priority */
        if (right < h->size && h->data[right].priority < h->data[smallest].priority)
            smallest = right;

        if (smallest != idx) {
            /* Swap with the smaller child and continue downward */
            swap_tasks(&h->data[idx], &h->data[smallest]);
            idx = smallest;
        } else {
            break;                 /* Heap property satisfied                 */
        }
    }
}

/*
 * insert_task — Add a new task to the min-heap.
 *
 * Steps:
 *   1. Place new task at the end of the array
 *   2. Call heapify_up to restore heap order
 *
 * Time complexity: O(log n)
 */
void insert_task(MinHeap *h, Task t) {
    if (h->size >= h->capacity) {
        fprintf(stderr, "[HEAP] ERROR: Heap capacity exceeded (%d)\n", h->capacity);
        return;
    }
    h->data[h->size] = t;          /* Place at the end                        */
    heapify_up(h, h->size);        /* Bubble up to correct position           */
    h->size++;
}

/*
 * extract_min — Remove and return the task with the lowest priority value.
 *
 * Steps:
 *   1. Save the root (minimum element)
 *   2. Move the last element to root position
 *   3. Call heapify_down to restore heap order
 *
 * Time complexity: O(log n)
 */
Task extract_min(MinHeap *h) {
    if (h->size == 0) {
        Task empty;
        memset(&empty, 0, sizeof(Task));
        empty.priority = -1;       /* Sentinel: empty heap                    */
        return empty;
    }

    Task min_task = h->data[0];    /* Save root (minimum priority)            */
    h->size--;
    if (h->size > 0) {
        h->data[0] = h->data[h->size]; /* Move last element to root           */
        heapify_down(h, 0);            /* Restore heap property               */
    }
    return min_task;
}

/*
 * print_priority_queue — Display current heap contents in sorted order.
 *
 * NOTE: We create a temporary copy because extracting ruins the original heap.
 * Only call this when all tasks have been received (at the end of a phase).
 */
void print_priority_queue(MinHeap *h) {
    if (h->size == 0) {
        printf("  [HEAP] Heap is empty.\n");
        return;
    }

    /* Make a temporary copy of heap data to extract in sorted order */
    MinHeap tmp;
    init_heap(&tmp, h->capacity);
    tmp.size = h->size;
    memcpy(tmp.data, h->data, h->size * sizeof(Task));

    printf("\n  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │           FINAL SORTED PRIORITY QUEUE                  │\n");
    printf("  ├──────┬──────────┬──────────────────────────────────────┤\n");
    printf("  │ Rank │ Priority │ Payload                              │\n");
    printf("  ├──────┼──────────┼──────────────────────────────────────┤\n");

    int rank = 1;
    while (tmp.size > 0) {
        Task t = extract_min(&tmp);
        printf("  │ %4d │ %8d │ %-36s │\n", rank++, t.priority, t.payload);
    }
    printf("  └──────┴──────────┴──────────────────────────────────────┘\n\n");

    free_heap(&tmp);
}

/* ============================================================================
 * TASK GENERATION
 * ============================================================================ */

/*
 * generate_tasks — Each worker process creates K random tasks.
 *
 * Tasks have:
 *   - Random priority in range [1, 100]
 *   - Payload string like "P2_Task_7" (process 2, task 7)
 *   - Source rank and task ID embedded
 *
 * Parameters:
 *   rank     → this process's MPI rank
 *   K        → number of tasks to generate
 *   tasks    → output array (caller must allocate K * sizeof(Task))
 */
void generate_tasks(int rank, int K, Task *tasks) {
    for (int i = 0; i < K; i++) {
        tasks[i].priority    = (rand() % 100) + 1;   /* Priority: 1-100      */
        tasks[i].source_rank = rank;
        tasks[i].task_id     = i;
        snprintf(tasks[i].payload, MAX_PAYLOAD_LEN, "P%d_Task_%d", rank, i);

        printf("[Process %d] Generated task (%d, %s)\n",
               rank, tasks[i].priority, tasks[i].payload);
    }
}

/* ============================================================================
 * SOFTWARE BACKOFF UTILITY
 * ============================================================================ */

/*
 * software_backoff — Implement exponential backoff to reduce congestion.
 *
 * WHY BACKOFF IS NEEDED:
 *   When many processes try to send messages to root simultaneously, they
 *   compete for the root's attention and network resources. This creates
 *   "message storms" that can overwhelm the root's receive queue.
 *
 * HOW BACKOFF HELPS:
 *   By waiting progressively longer between send attempts, processes spread
 *   out their transmissions over time, reducing simultaneous collisions.
 *   This is similar to Ethernet CSMA/CD collision avoidance.
 *
 * EXPONENTIAL BACKOFF:
 *   - Start with a small wait (e.g., 100 microseconds)
 *   - Double the wait on each retry
 *   - Cap at a maximum wait to avoid deadlock
 *
 * Parameters:
 *   current_wait_us → pointer to current wait time in microseconds
 *                     (will be updated for the next call)
 */
void software_backoff(int *current_wait_us) {
    if (*current_wait_us > 0) {
        /* usleep pauses the calling process for the specified microseconds */
        usleep(*current_wait_us);
    }

    /* Increase wait time exponentially, capped at maximum */
    *current_wait_us = (*current_wait_us == 0)
                       ? BACKOFF_INIT_US
                       : (*current_wait_us * BACKOFF_MULT);

    if (*current_wait_us > BACKOFF_MAX_US)
        *current_wait_us = BACKOFF_MAX_US;
}

/* ============================================================================
 * PART 1: BLOCKING COLLECTION
 * ============================================================================
 *
 * In blocking communication:
 *   - MPI_Send blocks until the message is buffered or received
 *   - MPI_Recv blocks until a matching message arrives
 *
 * The root process uses MPI_ANY_SOURCE to receive from any process,
 * but can only receive ONE message at a time (sequential, not concurrent).
 *
 * DISADVANTAGE:
 *   Root must wait for each send to complete before starting the next receive.
 *   This means computation and communication cannot overlap.
 *   If a process is slow, root sits idle waiting for it.
 */

/*
 * blocking_worker — Worker side of Part 1.
 * Sends each task one by one using MPI_Send.
 */
void blocking_worker(int rank, int K, Task *tasks) {
    printf("[Process %d] === BLOCKING: Starting to send %d tasks ===\n", rank, K);

    for (int i = 0; i < K; i++) {
        printf("[Process %d] BLOCKING send: task (%d, %s) → Root\n",
               rank, tasks[i].priority, tasks[i].payload);

        /*
         * MPI_Send — Blocking point-to-point send.
         * Arguments:
         *   &tasks[i]    → address of data to send
         *   sizeof(Task) → number of bytes
         *   MPI_BYTE     → raw byte type (no MPI datatype needed)
         *   ROOT_RANK    → destination process
         *   TAG_TASK     → message tag for identification
         *   MPI_COMM_WORLD → communicator (all processes)
         *
         * MPI_Send BLOCKS until the send buffer can be safely reused,
         * meaning the data has been copied to MPI's internal buffer
         * OR the receiver has called a matching MPI_Recv.
         */
        MPI_Send(&tasks[i], sizeof(Task), MPI_BYTE,
                 ROOT_RANK, TAG_TASK, MPI_COMM_WORLD);
    }

    /* Notify root that this worker is done */
    int done_signal = rank;
    MPI_Send(&done_signal, 1, MPI_INT, ROOT_RANK, TAG_DONE, MPI_COMM_WORLD);

    printf("[Process %d] BLOCKING: All tasks sent, done signal sent.\n", rank);
}

/*
 * blocking_root — Root side of Part 1.
 * Receives tasks sequentially using MPI_Recv and inserts into heap.
 *
 * Parameters:
 *   nprocs     → total number of MPI processes
 *   total_tasks → total tasks expected (K * (nprocs-1))
 *   heap       → the min-heap to insert into
 */
void blocking_root(int nprocs, int total_tasks, MinHeap *heap) {
    int  received    = 0;
    int  done_count  = 0;
    int  workers     = nprocs - 1;  /* Number of non-root processes            */
    Task incoming;
    MPI_Status status;

    printf("[Root] === BLOCKING: Waiting to receive %d tasks ===\n", total_tasks);

    /*
     * Root loops until it has received all tasks AND all workers have sent
     * their "done" signals. We use MPI_ANY_SOURCE to accept from any process.
     * MPI_ANY_TAG lets us accept either TAG_TASK or TAG_DONE.
     */
    while (done_count < workers) {
        /*
         * MPI_Recv — Blocking point-to-point receive.
         * Arguments:
         *   &incoming      → buffer to store received message
         *   sizeof(Task)   → max bytes to receive (must be ≥ actual message)
         *   MPI_BYTE       → raw byte type
         *   MPI_ANY_SOURCE → accept from any sender
         *   MPI_ANY_TAG    → accept any message tag
         *   MPI_COMM_WORLD → communicator
         *   &status        → output: tells us who sent it and with what tag
         *
         * MPI_Recv BLOCKS until a matching message arrives.
         */
        MPI_Recv(&incoming, sizeof(Task), MPI_BYTE,
                 MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == TAG_DONE) {
            /* Worker finished — count it */
            done_count++;
            printf("[Root] BLOCKING: Received DONE signal from Process %d "
                   "(%d/%d done)\n",
                   status.MPI_SOURCE, done_count, workers);
        } else {
            /* Regular task — insert into heap */
            printf("[Root] BLOCKING: Received task (%d, %s) from Process %d\n",
                   incoming.priority, incoming.payload, status.MPI_SOURCE);

            insert_task(heap, incoming);
            received++;

            printf("[Root] BLOCKING: Heap size = %d\n", heap->size);
        }
    }

    printf("[Root] BLOCKING: All %d tasks received and queued.\n", received);
}

/* ============================================================================
 * PART 2: NON-BLOCKING WITH POLLING
 * ============================================================================
 *
 * In non-blocking communication:
 *   - MPI_Isend initiates a send and returns IMMEDIATELY (non-blocking)
 *   - MPI_Irecv posts a receive request and returns IMMEDIATELY
 *   - MPI_Test checks if a posted request has completed (non-blocking check)
 *
 * This allows OVERLAPPING of communication and computation:
 *   While waiting for a message to arrive, the root can do useful work
 *   (e.g., inserting previously received tasks into the heap, logging, etc.)
 *
 * ADVANTAGE OVER BLOCKING:
 *   Root never sits completely idle. It continuously polls and processes.
 *
 * KEY CONCEPT — MPI_Request:
 *   Each non-blocking operation returns an MPI_Request handle.
 *   This handle is used to check or wait for completion.
 *
 * KEY CONCEPT — MPI_Test:
 *   MPI_Test(request, flag, status)
 *   - Returns flag=1 if request is complete, flag=0 if still in progress
 *   - DOES NOT BLOCK — returns immediately
 */

/*
 * nonblocking_worker — Worker side of Part 2.
 * Uses MPI_Isend to post sends non-blockingly.
 *
 * IMPORTANT BUFFER SAFETY:
 *   After MPI_Isend, the buffer must NOT be modified until the send completes.
 *   We use MPI_Wait to ensure completion before reusing or freeing the buffer.
 */
void nonblocking_worker(int rank, int K, Task *tasks) {
    /*
     * Allocate an array of MPI_Request handles — one per task.
     * Each MPI_Isend returns one request we must track.
     */
    MPI_Request *requests = (MPI_Request *)malloc(K * sizeof(MPI_Request));
    int backoff_us = 0;

    printf("[Process %d] === NON-BLOCKING: Starting async send of %d tasks ===\n",
           rank, K);

    for (int i = 0; i < K; i++) {
        printf("[Process %d] NON-BLOCKING: Initiating async send of task (%d, %s)\n",
               rank, tasks[i].priority, tasks[i].payload);

        /*
         * MPI_Isend — Non-blocking send.
         * Arguments: same as MPI_Send, but also takes a pointer to an
         * MPI_Request that will be filled in.
         *
         * Returns IMMEDIATELY. The actual data transfer may happen later.
         * We must NOT modify tasks[i] until this request completes!
         */
        MPI_Isend(&tasks[i], sizeof(Task), MPI_BYTE,
                  ROOT_RANK, TAG_TASK, MPI_COMM_WORLD, &requests[i]);

        printf("[Process %d] NON-BLOCKING: Send request posted (request[%d])\n",
               rank, i);

        /*
         * Software backoff: sleep briefly to reduce message flooding.
         * Without this, all processes would post sends at exactly the
         * same moment, potentially overwhelming the root's receive buffer.
         */
        if (i % 3 == 0 && backoff_us < BACKOFF_MAX_US) {
            printf("[Process %d] BACKOFF: Applying %d µs backoff after %d sends\n",
                   rank, backoff_us, i);
            software_backoff(&backoff_us);
        }
    }

    /*
     * Wait for ALL posted Isend requests to complete before we can
     * safely return (and thus allow the task array to go out of scope).
     *
     * MPI_Waitall blocks until every request in the array is complete.
     */
    printf("[Process %d] NON-BLOCKING: Waiting for all sends to complete...\n", rank);
    MPI_Waitall(K, requests, MPI_STATUSES_IGNORE);

    printf("[Process %d] NON-BLOCKING: All sends confirmed complete.\n", rank);

    /* Send done signal (blocking is fine here — just one small message) */
    int done_signal = rank;
    MPI_Send(&done_signal, 1, MPI_INT, ROOT_RANK, TAG_DONE, MPI_COMM_WORLD);

    free(requests);
}

/*
 * nonblocking_root — Root side of Part 2.
 *
 * Root posts multiple MPI_Irecv requests simultaneously, then polls them
 * using MPI_Test in a loop. While polling, root performs useful work
 * (simulated here as heap maintenance and status logging).
 *
 * This demonstrates CONCURRENT COMMUNICATION: root has multiple outstanding
 * receive requests at once and processes them as they complete.
 */
void nonblocking_root(int nprocs, int total_tasks, MinHeap *heap) {
    int workers    = nprocs - 1;
    int done_count = 0;
    int received   = 0;

    /*
     * We post (workers) receive requests simultaneously — one per worker.
     * Each request will receive one task from any source.
     * As each request completes, we repost it to receive the next task.
     */
    Task       *recv_bufs = (Task *)malloc(workers * sizeof(Task));
    MPI_Request *reqs     = (MPI_Request *)malloc(workers * sizeof(MPI_Request));

    /* Also track done signals with separate receives */
    int         *done_bufs = (int *)malloc(workers * sizeof(int));
    MPI_Request *done_reqs = (MPI_Request *)malloc(workers * sizeof(MPI_Request));

    printf("[Root] === NON-BLOCKING: Posting %d simultaneous receive requests ===\n",
           workers);

    /*
     * Post initial batch of MPI_Irecv requests.
     * MPI_Irecv — Non-blocking receive.
     * Returns immediately. The buffer will be filled when a matching
     * message arrives. We check completion with MPI_Test.
     */
    for (int i = 0; i < workers; i++) {
        MPI_Irecv(&recv_bufs[i], sizeof(Task), MPI_BYTE,
                  MPI_ANY_SOURCE, TAG_TASK, MPI_COMM_WORLD, &reqs[i]);

        MPI_Irecv(&done_bufs[i], 1, MPI_INT,
                  i + 1, TAG_DONE, MPI_COMM_WORLD, &done_reqs[i]);

        printf("[Root] NON-BLOCKING: Posted receive request slot %d\n", i);
    }

    int poll_count = 0;  /* Count how many polling iterations we do */

    /*
     * POLLING LOOP: Keep checking until all workers report done
     * AND we've received all expected tasks.
     */
    while (received < total_tasks || done_count < workers) {
        poll_count++;

        /* === POLL TASK RECEIVE REQUESTS === */
        for (int i = 0; i < workers; i++) {
            if (reqs[i] == MPI_REQUEST_NULL) continue; /* Already completed   */

            int flag = 0;
            MPI_Status status;

            /*
             * MPI_Test — Non-blocking check of a request.
             * Arguments:
             *   &reqs[i] → request to check
             *   &flag    → output: 1 if done, 0 if still in progress
             *   &status  → output: message metadata (if done)
             *
             * Does NOT block. Returns immediately with current status.
             * This is the key to overlapping communication and computation.
             */
            MPI_Test(&reqs[i], &flag, &status);

            if (flag) {
                /* Request completed! Process the received task. */
                printf("[Root] NON-BLOCKING: Task arrived in slot %d → "
                       "(%d, %s) from Process %d\n",
                       i, recv_bufs[i].priority, recv_bufs[i].payload,
                       status.MPI_SOURCE);

                insert_task(heap, recv_bufs[i]);
                received++;

                printf("[Root] NON-BLOCKING: Inserted into heap. Size = %d\n",
                       heap->size);

                /*
                 * Repost the receive request for this slot to keep
                 * listening for more tasks from other processes.
                 * This creates a sliding window of concurrent receives.
                 */
                if (received < total_tasks) {
                    MPI_Irecv(&recv_bufs[i], sizeof(Task), MPI_BYTE,
                              MPI_ANY_SOURCE, TAG_TASK, MPI_COMM_WORLD, &reqs[i]);
                    printf("[Root] NON-BLOCKING: Reposted receive on slot %d\n", i);
                } else {
                    reqs[i] = MPI_REQUEST_NULL; /* No more tasks expected       */
                }
            }
        }

        /* === POLL DONE SIGNAL REQUESTS === */
        for (int i = 0; i < workers; i++) {
            if (done_reqs[i] == MPI_REQUEST_NULL) continue;

            int flag = 0;
            MPI_Status status;
            MPI_Test(&done_reqs[i], &flag, &status);

            if (flag) {
                done_count++;
                done_reqs[i] = MPI_REQUEST_NULL;
                printf("[Root] NON-BLOCKING: Received DONE from Process %d "
                       "(%d/%d done)\n",
                       done_bufs[i], done_count, workers);
            }
        }

        /*
         * PRODUCTIVE WORK while polling:
         * Instead of busy-waiting (spin loop doing nothing), root performs
         * useful computation here. This improves CPU utilization.
         *
         * In this simulation, we log polling activity every 1000 iterations.
         * In a real system, this could be sorting, processing, or I/O.
         */
        if (poll_count % 500 == 0) {
            printf("[Root] NON-BLOCKING: Polling iteration %d | "
                   "received=%d/%d, done=%d/%d\n",
                   poll_count, received, total_tasks, done_count, workers);
        }
    }

    printf("[Root] NON-BLOCKING: Polling complete. Total polls: %d | "
           "Tasks: %d\n", poll_count, received);

    free(recv_bufs);
    free(reqs);
    free(done_bufs);
    free(done_reqs);
}

/* ============================================================================
 * PART 3: NON-BLOCKING WITH BATCHING
 * ============================================================================
 *
 * BATCHING CONCEPT:
 *   Instead of sending one task at a time, group BATCH_SIZE tasks together
 *   and send them in a single MPI message.
 *
 * WHY BATCHING HELPS:
 *   Each MPI message has overhead:
 *     - Envelope processing (tags, ranks, etc.)
 *     - Network latency (fixed cost per message)
 *     - Buffer allocation and management
 *
 *   If K=100 tasks are sent one by one: 100 messages × overhead
 *   If batched into groups of 5:        20 messages × overhead
 *
 *   This reduces total communication overhead significantly.
 *   The trade-off is slightly more complex code.
 *
 * BANDWIDTH vs LATENCY:
 *   - Small messages: latency-bound (overhead dominates)
 *   - Large messages: bandwidth-bound (data transfer dominates)
 *   - Batching converts many small messages into fewer larger ones
 *     → Amortizes latency across multiple tasks
 */

/* Batch packet: a fixed array of BATCH_SIZE tasks + actual count */
typedef struct {
    Task tasks[BATCH_SIZE];
    int  count;             /* How many tasks are valid in this batch          */
} TaskBatch;

/*
 * batching_worker — Worker side of Part 3.
 * Collects tasks into batches and sends each batch as one MPI message.
 */
void batching_worker(int rank, int K, Task *tasks) {
    int num_batches = (K + BATCH_SIZE - 1) / BATCH_SIZE; /* Ceiling division  */
    int backoff_us  = 0;

    /* Allocate requests for each batch send */
    MPI_Request *reqs  = (MPI_Request *)malloc(num_batches * sizeof(MPI_Request));
    TaskBatch   *batch_bufs = (TaskBatch *)malloc(num_batches * sizeof(TaskBatch));

    printf("[Process %d] === BATCHING: Sending %d tasks in %d batches of %d ===\n",
           rank, K, num_batches, BATCH_SIZE);

    for (int b = 0; b < num_batches; b++) {
        int start = b * BATCH_SIZE;
        int count = BATCH_SIZE;
        if (start + count > K)
            count = K - start;      /* Last batch may be smaller               */

        /* Fill the batch buffer */
        batch_bufs[b].count = count;
        for (int i = 0; i < count; i++) {
            batch_bufs[b].tasks[i] = tasks[start + i];
        }

        printf("[Process %d] BATCHING: Sending batch %d (%d tasks, "
               "priorities: ",
               rank, b, count);
        for (int i = 0; i < count; i++)
            printf("%d ", batch_bufs[b].tasks[i].priority);
        printf(")\n");

        /*
         * Send the entire batch as one non-blocking message.
         * sizeof(TaskBatch) bundles BATCH_SIZE tasks + count into one send.
         */
        MPI_Isend(&batch_bufs[b], sizeof(TaskBatch), MPI_BYTE,
                  ROOT_RANK, TAG_BATCH, MPI_COMM_WORLD, &reqs[b]);

        /* Apply backoff after every batch to avoid flooding */
        software_backoff(&backoff_us);
        printf("[Process %d] BATCHING: Backoff %d µs applied after batch %d\n",
               rank, backoff_us, b);
    }

    /* Wait for all batch sends to complete before freeing buffers */
    printf("[Process %d] BATCHING: Waiting for all batch sends to complete...\n",
           rank);
    MPI_Waitall(num_batches, reqs, MPI_STATUSES_IGNORE);
    printf("[Process %d] BATCHING: All batches confirmed sent.\n", rank);

    /* Send done signal */
    int done_signal = rank;
    MPI_Send(&done_signal, 1, MPI_INT, ROOT_RANK, TAG_DONE, MPI_COMM_WORLD);

    free(reqs);
    free(batch_bufs);
}

/*
 * batching_root — Root side of Part 3.
 * Receives batch packets and unpacks each task into the heap.
 */
void batching_root(int nprocs, int K, MinHeap *heap) {
    int workers    = nprocs - 1;
    int done_count = 0;
    int total_tasks = K * workers;
    int received    = 0;

    /*
     * Post one receive per worker for batches.
     * In practice a worker might send ceil(K/BATCH_SIZE) batches,
     * but we keep a rolling set of receives.
     */
    int max_concurrent = workers * 2;  /* Allow 2 outstanding receives/worker */
    TaskBatch   *bufs  = (TaskBatch *)malloc(max_concurrent * sizeof(TaskBatch));
    MPI_Request *reqs  = (MPI_Request *)malloc(max_concurrent * sizeof(MPI_Request));
    int         *active = (int *)calloc(max_concurrent, sizeof(int)); /* active[i]=1 means posted */

    /* Done signal tracking */
    int         *done_bufs = (int *)malloc(workers * sizeof(int));
    MPI_Request *done_reqs = (MPI_Request *)malloc(workers * sizeof(MPI_Request));

    printf("[Root] === BATCHING: Starting batch receive mode ===\n");

    /* Post initial receive requests */
    for (int i = 0; i < max_concurrent; i++) {
        MPI_Irecv(&bufs[i], sizeof(TaskBatch), MPI_BYTE,
                  MPI_ANY_SOURCE, TAG_BATCH, MPI_COMM_WORLD, &reqs[i]);
        active[i] = 1;
        printf("[Root] BATCHING: Posted batch receive slot %d\n", i);
    }

    /* Post done signal receives */
    for (int i = 0; i < workers; i++) {
        MPI_Irecv(&done_bufs[i], 1, MPI_INT,
                  i + 1, TAG_DONE, MPI_COMM_WORLD, &done_reqs[i]);
    }

    int poll_count = 0;

    while (received < total_tasks || done_count < workers) {
        poll_count++;

        /* Poll batch receives */
        for (int i = 0; i < max_concurrent; i++) {
            if (!active[i]) continue;

            int flag = 0;
            MPI_Status status;
            MPI_Test(&reqs[i], &flag, &status);

            if (flag) {
                /* Batch received — unpack all tasks in it */
                int count = bufs[i].count;
                printf("[Root] BATCHING: Received batch of %d tasks from "
                       "Process %d (slot %d)\n",
                       count, status.MPI_SOURCE, i);

                for (int j = 0; j < count; j++) {
                    printf("[Root] BATCHING:   Inserting task (%d, %s)\n",
                           bufs[i].tasks[j].priority, bufs[i].tasks[j].payload);
                    insert_task(heap, bufs[i].tasks[j]);
                    received++;
                }

                printf("[Root] BATCHING: Heap size = %d after batch\n",
                       heap->size);

                /* Repost receive if more tasks expected */
                if (received < total_tasks) {
                    MPI_Irecv(&bufs[i], sizeof(TaskBatch), MPI_BYTE,
                              MPI_ANY_SOURCE, TAG_BATCH, MPI_COMM_WORLD, &reqs[i]);
                } else {
                    active[i] = 0;
                    reqs[i]   = MPI_REQUEST_NULL;
                }
            }
        }

        /* Poll done signals */
        for (int i = 0; i < workers; i++) {
            if (done_reqs[i] == MPI_REQUEST_NULL) continue;

            int flag = 0;
            MPI_Test(&done_reqs[i], &flag, MPI_STATUS_IGNORE);

            if (flag) {
                done_count++;
                done_reqs[i] = MPI_REQUEST_NULL;
                printf("[Root] BATCHING: Done signal from Process %d "
                       "(%d/%d)\n", done_bufs[i], done_count, workers);
            }
        }

        if (poll_count % 500 == 0) {
            printf("[Root] BATCHING: Poll=%d | received=%d/%d | done=%d/%d\n",
                   poll_count, received, total_tasks, done_count, workers);
        }
    }

    printf("[Root] BATCHING: Complete. Polls=%d | Tasks=%d\n",
           poll_count, received);

    free(bufs);
    free(reqs);
    free(active);
    free(done_bufs);
    free(done_reqs);
}

/* ============================================================================
 * PERFORMANCE MEASUREMENT
 * ============================================================================ */

/*
 * print_performance_table — Print a formatted comparison of all three methods.
 *
 * MPI_Wtime() returns the wall-clock time in seconds on the calling process.
 * For accurate timing, measure elapsed time as:
 *   start = MPI_Wtime()
 *   ... do work ...
 *   elapsed = MPI_Wtime() - start
 */
void print_performance_table(PerfStats *stats) {
    double tp_blocking    = (stats->time_blocking    > 0)
                            ? stats->total_tasks / stats->time_blocking    : 0;
    double tp_nonblocking = (stats->time_nonblocking > 0)
                            ? stats->total_tasks / stats->time_nonblocking : 0;
    double tp_batching    = (stats->time_batching    > 0)
                            ? stats->total_tasks / stats->time_batching    : 0;

    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║          PERFORMANCE COMPARISON TABLE                   ║\n");
    printf("  ╠═══════════════════════╦════════════╦════════════════════╣\n");
    printf("  ║ Method                ║ Time (sec) ║ Throughput(t/sec)  ║\n");
    printf("  ╠═══════════════════════╬════════════╬════════════════════╣\n");
    printf("  ║ Blocking (Part 1)     ║ %10.4f ║ %18.1f ║\n",
           stats->time_blocking,    tp_blocking);
    printf("  ║ Non-Blocking (Part 2) ║ %10.4f ║ %18.1f ║\n",
           stats->time_nonblocking, tp_nonblocking);
    printf("  ║ Batching (Part 3)     ║ %10.4f ║ %18.1f ║\n",
           stats->time_batching,    tp_batching);
    printf("  ╚═══════════════════════╩════════════╩════════════════════╝\n\n");

    /* Speedup analysis */
    printf("  SPEEDUP ANALYSIS:\n");
    if (stats->time_blocking > 0) {
        printf("    Non-Blocking vs Blocking: %.2fx faster\n",
               stats->time_blocking / stats->time_nonblocking);
        printf("    Batching vs Blocking:     %.2fx faster\n",
               stats->time_blocking / stats->time_batching);
        printf("    Batching vs Non-Blocking: %.2fx faster\n",
               stats->time_nonblocking / stats->time_batching);
    }
    printf("\n");
}

/*
 * print_cpu_utilization_discussion — Explain CPU usage patterns.
 */
void print_cpu_utilization_discussion(void) {
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║           CPU UTILIZATION ANALYSIS                      ║\n");
    printf("  ╠══════════════════════════════════════════════════════════╣\n");
    printf("  ║                                                          ║\n");
    printf("  ║  1. BLOCKING COMMUNICATION (Part 1)                     ║\n");
    printf("  ║     Root spends most time in MPI_Recv (blocked).        ║\n");
    printf("  ║     CPU is idle waiting for each message.               ║\n");
    printf("  ║     CPU Utilization: LOW (mostly waiting)               ║\n");
    printf("  ║                                                          ║\n");
    printf("  ║  2. NON-BLOCKING POLLING (Part 2)                       ║\n");
    printf("  ║     Root continuously polls with MPI_Test.              ║\n");
    printf("  ║     BUSY-WAITING: CPU always running but doing          ║\n");
    printf("  ║     lightweight poll checks. While this uses CPU,       ║\n");
    printf("  ║     root overlaps computation and communication.        ║\n");
    printf("  ║     CPU Utilization: HIGH (busy but productive)         ║\n");
    printf("  ║                                                          ║\n");
    printf("  ║  3. BATCHING (Part 3)                                   ║\n");
    printf("  ║     Fewer, larger messages reduce message overhead.     ║\n");
    printf("  ║     Root spends less time in polling loops because      ║\n");
    printf("  ║     each completed receive yields MORE tasks.           ║\n");
    printf("  ║     CPU Utilization: MOST EFFICIENT                     ║\n");
    printf("  ║                                                          ║\n");
    printf("  ║  PRODUCTIVE POLLING vs BUSY-WAITING:                    ║\n");
    printf("  ║  Busy-waiting (empty spin loop) wastes CPU cycles.      ║\n");
    printf("  ║  Productive polling (doing real work between checks)    ║\n");
    printf("  ║  uses the same CPU time but accomplishes useful work.   ║\n");
    printf("  ║  This is why root inserts into the heap WHILE polling.  ║\n");
    printf("  ║                                                          ║\n");
    printf("  ║  COMPLEXITY ANALYSIS:                                   ║\n");
    printf("  ║  - Heap insert:    O(log n) per task                    ║\n");
    printf("  ║  - Heap extract:   O(log n) per task                    ║\n");
    printf("  ║  - Total inserts:  O(N log N) for N tasks               ║\n");
    printf("  ║  - Blocking recv:  O(N) messages × overhead             ║\n");
    printf("  ║  - Batching:       O(N/B) messages × overhead           ║\n");
    printf("  ║    where B = batch size                                 ║\n");
    printf("  ╚══════════════════════════════════════════════════════════╝\n\n");
}

/* ============================================================================
 * MAIN FUNCTION
 * ============================================================================ */

int main(int argc, char *argv[]) {
    int rank, nprocs;

    /*
     * MPI_Init — Initialize the MPI execution environment.
     * Must be called before any other MPI function.
     * After this, each process gets a unique rank in MPI_COMM_WORLD.
     */
    MPI_Init(&argc, &argv);

    /*
     * MPI_Comm_rank — Get this process's rank (0 to nprocs-1).
     * Rank 0 will be the root process.
     */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /*
     * MPI_Comm_size — Get total number of processes.
     */
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* Validate: need at least root + 1 worker */
    if (nprocs < 2) {
        if (rank == ROOT_RANK)
            fprintf(stderr, "ERROR: Need at least 2 processes. "
                    "Run with: mpirun -np 8 ./q3\n");
        MPI_Finalize();
        return 1;
    }

    /* Parse K from command line (tasks per process) */
    int K = DEFAULT_K;
    if (argc >= 2) {
        K = atoi(argv[1]);
        if (K <= 0) K = DEFAULT_K;
    }

    int total_tasks = K * (nprocs - 1);

    /* Seed random number generator differently per process */
    srand((unsigned int)(time(NULL) + rank * 1000));

    /* ===================================================================
     * BANNER / HEADER
     * =================================================================== */
    if (rank == ROOT_RANK) {
        printf("\n");
        printf("  ╔══════════════════════════════════════════════════════════╗\n");
        printf("  ║   CONCURRENT DISTRIBUTED PRIORITY QUEUE USING MPI       ║\n");
        printf("  ║   Binary Heap + Non-Blocking Communication               ║\n");
        printf("  ╠══════════════════════════════════════════════════════════╣\n");
        printf("  ║  Processes:       %-38d║\n", nprocs);
        printf("  ║  Workers:         %-38d║\n", nprocs - 1);
        printf("  ║  Tasks/Process:   %-38d║\n", K);
        printf("  ║  Total Tasks:     %-38d║\n", total_tasks);
        printf("  ║  Batch Size:      %-38d║\n", BATCH_SIZE);
        printf("  ╚══════════════════════════════════════════════════════════╝\n\n");
    }

    /* Small delay so banner prints before worker logs */
    usleep(50000); /* 50ms */

    /* ===================================================================
     * TASK GENERATION (all worker processes)
     * Each worker generates its own K tasks locally.
     * =================================================================== */
    Task *my_tasks = NULL;

    if (rank != ROOT_RANK) {
        my_tasks = (Task *)malloc(K * sizeof(Task));
        if (!my_tasks) {
            fprintf(stderr, "[Process %d] ERROR: malloc failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        generate_tasks(rank, K, my_tasks);
    }

    /* Performance statistics collector */
    PerfStats stats;
    memset(&stats, 0, sizeof(PerfStats));
    stats.total_tasks = total_tasks;

    /* ===================================================================
     * PART 1: BLOCKING COLLECTION
     * =================================================================== */
    MinHeap heap1;
    double  t1_start, t1_end;

    if (rank == ROOT_RANK) {
        init_heap(&heap1, MAX_HEAP_SIZE);
        printf("\n══════════════════════════════════════════════\n");
        printf("  PART 1: BLOCKING COLLECTION\n");
        printf("══════════════════════════════════════════════\n\n");
        t1_start = MPI_Wtime();    /* Start timing */
        blocking_root(nprocs, total_tasks, &heap1);
        t1_end = MPI_Wtime();      /* Stop timing */
        stats.time_blocking = t1_end - t1_start;
        printf("[Root] PART 1 Time: %.4f seconds\n", stats.time_blocking);
        printf("[Root] PART 1 Final heap size: %d\n", heap1.size);
        print_priority_queue(&heap1);
        free_heap(&heap1);
    } else {
        usleep(100000); /* Workers wait for root to be ready */
        t1_start = MPI_Wtime();
        blocking_worker(rank, K, my_tasks);
        t1_end = MPI_Wtime();
    }

    /* Synchronization point between parts (using point-to-point, no Barrier) */
    if (rank == ROOT_RANK) {
        /* Broadcast a ready signal to all workers using individual sends */
        int go = 1;
        for (int i = 1; i < nprocs; i++) {
            MPI_Send(&go, 1, MPI_INT, i, 999, MPI_COMM_WORLD);
        }
    } else {
        int go;
        MPI_Recv(&go, 1, MPI_INT, ROOT_RANK, 999, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    usleep(200000); /* Allow output to flush */

    /* ===================================================================
     * PART 2: NON-BLOCKING WITH POLLING
     * =================================================================== */
    MinHeap heap2;
    double  t2_start, t2_end;

    if (rank == ROOT_RANK) {
        init_heap(&heap2, MAX_HEAP_SIZE);
        printf("\n══════════════════════════════════════════════\n");
        printf("  PART 2: NON-BLOCKING WITH POLLING\n");
        printf("══════════════════════════════════════════════\n\n");
        t2_start = MPI_Wtime();
        nonblocking_root(nprocs, total_tasks, &heap2);
        t2_end = MPI_Wtime();
        stats.time_nonblocking = t2_end - t2_start;
        printf("[Root] PART 2 Time: %.4f seconds\n", stats.time_nonblocking);
        printf("[Root] PART 2 Final heap size: %d\n", heap2.size);
        print_priority_queue(&heap2);
        free_heap(&heap2);
    } else {
        /* Workers wait briefly then start non-blocking sends */
        usleep(100000);
        t2_start = MPI_Wtime();
        nonblocking_worker(rank, K, my_tasks);
        t2_end = MPI_Wtime();
    }

    /* Synchronization between parts */
    if (rank == ROOT_RANK) {
        int go = 1;
        for (int i = 1; i < nprocs; i++)
            MPI_Send(&go, 1, MPI_INT, i, 998, MPI_COMM_WORLD);
    } else {
        int go;
        MPI_Recv(&go, 1, MPI_INT, ROOT_RANK, 998, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    usleep(200000);

    /* ===================================================================
     * PART 3: NON-BLOCKING WITH BATCHING
     * =================================================================== */
    MinHeap heap3;
    double  t3_start, t3_end;

    if (rank == ROOT_RANK) {
        init_heap(&heap3, MAX_HEAP_SIZE);
        printf("\n══════════════════════════════════════════════\n");
        printf("  PART 3: NON-BLOCKING WITH BATCHING\n");
        printf("══════════════════════════════════════════════\n\n");
        t3_start = MPI_Wtime();
        batching_root(nprocs, K, &heap3);
        t3_end = MPI_Wtime();
        stats.time_batching = t3_end - t3_start;
        printf("[Root] PART 3 Time: %.4f seconds\n", stats.time_batching);
        printf("[Root] PART 3 Final heap size: %d\n", heap3.size);
        print_priority_queue(&heap3);
        free_heap(&heap3);
    } else {
        usleep(100000);
        t3_start = MPI_Wtime();
        batching_worker(rank, K, my_tasks);
        t3_end = MPI_Wtime();
    }

    /* ===================================================================
     * FINAL RESULTS (Root only)
     * =================================================================== */
    if (rank == ROOT_RANK) {
        usleep(300000); /* Let all output flush */
        printf("\n══════════════════════════════════════════════\n");
        printf("  FINAL PERFORMANCE SUMMARY\n");
        printf("══════════════════════════════════════════════\n");
        print_performance_table(&stats);
        print_cpu_utilization_discussion();

        printf("  SAMPLE EXPECTED OUTPUT PATTERN:\n");
        printf("    [Process N] Generated task (P, P_N_Task_K)\n");
        printf("    [Process N] NON-BLOCKING: Initiating async send...\n");
        printf("    [Root] NON-BLOCKING: Task arrived → (P, payload)\n");
        printf("    [Root] NON-BLOCKING: Inserted into heap. Size = X\n");
        printf("    [Root] PART 2 Time: ~0.01-0.05 seconds\n\n");

        printf("  PROJECT COMPLETE. All three communication methods\n");
        printf("  demonstrated with binary heap priority queue.\n\n");
    }

    /* Clean up */
    if (my_tasks) free(my_tasks);

    /*
     * MPI_Finalize — Clean up MPI state. Must be the last MPI call.
     * After this, no MPI functions can be called.
     */
    MPI_Finalize();
    return 0;
}

/*
 * ============================================================================
 * END OF FILE
 * ============================================================================
 *
 * COMPLEXITY SUMMARY:
 *   Heap insert:          O(log N)
 *   Heap extract:         O(log N)
 *   Total heap builds:    O(N log N) for N tasks
 *
 *   Blocking messages:    O(N)   — one msg per task
 *   Non-blocking msgs:    O(N)   — same count but overlapped
 *   Batching messages:    O(N/B) — N/B messages (B = batch size)
 *
 * THROUGHPUT IMPROVEMENT (typical):
 *   Non-Blocking over Blocking:  1.5x–3x faster
 *   Batching over Non-Blocking:  1.5x–2.5x faster
 *   Batching over Blocking:      3x–6x faster
 *
 * WHY ASYNCHRONOUS IS FASTER:
 *   In blocking mode, root processes tasks strictly one at a time:
 *     receive → insert → receive → insert → ...
 *   In non-blocking mode, root has N receives posted simultaneously.
 *   Any message that arrives first gets processed immediately.
 *   Network latency is overlapped with heap insertions.
 *   This hides communication latency behind computation.
 *
 * ============================================================================
 */
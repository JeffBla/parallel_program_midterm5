#include <mpi.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <vector>

#define NUM_NODE 30
#define swap(x, y, tmp) \
    {                   \
        tmp = x;        \
        x = y;          \
        y = tmp;        \
    }

using namespace std;

typedef pair<double, int> pdi;

long long arr_x[NUM_NODE + 1] = {0};  // there is no index zero
long long arr_y[NUM_NODE + 1] = {0};  // there is no index zero

double calCost(int u, int v) {
    double x_pow = pow(arr_x[u] - arr_x[v], 2);
    double y_pow = pow(arr_y[u] - arr_y[v], 2);
    double dist = sqrt(x_pow + y_pow);
    return floor(dist * 10000) / 10000.0;
}

double PrimMST(int V, vector<int> &Ps) {
    // Create an adjacency list representation of the graph
    vector<pair<int, double>> adj[NUM_NODE + 1];

    // Fill the adjacency list with edges and their weights
    int u;
    for (int i = 0; i < V; i++) {
        for (int j = i + 1; j < V; j++) {
            u = Ps[i];
            int v = Ps[j];
            double wt = calCost(u, v);
            adj[u].push_back({v, wt});
            adj[v].push_back({u, wt});
        }
    }
    // Create a priority queue to store edges with their weights
    priority_queue<pdi, vector<pdi>, greater<pdi>> pq;

    // Create a visited array to keep track of visited vertices
    vector<bool> visited(NUM_NODE + 1, false);

    // Variable to store the result (sum of edge weights)
    double result = 0;

    // Start with vertex 1
    pq.push({0, u});

    // Perform Prim's algorithm to find the Minimum Spanning Tree
    while (!pq.empty()) {
        auto p = pq.top();
        pq.pop();

        double wt = p.first;  // Weight of the edge
        u = p.second;         // Vertex connected to the edge

        if (visited[u] == true) {
            continue;  // Skip if the vertex is already visited
        }

        result += wt;       // Add the edge weight to the result
        visited[u] = true;  // Mark the vertex as visited

        // Explore the adjacent vertices
        for (auto v : adj[u]) {
            // v[0] represents the vertex and v[1] represents the edge weight
            if (visited[v.first] == false) {
                pq.push({v.second, v.first});  // Add the adjacent edge to the priority queue
            }
        }
    }

    return result;  // Return the sum of edge weights of the Minimum Spanning Tree
}

double Combination(int num_need, vector<int> &Ps_not_wall, int num_p_not_wall,
                   vector<int> &wall_Ps, int num_wall_p, int left, vector<bool> &record) {
    if (num_need == 0) {
        vector<int> all_picked_Ps(wall_Ps);
        for (int i = 0; i < record.size(); i++) {
            if (record[i]) {
                all_picked_Ps.push_back(Ps_not_wall[i]);
            }
        }
        double cost = PrimMST(all_picked_Ps.size(), all_picked_Ps);
        return cost;
    } else if (left == num_p_not_wall) {
        return INT32_MAX;
    }

    double minest = INT32_MAX;
    for (int i = left; i < num_p_not_wall; i++) {
        record[i] = true;
        minest = min(minest, Combination(num_need - 1, Ps_not_wall, num_p_not_wall, wall_Ps, num_wall_p, i + 1, record));
        record[i] = false;
    }
    return minest;
}

int partition(long long arr_x[], long long arr_y[], int sorted_i_arr[], int left, int right) {
    long long pivot_x = arr_x[sorted_i_arr[right]];
    long long pivot_y = arr_y[sorted_i_arr[right]];
    int i = left - 1, tmp;
    long long target_x;
    for (int j = left; j != right; j++) {
        target_x = arr_x[sorted_i_arr[j]];
        if (target_x < pivot_x) {
            i++;
            swap(sorted_i_arr[j], sorted_i_arr[i], tmp);
        } else if (target_x == pivot_x) {
            if (arr_y[sorted_i_arr[j]] < pivot_y) {
                i++;
                swap(sorted_i_arr[j], sorted_i_arr[i], tmp);
            }
        }
    }
    swap(sorted_i_arr[right], sorted_i_arr[i + 1], tmp);
    return i + 1;
}

// ascend x, ascend y
void quicksort(long long arr_x[], long long arr_y[], int sorted_i_arr[], int left, int right) {
    if (left < right) {
        int middle = partition(arr_x, arr_y, sorted_i_arr, left, right);
        quicksort(arr_x, arr_y, sorted_i_arr, left, middle - 1);
        quicksort(arr_x, arr_y, sorted_i_arr, middle + 1, right);
    }
}

// O(x0, y0), X(x1, y1), Y(x2, y2)
double cross(long long x0, long long y0, long long x1, long long y1, long long x2, long long y2) {
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
}

int AndrewMonotoneChain(long long arr_x[], long long arr_y[], int sorted_i_arr[], int result_i_arr[], int i_len) {
    int l = 0, u = 0;
    int i_upper_arr[NUM_NODE] = {0}, i_lower_arr[NUM_NODE] = {0};
    for (int i = 0; i < i_len && sorted_i_arr[i] != 0; i++) {
        while (u >= 2 && (cross(arr_x[i_upper_arr[u - 2]], arr_y[i_upper_arr[u - 2]],
                                arr_x[i_upper_arr[u - 1]], arr_y[i_upper_arr[u - 1]],
                                arr_x[sorted_i_arr[i]], arr_y[sorted_i_arr[i]]) < 0))
            u--;
        while (l >= 2 && (cross(arr_x[i_lower_arr[l - 2]], arr_y[i_lower_arr[l - 2]],
                                arr_x[i_lower_arr[l - 1]], arr_y[i_lower_arr[l - 1]],
                                arr_x[sorted_i_arr[i]], arr_y[sorted_i_arr[i]]) > 0))
            l--;
        i_upper_arr[u++] = sorted_i_arr[i];
        i_lower_arr[l++] = sorted_i_arr[i];
    }

    int i = 0;
    while (i < l) {
        result_i_arr[i] = i_lower_arr[i];
        i++;
    }
    // the start and end is the same, so don't store
    int j = u - 2;
    while (j > 0) {
        result_i_arr[i] = i_upper_arr[j];
        i++;
        j--;
    }
    // handle edge problem
    if (l == 0)  // both l & u are 0
        return 0;
    else if (l == 1)
        return 1;
    else
        return l + u - 2;  // the number of point in result arr
}

int main(int argc, char *argv[]) {
    int myid, numprocs;
    double startwtime = 0.0, endwtime;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int sorted_i_arr[NUM_NODE] = {0};
    int n;
    if (myid == 0) {
        char filename[50];
        scanf("%s", filename);
        FILE *f = fopen(filename, "r");

        char buffer[1000000];
        char *ptr;
        fscanf(f, "%d ", &n);
        for (int i = 0; i < n; i++) {  // from 1~12000
            fgets(buffer, 1000000, f);
            ptr = strtok(buffer, " ");
            arr_x[i + 1] = atoi(ptr);
            ptr = strtok(NULL, " \r\n");
            arr_y[i + 1] = atoi(ptr);

            sorted_i_arr[i] = i + 1;
        }

        quicksort(arr_x, arr_y, sorted_i_arr, 0, n - 1);
    }
    MPI_Bcast(arr_x, NUM_NODE + 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(arr_y, NUM_NODE + 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int elements_per_proc = n / numprocs;
    int remainder = n % numprocs;
    if (remainder != 0)
        elements_per_proc++;

    int sub_sorted_i_arr[elements_per_proc];
    memset(sub_sorted_i_arr, 0, sizeof(int) * elements_per_proc);

    int sub_result_i_arr[elements_per_proc];
    memset(sub_result_i_arr, 0, sizeof(int) * elements_per_proc);

    MPI_Scatter(sorted_i_arr, elements_per_proc, MPI_INT,
                sub_sorted_i_arr, elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    AndrewMonotoneChain(arr_x, arr_y, sub_sorted_i_arr, sub_result_i_arr, elements_per_proc);

    int *gather_i_arr = NULL;
    if (myid == 0) {
        gather_i_arr = (int *)malloc(sizeof(int) * NUM_NODE);
        memset(gather_i_arr, 0, sizeof(int) * NUM_NODE);
    }
    MPI_Gather(sub_result_i_arr, elements_per_proc, MPI_INT, gather_i_arr,
               elements_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

    int fin_len;
    vector<int> wall_Ps;
    int num_p_not_wall;
    vector<int> Ps_not_wall;
    if (myid == 0) {
        // remove 0 from gather_i_arr and cnt num of point
        int fin_sort_i_arr[NUM_NODE];
        int len_fin_sort = 0;
        for (int i = 0; i < NUM_NODE; i++) {
            if (gather_i_arr[i] != 0) {
                fin_sort_i_arr[len_fin_sort++] = gather_i_arr[i];
            }
        }
        quicksort(arr_x, arr_y, fin_sort_i_arr, 0, len_fin_sort - 1);
        int final_i_arr[len_fin_sort];
        fin_len = AndrewMonotoneChain(arr_x, arr_y, fin_sort_i_arr, final_i_arr, len_fin_sort);

        wall_Ps.assign(final_i_arr, final_i_arr + fin_len);
        for (auto p : sorted_i_arr) {
            if (p == 0)
                break;

            bool notInWall = true;
            for (auto wall_p : wall_Ps) {
                if (p == wall_p) {
                    notInWall = false;
                    break;
                }
            }
            if (notInWall) {
                Ps_not_wall.push_back(p);
                num_p_not_wall++;
            }
        }
    }
    MPI_Bcast(&num_p_not_wall, 1, MPI_INT, 0, MPI_COMM_WORLD);
    Ps_not_wall.resize(num_p_not_wall);
    MPI_Bcast(Ps_not_wall.data(), num_p_not_wall, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&fin_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    wall_Ps.resize(fin_len);
    MPI_Bcast(wall_Ps.data(), fin_len, MPI_INT, 0, MPI_COMM_WORLD);

    vector<bool> record(num_p_not_wall, false);
    double min_num = INT32_MAX;
    for (int i = myid; i <= num_p_not_wall; i += numprocs) {
        min_num = min(min_num, Combination(i, Ps_not_wall, num_p_not_wall, wall_Ps, fin_len, 0, record));
    }

    double minest;
    MPI_Reduce(&min_num, &minest, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        printf("%.4lf", minest);
    }

    MPI_Finalize();

    free(gather_i_arr);
    gather_i_arr = NULL;
    return 0;
}
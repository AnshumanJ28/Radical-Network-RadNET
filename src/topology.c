#include "radnet.h"
#include <math.h>
#include <stdio.h>


int radnet_shell_node_count(int shell_index_1based) {
    double s = (double)shell_index_1based;
    double val = 2.0 * exp(0.3 * (s - 1.0));
    return (int)ceil(val);
}

void radnet_print_topology(int num_shells) {
    int total = 0;
    printf("Shell |  N(s)\n");
    printf("------+------\n");
    for (int s = 1; s <= num_shells; s++) {
        int n = radnet_shell_node_count(s);
        total += n;
        printf("%5d | %5d\n", s, n);
    }
    printf("------+------\n");
    printf("Total nodes per web: %d\n", total);
}


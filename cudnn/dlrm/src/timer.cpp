#include "timer.h"


int timerCnt = 0;
char timerName[MAXTIMER][128];
struct timespec startTime[MAXTIMER];
double totalTime[MAXTIMER];

void startTimer(const char *s) {
    for (int i = 0; i < timerCnt; i++) {
        if( strcmp(timerName[i], s) == 0 ){
            clock_gettime(CLOCK_REALTIME, &startTime[i]);
            return;
        }
    }
    strcpy(timerName[timerCnt], s);
    clock_gettime(CLOCK_REALTIME, &startTime[timerCnt]);
    timerCnt++;
}

double stopTimer(const char *s) {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    int idx = -1;
    
    for (int i = 0; i < timerCnt; i++) {
        if( strcmp(timerName[i], s) == 0 ) {
            idx = i; break;
        } 
    }

    double elapsed = (tp.tv_sec - startTime[idx].tv_sec) + (tp.tv_nsec - startTime[idx].tv_nsec) * 1e-9;
    totalTime[idx] += elapsed;

    return elapsed;
}

void printAllTimer() {
    for (int i = 0; i < timerCnt; i++) {
        fprintf(stdout, "[%s] %.3f ms\n", timerName[i], totalTime[i] * 1000);
    }
}

void printAllTimer(double N) {
    for (int i = 0; i < timerCnt; i++) {
        fprintf(stdout, "[%s] %.3f ms\n", timerName[i], totalTime[i] * 1000 / N);
    }
}

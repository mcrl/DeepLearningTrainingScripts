#ifndef _TIMER_H_
#define _TIMER_H_

#include <ctime>
#include <string.h>
#include <cstdio>

#define MAXTIMER 128

extern int timerCnt;
extern char timerName[MAXTIMER][128];
extern struct timespec startTime[MAXTIMER];
extern double totalTime[MAXTIMER];

void startTimer(const char *s);
double stopTimer(const char *s);

void printAllTimer();
void printAllTimer(double N);



#endif // _TIMER_H_
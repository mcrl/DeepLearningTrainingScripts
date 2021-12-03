//polybenchUtilFuncts.h
//Scott Grauer-Gray (sgrauerg@gmail.com)
//Functions used across hmpp codes

#ifndef POLYBENCH_UTIL_FUNCTS_H
#define POLYBENCH_UTIL_FUNCTS_H

//define a small float value
#define SMALL_FLOAT_VAL 0.00000001f

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


float absVal(float a)
{
	if(a < 0)
	{
		return (a * -1);
	}
   	else
	{ 
		return a;
	}
}



float percentDiff(double val1, double val2)
{
	if ((absVal(val1) < 0.01) && (absVal(val2) < 0.01))
	{
		return 0.0f;
	}

	else
	{
    		return 100.0f * (absVal(absVal(val1 - val2) / absVal(val1 + SMALL_FLOAT_VAL)));
	}
}

#include <time.h>

double measure_get_time() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

void measure_integrate_start() {
  FILE* f = popen("python -c 'from hioki3334 import hioki3334; dev = hioki3334(); dev.integrate_reset(); dev.integrate_start()'", "r");
  if (f == NULL) {
    printf("Failed to run command\n");
    exit(1);
  }
  pclose(f);
}

void measure_get_wh_time(double* wh, double* t) {
  FILE* f = popen("python -c 'from hioki3334 import hioki3334; dev = hioki3334(); wh, time = dev.measure_wh_time(); print(wh); print(time)'", "r");
  if (f == NULL) {
    printf("Failed to run command\n");
    exit(1);
  }
  int res = fscanf(f, "%lf%lf", wh, t);
  if (res != 2) {
    printf("Failed to read result\n");
    exit(1);
  }
  pclose(f);
}

static double st_high, st_low, swh;
static int iter;
static int disabled = 0;

void measure_start() {
  if (disabled) return;
  measure_integrate_start();
  measure_get_wh_time(&swh, &st_low);
  st_high = measure_get_time();
  iter = 0;
  printf("[%s:%d] measure_start time(high precision)=%f time(low precision)=%f energy=%f\n", __FILE__, __LINE__, st_high, st_low, swh);
}

int measure_continue() {
  if (disabled) return 0;
  ++iter;
  double et_high = measure_get_time();
  return et_high - st_high < MEASURE_TIME_THRESHOLD ? 1 : 0;
}

void measure_end() {
  if (disabled) return;
  double et_low, ewh;
  measure_get_wh_time(&ewh, &et_low);
  double et_high = measure_get_time();
  printf("[%s:%d] measure_end time(high precision)=%f time(low precision)=%f energy=%f iter=%d\n", __FILE__, __LINE__, et_high, et_low, ewh, iter);

  double tt = et_high - st_high;
  double atp = (ewh - swh) / (tt / 3600);
  double aip = MEASURE_IDLE_WATT;
  double adp = atp - aip;
  double t = tt / iter;
  double wh = adp * (t / 3600);
  printf("[%s:%d] Total Time(s) = %f\n", __FILE__, __LINE__, tt);
  printf("[%s:%d] Average Total Power(W) = %f\n", __FILE__, __LINE__, atp);
  printf("[%s:%d] Average Idle Power(W) = %f\n", __FILE__, __LINE__, aip);
  printf("[%s:%d] Average Device Power(W) = %f\n", __FILE__, __LINE__, adp);
  printf("[%s:%d] RESULT time(s) = %f\n", __FILE__, __LINE__, t);
  printf("[%s:%d] RESULT energy(Wh) = %f\n", __FILE__, __LINE__, wh);
}

void measure_disable() {
  disabled = 1;
}

#endif //POLYBENCH_UTIL_FUNCTS_H

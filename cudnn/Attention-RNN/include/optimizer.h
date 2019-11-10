#ifndef __OPTIMIZER_H__
#define __OPTIMIZER_H__

void optimizer_step(const void *dw, void *w, size_t nelem, size_t N);
#endif //__OPTIMIZER_H__

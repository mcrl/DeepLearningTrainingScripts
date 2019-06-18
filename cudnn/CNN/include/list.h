#ifndef _LIST_H_
#define _LIST_H_

#include <stdlib.h>
#include <stdbool.h>

typedef struct iterator {
  struct iterator *prev;
  struct iterator *next;
} iterator_t;

typedef struct {
  iterator_t head;
  iterator_t tail;
  size_t size;
} list_t;

void list_init(list_t *l);

bool list_empty(list_t *l);

void list_erase(list_t *l, iterator_t *it);

void list_push_back(list_t *l, iterator_t *it);

void list_push_front(list_t *l, iterator_t *it);

#define list_data(type, it) \
  ( (type *)((char *)&(it)->next - (size_t)&((type *)0)->iterator.next) )

#define list_iter(list, it) \
  for (iterator_t *it = (list)->head.next; it != &(list)->tail; it = it->next)

///////////////////////////////////////////////////////////////
//
//   struct data {
//     var member;
//     iterator_t iterator;
//   } item1, item2, item3;
//
//   list_t *l = (list_t *)malloc(sizeof(list_t));
//   list_init(l);
//
//   list_push_back(l, &item1->iterator);
//   list_push_back(l, &item2->iterator);
//   list_push_back(l, &item3->iterator);
//
//   list_erase(l, &item2->iterator);
//
//   list_iter(l, it) {
//     struct data *item = list_data(struct data, it);
//     job(item);
//   }
//
//   free(l);
//
///////////////////////////////////////////////////////////////

#endif // _LIST_H_

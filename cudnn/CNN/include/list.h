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

#define list_first(list, type, it_name) \
  ( (type *)((char *)&(list)->head.next->next - (size_t)&((type *)0)->it_name.next) )

#define list_last(list, type, it_name) \
  ( (type *)((char *)&(list)->tail.prev->next - (size_t)&((type *)0)->it_name.next) )

#define list_data(type, it_name) \
  ( (type *)((char *)&(__i)->next - (size_t)&((type *)0)->it_name.next) )

#define list_iter(list) \
  for (iterator_t *__i = (list)->head.next; __i != &(list)->tail; __i = __i->next)

///////////////////////////////////////////////////////////////
//
//   struct data {
//     void *payload;
//     iterator_t iterator;
//   } item1, item2, item3;
//
//   list_t *l = (list_t *)malloc(sizeof(list_t));
//   list_init(l);
//
//   /* must use distinct iterator for different list */
//   list_push_back(l, &item1->iterator);
//   list_push_back(l, &item2->iterator);
//   list_push_back(l, &item3->iterator);
//
//   list_erase(l, &item2->iterator);
//
//   list_iter(l) {
//     struct data *item = list_data(struct data, iterator);
//     job(item->payload);
//   }
//
//   free(l);
//
///////////////////////////////////////////////////////////////

#endif // _LIST_H_

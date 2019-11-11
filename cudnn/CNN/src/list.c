
#include "list.h"

void list_init(list_t *l)
{
  l->head.prev = NULL;
  l->head.next = &l->tail;
  l->tail.prev = &l->head;
  l->tail.next = NULL;
  l->size = 0;
}

bool list_empty(list_t *l)
{
  return l->size == 0;
}

void list_erase(list_t *l, iterator_t *it)
{
  it->prev->next = it->next;
  it->next->prev = it->prev;

  l->size--;
}

void list_push_back(list_t *l, iterator_t *it)
{
  iterator_t *em = l->tail.prev;

  it->next = em->next;
  it->prev = em;
  em->next->prev = it;
  em->next = it;

  l->size++;
}

void list_push_front(list_t *l, iterator_t *it)
{
  iterator_t *em = l->head.next;

  it->prev = em->prev;
  it->next = em;
  em->prev->next = it;
  em->prev = it;

  l->size++;
}
